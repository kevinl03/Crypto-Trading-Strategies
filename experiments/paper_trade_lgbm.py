"""
Live paper trading for the trained StatArb LightGBM model.

Tails a running `collect_statarb_data` JSONL run, rebuilds the same features
used in current `cex_gbm_new` training (z-score lags + ticker/orderbook/trades
+ cross-exchange / momentum / accel), and opens paper bets when |pred| >= entry
threshold.

Settles each bet after HORIZON snapshots against the realized future z-score
(the model's training target), logging DirAcc / PnL-proxy live.

Usage:
    python -m experiments.paper_trade_lgbm \\
        --model path/to/statarb_lgbm.txt \\
        --run-dir data/statarb/<run_id> \\
        --hours 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TOP_EXCHANGES = ["binance", "bybit", "okx", "coinbase", "kraken", "mexc"]
# Match current cex_gbm_new.ipynb / statarb/outputs/statarb_lgbm.txt
ZSCORE_WINDOW = 300
MIN_PERIODS = 90
N_LAGS = 3
HORIZON = 1
SIGNALS = ("spread_matrix", "ticker", "orderbook", "trades")
# Rotate output JSONL shards before they grow past this many lines.
JSONL_SHARD_MAX_LINES = 50_000


@dataclass
class Config:
    model_path: Path
    run_dir: Path
    output_dir: Path
    hours: float = 8.0
    entry_tau: float = 0.5
    poll_sec: float = 15.0
    max_open: int = 50
    fee_bps_per_leg: float = 4.0  # rough taker fee; informational only


@dataclass
class OpenBet:
    coin: str
    pair: str
    entry_snap: int
    exit_snap: int
    entry_ts: str
    direction: int
    pred: float
    entry_z: float
    entry_spread_bps: float


@dataclass
class State:
    file_offsets: Dict[str, int] = field(default_factory=dict)
    # snapshot -> list of flat records per signal
    spreads: Dict[int, List[dict]] = field(default_factory=lambda: defaultdict(list))
    tickers: Dict[int, List[dict]] = field(default_factory=lambda: defaultdict(list))
    orderbooks: Dict[int, List[dict]] = field(default_factory=lambda: defaultdict(list))
    trades: Dict[int, List[dict]] = field(default_factory=lambda: defaultdict(list))
    snap_ts: Dict[int, str] = field(default_factory=dict)
    known_snaps: Deque[int] = field(default_factory=deque)
    last_scored_snap: int = 0
    open_bets: List[OpenBet] = field(default_factory=list)
    closed: List[dict] = field(default_factory=list)
    signals: List[dict] = field(default_factory=list)
    n_preds: int = 0
    # How many in-memory rows have already been appended to disk shards.
    n_signals_flushed: int = 0
    n_closed_flushed: int = 0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def jsonl_shard_path(out_dir: Path, stem: str, part: int) -> Path:
    if part <= 0:
        return out_dir / f"{stem}.jsonl"
    return out_dir / f"{stem}_{part:03d}.jsonl"


def load_jsonl_writer_state(out_dir: Path) -> dict:
    path = out_dir / "jsonl_writer_state.json"
    if not path.exists():
        return {
            "signals_part": 0,
            "signals_lines": 0,
            "trades_part": 0,
            "trades_lines": 0,
        }
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {
            "signals_part": 0,
            "signals_lines": 0,
            "trades_part": 0,
            "trades_lines": 0,
        }


def save_jsonl_writer_state(out_dir: Path, meta: dict) -> None:
    (out_dir / "jsonl_writer_state.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


def append_jsonl_sharded(
    out_dir: Path,
    stem: str,
    rows: List[dict],
    part_key: str,
    lines_key: str,
    meta: dict,
    max_lines: int = JSONL_SHARD_MAX_LINES,
) -> None:
    """Append rows to rotating JSONL shards; never drop older lines."""
    if not rows:
        return
    part = int(meta.get(part_key, 0))
    lines = int(meta.get(lines_key, 0))
    idx = 0
    while idx < len(rows):
        path = jsonl_shard_path(out_dir, stem, part)
        room = max_lines - lines
        if room <= 0:
            part += 1
            lines = 0
            continue
        chunk = rows[idx : idx + room]
        with path.open("a", encoding="utf-8") as f:
            for row in chunk:
                f.write(json.dumps(row) + "\n")
        lines += len(chunk)
        idx += len(chunk)
        if lines >= max_lines:
            part += 1
            lines = 0
    meta[part_key] = part
    meta[lines_key] = lines


def list_jsonl_shards(out_dir: Path, stem: str) -> List[Path]:
    """Return shard files in write order: stem.jsonl, stem_001.jsonl, ..."""
    primary = out_dir / f"{stem}.jsonl"
    numbered = sorted(out_dir.glob(f"{stem}_*.jsonl"))
    out: List[Path] = []
    if primary.exists():
        out.append(primary)
    out.extend(numbered)
    return out


def signal_day_files(run_dir: Path, signal: str) -> List[Path]:
    d = run_dir / signal
    if not d.is_dir():
        return []
    return sorted(d.glob("*.jsonl"))


def ingest_new_lines(state: State, run_dir: Path) -> int:
    """Append newly written JSONL records into in-memory buffers. Returns #lines read."""
    n = 0
    for signal in SIGNALS:
        for path in signal_day_files(run_dir, signal):
            key = str(path)
            offset = state.file_offsets.get(key, 0)
            size = path.stat().st_size
            if size < offset:
                offset = 0  # file rotated/truncated
            if size == offset:
                continue
            with path.open("r", encoding="utf-8") as f:
                f.seek(offset)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    n += 1
                    _absorb(state, signal, rec)
                state.file_offsets[key] = f.tell()
    return n


def _absorb(state: State, signal: str, rec: dict) -> None:
    if rec.get("error"):
        return
    snap = rec.get("snapshot_idx")
    if not isinstance(snap, int):
        return
    ts = rec.get("ts") or utc_now()
    state.snap_ts[snap] = ts
    if not state.known_snaps or snap != state.known_snaps[-1]:
        if snap not in state.known_snaps:
            state.known_snaps.append(snap)

    if signal == "spread_matrix":
        coin = rec.get("coin")
        pairs = rec.get("pairwise_spreads") or []
        for p in pairs:
            ex1, ex2 = p.get("ex1"), p.get("ex2")
            if ex1 not in TOP_EXCHANGES or ex2 not in TOP_EXCHANGES:
                continue
            state.spreads[snap].append(
                {
                    "snapshot_idx": snap,
                    "coin": coin,
                    "exchange_a": ex1,
                    "exchange_b": ex2,
                    "spread_bps": float(p.get("spread_bps") or 0),
                    "p1": float(p.get("p1") or 0),
                    "p2": float(p.get("p2") or 0),
                    "window_id": "live",
                }
            )
    elif signal == "ticker":
        exch = rec.get("exchange")
        if exch not in TOP_EXCHANGES:
            return
        state.tickers[snap].append(
            {
                "snapshot_idx": snap,
                "coin": rec.get("coin"),
                "exchange": exch,
                "mid": float(rec.get("mid") or 0),
                "spread_bps": float(rec.get("spread_bps") or 0),
                "bid_volume": float(rec.get("bid_volume") or 0),
                "ask_volume": float(rec.get("ask_volume") or 0),
                "window_id": "live",
            }
        )
    elif signal == "orderbook":
        exch = rec.get("exchange")
        if exch not in TOP_EXCHANGES:
            return
        state.orderbooks[snap].append(
            {
                "snapshot_idx": snap,
                "coin": rec.get("coin"),
                "exchange": exch,
                "imbalance": float(rec.get("imbalance") or 0),
                "window_id": "live",
            }
        )
    elif signal == "trades":
        exch = rec.get("exchange")
        if exch not in TOP_EXCHANGES:
            return
        state.trades[snap].append(
            {
                "snapshot_idx": snap,
                "coin": rec.get("coin"),
                "exchange": exch,
                "buy_sell_ratio": float(rec.get("buy_sell_ratio") or 0),
                "total_volume": float(rec.get("total_volume") or 0),
                "window_id": "live",
            }
        )


def _to_wide(df: pd.DataFrame, index_cols, col_col, value_cols, prefix) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values(index_cols + [col_col]).reset_index(drop=True)
    lag_cols = []
    grp = df.groupby(["window_id", "coin", col_col])
    for col in value_cols:
        for lag in range(1, N_LAGS + 1):
            name = f"{col}_lag{lag}"
            df[name] = grp[col].transform(lambda x, l=lag: x.shift(l))
            lag_cols.append(name)
    df = df.drop(columns=value_cols, errors="ignore")
    df[lag_cols] = df[lag_cols].astype("float32")
    keep = index_cols + [col_col] + lag_cols
    wide = df[keep].groupby(index_cols + [col_col])[lag_cols].mean().unstack(col_col)
    wide.columns = [f"{prefix}{col}_{exch}" for col, exch in wide.columns]
    wide.columns.name = None
    return wide.reset_index()


def build_spread_features(sm: pd.DataFrame) -> pd.DataFrame:
    if sm.empty:
        return pd.DataFrame()
    sm = sm.copy()
    sm["pair"] = sm["exchange_a"] + "__" + sm["exchange_b"]
    sm = sm.sort_values(["window_id", "coin", "pair", "snapshot_idx"]).reset_index(drop=True)
    grp_keys = ["window_id", "coin", "pair"]
    grp = sm.groupby(grp_keys)["spread_bps"]
    roll_mean = grp.transform(lambda x: x.rolling(ZSCORE_WINDOW, min_periods=MIN_PERIODS).mean())
    roll_std = grp.transform(lambda x: x.rolling(ZSCORE_WINDOW, min_periods=MIN_PERIODS).std())
    sm["zscore"] = (sm["spread_bps"] - roll_mean) / roll_std.replace(0, np.nan)
    for lag in range(1, N_LAGS + 1):
        sm[f"spread_bps_lag{lag}"] = grp.transform(lambda x, l=lag: x.shift(l))
        sm[f"zscore_lag{lag}"] = sm.groupby(grp_keys)["zscore"].transform(lambda x, l=lag: x.shift(l))
    return sm.drop(columns=["exchange_a", "exchange_b", "p1", "p2"], errors="ignore")


def add_cross_exchange_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    exchanges = ["binance", "bybit", "okx", "coinbase", "kraken"]
    mid_cols = [f"tk_mid_lag1_{ex}" for ex in exchanges if f"tk_mid_lag1_{ex}" in df.columns]
    if mid_cols:
        df["cross_mid_std"] = df[mid_cols].std(axis=1)
        df["cross_mid_range"] = df[mid_cols].max(axis=1) - df[mid_cols].min(axis=1)
        ranks = df[mid_cols].rank(axis=1, pct=True)
        for col, ex in zip(mid_cols, [c.split("_")[-1] for c in mid_cols]):
            df[f"price_rank_{ex}"] = ranks[col]
    ba_cols = [f"tk_spread_bps_lag1_{ex}" for ex in exchanges if f"tk_spread_bps_lag1_{ex}" in df.columns]
    if ba_cols:
        df["cross_ba_std"] = df[ba_cols].std(axis=1)
        df["tightest_ba"] = df[ba_cols].min(axis=1)
        df["widest_ba"] = df[ba_cols].max(axis=1)
        df["ba_range"] = df["widest_ba"] - df["tightest_ba"]
    ob_cols = [f"ob_imbalance_lag1_{ex}" for ex in exchanges if f"ob_imbalance_lag1_{ex}" in df.columns]
    if ob_cols:
        df["cross_ob_std"] = df[ob_cols].std(axis=1)
        df["cross_ob_range"] = df[ob_cols].max(axis=1) - df[ob_cols].min(axis=1)
        df["net_ob_pressure"] = df[ob_cols].mean(axis=1)
    flow_cols = [
        f"tr_buy_sell_ratio_lag1_{ex}" for ex in exchanges if f"tr_buy_sell_ratio_lag1_{ex}" in df.columns
    ]
    if flow_cols:
        df["cross_flow_std"] = df[flow_cols].std(axis=1)
        df["net_flow_signal"] = df[flow_cols].mean(axis=1)
        df["flow_divergence"] = df[flow_cols].max(axis=1) - df[flow_cols].min(axis=1)
    # Spread / z-score momentum (notebook §6) — lag5 no-ops when N_LAGS < 5
    for lag_a, lag_b in [(1, 3), (1, 5)]:
        if f"spread_bps_lag{lag_a}" in df.columns and f"spread_bps_lag{lag_b}" in df.columns:
            df[f"spread_momentum_{lag_a}_{lag_b}"] = (
                df[f"spread_bps_lag{lag_a}"] - df[f"spread_bps_lag{lag_b}"]
            )
        if f"zscore_lag{lag_a}" in df.columns and f"zscore_lag{lag_b}" in df.columns:
            df[f"zscore_momentum_{lag_a}_{lag_b}"] = (
                df[f"zscore_lag{lag_a}"] - df[f"zscore_lag{lag_b}"]
            )
    # Z-score acceleration (notebook §7)
    if all(f"zscore_lag{lag}" in df.columns for lag in (1, 2, 3)):
        df["zscore_accel"] = df["zscore_lag1"] - 2 * df["zscore_lag2"] + df["zscore_lag3"]
    return df


def frames_for_window(state: State, snaps: Iterable[int]) -> Dict[str, pd.DataFrame]:
    snaps = list(snaps)
    sm = [r for s in snaps for r in state.spreads.get(s, [])]
    tk = [r for s in snaps for r in state.tickers.get(s, [])]
    ob = [r for s in snaps for r in state.orderbooks.get(s, [])]
    tr = [r for s in snaps for r in state.trades.get(s, [])]
    return {
        "spread_matrix": pd.DataFrame(sm),
        "ticker": pd.DataFrame(tk),
        "orderbook": pd.DataFrame(ob),
        "trades": pd.DataFrame(tr),
    }


def build_live_matrix(raw: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = build_spread_features(raw["spread_matrix"])
    if base.empty:
        return base
    merge_keys = ["window_id", "snapshot_idx", "coin"]
    df = base
    for label, data, cols, prefix in [
        ("ticker", raw["ticker"], ["mid", "spread_bps", "bid_volume", "ask_volume"], "tk_"),
        ("orderbook", raw["orderbook"], ["imbalance"], "ob_"),
        ("trades", raw["trades"], ["buy_sell_ratio", "total_volume"], "tr_"),
    ]:
        if data is None or data.empty:
            continue
        aux = _to_wide(data, merge_keys, "exchange", cols, prefix)
        if not aux.empty:
            df = pd.merge(df, aux, on=merge_keys, how="left")
    df = add_cross_exchange_features(df)
    return df.drop(columns=["window_id"], errors="ignore")


def prepare_X(df: pd.DataFrame, feature_names: List[str], cat_maps: List[List[str]]) -> pd.DataFrame:
    X = df.reindex(columns=feature_names).copy()
    # restore training categorical vocabularies
    if "coin" in X.columns and len(cat_maps) >= 1:
        X["coin"] = pd.Categorical(X["coin"], categories=cat_maps[0])
    if "pair" in X.columns and len(cat_maps) >= 2:
        X["pair"] = pd.Categorical(X["pair"], categories=cat_maps[1])
    return X


def prune_old(state: State, keep_last: int = ZSCORE_WINDOW + 40) -> None:
    if len(state.known_snaps) <= keep_last:
        return
    drop = list(state.known_snaps)[:-keep_last]
    for s in drop:
        state.spreads.pop(s, None)
        state.tickers.pop(s, None)
        state.orderbooks.pop(s, None)
        state.trades.pop(s, None)
        state.snap_ts.pop(s, None)
        state.known_snaps.popleft()


def settle_bets(state: State, z_lookup: Dict[Tuple[str, str, int], Tuple[float, float]]) -> None:
    still: List[OpenBet] = []
    for bet in state.open_bets:
        key = (bet.coin, bet.pair, bet.exit_snap)
        if key not in z_lookup:
            still.append(bet)
            continue
        z_exit, spread_exit = z_lookup[key]
        direction = bet.direction
        hit = int(np.sign(z_exit) == direction) if z_exit == z_exit and z_exit != 0 else 0
        pnl_proxy = float(direction * z_exit) if z_exit == z_exit else float("nan")
        spread_delta = float(spread_exit - bet.entry_spread_bps)
        closed = {
            "entry_ts": bet.entry_ts,
            "exit_ts": state.snap_ts.get(bet.exit_snap, utc_now()),
            "coin": bet.coin,
            "pair": bet.pair,
            "entry_snap": bet.entry_snap,
            "exit_snap": bet.exit_snap,
            "direction": direction,
            "pred": bet.pred,
            "entry_z": bet.entry_z,
            "exit_z": float(z_exit) if z_exit == z_exit else None,
            "entry_spread_bps": bet.entry_spread_bps,
            "exit_spread_bps": float(spread_exit),
            "spread_delta_bps": spread_delta,
            "pnl_proxy": pnl_proxy,
            "dir_hit": hit,
        }
        state.closed.append(closed)
    state.open_bets = still


def score_snapshot(
    state: State,
    model: lgb.Booster,
    feature_names: List[str],
    cat_maps: List[List[str]],
    cfg: Config,
    snap: int,
) -> bool:
    """Return True once this snap has been fully handled (predicted or permanently empty)."""
    snaps = [s for s in state.known_snaps if s <= snap]
    # Permanently skip pre-warmup snaps so the cursor can advance.
    if snap < MIN_PERIODS + N_LAGS:
        return True
    if len(snaps) < MIN_PERIODS + N_LAGS:
        return False
    if not state.spreads.get(snap):
        return False

    snaps = snaps[-(ZSCORE_WINDOW + N_LAGS + 10) :]
    raw = frames_for_window(state, snaps)
    if raw["spread_matrix"].empty:
        return False
    df = build_live_matrix(raw)
    if df.empty:
        return False

    z_lookup: Dict[Tuple[str, str, int], Tuple[float, float]] = {}
    for _, row in df[["coin", "pair", "snapshot_idx", "zscore", "spread_bps"]].iterrows():
        z_lookup[(row["coin"], row["pair"], int(row["snapshot_idx"]))] = (
            float(row["zscore"]) if pd.notna(row["zscore"]) else np.nan,
            float(row["spread_bps"]),
        )
    settle_bets(state, z_lookup)

    cur = df[df["snapshot_idx"] == snap].copy()
    if cur.empty:
        # Spreads were ingested but fell out of the feature frame — retry next poll.
        return False
    lag_cols = [f"zscore_lag{i}" for i in range(1, N_LAGS + 1)]
    cur = cur.dropna(subset=lag_cols, how="any")
    if cur.empty:
        return False

    X = prepare_X(cur, feature_names, cat_maps)
    preds = model.predict(X)
    state.n_preds += len(preds)

    open_keys = {(b.coin, b.pair) for b in state.open_bets}
    for (_, row), pred in zip(cur.iterrows(), preds):
        coin, pair = row["coin"], row["pair"]
        z = float(row["zscore"]) if pd.notna(row["zscore"]) else float("nan")
        sig = {
            "ts": state.snap_ts.get(snap, utc_now()),
            "snapshot_idx": snap,
            "coin": coin,
            "pair": pair,
            "pred": float(pred),
            "zscore": z,
            "spread_bps": float(row["spread_bps"]),
            "abs_pred": float(abs(pred)),
        }
        state.signals.append(sig)

        if abs(pred) < cfg.entry_tau:
            continue
        if (coin, pair) in open_keys:
            continue
        if len(state.open_bets) >= cfg.max_open:
            continue
        direction = 1 if pred > 0 else -1
        bet = OpenBet(
            coin=coin,
            pair=pair,
            entry_snap=snap,
            exit_snap=snap + HORIZON,
            entry_ts=state.snap_ts.get(snap, utc_now()),
            direction=direction,
            pred=float(pred),
            entry_z=z,
            entry_spread_bps=float(row["spread_bps"]),
        )
        state.open_bets.append(bet)
        open_keys.add((coin, pair))
    return True


def save_checkpoint(state: State, cfg: Config, force_summary: bool = False) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    meta = load_jsonl_writer_state(cfg.output_dir)
    new_signals = state.signals[state.n_signals_flushed :]
    new_closed = state.closed[state.n_closed_flushed :]
    append_jsonl_sharded(
        cfg.output_dir, "signals", new_signals, "signals_part", "signals_lines", meta
    )
    append_jsonl_sharded(
        cfg.output_dir, "trades", new_closed, "trades_part", "trades_lines", meta
    )
    state.n_signals_flushed = len(state.signals)
    state.n_closed_flushed = len(state.closed)
    save_jsonl_writer_state(cfg.output_dir, meta)

    summary = summarize(state, cfg)
    (cfg.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if force_summary or len(state.closed) % 25 == 0:
        print(
            f"  [{utc_now()}] snaps={len(state.known_snaps)} preds={state.n_preds} "
            f"open={len(state.open_bets)} closed={len(state.closed)} "
            f"dir_acc={summary.get('dir_acc')} pnl_proxy={summary.get('mean_pnl_proxy')} "
            f"jsonl_signals={meta.get('signals_part')}:{meta.get('signals_lines')} "
            f"jsonl_trades={meta.get('trades_part')}:{meta.get('trades_lines')}",
            flush=True,
        )


def summarize(state: State, cfg: Config) -> dict:
    closed = state.closed
    if not closed:
        return {
            "n_closed": 0,
            "n_open": len(state.open_bets),
            "n_preds": state.n_preds,
            "n_snaps": len(state.known_snaps),
            "model": str(cfg.model_path),
            "run_dir": str(cfg.run_dir),
            "entry_tau": cfg.entry_tau,
            "updated_at": utc_now(),
        }
    hits = [t["dir_hit"] for t in closed if t.get("exit_z") is not None]
    pnls = [t["pnl_proxy"] for t in closed if t.get("pnl_proxy") == t.get("pnl_proxy")]
    return {
        "n_closed": len(closed),
        "n_open": len(state.open_bets),
        "n_preds": state.n_preds,
        "n_snaps": len(state.known_snaps),
        "n_signals": len(state.signals),
        "dir_acc": float(np.mean(hits)) if hits else None,
        "mean_pnl_proxy": float(np.mean(pnls)) if pnls else None,
        "median_abs_pred_entries": float(np.median([abs(t["pred"]) for t in closed])),
        "model": str(cfg.model_path),
        "run_dir": str(cfg.run_dir),
        "entry_tau": cfg.entry_tau,
        "horizon": HORIZON,
        "updated_at": utc_now(),
    }


def find_newest_run(statarb_root: Path) -> Optional[Path]:
    if not statarb_root.is_dir():
        return None
    dirs = [p for p in statarb_root.iterdir() if p.is_dir() and (p / "spread_matrix").exists()]
    if not dirs:
        return None
    return max(dirs, key=lambda p: p.stat().st_mtime)


def wait_for_run(path: Optional[Path], statarb_root: Path, timeout_sec: float = 600) -> Path:
    if path and path.exists():
        return path
    print(f"  Waiting for collector run under {statarb_root} …", flush=True)
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout_sec:
        cand = find_newest_run(statarb_root)
        if cand and cand != last:
            # require at least one spread line
            files = list((cand / "spread_matrix").glob("*.jsonl"))
            if files and files[-1].stat().st_size > 0:
                print(f"  Found run: {cand}", flush=True)
                return cand
            last = cand
        time.sleep(5)
    raise SystemExit(f"No collector run appeared under {statarb_root} within {timeout_sec}s")


def main() -> None:
    ap = argparse.ArgumentParser(description="Live LightGBM paper trader")
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--run-dir", type=Path, default=None, help="Collector run dir; auto-detect if omitted")
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--entry-tau", type=float, default=0.5)
    ap.add_argument("--poll-sec", type=float, default=15.0)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--max-open", type=int, default=50)
    args = ap.parse_args()

    model_path = args.model.resolve()
    assert model_path.exists(), model_path
    model = lgb.Booster(model_file=str(model_path))
    feature_names = model.feature_name()
    cat_maps = list(model.pandas_categorical or [])

    statarb_root = ROOT / "data" / "statarb"
    run_dir = wait_for_run(args.run_dir.resolve() if args.run_dir else None, statarb_root)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = args.output_dir or (ROOT / "data" / "paper_trading" / f"lgbm_{stamp}")
    cfg = Config(
        model_path=model_path,
        run_dir=run_dir,
        output_dir=out.resolve(),
        hours=args.hours,
        entry_tau=args.entry_tau,
        poll_sec=args.poll_sec,
        max_open=args.max_open,
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    # Fresh in-memory state always replays the collector from offset 0. Archive any
    # prior JSONL shards in this output dir so append-mode cannot duplicate rows.
    existing_shards = list_jsonl_shards(cfg.output_dir, "signals") + list_jsonl_shards(
        cfg.output_dir, "trades"
    )
    if existing_shards or (cfg.output_dir / "jsonl_writer_state.json").exists():
        bak = cfg.output_dir / f"pre_restart_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        bak.mkdir(parents=True, exist_ok=True)
        for p in existing_shards:
            p.replace(bak / p.name)
        wstate = cfg.output_dir / "jsonl_writer_state.json"
        if wstate.exists():
            wstate.replace(bak / wstate.name)
        print(f"  Archived prior JSONL shards -> {bak}", flush=True)

    prev_cfg: dict = {}
    cfg_path = cfg.output_dir / "config.json"
    if cfg_path.exists():
        try:
            prev_cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError:
            prev_cfg = {}
    cfg_doc = {
        "model": str(cfg.model_path),
        "run_dir": str(cfg.run_dir),
        "hours": cfg.hours,
        "entry_tau": cfg.entry_tau,
        "horizon": HORIZON,
        "zscore_window": ZSCORE_WINDOW,
        "started_at": prev_cfg.get("started_at") or utc_now(),
        "jsonl_shard_max_lines": JSONL_SHARD_MAX_LINES,
    }
    for key in ("hours_planned_total", "deadline_utc", "extended_at"):
        if key in prev_cfg:
            cfg_doc[key] = prev_cfg[key]
    if prev_cfg:
        cfg_doc["restarted_at"] = utc_now()
    cfg_path.write_text(json.dumps(cfg_doc, indent=2), encoding="utf-8")

    state = State()
    deadline = time.time() + cfg.hours * 3600
    print("=" * 60, flush=True)
    print(f"  LGBM PAPER TRADER", flush=True)
    print(f"  model:   {cfg.model_path}", flush=True)
    print(f"  run:     {cfg.run_dir}", flush=True)
    print(f"  output:  {cfg.output_dir}", flush=True)
    print(f"  hours:   {cfg.hours}  entry_tau={cfg.entry_tau}  horizon={HORIZON}", flush=True)
    print(f"  features:{len(feature_names)}  cats={ [len(c) for c in cat_maps] }", flush=True)
    print("=" * 60, flush=True)

    last_ckpt = 0.0
    err_streak = 0
    try:
        while time.time() < deadline:
            n = 0
            try:
                n = ingest_new_lines(state, cfg.run_dir)
                snaps = list(state.known_snaps)
                for snap in snaps:
                    if snap <= state.last_scored_snap:
                        continue
                    if not state.spreads.get(snap):
                        continue
                    if score_snapshot(state, model, feature_names, cat_maps, cfg, snap):
                        state.last_scored_snap = snap
                    else:
                        # Not ready yet (warmup / incomplete snap). Stop advancing so we retry.
                        break
                prune_old(state)
                err_streak = 0
            except Exception as e:
                err_streak += 1
                print(f"  [WARN] loop error ({err_streak}): {type(e).__name__}: {e}", flush=True)
                import traceback

                traceback.print_exc()
                if err_streak >= 20:
                    raise
                time.sleep(5)
                continue
            if time.time() - last_ckpt > 60:
                save_checkpoint(state, cfg)
                last_ckpt = time.time()
            if n == 0:
                time.sleep(cfg.poll_sec)
            else:
                time.sleep(min(cfg.poll_sec, 5.0))
    except KeyboardInterrupt:
        print("\n  Interrupted — flushing …", flush=True)

    save_checkpoint(state, cfg, force_summary=True)
    summary = summarize(state, cfg)
    print("\n  PAPER SESSION COMPLETE", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
