"""Strengthening analyses for LGBM vs mechanical baselines (Jul 31).

Computes two optional peer improvements for later paper use:
  1) Cost grid on settled trades: net_bps = direction * spread_delta_bps - c
     for c in {0, 5, 10, 15} bps round-turn.
  2) Ranked capacity fill (max_open=50):
     - LGBM: among |pred|>=tau candidates each snap, fill by |pred| desc
     - Mechanical: among |z|>=tau candidates, fill by |z| desc
       (persistence and mean-reversion directions)

Usage:
  python scripts/baseline_strengthening_jul31.py data/paper_trading/July31st_8_hr
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from mechanical_z_baseline_paper_session import (  # noqa: E402
    DirectionMode,
    align_forward_z,
    direction_from_z,
    load_signals,
    summarize_book,
)
from portfolio_sharpe_paper_session import (  # noqa: E402
    build_z_panel,
    load_jsonl_rows,
    sharpe,
)

COST_GRID_BPS = (0.0, 5.0, 10.0, 15.0)


@dataclass
class OpenBet:
    coin: str
    pair: str
    entry_snap: int
    exit_snap: int
    entry_ts: pd.Timestamp
    direction: int
    score: float  # |pred| or |z| used for ranking / logging
    entry_z: float
    entry_spread_bps: float
    pred: float


def cost_grid(trades: pd.DataFrame, costs: tuple[float, ...] = COST_GRID_BPS) -> dict:
    """Fee sensitivity on direction * spread_delta_bps (gross bps proxy)."""
    if trades.empty or "spread_delta_bps" not in trades.columns:
        return {"n": 0, "note": "no spread_delta_bps"}
    d = trades["direction"].astype(float)
    delta = trades["spread_delta_bps"].astype(float)
    mask = d.notna() & delta.notna()
    gross = (d[mask] * delta[mask]).to_numpy(float)
    out: dict = {
        "n": int(mask.sum()),
        "definition": "net_bps = direction * spread_delta_bps - c  (c = round-turn bps)",
        "gross_mean_bps": float(np.mean(gross)) if len(gross) else None,
        "gross_win_rate": float(np.mean(gross > 0)) if len(gross) else None,
        "by_cost_bps": {},
    }
    for c in costs:
        net = gross - c
        out["by_cost_bps"][str(c)] = {
            "mean_net_bps": float(np.mean(net)),
            "median_net_bps": float(np.median(net)),
            "win_rate": float(np.mean(net > 0)),
            "per_trade_sharpe": sharpe(net),
            "frac_surviving_positive_mean": bool(np.mean(net) > 0),
        }
    # Also z-proxy net isn't fee-native; report break-even c where mean net_bps=0
    if len(gross) and float(np.mean(gross)) != 0:
        out["break_even_round_turn_bps"] = float(np.mean(gross))
    return out


def settle_bet(
    bet: OpenBet,
    z_exit: float,
    spread_exit: float | None,
    exit_ts: pd.Timestamp,
) -> dict:
    hit = int(np.sign(z_exit) == bet.direction) if z_exit != 0 else 0
    return {
        "entry_ts": bet.entry_ts,
        "exit_ts": exit_ts,
        "coin": bet.coin,
        "pair": bet.pair,
        "entry_snap": bet.entry_snap,
        "exit_snap": bet.exit_snap,
        "direction": bet.direction,
        "pred": bet.pred,
        "entry_z": bet.entry_z,
        "exit_z": z_exit,
        "entry_spread_bps": bet.entry_spread_bps,
        "exit_spread_bps": spread_exit,
        "spread_delta_bps": (
            None if spread_exit is None else float(spread_exit) - float(bet.entry_spread_bps)
        ),
        "pnl_proxy": float(bet.direction * z_exit),
        "dir_hit": hit,
        "rank_score": bet.score,
    }


def ranked_capacity_book(
    rows: pd.DataFrame,
    *,
    horizon: int,
    max_open: int,
    tau: float,
    score_col: str,
    eligible: Callable[[pd.Series], bool],
    direction_fn: Callable[[pd.Series], int],
) -> tuple[pd.DataFrame, dict]:
    """Settle-then-open per snap; fill free slots by score_col descending."""
    work = rows.sort_values(["snapshot_idx", "ts"]).reset_index(drop=True)
    exit_lookup: dict[tuple[str, str, int], tuple[float, float | None, pd.Timestamp]] = {}
    for row in work.itertuples(index=False):
        exit_lookup[(row.coin, row.pair, int(row.snapshot_idx))] = (
            float(row.zscore),
            float(row.spread_bps) if pd.notna(row.spread_bps) else None,
            row.ts,
        )

    open_bets: list[OpenBet] = []
    closed: list[dict] = []
    skipped_pair = skipped_cap = candidates = 0

    for snap in sorted(work["snapshot_idx"].unique()):
        still: list[OpenBet] = []
        for bet in open_bets:
            if bet.exit_snap > snap:
                still.append(bet)
                continue
            key = (bet.coin, bet.pair, bet.exit_snap)
            if key not in exit_lookup:
                still.append(bet)
                continue
            z_exit, spread_exit, exit_ts = exit_lookup[key]
            closed.append(settle_bet(bet, z_exit, spread_exit, exit_ts))
        open_bets = still

        open_keys = {(b.coin, b.pair) for b in open_bets}
        batch = work[work["snapshot_idx"] == snap].copy()
        # Rank candidates
        cand_recs = []
        for row in batch.itertuples(index=False):
            s = pd.Series(row._asdict())
            if not eligible(s):
                continue
            z = float(s["zscore"])
            if z != z:
                continue
            exit_key = (s["coin"], s["pair"], int(snap) + horizon)
            if "z_fwd" in s and s["z_fwd"] == s["z_fwd"]:
                pass
            elif exit_key not in exit_lookup:
                continue
            candidates += 1
            if (s["coin"], s["pair"]) in open_keys:
                skipped_pair += 1
                continue
            cand_recs.append(s)

        if not cand_recs:
            continue
        cand_df = pd.DataFrame(cand_recs)
        cand_df["_score"] = cand_df[score_col].astype(float).abs()
        cand_df = cand_df.sort_values("_score", ascending=False)

        for _, s in cand_df.iterrows():
            if len(open_bets) >= max_open:
                skipped_cap += 1
                continue
            if (s["coin"], s["pair"]) in open_keys:
                skipped_pair += 1
                continue
            direction = direction_fn(s)
            score = float(abs(s[score_col]))
            bet = OpenBet(
                coin=s["coin"],
                pair=s["pair"],
                entry_snap=int(snap),
                exit_snap=int(snap) + horizon,
                entry_ts=s["ts"],
                direction=direction,
                score=score,
                entry_z=float(s["zscore"]),
                entry_spread_bps=float(s["spread_bps"]) if pd.notna(s["spread_bps"]) else float("nan"),
                pred=float(s["pred"]) if "pred" in s and pd.notna(s["pred"]) else float(s["zscore"]),
            )
            open_bets.append(bet)
            open_keys.add((bet.coin, bet.pair))

    still_open = 0
    for bet in open_bets:
        key = (bet.coin, bet.pair, bet.exit_snap)
        if key not in exit_lookup:
            still_open += 1
            continue
        z_exit, spread_exit, exit_ts = exit_lookup[key]
        closed.append(settle_bet(bet, z_exit, spread_exit, exit_ts))

    meta = {
        "candidates": candidates,
        "skipped_pair_busy": skipped_pair,
        "skipped_at_capacity": skipped_cap,
        "n_still_open_unsettled": still_open,
        "max_open": max_open,
        "rank_by": score_col,
        "tau": tau,
    }
    return pd.DataFrame(closed), meta


def book_summary(trades: pd.DataFrame, z_panel: pd.DataFrame, max_open: int, label: str) -> dict:
    base = summarize_book(trades, z_panel, max_open=max_open, label=label)
    base["cost_grid"] = cost_grid(trades)
    return base


def load_trades_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir", type=Path)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--max-open", type=int, default=50)
    ap.add_argument("--horizon", type=int, default=None)
    args = ap.parse_args()
    out = args.session_dir.resolve()

    cfg_path = out / "config.json"
    horizon = args.horizon
    if horizon is None and cfg_path.exists():
        horizon = int(json.loads(cfg_path.read_text(encoding="utf-8-sig")).get("horizon", 1))
    if horizon is None:
        horizon = 1

    sig = load_signals(out)
    aligned = align_forward_z(sig, horizon=horizon)
    z_fwd = aligned[
        ["coin", "pair", "snapshot_idx", "z_fwd", "exit_ts", "exit_spread_bps"]
    ].drop_duplicates(["coin", "pair", "snapshot_idx"])
    panel = sig.merge(z_fwd, on=["coin", "pair", "snapshot_idx"], how="left")
    z_panel = build_z_panel(sig)

    dest = out / "baseline_strengthening"
    dest.mkdir(parents=True, exist_ok=True)

    # --- 1) Cost grid on campaign + existing mechanical FIFO capacity books ---
    lgbm_trades = pd.DataFrame(load_jsonl_rows(out, "trades"))
    mech_dir = out / "mechanical_z_baseline"
    mech_persist = load_trades_df(mech_dir / "trades_persistence_capacity_max_open.jsonl")
    if mech_persist.empty:
        mech_persist = load_trades_df(mech_dir / "trades_capacity_max_open.jsonl")
    mech_mr = load_trades_df(mech_dir / "trades_mean_reversion_capacity_max_open.jsonl")

    cost_section = {
        "lgbm_campaign_fifo_capacity": cost_grid(lgbm_trades),
        "mechanical_persistence_fifo_capacity": cost_grid(mech_persist),
        "mechanical_mean_reversion_fifo_capacity": cost_grid(mech_mr),
    }

    # --- 2) Ranked capacity replays ---
    def lgbm_eligible(s: pd.Series) -> bool:
        return pd.notna(s.get("pred")) and abs(float(s["pred"])) >= args.tau

    def lgbm_direction(s: pd.Series) -> int:
        return 1 if float(s["pred"]) > 0 else -1

    lgbm_ranked, lgbm_meta = ranked_capacity_book(
        panel,
        horizon=horizon,
        max_open=args.max_open,
        tau=args.tau,
        score_col="pred",
        eligible=lgbm_eligible,
        direction_fn=lgbm_direction,
    )

    ranked_mech: dict[str, dict] = {}
    for mode in ("persistence", "mean_reversion"):
        mode_t: DirectionMode = mode  # type: ignore[assignment]

        def mech_eligible(s: pd.Series, _tau=args.tau) -> bool:
            z = float(s["zscore"])
            return z == z and abs(z) >= _tau and z != 0.0

        def mech_direction(s: pd.Series, _mode=mode_t) -> int:
            return direction_from_z(float(s["zscore"]), _mode)

        trades, meta = ranked_capacity_book(
            panel,
            horizon=horizon,
            max_open=args.max_open,
            tau=args.tau,
            score_col="zscore",
            eligible=mech_eligible,
            direction_fn=mech_direction,
        )
        ranked_mech[mode] = {
            **book_summary(trades, z_panel, args.max_open, f"mech_{mode}_ranked"),
            "capacity_meta": meta,
        }
        trades.to_json(dest / f"trades_mechanical_{mode}_ranked_capacity.jsonl", orient="records", lines=True)

    lgbm_ranked_summary = book_summary(
        lgbm_ranked, z_panel, args.max_open, "lgbm_ranked_capacity"
    )
    lgbm_ranked_summary["capacity_meta"] = lgbm_meta
    lgbm_ranked.to_json(dest / "trades_lgbm_ranked_capacity.jsonl", orient="records", lines=True)

    # Campaign FIFO summary for side-by-side
    campaign_fifo = {
        "n_closed": int(len(lgbm_trades)),
        "dir_acc": float(lgbm_trades["dir_hit"].mean()) if len(lgbm_trades) else None,
        "mean_pnl_proxy": float(lgbm_trades["pnl_proxy"].mean()) if len(lgbm_trades) else None,
        "note": "Live campaign fill order (encounter order), max_open=50",
    }

    report = {
        "session": str(out),
        "purpose": (
            "Optional strengthening analyses: fee grid + confidence/|z|-ranked "
            "capacity fill. Not the primary headline table; keep for later use."
        ),
        "tau": args.tau,
        "horizon": horizon,
        "max_open": args.max_open,
        "cost_grid_fifo_capacity": cost_section,
        "ranked_capacity": {
            "lgbm_campaign_fifo_reference": campaign_fifo,
            "lgbm_ranked_by_abs_pred": lgbm_ranked_summary,
            "mechanical_persistence_ranked_by_abs_z": ranked_mech["persistence"],
            "mechanical_mean_reversion_ranked_by_abs_z": ranked_mech["mean_reversion"],
        },
        "how_to_read": {
            "cost_grid": (
                "Compare mean_net_bps and win_rate across c. Higher mean pnl "
                "selection should survive costs longer."
            ),
            "ranked_capacity": (
                "Fairer scarce-slot peer: LGBM fills by |pred|, mechanical by |z|. "
                "Compare DirAcc / mean_pnl_proxy / Sharpe B under max_open."
            ),
        },
    }

    report_path = dest / "baseline_strengthening_report.json"
    # JSON-safe: drop huge hourly paths from nested summarize_book if present
    def strip_paths(obj):
        if isinstance(obj, dict):
            return {
                k: strip_paths(v)
                for k, v in obj.items()
                if k != "hourly_path"
            }
        if isinstance(obj, list):
            return [strip_paths(x) for x in obj]
        return obj

    slim = strip_paths(report)
    report_path.write_text(json.dumps(slim, indent=2, default=float), encoding="utf-8")

    print(json.dumps(slim, indent=2, default=float))
    print(f"\nsaved {report_path}", flush=True)

    print("\n=== COST GRID (mean_net_bps / win_rate) ===", flush=True)
    for name, block in cost_section.items():
        if not block.get("by_cost_bps"):
            print(f"{name}: no data", flush=True)
            continue
        cells = [
            f"c={c}: mean={v['mean_net_bps']:+.3f} wr={v['win_rate']:.3f}"
            for c, v in block["by_cost_bps"].items()
        ]
        print(f"{name}: " + " | ".join(cells), flush=True)

    print("\n=== RANKED CAPACITY ===", flush=True)
    for label, block in [
        ("LGBM ranked", lgbm_ranked_summary),
        ("Mech persist ranked", ranked_mech["persistence"]),
        ("Mech MR ranked", ranked_mech["mean_reversion"]),
    ]:
        sb = block["sharpe"]["B_hourly_equity_pnl_with_open_mtm"]["sharpe"]
        print(
            f"{label}: n={block['n_closed']} DirAcc={block['dir_acc']:.4f} "
            f"mean_pnl={block['mean_pnl_proxy']:.4f} SharpeB={sb:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
