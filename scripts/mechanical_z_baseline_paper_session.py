"""Mechanical |z_t|>=tau persistence peer on an LGBM paper session.

Policy (persistence only — no mean-reversion):
  - Enter when |z_t| >= tau
  - direction = sign(z_t)
  - Settle at t+H with pnl_proxy = direction * z_{t+H}

Runs two books on the same signals panel:
  A) unconstrained — every eligible row with a forward z
  B) capacity-matched — max_open slots, one open bet per (coin, pair),
     settle-before-open per snapshot (mirrors experiments/paper_trade_lgbm.py)

Usage:
  python scripts/mechanical_z_baseline_paper_session.py data/paper_trading/July31st_8_hr
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from portfolio_sharpe_paper_session import (  # noqa: E402
    build_z_panel,
    closed_only_hourly,
    load_jsonl_rows,
    period_pnl_series,
    sharpe,
)


@dataclass
class OpenBet:
    coin: str
    pair: str
    entry_snap: int
    exit_snap: int
    entry_ts: pd.Timestamp
    direction: int
    entry_z: float
    entry_spread_bps: float


def load_signals(out: Path) -> pd.DataFrame:
    rows = load_jsonl_rows(out, "signals")
    if not rows:
        raise SystemExit(f"no signals in {out}")
    sig = pd.DataFrame(rows)
    sig["ts"] = pd.to_datetime(sig["ts"], utc=True)
    sig["snapshot_idx"] = sig["snapshot_idx"].astype(int)
    return sig


def align_forward_z(sig: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Attach z at snapshot_idx+horizon as z_fwd / exit metadata."""
    z = sig.dropna(subset=["zscore"])[
        ["coin", "pair", "snapshot_idx", "zscore", "ts", "spread_bps"]
    ].copy()
    z["snapshot_idx"] = z["snapshot_idx"].astype(int) - horizon
    z = z.rename(
        columns={
            "zscore": "z_fwd",
            "ts": "exit_ts",
            "spread_bps": "exit_spread_bps",
        }
    )
    # If multiple signal rows share a key, keep last
    z = z.sort_values(["coin", "pair", "snapshot_idx", "exit_ts"]).drop_duplicates(
        ["coin", "pair", "snapshot_idx"], keep="last"
    )
    merged = sig.merge(z, on=["coin", "pair", "snapshot_idx"], how="inner")
    return merged.dropna(subset=["zscore", "z_fwd"])


def trade_row(row: pd.Series, direction: int) -> dict:
    entry_z = float(row["zscore"])
    exit_z = float(row["z_fwd"])
    pnl = float(direction * exit_z)
    hit = int(np.sign(exit_z) == direction) if exit_z != 0 else 0
    return {
        "entry_ts": row["ts"],
        "exit_ts": row["exit_ts"],
        "coin": row["coin"],
        "pair": row["pair"],
        "entry_snap": int(row["snapshot_idx"]),
        "exit_snap": int(row["snapshot_idx"]) + int(row["horizon"]),
        "direction": int(direction),
        "pred": entry_z,  # mechanical "prediction" = z_t (persistence)
        "entry_z": entry_z,
        "exit_z": exit_z,
        "entry_spread_bps": float(row["spread_bps"]) if pd.notna(row["spread_bps"]) else None,
        "exit_spread_bps": float(row["exit_spread_bps"])
        if pd.notna(row.get("exit_spread_bps"))
        else None,
        "spread_delta_bps": (
            float(row["exit_spread_bps"]) - float(row["spread_bps"])
            if pd.notna(row.get("exit_spread_bps")) and pd.notna(row["spread_bps"])
            else None
        ),
        "pnl_proxy": pnl,
        "dir_hit": hit,
    }


def unconstrained_trades(df: pd.DataFrame, tau: float, horizon: int) -> pd.DataFrame:
    """Every row with |z_t| >= tau and finite z_fwd."""
    rows = []
    for row in df.itertuples(index=False):
        z = float(row.zscore)
        if abs(z) < tau or z == 0.0:
            continue
        direction = 1 if z > 0 else -1
        s = pd.Series(row._asdict())
        s["horizon"] = horizon
        rows.append(trade_row(s, direction))
    return pd.DataFrame(rows)


def capacity_trades(
    df: pd.DataFrame, tau: float, horizon: int, max_open: int
) -> tuple[pd.DataFrame, dict]:
    """Replay live capacity: settle then open; max_open; one per (coin, pair)."""
    # Preserve within-snap signal order from the session file.
    work = df.sort_values(["snapshot_idx", "ts"]).reset_index(drop=True)
    open_bets: list[OpenBet] = []
    closed: list[dict] = []
    skipped_pair = 0
    skipped_cap = 0
    candidates = 0

    # Exit z lookup: (coin, pair, exit_snap) -> (z, spread, ts)
    exit_lookup: dict[tuple[str, str, int], tuple[float, float | None, pd.Timestamp]] = {}
    for row in work.itertuples(index=False):
        # work rows are at entry snap; zscore is z at that snap.
        # We also need z at each snap as a possible exit — use zscore keyed by this snap.
        key = (row.coin, row.pair, int(row.snapshot_idx))
        exit_lookup[key] = (
            float(row.zscore),
            float(row.spread_bps) if pd.notna(row.spread_bps) else None,
            row.ts,
        )

    snaps = sorted(work["snapshot_idx"].unique())
    for snap in snaps:
        # Settle bets whose exit_snap == this snap
        still: list[OpenBet] = []
        for bet in open_bets:
            if bet.exit_snap > snap:
                still.append(bet)
                continue
            if bet.exit_snap < snap:
                # Should have settled earlier; keep trying
                key = (bet.coin, bet.pair, bet.exit_snap)
                if key not in exit_lookup:
                    still.append(bet)
                    continue
            key = (bet.coin, bet.pair, bet.exit_snap)
            if key not in exit_lookup:
                still.append(bet)
                continue
            z_exit, spread_exit, exit_ts = exit_lookup[key]
            hit = int(np.sign(z_exit) == bet.direction) if z_exit != 0 else 0
            pnl = float(bet.direction * z_exit)
            closed.append(
                {
                    "entry_ts": bet.entry_ts,
                    "exit_ts": exit_ts,
                    "coin": bet.coin,
                    "pair": bet.pair,
                    "entry_snap": bet.entry_snap,
                    "exit_snap": bet.exit_snap,
                    "direction": bet.direction,
                    "pred": bet.entry_z,
                    "entry_z": bet.entry_z,
                    "exit_z": z_exit,
                    "entry_spread_bps": bet.entry_spread_bps,
                    "exit_spread_bps": spread_exit,
                    "spread_delta_bps": (
                        None
                        if spread_exit is None
                        else float(spread_exit) - float(bet.entry_spread_bps)
                    ),
                    "pnl_proxy": pnl,
                    "dir_hit": hit,
                }
            )
        open_bets = still

        open_keys = {(b.coin, b.pair) for b in open_bets}
        batch = work[work["snapshot_idx"] == snap]
        for row in batch.itertuples(index=False):
            z = float(row.zscore)
            if abs(z) < tau or z == 0.0:
                continue
            # Need forward z available eventually; skip if we know it will never settle
            exit_key = (row.coin, row.pair, int(snap) + horizon)
            # Prefer pre-aligned z_fwd when present
            if hasattr(row, "z_fwd") and row.z_fwd == row.z_fwd:
                pass
            elif exit_key not in exit_lookup:
                continue
            candidates += 1
            if (row.coin, row.pair) in open_keys:
                skipped_pair += 1
                continue
            if len(open_bets) >= max_open:
                skipped_cap += 1
                continue
            direction = 1 if z > 0 else -1
            bet = OpenBet(
                coin=row.coin,
                pair=row.pair,
                entry_snap=int(snap),
                exit_snap=int(snap) + horizon,
                entry_ts=row.ts,
                direction=direction,
                entry_z=z,
                entry_spread_bps=float(row.spread_bps) if pd.notna(row.spread_bps) else float("nan"),
            )
            open_bets.append(bet)
            open_keys.add((row.coin, row.pair))

    # Attempt final settle for any leftovers that have exit z
    still_open = 0
    for bet in open_bets:
        key = (bet.coin, bet.pair, bet.exit_snap)
        if key not in exit_lookup:
            still_open += 1
            continue
        z_exit, spread_exit, exit_ts = exit_lookup[key]
        hit = int(np.sign(z_exit) == bet.direction) if z_exit != 0 else 0
        closed.append(
            {
                "entry_ts": bet.entry_ts,
                "exit_ts": exit_ts,
                "coin": bet.coin,
                "pair": bet.pair,
                "entry_snap": bet.entry_snap,
                "exit_snap": bet.exit_snap,
                "direction": bet.direction,
                "pred": bet.entry_z,
                "entry_z": bet.entry_z,
                "exit_z": z_exit,
                "entry_spread_bps": bet.entry_spread_bps,
                "exit_spread_bps": spread_exit,
                "spread_delta_bps": (
                    None
                    if spread_exit is None
                    else float(spread_exit) - float(bet.entry_spread_bps)
                ),
                "pnl_proxy": float(bet.direction * z_exit),
                "dir_hit": hit,
            }
        )

    meta = {
        "candidates_abs_z_ge_tau": candidates,
        "skipped_pair_busy": skipped_pair,
        "skipped_at_capacity": skipped_cap,
        "n_still_open_unsettled": still_open,
        "max_open": max_open,
    }
    return pd.DataFrame(closed), meta


def summarize_book(trades: pd.DataFrame, z_panel: pd.DataFrame, max_open: int, label: str) -> dict:
    if trades.empty:
        return {"label": label, "n_closed": 0}

    trades = trades.copy()
    trades["entry_ts"] = pd.to_datetime(trades["entry_ts"], utc=True)
    trades["exit_ts"] = pd.to_datetime(trades["exit_ts"], utc=True)
    trades["pnl_proxy"] = trades["pnl_proxy"].astype(float)
    trades["direction"] = trades["direction"].astype(int)

    a = closed_only_hourly(trades)
    hourly = period_pnl_series(trades, z_panel, "1h")
    h_pnl = hourly["pnl"].dropna().to_numpy()

    entry_z = trades["entry_z"].to_numpy(float)
    exit_z = trades["exit_z"].to_numpy(float)
    # Persistence forecast quality on the traded rows: predict z_fwd with z_t
    return {
        "label": label,
        "n_closed": int(len(trades)),
        "dir_acc": float(trades["dir_hit"].mean()),
        "mean_pnl_proxy": float(trades["pnl_proxy"].mean()),
        "median_pnl_proxy": float(trades["pnl_proxy"].median()),
        "per_trade_sharpe": sharpe(trades["pnl_proxy"].to_numpy(float)),
        "forecast_on_entries": {
            "r2_z_t_predicts_z_fwd": float(r2_score(exit_z, entry_z)),
            "mae": float(mean_absolute_error(exit_z, entry_z)),
            "rmse": float(mean_squared_error(exit_z, entry_z) ** 0.5),
            "dir_acc_sign_match": float(np.mean(np.sign(entry_z) == np.sign(exit_z))),
        },
        "sharpe": {
            "A_closed_only_hourly_pnl": {
                "sharpe": sharpe(a.to_numpy()),
                "n_periods": int(len(a)),
                "mean_pnl": float(a.mean()) if len(a) else None,
            },
            "B_hourly_equity_pnl_with_open_mtm": {
                "sharpe": sharpe(h_pnl),
                "n_periods": int(np.isfinite(h_pnl).sum()),
                "mean_pnl": float(np.nanmean(h_pnl)) if len(h_pnl) else None,
            },
            "C_hourly_return_capital_max_open": {
                "sharpe": sharpe(h_pnl / max_open) if len(h_pnl) else float("nan"),
                "max_open_capital": max_open,
            },
        },
        "hourly_path": [
            {
                **{k: (pd.Timestamp(v).isoformat() if k == "t" else v) for k, v in rec.items()}
            }
            for rec in hourly.dropna(subset=["pnl"]).to_dict(orient="records")
        ],
    }


def write_trades_jsonl(path: Path, trades: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in trades.to_dict(orient="records"):
            for k in ("entry_ts", "exit_ts"):
                if k in rec and rec[k] is not None:
                    rec[k] = pd.Timestamp(rec[k]).isoformat()
            f.write(json.dumps(rec, default=float) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir", type=Path)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--max-open", type=int, default=50)
    args = ap.parse_args()
    out = args.session_dir.resolve()

    cfg_path = out / "config.json"
    horizon = args.horizon
    if horizon is None and cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
        horizon = int(cfg.get("horizon", 1))
    if horizon is None:
        horizon = 1

    sig = load_signals(out)
    aligned = align_forward_z(sig, horizon=horizon)
    aligned["horizon"] = horizon

    # Unconstrained book uses only rows with known z_fwd
    trades_free = unconstrained_trades(aligned, tau=args.tau, horizon=horizon)

    # Capacity book uses full signal stream (for settle lookup) but only opens
    # when |z|>=tau; align_forward already dropped terminal snaps without fwd z
    # for unconstrained — for capacity, prefer full sig panel + exit lookup.
    # Rebuild capacity input from all signals so mid-session exits resolve,
    # then only open when forward exists via z_fwd merge where possible.
    sig_cap = sig.copy()
    # Attach z_fwd when available (same merge)
    z_fwd = aligned[
        ["coin", "pair", "snapshot_idx", "z_fwd", "exit_ts", "exit_spread_bps"]
    ].drop_duplicates(["coin", "pair", "snapshot_idx"])
    sig_cap = sig_cap.merge(
        z_fwd, on=["coin", "pair", "snapshot_idx"], how="left"
    )
    trades_cap, cap_meta = capacity_trades(
        sig_cap, tau=args.tau, horizon=horizon, max_open=args.max_open
    )

    z_panel = build_z_panel(sig)

    report = {
        "session": str(out),
        "policy": {
            "name": "mechanical_abs_z_persistence",
            "enter": f"|z_t| >= {args.tau}",
            "direction": "sign(z_t)",
            "mean_reversion": False,
            "horizon": horizon,
            "settle": "pnl_proxy = direction * z_{t+H}",
        },
        "n_signal_rows": int(len(sig)),
        "n_rows_with_z_fwd": int(len(aligned)),
        "unconstrained": summarize_book(
            trades_free, z_panel, max_open=args.max_open, label="unconstrained"
        ),
        "capacity_matched": {
            **summarize_book(
                trades_cap, z_panel, max_open=args.max_open, label=f"max_open={args.max_open}"
            ),
            "capacity_meta": cap_meta,
        },
    }

    dest_dir = out / "mechanical_z_baseline"
    dest_dir.mkdir(parents=True, exist_ok=True)
    write_trades_jsonl(dest_dir / "trades_unconstrained.jsonl", trades_free)
    write_trades_jsonl(dest_dir / "trades_capacity_max_open.jsonl", trades_cap)
    report_path = dest_dir / "mechanical_z_baseline_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")

    print(json.dumps(report, indent=2, default=float))
    print(f"\nsaved {report_path}", flush=True)
    print(f"saved trades under {dest_dir}", flush=True)

    u = report["unconstrained"]
    c = report["capacity_matched"]
    print("\n=== HEADLINES (persistence |z|>=tau -> t+H) ===", flush=True)
    print(
        f"Unconstrained: n={u['n_closed']}  DirAcc={u.get('dir_acc')}  "
        f"mean_pnl={u.get('mean_pnl_proxy')}  SharpeB={u['sharpe']['B_hourly_equity_pnl_with_open_mtm']['sharpe']}",
        flush=True,
    )
    print(
        f"Capacity max_open={args.max_open}: n={c['n_closed']}  DirAcc={c.get('dir_acc')}  "
        f"mean_pnl={c.get('mean_pnl_proxy')}  SharpeB={c['sharpe']['B_hourly_equity_pnl_with_open_mtm']['sharpe']}  "
        f"skip_cap={cap_meta['skipped_at_capacity']} skip_pair={cap_meta['skipped_pair_busy']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
