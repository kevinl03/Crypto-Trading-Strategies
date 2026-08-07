"""Offline sim: H=1 vs persistence hold on a live LGBM paper session's signals.

Does NOT touch the running paper trader / collector.

Persistence exit rule (checked from the next snap onward):
  stay open while sign(pred) == position and |pred| >= tau
  else exit at that snap (min hold = 1)

Usage:
  python scripts/paper_trading_3day/sim_persistence_hold.py data/paper_trading/5day_Aug4_2026
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def load_jsonl_rows(out: Path, stem: str) -> list[dict]:
    paths: list[Path] = []
    primary = out / f"{stem}.jsonl"
    if primary.exists():
        paths.append(primary)
    paths.extend(sorted(out.glob(f"{stem}_*.jsonl")))
    rows: list[dict] = []
    for path in paths:
        with path.open(encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def sharpe(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    s = float(np.std(x, ddof=1))
    return float(np.mean(x) / s) if s > 0 else float("nan")


@dataclass
class Bet:
    coin: str
    pair: str
    direction: int
    entry_snap: int
    entry_ts: pd.Timestamp
    entry_z: float
    entry_spread: float
    entry_pred: float


def simulate(
    panel: dict[tuple[str, str, int], dict],
    snaps: list[int],
    snap_ts: dict[int, pd.Timestamp],
    *,
    tau: float,
    max_open: int,
    mode: str,
    max_hold: int | None,
) -> pd.DataFrame:
    """Return closed trades dataframe."""
    open_bets: dict[tuple[str, str], Bet] = {}
    closed: list[dict] = []

    for i, snap in enumerate(snaps):
        # --- exits first ---
        to_close: list[tuple[str, str]] = []
        for key, bet in open_bets.items():
            hold = snap - bet.entry_snap
            if hold < 1:
                continue
            row = panel.get((bet.coin, bet.pair, snap))
            if row is None:
                # missing quote: force exit at last available prior snap if any
                prev = None
                for s in range(snap - 1, bet.entry_snap, -1):
                    prev = panel.get((bet.coin, bet.pair, s))
                    if prev is not None:
                        break
                if prev is None:
                    continue
                exit_snap = int(prev["snapshot_idx"])
                exit_z = float(prev["zscore"])
                exit_spread = float(prev["spread_bps"])
                exit_ts = snap_ts.get(exit_snap, bet.entry_ts)
                reason = "missing"
            else:
                pred = float(row["pred"])
                exit_z = float(row["zscore"])
                exit_spread = float(row["spread_bps"])
                exit_snap = snap
                exit_ts = snap_ts.get(snap, pd.Timestamp(row["ts"]))
                reason = "rule"

                if mode == "h1":
                    # always exit at first eligible snap (>= entry+1)
                    pass
                else:
                    keep = np.sign(pred) == bet.direction and abs(pred) >= tau
                    if max_hold is not None and hold >= max_hold:
                        keep = False
                        reason = "max_hold"
                    if keep:
                        continue

            pnl = float(bet.direction * exit_z)
            d_z = float(bet.direction * (exit_z - bet.entry_z))
            d_bps = float(bet.direction * (exit_spread - bet.entry_spread))
            closed.append(
                {
                    "coin": bet.coin,
                    "pair": bet.pair,
                    "direction": bet.direction,
                    "entry_snap": bet.entry_snap,
                    "exit_snap": exit_snap,
                    "hold_snaps": exit_snap - bet.entry_snap,
                    "entry_ts": bet.entry_ts,
                    "exit_ts": exit_ts,
                    "entry_z": bet.entry_z,
                    "exit_z": exit_z,
                    "entry_pred": bet.entry_pred,
                    "pnl_proxy": pnl,
                    "delta_z": d_z,
                    "dir_bps": d_bps,
                    "dir_hit": int(np.sign(exit_z) == bet.direction) if exit_z != 0 else 0,
                    "exit_reason": reason,
                }
            )
            to_close.append(key)

        for key in to_close:
            open_bets.pop(key, None)

        # --- entries ---
        # candidates at this snap with |pred| >= tau, not already open
        cands: list[tuple[float, str, str, dict]] = []
        # We don't have a full coin/pair list; scan is expensive if we iterate all keys.
        # Build from a precomputed per-snap index instead (passed via closure).
        for coin, pair, row in _snap_rows.get(snap, []):
            if abs(float(row["pred"])) < tau:
                continue
            if (coin, pair) in open_bets:
                continue
            cands.append((abs(float(row["pred"])), coin, pair, row))

        # Match live FIFO-ish: live takes row order from score_snapshot; we rank by |pred|
        # descending so capacity goes to strongest signals (slightly optimistic vs FIFO).
        # Also report count; ranking is a reasonable capacity policy for comparison.
        cands.sort(reverse=True, key=lambda x: x[0])
        for _, coin, pair, row in cands:
            if len(open_bets) >= max_open:
                break
            direction = 1 if float(row["pred"]) > 0 else -1
            open_bets[(coin, pair)] = Bet(
                coin=coin,
                pair=pair,
                direction=direction,
                entry_snap=snap,
                entry_ts=snap_ts.get(snap, pd.Timestamp(row["ts"])),
                entry_z=float(row["zscore"]),
                entry_spread=float(row["spread_bps"]),
                entry_pred=float(row["pred"]),
            )

    # drop still-open at end (same as incomplete inventory)
    return pd.DataFrame(closed)


# filled in main before simulate()
_snap_rows: dict[int, list[tuple[str, str, dict]]] = {}


def hourly_sharpe_from_trades(trades: pd.DataFrame) -> float:
    if trades.empty:
        return float("nan")
    s = trades.set_index("exit_ts").resample("1h")["pnl_proxy"].sum()
    return sharpe(s.to_numpy(dtype=float))


def summarize(name: str, trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"policy": name, "n": 0}
    return {
        "policy": name,
        "n_closed": int(len(trades)),
        "mean_hold_snaps": float(trades["hold_snaps"].mean()),
        "median_hold_snaps": float(trades["hold_snaps"].median()),
        "p90_hold_snaps": float(trades["hold_snaps"].quantile(0.9)),
        "dir_acc": float(trades["dir_hit"].mean()),
        "mean_pnl_proxy": float(trades["pnl_proxy"].mean()),
        "sum_pnl_proxy": float(trades["pnl_proxy"].sum()),
        "mean_delta_z": float(trades["delta_z"].mean()),
        "sum_delta_z": float(trades["delta_z"].sum()),
        "mean_dir_bps": float(trades["dir_bps"].mean()),
        "sum_dir_bps": float(trades["dir_bps"].sum()),
        "pct_dir_bps_pos": float((trades["dir_bps"] > 0).mean()),
        "hourly_sharpe_closed": hourly_sharpe_from_trades(trades),
        "per_trade_sharpe": sharpe(trades["pnl_proxy"].to_numpy()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir", type=Path)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--max-open", type=int, default=50)
    ap.add_argument("--max-hold", type=int, default=20, help="Cap for persistence policy")
    args = ap.parse_args()
    out = args.session_dir.resolve()

    print(f"Loading signals from {out} ...", flush=True)
    rows = load_jsonl_rows(out, "signals")
    if not rows:
        raise SystemExit("no signals")
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["snapshot_idx"] = df["snapshot_idx"].astype(int)
    df = df.dropna(subset=["pred", "zscore"])
    df = df.sort_values(["snapshot_idx", "coin", "pair", "ts"])
    df = df.groupby(["coin", "pair", "snapshot_idx"], as_index=False).last()

    global _snap_rows
    _snap_rows = {}
    panel: dict[tuple[str, str, int], dict] = {}
    for r in df.itertuples(index=False):
        d = {
            "snapshot_idx": int(r.snapshot_idx),
            "ts": r.ts.isoformat(),
            "pred": float(r.pred),
            "zscore": float(r.zscore),
            "spread_bps": float(r.spread_bps),
        }
        panel[(r.coin, r.pair, int(r.snapshot_idx))] = d
        _snap_rows.setdefault(int(r.snapshot_idx), []).append((r.coin, r.pair, d))

    snaps = sorted(_snap_rows)
    snap_ts = {
        int(s): pd.Timestamp(ts)
        for s, ts in df.groupby("snapshot_idx")["ts"].max().items()
    }
    print(
        f"panel: {len(panel):,} rows, snaps {snaps[0]}..{snaps[-1]} ({len(snaps)} unique)",
        flush=True,
    )

    print("Simulating H=1 ...", flush=True)
    h1 = simulate(
        panel, snaps, snap_ts, tau=args.tau, max_open=args.max_open, mode="h1", max_hold=1
    )
    print("Simulating persistence hold ...", flush=True)
    pers = simulate(
        panel,
        snaps,
        snap_ts,
        tau=args.tau,
        max_open=args.max_open,
        mode="persist",
        max_hold=args.max_hold,
    )
    print("Simulating persistence hold (no max) ...", flush=True)
    pers_free = simulate(
        panel,
        snaps,
        snap_ts,
        tau=args.tau,
        max_open=args.max_open,
        mode="persist",
        max_hold=None,
    )

    summaries = [
        summarize("H=1 (baseline)", h1),
        summarize(f"persist (max_hold={args.max_hold})", pers),
        summarize("persist (no max_hold)", pers_free),
    ]

    # Live trades for reference
    live = pd.DataFrame(load_jsonl_rows(out, "trades"))
    if not live.empty:
        live["exit_ts"] = pd.to_datetime(live["exit_ts"], utc=True)
        live["pnl_proxy"] = live["pnl_proxy"].astype(float)
        live["dir_hit"] = live["dir_hit"].astype(int)
        live["hold_snaps"] = (live["exit_snap"] - live["entry_snap"]).astype(int)
        live["delta_z"] = live["direction"] * (live["exit_z"] - live["entry_z"])
        live["dir_bps"] = live["direction"] * live["spread_delta_bps"]
        summaries.insert(0, summarize("LIVE H=1 (recorded)", live))

    print("\n" + "=" * 72)
    print("PERSISTENCE HOLD SIM (offline; live session untouched)")
    print(f"session={out}")
    print(f"tau={args.tau}  max_open={args.max_open}")
    print("=" * 72)
    cols = [
        "policy",
        "n_closed",
        "mean_hold_snaps",
        "median_hold_snaps",
        "p90_hold_snaps",
        "dir_acc",
        "mean_pnl_proxy",
        "sum_pnl_proxy",
        "mean_delta_z",
        "sum_delta_z",
        "mean_dir_bps",
        "sum_dir_bps",
        "pct_dir_bps_pos",
        "hourly_sharpe_closed",
        "per_trade_sharpe",
    ]
    tab = pd.DataFrame(summaries)[cols]
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", lambda x: f"{x:0.4f}")
    print(tab.to_string(index=False))
    print(
        "\nNotes:\n"
        "- pnl_proxy = direction * z_exit (same unit as live; favors short holds with large |z|)\n"
        "- delta_z = direction * (z_exit - z_entry) (move captured over the hold)\n"
        "- dir_bps = direction * (spread_exit - spread_entry) (crude bps, no fees)\n"
        "- Entries ranked by |pred| when capacity-constrained (slightly optimistic vs live FIFO)\n"
        "- Live paper trading / collector were not modified"
    )

    out_json = out / "sim_persistence_hold_report.json"
    out_json.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
