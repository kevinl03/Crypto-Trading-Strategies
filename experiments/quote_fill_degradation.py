#!/usr/bin/env python3
"""
Measured execution degradation: quote-mid fills vs bid/ask fills.

For every viable-region pair-model (sigma/fee >= 0.5) in the HF snapshot
windows, runs the paper's OU/z-score protocol with signals computed on
quote-mid spreads, then re-prices each trade's fills at the touch:

    short spread:  sell venue1 at bid, buy venue2 at ask  (entry)
                   buy venue1 at ask, sell venue2 at bid  (exit)

so each round trip additionally pays the sum of both venues' quoted
half-widths at entry and at exit, measured from the order book snapshot
at those instants (not an assumed constant).

Also sweeps the maximum-holding cap (30/60/120/1440 bars) for the May
headline pair (WIF binance-mexc, OHLCV) to measure win-rate under
bounded holding.

Output: paper/quote_fill_data.json (provenance-stamped).
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments.backtest_historical import (  # noqa: E402
    run_ou_strategy,
    run_zscore_strategy,
    compute_fees_bps,
    load_parquet_data,
    build_spread_from_ohlcv,
)

CEX = ROOT / "datasets" / "statarb-crypto-research"
MIN_JOINT_SNAPSHOTS = 2000
MIN_RATIO = 0.5


def load_quotes(path: Path) -> pd.DataFrame:
    t = pd.read_parquet(path, columns=["snapshot_idx", "exchange", "coin", "error", "payload"])
    t = t[t.error.isna()]
    recs = []
    for r in t.itertuples():
        try:
            p = json.loads(r.payload) if isinstance(r.payload, str) else r.payload
            bid, ask = p.get("bid"), p.get("ask")
            if bid and ask and 0 < float(bid) <= float(ask):
                bid, ask = float(bid), float(ask)
                mid = (bid + ask) / 2
                recs.append((r.snapshot_idx, r.coin, r.exchange, mid,
                             (ask - bid) / 2 / mid * 1e4))  # half-width bps
        except Exception:
            continue
    return pd.DataFrame(recs, columns=["snap", "coin", "exchange", "mid", "hw_bps"])


def run_window(name: str, q: pd.DataFrame) -> list[dict]:
    rows = []
    for coin, grp in q.groupby("coin"):
        piv_m = grp.pivot_table(index="snap", columns="exchange", values="mid")
        piv_h = grp.pivot_table(index="snap", columns="exchange", values="hw_bps")
        venues = [c for c in piv_m.columns if piv_m[c].notna().sum() >= MIN_JOINT_SNAPSHOTS]
        for e1, e2 in combinations(sorted(venues), 2):
            j = pd.concat([piv_m[e1].rename("m1"), piv_m[e2].rename("m2"),
                           piv_h[e1].rename("h1"), piv_h[e2].rename("h2")], axis=1).dropna()
            if len(j) < MIN_JOINT_SNAPSHOTS:
                continue
            spread = ((j.m1 - j.m2) / ((j.m1 + j.m2) / 2) * 1e4).to_numpy()
            cross = (j.h1 + j.h2).to_numpy()  # bps paid to cross both books once
            fee = compute_fees_bps(e1, e2)
            std = float(np.std(spread))
            if std / fee < MIN_RATIO:
                continue
            snaps = j.index.to_numpy()
            for model, fn in (("ou", run_ou_strategy), ("zscore", run_zscore_strategy)):
                trades = fn(spread, snaps, fee)
                if len(trades) < 5:
                    continue
                mid_pnls = np.array([t.pnl_net_bps for t in trades])
                exe_pnls = np.array([t.pnl_net_bps - cross[t.entry_idx] - cross[t.exit_idx]
                                     for t in trades])
                rows.append({
                    "window": name, "coin": coin, "ex1": e1, "ex2": e2, "model": model,
                    "spread_std_bps": round(std, 2), "fee_bps": round(fee, 1),
                    "n_trades": len(trades),
                    "median_cross_bps": round(float(np.median(cross)), 2),
                    "mid_net_bps": round(float(mid_pnls.sum()), 1),
                    "exe_net_bps": round(float(exe_pnls.sum()), 1),
                    "mid_win": round(float((mid_pnls > 0).mean()), 3),
                    "exe_win": round(float((exe_pnls > 0).mean()), 3),
                })
    return rows


def main():
    all_rows = []
    for name, path in (("train", CEX / "ticker.parquet"),
                       ("test", CEX / "test" / "ticker.parquet")):
        print(f"Loading {name} quotes...")
        q = load_quotes(path)
        rows = run_window(name, q)
        all_rows.extend(rows)
        print(f"  {len(rows)} viable pair-models (sigma/fee >= {MIN_RATIO}, >=5 trades)")

    mid_prof = [r for r in all_rows if r["mid_net_bps"] > 0]
    exe_prof = [r for r in all_rows if r["exe_net_bps"] > 0]
    retention = [r["exe_net_bps"] / r["mid_net_bps"] for r in mid_prof if r["mid_net_bps"] > 0]
    dwin = [r["mid_win"] - r["exe_win"] for r in all_rows]
    print(f"\n=== Quote-fill degradation ({len(all_rows)} viable pair-models) ===")
    print(f"  profitable @ mid fills:     {len(mid_prof)}/{len(all_rows)}")
    print(f"  profitable @ bid/ask fills: {len(exe_prof)}/{len(all_rows)}")
    print(f"  P&L retention (mid-profitable set): median {np.median(retention)*100:.0f}%, "
          f"IQR [{np.percentile(retention,25)*100:.0f}%, {np.percentile(retention,75)*100:.0f}%]")
    print(f"  win-rate drop: median {np.median(dwin)*100:.1f} pp")
    print(f"  median crossing cost per book-touch: "
          f"{np.median([r['median_cross_bps'] for r in all_rows]):.1f} bps")

    # ── Max-holding sweep on the May headline pair ──────────────────────
    print("\n=== Max-holding sweep: WIF binance-mexc OU, May OHLCV ===")
    data = load_parquet_data(str(ROOT / "data" / "historical"))
    sdf = build_spread_from_ohlcv(data["WIF"]["binance"], data["WIF"]["mexc"])
    spread = sdf["spread_bps"].values
    ts = sdf["datetime"].values if "datetime" in sdf.columns else sdf["timestamp"].values
    fee = compute_fees_bps("binance", "mexc")
    sweep = []
    for cap in (30, 60, 120, 1440):
        trades = run_ou_strategy(spread, ts, fee, max_holding=cap)
        pnls = np.array([t.pnl_net_bps for t in trades])
        stopped = sum(1 for t in trades if t.holding_periods >= cap)
        row = {"max_holding_bars": cap, "n_trades": len(trades),
               "win_rate": round(float((pnls > 0).mean()), 3),
               "net_bps": round(float(pnls.sum()), 1),
               "stopped_out": stopped}
        sweep.append(row)
        print(f"  cap={cap:>5}: {row['n_trades']} trades, win {row['win_rate']:.1%}, "
              f"net {row['net_bps']:+.0f}, stop-outs {stopped}")

    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                      text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        sha = None
    payload = {
        "provenance": {
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_sha": sha, "script": "experiments/quote_fill_degradation.py",
            "params": {"min_ratio": MIN_RATIO, "min_joint_snapshots": MIN_JOINT_SNAPSHOTS,
                       "protocol": "window=60, entry_z=2, exit_z=0.5, max_holding=1440"},
        },
        "summary": {
            "viable_pair_models": len(all_rows),
            "profitable_mid": len(mid_prof), "profitable_exe": len(exe_prof),
            "pnl_retention_median": round(float(np.median(retention)), 3),
            "pnl_retention_iqr": [round(float(np.percentile(retention, 25)), 3),
                                   round(float(np.percentile(retention, 75)), 3)],
            "win_drop_pp_median": round(float(np.median(dwin)) * 100, 1),
        },
        "max_holding_sweep": sweep,
        "results": all_rows,
    }
    dest = ROOT / "paper" / "quote_fill_data.json"
    with open(dest, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {dest}")


if __name__ == "__main__":
    main()
