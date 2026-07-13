#!/usr/bin/env python3
"""
Generate the paper's secondary tables from canonical data:

  A. Entry-threshold (k) ablation      — WIF binance-mexc OU
  B. Subperiod stability (3 x ~10d)    — WIF binance-mexc OU
  C. Extended baselines                — buy-and-hold, random-entry MC

Prints LaTeX-ready rows and writes paper/table_data.json with provenance.

Usage:
    python experiments/paper_tables.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.backtest_historical import (  # noqa: E402
    load_parquet_data,
    build_spread_from_ohlcv,
    run_ou_strategy,
    compute_fees_bps,
)

DATA_DIR = "data/historical"
ASSET, EX1, EX2 = "WIF", "binance", "mexc"
RNG = np.random.default_rng(42)


def summarize(trades):
    if not trades:
        return {"trades": 0, "win_rate": 0.0, "net_bps": 0.0, "sharpe": 0.0}
    pnls = np.array([t.pnl_net_bps for t in trades])
    return {
        "trades": len(pnls),
        "win_rate": float(np.mean(pnls > 0)),
        "net_bps": float(np.sum(pnls)),
        "sharpe": float(np.mean(pnls) / np.std(pnls)) if len(pnls) > 1 and np.std(pnls) > 0 else 0.0,
    }


def main():
    data = load_parquet_data(DATA_DIR)
    spread_df = build_spread_from_ohlcv(data[ASSET][EX1], data[ASSET][EX2])
    spread = spread_df["spread_bps"].values
    ts = (
        spread_df["datetime"].values
        if "datetime" in spread_df.columns
        else spread_df["timestamp"].values
    )
    fee = compute_fees_bps(EX1, EX2)
    print(f"{ASSET} {EX1}-{EX2}: {len(spread)} bars, fee={fee:.1f} bps\n")

    out = {"asset": ASSET, "pair": f"{EX1}-{EX2}", "fee_bps": fee}

    # ── A. Entry-threshold ablation ─────────────────────────────────────
    print("A. Entry-threshold ablation (OU, exit_z=0.5, max_holding=1440)")
    out["ablation_k"] = []
    for k in [1.0, 1.5, 2.0, 2.5, 3.0]:
        s = summarize(run_ou_strategy(spread, ts, fee, entry_z=k, exit_z=0.5))
        s["k"] = k
        out["ablation_k"].append(s)
        print(f"  k={k}: {s['trades']} trades, win {s['win_rate']:.0%}, "
              f"net {s['net_bps']:+.0f}, sharpe {s['sharpe']:.2f}")

    # ── B. Subperiod stability (three equal thirds ≈ 10 days each) ──────
    print("\nB. Subperiod stability (defaults: entry_z=2, exit_z=0.5)")
    out["subperiods"] = []
    third = len(spread) // 3
    for i in range(3):
        lo = i * third
        hi = (i + 1) * third if i < 2 else len(spread)
        s = summarize(run_ou_strategy(spread[lo:hi], ts[lo:hi], fee))
        s["period"] = f"days {i*10+1}-{(i+1)*10}"
        out["subperiods"].append(s)
        print(f"  {s['period']}: {s['trades']} trades, win {s['win_rate']:.0%}, "
              f"net {s['net_bps']:+.0f}, sharpe {s['sharpe']:.2f}")
    s = summarize(run_ou_strategy(spread, ts, fee))
    s["period"] = "full 30d"
    out["subperiods"].append(s)
    print(f"  full: {s['trades']} trades, win {s['win_rate']:.0%}, "
          f"net {s['net_bps']:+.0f}, sharpe {s['sharpe']:.2f}")

    # ── C. Baselines ────────────────────────────────────────────────────
    print("\nC. Baselines")
    ou_full = run_ou_strategy(spread, ts, fee)
    n_trades = len(ou_full)
    avg_hold = int(np.mean([t.holding_periods for t in ou_full])) or 1

    close = data[ASSET][EX1]["close"].values
    bh_bps = (close[-1] / close[0] - 1.0) * 10_000
    out["buy_and_hold_bps"] = float(bh_bps)
    print(f"  Buy-and-hold ({EX1}): {bh_bps:+.0f} bps over window")

    # Random entry: same trade count and holding period, pays fee per trade
    sims = []
    for _ in range(1000):
        idx = RNG.integers(0, len(spread) - avg_hold - 1, size=n_trades)
        side = RNG.choice([-1.0, 1.0], size=n_trades)
        gross = side * (spread[idx] - spread[idx + avg_hold])
        sims.append(np.sum(gross - fee))
    sims = np.array(sims)
    out["random_entry"] = {
        "mean_net_bps": float(np.mean(sims)),
        "p5": float(np.percentile(sims, 5)),
        "p95": float(np.percentile(sims, 95)),
        "win_rate_mean": float(np.mean(sims > 0)),
        "n_sims": 1000,
        "n_trades": n_trades,
        "avg_hold": avg_hold,
    }
    print(f"  Random entry (n={n_trades}, hold={avg_hold}): "
          f"mean {np.mean(sims):+.0f} bps  [p5 {np.percentile(sims, 5):+.0f}, "
          f"p95 {np.percentile(sims, 95):+.0f}]")

    # ── provenance + write ──────────────────────────────────────────────
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        git_sha = None
    out["provenance"] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha,
        "script": "experiments/paper_tables.py",
        "seed": 42,
    }
    out_path = Path("paper") / "table_data.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
