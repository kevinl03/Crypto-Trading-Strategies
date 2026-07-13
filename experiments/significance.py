#!/usr/bin/env python3
"""
Statistical significance for the canonical 100 pair-model backtest.

For every pair-model in the paper's protocol, captures the trade list,
aggregates net P&L by entry day, and runs a day-block bootstrap
(resampling days with replacement, B=10,000) to obtain:

  - 95% CI on total net P&L
  - one-sided bootstrap p-value for net > 0

Then applies Benjamini-Hochberg FDR control (q=0.05) across all
pair-models with >=5 trades to report how many profitable pair-models
survive multiplicity correction.

Output: paper/significance_data.json (provenance-stamped).
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
    load_parquet_data,
    build_spread_from_ohlcv,
    run_ou_strategy,
    run_zscore_strategy,
    compute_fees_bps,
)

EXCLUDE = {"kraken", "coinbase", "phemex", "gateio"}
TOP_N_PAIRS = 10
B = 10_000
RNG = np.random.default_rng(7)
HEADLINE = {("WIF", "binance", "mexc", "ou"), ("PEPE", "binance", "cryptocom", "ou"),
            ("CRV", "cryptocom", "mexc", "zscore")}


def day_bootstrap(daily: np.ndarray) -> tuple[float, float, float]:
    n = len(daily)
    idx = RNG.integers(0, n, size=(B, n))
    sums = daily[idx].sum(axis=1)
    return (float(np.percentile(sums, 2.5)), float(np.percentile(sums, 97.5)),
            float(np.mean(sums <= 0)))


def main():
    data = load_parquet_data(str(ROOT / "data" / "historical"))
    rows = []
    for asset in sorted(data):
        venues = [e for e in sorted(data[asset]) if e not in EXCLUDE]
        pairs = []
        for e1, e2 in combinations(venues, 2):
            sdf = build_spread_from_ohlcv(data[asset][e1], data[asset][e2])
            if len(sdf) >= 500:
                pairs.append((e1, e2, sdf))
        pairs.sort(key=lambda x: x[2]["spread_bps"].std(), reverse=True)
        for e1, e2, sdf in pairs[:TOP_N_PAIRS]:
            spread = sdf["spread_bps"].values
            ts = sdf["datetime"].values if "datetime" in sdf.columns else sdf["timestamp"].values
            fee = compute_fees_bps(e1, e2)
            for model, fn in (("ou", run_ou_strategy), ("zscore", run_zscore_strategy)):
                trades = fn(spread, ts, fee)
                if len(trades) < 5:
                    continue
                df = pd.DataFrame({
                    "day": [str(t.entry_time)[:10] for t in trades],
                    "pnl": [t.pnl_net_bps for t in trades],
                })
                daily = df.groupby("day").pnl.sum().to_numpy()
                lo, hi, p = day_bootstrap(daily)
                rows.append({
                    "asset": asset, "ex1": e1, "ex2": e2, "model": model,
                    "n_trades": len(trades), "n_days": len(daily),
                    "net_bps": round(float(df.pnl.sum()), 1),
                    "ci95": [round(lo, 1), round(hi, 1)],
                    "p_one_sided": round(p, 5),
                })

    # Benjamini-Hochberg at q=0.05
    q = 0.05
    ps = sorted((r["p_one_sided"], i) for i, r in enumerate(rows))
    m = len(ps)
    thresh_idx = -1
    for rank, (p, _) in enumerate(ps, start=1):
        if p <= q * rank / m:
            thresh_idx = rank
    surviving = {i for _, i in ps[:thresh_idx]} if thresh_idx > 0 else set()
    for i, r in enumerate(rows):
        r["bh_significant"] = i in surviving

    prof = [r for r in rows if r["net_bps"] > 0]
    ci_pos = [r for r in prof if r["ci95"][0] > 0]
    bh_pos = [r for r in prof if r["bh_significant"]]
    print(f"pair-models evaluated: {m} | profitable: {len(prof)}")
    print(f"profitable with 95% CI lower bound > 0: {len(ci_pos)}")
    print(f"profitable surviving BH (q=0.05):        {len(bh_pos)}")
    print("\nHeadline pairs:")
    for r in rows:
        if (r["asset"], r["ex1"], r["ex2"], r["model"]) in HEADLINE:
            print(f"  {r['asset']} {r['ex1']}-{r['ex2']} {r['model']}: net {r['net_bps']:+.0f} "
                  f"CI95 [{r['ci95'][0]:+.0f}, {r['ci95'][1]:+.0f}] p={r['p_one_sided']} "
                  f"BH={'yes' if r['bh_significant'] else 'no'}")

    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                      text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        sha = None
    payload = {
        "provenance": {
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_sha": sha, "script": "experiments/significance.py",
            "method": f"day-block bootstrap B={B}, seed=7; BH q=0.05 across {m} pair-models",
        },
        "summary": {"evaluated": m, "profitable": len(prof),
                    "ci_lower_positive": len(ci_pos), "bh_significant": len(bh_pos)},
        "results": rows,
    }
    dest = ROOT / "paper" / "significance_data.json"
    with open(dest, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {dest}")


if __name__ == "__main__":
    main()
