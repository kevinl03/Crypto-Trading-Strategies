#!/usr/bin/env python3
"""
Out-of-sample replication of the fee-floor viability law on quote data.

Builds pairwise cross-venue spread series from ticker QUOTE MIDS (the
executable instrument) for all 23 assets x 12 venues in the HF snapshot
dataset, and runs the paper's exact backtest protocol (OU + z-score,
window=60 bars, entry_z=2, exit_z=0.5, max_holding=1440, per-pair CCXT
taker fees) independently on the two chronologically separated windows:

    train: 2026-06-13 -> 06-16 (~65h, ~60s cadence)
    test:  2026-06-22 -> 06-24 (~57h)

This tests the paper's threshold claim at 23-asset breadth on the quote
instrument, including the 10-25 bps "gray zone" the 5-asset design skipped.

Output: paper/hf_replication_data.json (provenance-stamped) + stdout summary.
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
)

CEX = ROOT / "datasets" / "statarb-crypto-research"
MIN_JOINT_SNAPSHOTS = 2000
BUCKETS = [(0, 10), (10, 15), (15, 25), (25, 50), (50, float("inf"))]


def load_mids(path: Path) -> pd.DataFrame:
    """Extract quote mids per (snapshot, coin, exchange) from ticker payloads."""
    t = pd.read_parquet(path, columns=["snapshot_idx", "exchange", "coin", "error", "payload"])
    t = t[t.error.isna()]
    recs = []
    for r in t.itertuples():
        try:
            p = json.loads(r.payload) if isinstance(r.payload, str) else r.payload
            bid, ask = p.get("bid"), p.get("ask")
            if bid and ask and float(bid) > 0:
                mid = (float(bid) + float(ask)) / 2
            elif p.get("last"):
                mid = float(p["last"])
            else:
                continue
            recs.append((r.snapshot_idx, r.coin, r.exchange, mid))
        except Exception:
            continue
    return pd.DataFrame(recs, columns=["snap", "coin", "exchange", "mid"])


def run_window(name: str, mids: pd.DataFrame) -> list[dict]:
    rows = []
    for coin, grp in mids.groupby("coin"):
        piv = grp.pivot_table(index="snap", columns="exchange", values="mid")
        venues = [c for c in piv.columns if piv[c].notna().sum() >= MIN_JOINT_SNAPSHOTS]
        for e1, e2 in combinations(sorted(venues), 2):
            j = piv[[e1, e2]].dropna()
            if len(j) < MIN_JOINT_SNAPSHOTS:
                continue
            spread = ((j[e1] - j[e2]) / ((j[e1] + j[e2]) / 2) * 1e4).to_numpy()
            snaps = j.index.to_numpy()
            fee = compute_fees_bps(e1, e2)
            std = float(np.std(spread))
            for model, fn in (("ou", run_ou_strategy), ("zscore", run_zscore_strategy)):
                trades = fn(spread, snaps, fee)
                pnls = [t.pnl_net_bps for t in trades]
                rows.append({
                    "window": name, "coin": coin, "ex1": e1, "ex2": e2,
                    "model": model, "n_bars": len(spread),
                    "spread_std_bps": round(std, 2), "fee_bps": round(fee, 1),
                    "n_trades": len(pnls),
                    "net_bps": round(float(np.sum(pnls)), 1) if pnls else 0.0,
                    "win_rate": round(float(np.mean([p > 0 for p in pnls])), 3) if pnls else None,
                    "sharpe": round(float(np.mean(pnls) / np.std(pnls)), 2)
                              if len(pnls) > 1 and np.std(pnls) > 0 else None,
                })
    return rows


def bucket_summary(rows: list[dict], min_trades: int = 5) -> list[dict]:
    out = []
    active = [r for r in rows if r["n_trades"] >= min_trades]
    for lo, hi in BUCKETS:
        b = [r for r in active if lo <= r["spread_std_bps"] < hi]
        if not b:
            continue
        prof = [r for r in b if r["net_bps"] > 0]
        out.append({
            "bucket_bps": f"{lo}-{hi if hi != float('inf') else 'inf'}",
            "pair_models": len(b), "profitable": len(prof),
            "rate": round(len(prof) / len(b), 3),
        })
    return out


def main():
    results = []
    for name, path in (("train", CEX / "ticker.parquet"),
                       ("test", CEX / "test" / "ticker.parquet")):
        print(f"Loading {name} ticker payloads...")
        mids = load_mids(path)
        print(f"  {len(mids):,} mid observations, "
              f"{mids.coin.nunique()} coins, {mids.exchange.nunique()} venues")
        rows = run_window(name, mids)
        results.extend(rows)
        print(f"  {len(rows)} pair-model backtests")

    print("\n=== Fee-floor law on quote data (pair-models with >=5 trades) ===")
    summary = {}
    for name in ("train", "test"):
        w = [r for r in results if r["window"] == name]
        s = bucket_summary(w)
        summary[name] = s
        print(f"\n{name.upper()}:")
        for b in s:
            print(f"  std {b['bucket_bps']:>8}: {b['profitable']:>3}/{b['pair_models']:<3} "
                  f"profitable ({b['rate']:.0%})")

    # cross-window consistency: pairs profitable in train — do they hold in test?
    key = lambda r: (r["coin"], r["ex1"], r["ex2"], r["model"])
    tr = {key(r): r for r in results if r["window"] == "train" and r["n_trades"] >= 5}
    te = {key(r): r for r in results if r["window"] == "test" and r["n_trades"] >= 5}
    common = set(tr) & set(te)
    tr_prof = {k for k in common if tr[k]["net_bps"] > 0}
    held = {k for k in tr_prof if te[k]["net_bps"] > 0}
    print(f"\nCross-window: {len(common)} pair-models evaluable in both; "
          f"{len(tr_prof)} profitable in train; {len(held)} remain profitable in test "
          f"({(len(held)/len(tr_prof)*100) if tr_prof else 0:.0f}% persistence)")

    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                      text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        sha = None
    payload = {
        "provenance": {
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_sha": sha, "script": "experiments/hf_grayzone_replication.py",
            "protocol": {"window": 60, "entry_z": 2.0, "exit_z": 0.5,
                         "max_holding": 1440, "min_joint_snapshots": MIN_JOINT_SNAPSHOTS,
                         "instrument": "ticker quote mids (~60s cadence)"},
        },
        "bucket_summary": summary,
        "cross_window": {"evaluable_both": len(common), "train_profitable": len(tr_prof),
                         "held_in_test": len(held)},
        "results": results,
    }
    dest = ROOT / "paper" / "hf_replication_data.json"
    with open(dest, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {dest}")


if __name__ == "__main__":
    main()
