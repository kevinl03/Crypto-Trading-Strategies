#!/usr/bin/env python3
"""
Cross-instrument executability verification on the HF snapshot datasets.

For the paper's headline exchange pairs, compares spread standard deviation
measured from two instruments over the SAME window:

  1. 1-minute OHLCV closes (last-trade prints)   — the paper's instrument
  2. Ticker quote mids (bid/ask midpoints, ~60s)  — the executable instrument

A pair whose print-dispersion >> quote-dispersion is dominated by stale
prints (non-executable); a pair where the two agree has genuine quote-level
dislocation. Also reports quoted bid-ask width per venue and the test-window
(Jun 22-24) quote dispersion for edge-decay measurement.

Output: paper/executability_data.json (provenance-stamped) + stdout table.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CEX = ROOT / "datasets" / "statarb-crypto-research"

PAIRS = [
    ("WIF", "binance", "mexc"),
    ("WIF", "binance", "cryptocom"),
    ("PEPE", "binance", "cryptocom"),
    ("CRV", "cryptocom", "mexc"),
]
# Pairwise spread std (bps) over the May 30-day OHLCV backtest window,
# from data/historical/backtest_results.json (git 47d7fa3).
MAY_OHLCV_STD = {
    ("WIF", "binance", "mexc"): 26.15,
    ("WIF", "binance", "cryptocom"): 93.94,
    ("PEPE", "binance", "cryptocom"): 24.95,
    ("CRV", "cryptocom", "mexc"): 76.66,
}
COINS = {p[0] for p in PAIRS}
EXES = {e for p in PAIRS for e in p[1:]}


def spread_bps(a: pd.Series, b: pd.Series) -> pd.Series:
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    return (j.a - j.b) / ((j.a + j.b) / 2) * 1e4


def ohlcv_pair_std(coin: str, e1: str, e2: str) -> tuple[int, float]:
    o = OHLCV[(OHLCV.coin == coin) & (OHLCV.exchange.isin({e1, e2}))]
    a = o[o.exchange == e1].set_index("timestamp")["close"].sort_index()
    b = o[o.exchange == e2].set_index("timestamp")["close"].sort_index()
    s = spread_bps(a, b)
    return len(s), float(s.std())


def ticker_mids(path: Path) -> pd.DataFrame:
    t = pd.read_parquet(path, columns=["snapshot_idx", "exchange", "coin", "error", "payload"])
    t = t[t.coin.isin(COINS) & t.exchange.isin(EXES) & t.error.isna()]
    recs = []
    for r in t.itertuples():
        try:
            p = json.loads(r.payload) if isinstance(r.payload, str) else r.payload
            bid, ask = p.get("bid"), p.get("ask")
            if bid and ask and float(bid) > 0:
                bid, ask = float(bid), float(ask)
                recs.append((r.snapshot_idx, r.coin, r.exchange,
                             (bid + ask) / 2, (ask - bid) / ((ask + bid) / 2) * 1e4))
        except Exception:
            continue
    return pd.DataFrame(recs, columns=["snap", "coin", "exchange", "mid", "quoted_bps"])


def quote_pair_std(mids: pd.DataFrame, coin: str, e1: str, e2: str) -> tuple[int, float]:
    piv = mids[mids.coin == coin].pivot_table(index="snap", columns="exchange", values="mid")
    if e1 not in piv.columns or e2 not in piv.columns:
        return 0, float("nan")
    s = spread_bps(piv[e1], piv[e2])
    return len(s), float(s.std())


print("Loading HF datasets...")
OHLCV = pd.read_parquet(CEX / "ohlcv.parquet")
OHLCV = OHLCV[OHLCV.coin.isin(COINS) & OHLCV.exchange.isin(EXES)]
train_mids = ticker_mids(CEX / "ticker.parquet")
test_mids = ticker_mids(CEX / "test" / "ticker.parquet")

out = {"pairs": [], "quoted_width_bps": {}}
hdr = f"{'pair':<26} {'May prints':>10} {'Jun prints':>10} {'Jun quotes':>10} {'Jun22-24 q':>10} {'stale?':>7}"
print("\nSpread std (bps) by instrument:")
print(hdr)
for coin, e1, e2 in PAIRS:
    n_o, std_o = ohlcv_pair_std(coin, e1, e2)
    n_q, std_q = quote_pair_std(train_mids, coin, e1, e2)
    n_t, std_t = quote_pair_std(test_mids, coin, e1, e2)
    stale = std_o > 3 * std_q  # prints disperse >3x more than quotes
    row = {
        "coin": coin, "ex1": e1, "ex2": e2,
        "may_ohlcv_std": MAY_OHLCV_STD[(coin, e1, e2)],
        "jun_train_ohlcv_std": round(std_o, 1), "jun_train_ohlcv_n": n_o,
        "jun_train_quote_std": round(std_q, 1), "jun_train_quote_n": n_q,
        "jun_test_quote_std": round(std_t, 1), "jun_test_quote_n": n_t,
        "stale_print_dominated": bool(stale),
    }
    out["pairs"].append(row)
    print(f"{coin} {e1}-{e2:<14} {row['may_ohlcv_std']:>10.1f} {std_o:>10.1f} "
          f"{std_q:>10.1f} {std_t:>10.1f} {str(stale):>7}")

print("\nQuoted bid-ask width (bps), train window:")
w = train_mids.groupby(["coin", "exchange"])["quoted_bps"].agg(["count", "median"]).round(1)
print(w.to_string())
out["quoted_width_bps"] = {
    f"{c}/{e}": {"n": int(r["count"]), "median": float(r["median"])}
    for (c, e), r in w.iterrows()
}

try:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                  text=True, stderr=subprocess.DEVNULL).strip()
except Exception:
    sha = None
out["provenance"] = {
    "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "git_sha": sha,
    "script": "experiments/analyze_executability.py",
    "train_window": "2026-06-13T23:48Z/2026-06-16T19:34Z",
    "test_window": "2026-06-22T03:10Z/2026-06-24T15:10Z",
}
dest = ROOT / "paper" / "executability_data.json"
with open(dest, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {dest}")
