#!/usr/bin/env python3
"""
Direct lead-lag measurement via return cross-correlation.

The paper currently infers venue lag from OU half-life. This script
measures it directly: for venue pair (A, B), compute the Pearson
correlation of A's log-return at time t with B's log-return at time
t+k for lags k in [-LMAX, +LMAX] bars. A peak at k > 0 means B follows
A (A leads). Reported on both instruments:

  1. May 30-day 1-minute OHLCV closes (prints) - the paper's data
  2. June HF snapshot quote mids (~60s)        - the executable data

Contrast expectation from the stale-print diagnosis: Crypto.com pairs
should show large print-based lags that collapse on quotes; the MEXC
lag should survive on both.

Output: paper/leadlag_data.json (provenance-stamped).
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments.backtest_historical import load_parquet_data  # noqa: E402

CEX = ROOT / "datasets" / "statarb-crypto-research"
LMAX = 45  # bars
PAIRS = [
    ("WIF", "binance", "mexc"),
    ("WIF", "binance", "cryptocom"),
    ("PEPE", "binance", "cryptocom"),
    ("CRV", "binance", "cryptocom"),
    ("CRV", "cryptocom", "mexc"),
    ("DOGE", "binance", "cryptocom"),  # efficient-asset control
]


def xcorr_profile(a: np.ndarray, b: np.ndarray, lmax: int) -> dict:
    """corr(a_t, b_{t+k}) for k in [-lmax, lmax]; peak k>0 => a leads b."""
    ra = np.diff(np.log(a))
    rb = np.diff(np.log(b))
    n = len(ra)
    out = {}
    for k in range(-lmax, lmax + 1):
        if k >= 0:
            x, y = ra[: n - k], rb[k:]
        else:
            x, y = ra[-k:], rb[: n + k]
        if len(x) < 100 or np.std(x) == 0 or np.std(y) == 0:
            out[k] = np.nan
            continue
        out[k] = float(np.corrcoef(x, y)[0, 1])
    ks = np.array(sorted(out))
    vs = np.array([out[k] for k in ks])
    peak_idx = int(np.nanargmax(vs))
    return {
        "peak_lag_bars": int(ks[peak_idx]),
        "peak_corr": round(float(vs[peak_idx]), 3),
        "corr_at_0": round(float(out[0]), 3),
        "n_obs": n,
        "profile": {int(k): (None if np.isnan(out[k]) else round(out[k], 4)) for k in ks},
    }


def hf_quote_mids() -> pd.DataFrame:
    frames = []
    for split in ("ticker.parquet", "test/ticker.parquet"):
        t = pd.read_parquet(CEX / split, columns=["snapshot_idx", "exchange", "coin", "error", "payload"])
        t = t[t.error.isna() & t.coin.isin({p[0] for p in PAIRS})
              & t.exchange.isin({e for p in PAIRS for e in p[1:]})]
        recs = []
        for r in t.itertuples():
            try:
                p = json.loads(r.payload) if isinstance(r.payload, str) else r.payload
                bid, ask = p.get("bid"), p.get("ask")
                if bid and ask and float(bid) > 0:
                    recs.append((split, r.snapshot_idx, r.coin, r.exchange,
                                 (float(bid) + float(ask)) / 2))
            except Exception:
                continue
        frames.append(pd.DataFrame(recs, columns=["split", "snap", "coin", "exchange", "mid"]))
    return pd.concat(frames)


def main():
    results = []

    print("Instrument 1: May OHLCV prints (1-min closes)")
    data = load_parquet_data(str(ROOT / "data" / "historical"))
    for coin, e1, e2 in PAIRS:
        if coin not in data or e1 not in data[coin] or e2 not in data[coin]:
            continue
        a = data[coin][e1].set_index("timestamp")["close"].sort_index()
        b = data[coin][e2].set_index("timestamp")["close"].sort_index()
        j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
        prof = xcorr_profile(j.a.to_numpy(), j.b.to_numpy(), LMAX)
        prof.update({"instrument": "ohlcv_prints_may", "coin": coin, "ex1": e1, "ex2": e2})
        results.append(prof)
        print(f"  {coin} {e1}->{e2}: peak lag {prof['peak_lag_bars']:+d} min "
              f"(corr {prof['peak_corr']}), corr@0 {prof['corr_at_0']}")

    print("\nInstrument 2: June HF quote mids (~60s bars, train+test pooled)")
    mids = hf_quote_mids()
    for coin, e1, e2 in PAIRS:
        rows = []
        for split, grp in mids[mids.coin == coin].groupby("split"):
            piv = grp.pivot_table(index="snap", columns="exchange", values="mid")
            if e1 not in piv.columns or e2 not in piv.columns:
                continue
            j = piv[[e1, e2]].dropna()
            if len(j) >= 1000:
                rows.append(j)
        if not rows:
            continue
        # analyse windows separately, average profiles (avoids cross-window jumps)
        profs = [xcorr_profile(j[e1].to_numpy(), j[e2].to_numpy(), LMAX) for j in rows]
        keys = sorted(int(k) for k in profs[0]["profile"])
        avg = {k: float(np.nanmean([p["profile"][k] for p in profs if p["profile"][k] is not None]))
               for k in keys}
        ks = np.array(keys)
        vs = np.array([avg[k] for k in keys])
        pk = int(np.nanargmax(vs))
        prof = {
            "instrument": "quote_mids_june", "coin": coin, "ex1": e1, "ex2": e2,
            "peak_lag_bars": int(ks[pk]), "peak_corr": round(float(vs[pk]), 3),
            "corr_at_0": round(float(avg[0]), 3),
            "n_obs": int(sum(p["n_obs"] for p in profs)),
            "profile": {k: round(avg[k], 4) for k in keys},
        }
        results.append(prof)
        print(f"  {coin} {e1}->{e2}: peak lag {prof['peak_lag_bars']:+d} min "
              f"(corr {prof['peak_corr']}), corr@0 {prof['corr_at_0']}")

    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                      text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        sha = None
    payload = {
        "provenance": {
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_sha": sha, "script": "experiments/leadlag_analysis.py",
            "method": "Pearson corr of log-returns, corr(rA_t, rB_{t+k}), k in [-45,45] bars; "
                      "peak k>0 => ex1 leads ex2",
        },
        "results": results,
    }
    dest = ROOT / "paper" / "leadlag_data.json"
    with open(dest, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {dest}")


if __name__ == "__main__":
    main()
