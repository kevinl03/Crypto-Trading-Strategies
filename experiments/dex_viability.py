#!/usr/bin/env python3
"""
DEX data repair + out-of-domain test of the fee-floor viability law.

REPAIR: From 2026-06-26 ~20:00 UTC some pools report price_usd at a
~5,000x scale error. Because corruption begins ~2h into the collection
window, a hard cutoff is not viable; we instead flag per-row scale
outliers against the cross-pool median price per (token, snapshot).

VALIDATION BATTERY (all reported to stdout + JSON):
  V1 root cause     - ratio distribution must be bimodal (~1 vs ~5e3),
                      i.e. a discrete scale error, not drift
  V2 external anchor- post-repair per-token median DEX price must sit
                      within [0.5x, 2x] of the CEX quote-mid median
                      (HF test window); corrupted rows are >100x off
  V3 conservation   - ~0 rows flagged before the corruption onset
  V4 continuity     - post-repair spread quantiles after onset match
                      the clean pre-onset distribution's magnitude
  V5 sensitivity    - viability conclusions must be stable across
                      detection thresholds R in {3, 10, 30} and both
                      cost models (impact-only vs impact+AMM fees)

VIABILITY TEST: liquidity-weighted venue prices -> pairwise venue
spreads -> identical backtest protocol as the paper (window=60 bars,
entry_z=2, exit_z=0.5, max_holding=1440) with per-token round-trip
costs. Measured costs (Jupiter $10k quotes) exist for BONK/SOL/WIF;
other tokens use a default and are flagged.

Output: paper/dex_viability_data.json (provenance-stamped).
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
from experiments.backtest_historical import run_ou_strategy, run_zscore_strategy  # noqa: E402

DEX = ROOT / "datasets" / "statarb-crypto-dex"
CEX = ROOT / "datasets" / "statarb-crypto-research"

MIN_LIQUIDITY_USD = 50_000
MIN_JOINT_SNAPSHOTS = 2000
DETECTION_R = 10.0          # primary scale-outlier threshold
SENSITIVITY_R = [3.0, 10.0, 30.0]
AMM_FEE_BPS_PER_SIDE = 25.0  # typical AMM tier; stated assumption
DEFAULT_IMPACT_BPS = 25.0    # per side, tokens without measured quotes
CORRUPTION_ONSET = pd.Timestamp("2026-06-26 20:00:00+00:00")


def load_pools() -> pd.DataFrame:
    po = pd.read_parquet(DEX / "dex_pools.parquet",
                         columns=["token", "chain", "dex", "pair_address", "price_usd",
                                  "liquidity_usd", "snapshot_idx", "timestamp"])
    po = po[(po.price_usd > 0) & (po.liquidity_usd >= MIN_LIQUIDITY_USD)].copy()
    po["ts"] = pd.to_datetime(po.timestamp)
    po["venue"] = po.dex + ":" + po.chain
    return po


def flag_outliers(po: pd.DataFrame, r: float) -> pd.DataFrame:
    med = po.groupby(["token", "snapshot_idx"]).price_usd.transform("median")
    po = po.assign(ratio=po.price_usd / med)
    po["bad"] = (po.ratio > r) | (po.ratio < 1.0 / r)
    return po


def cex_anchor() -> dict:
    """Median CEX quote-mid per token from the HF test window."""
    t = pd.read_parquet(CEX / "test" / "ticker.parquet",
                        columns=["coin", "error", "payload"])
    t = t[t.error.isna()]
    prices: dict[str, list] = {}
    for r in t.itertuples():
        try:
            p = json.loads(r.payload) if isinstance(r.payload, str) else r.payload
            bid, ask = p.get("bid"), p.get("ask")
            if bid and ask and float(bid) > 0:
                prices.setdefault(r.coin, []).append((float(bid) + float(ask)) / 2)
        except Exception:
            continue
    return {c: float(np.median(v)) for c, v in prices.items() if v}


def token_costs() -> tuple[dict, dict, set]:
    """Round-trip cost per token: impact-only and impact+AMM-fee models (bps)."""
    q = pd.read_parquet(DEX / "depth" / "dex_quotes.parquet",
                        columns=["token", "price_impact_pct"])
    q = q.dropna(subset=["price_impact_pct"])
    measured = (q.groupby("token").price_impact_pct.median() * 1e4).to_dict()  # bps/side
    impact_only, full = {}, {}
    for tok in ALL_TOKENS:
        side = measured.get(tok, DEFAULT_IMPACT_BPS)
        impact_only[tok] = 2 * side
        full[tok] = 2 * side + 2 * AMM_FEE_BPS_PER_SIDE
    return impact_only, full, set(measured)


def venue_spread_backtests(po: pd.DataFrame, costs: dict) -> list[dict]:
    """Liquidity-weighted venue price per snapshot -> pairwise venue backtests."""
    rows = []
    for token, grp in po[~po.bad].groupby("token"):
        # weighted median ~ weighted mean of log-price for stability
        agg = (grp.assign(w=grp.liquidity_usd,
                          wp=np.log(grp.price_usd) * grp.liquidity_usd)
                  .groupby(["venue", "snapshot_idx"])
                  .agg(wp=("wp", "sum"), w=("w", "sum")))
        agg["price"] = np.exp(agg.wp / agg.w)
        piv = agg.reset_index().pivot_table(index="snapshot_idx", columns="venue", values="price")
        venues = [c for c in piv.columns if piv[c].notna().sum() >= MIN_JOINT_SNAPSHOTS]
        cost = costs[token]
        for v1, v2 in combinations(sorted(venues), 2):
            j = piv[[v1, v2]].dropna()
            if len(j) < MIN_JOINT_SNAPSHOTS:
                continue
            spread = ((j[v1] - j[v2]) / ((j[v1] + j[v2]) / 2) * 1e4).to_numpy()
            snaps = j.index.to_numpy()
            std = float(np.std(spread))
            for model, fn in (("ou", run_ou_strategy), ("zscore", run_zscore_strategy)):
                trades = fn(spread, snaps, cost)
                pnls = [t.pnl_net_bps for t in trades]
                rows.append({
                    "token": token, "v1": v1, "v2": v2, "model": model,
                    "n_bars": len(spread), "spread_std_bps": round(std, 2),
                    "cost_bps": round(cost, 1), "ratio": round(std / cost, 3),
                    "n_trades": len(pnls),
                    "net_bps": round(float(np.sum(pnls)), 1) if pnls else 0.0,
                    "win_rate": round(float(np.mean([p > 0 for p in pnls])), 3) if pnls else None,
                })
    return rows


def ratio_buckets(rows: list[dict], min_trades: int = 5) -> list[dict]:
    out = []
    active = [r for r in rows if r["n_trades"] >= min_trades]
    for lo, hi, label in [(0, 0.5, "<0.5"), (0.5, 1.0, "0.5-1"), (1.0, 2.0, "1-2"), (2.0, 1e9, ">2")]:
        b = [r for r in active if lo <= r["ratio"] < hi]
        if b:
            prof = sum(1 for r in b if r["net_bps"] > 0)
            out.append({"ratio": label, "pair_models": len(b), "profitable": prof,
                        "rate": round(prof / len(b), 3)})
    return out


print("Loading DEX pools...")
po = load_pools()
ALL_TOKENS = sorted(po.token.unique())
po = flag_outliers(po, DETECTION_R)

# ── V1: root cause — discrete bimodality ────────────────────────────────
r = po.ratio
mid = r[(r > 3) & (r < 1000)]
print(f"\nV1 root cause: {po.bad.mean()*100:.1f}% rows flagged at R={DETECTION_R}")
print(f"  ratio distribution: {(np.abs(np.log10(r)) < 0.3).mean()*100:.1f}% within 2x of median; "
      f"{(r > 1000).mean()*100:.2f}% above 1000x; only {len(mid)/len(r)*100:.3f}% in 3x-1000x band"
      f" -> {'DISCRETE scale error' if len(mid)/len(r) < 0.01 else 'WARNING: non-discrete'}")

# ── V3: conservation — clean period untouched ───────────────────────────
pre = po[po.ts < CORRUPTION_ONSET]
print(f"V3 conservation: {pre.bad.sum()} of {len(pre):,} pre-onset rows flagged "
      f"({pre.bad.mean()*100:.3f}%) -> {'PASS' if pre.bad.mean() < 0.005 else 'FAIL'}")

# ── V2: external anchor — CEX quote medians ─────────────────────────────
print("V2 external anchor (post-repair DEX median vs CEX test-window quote median):")
anchor = cex_anchor()
v2_pass = True
v2 = {}
for tok in ALL_TOKENS:
    if tok not in anchor:
        continue
    dex_med = float(po[(po.token == tok) & (~po.bad)].price_usd.median())
    ratio = dex_med / anchor[tok]
    ok = 0.5 <= ratio <= 2.0
    v2[tok] = round(ratio, 3)
    v2_pass &= ok
    flag = "" if ok else "  <-- FAIL"
    print(f"  {tok:>5}: dex/cex = {ratio:6.3f}{flag}")
print(f"  -> {'PASS' if v2_pass else 'FAIL'}")

# ── V4: continuity — spread magnitude pre vs post onset ─────────────────
def snapshot_spread(pod):
    g = pod.groupby(["token", "snapshot_idx"]).price_usd
    return ((g.max() - g.min()) / g.mean() * 1e4)

clean = po[~po.bad]
s_pre = snapshot_spread(clean[clean.ts < CORRUPTION_ONSET])
s_post = snapshot_spread(clean[clean.ts >= CORRUPTION_ONSET])
print(f"V4 continuity: cross-pool spread median pre={s_pre.median():.0f} bps, "
      f"post={s_post.median():.0f} bps (p90: {s_pre.quantile(.9):.0f} vs {s_post.quantile(.9):.0f})"
      f" -> {'PASS (same magnitude)' if 0.2 < s_post.median()/max(s_pre.median(),1e-9) < 5 else 'FAIL'}")

# ── Viability test under both cost models ───────────────────────────────
impact_only, full_cost, measured_tokens = token_costs()
print(f"\nMeasured-impact tokens: {sorted(measured_tokens)}; "
      f"others use default {DEFAULT_IMPACT_BPS} bps/side")
results = {}
for cname, costs in (("impact_only", impact_only), ("impact_plus_amm_fees", full_cost)):
    rows = venue_spread_backtests(po, costs)
    results[cname] = rows
    print(f"\nViability ({cname}); sigma/cost buckets, pair-models with >=5 trades:")
    for b in ratio_buckets(rows):
        print(f"  ratio {b['ratio']:>6}: {b['profitable']:>3}/{b['pair_models']:<4} ({b['rate']:.0%})")
    m = [r for r in rows if r["token"] in measured_tokens and r["n_trades"] >= 5]
    mp = sum(1 for r in m if r["net_bps"] > 0)
    print(f"  measured-cost tokens only: {mp}/{len(m)} profitable")

# ── V5: sensitivity across detection thresholds ─────────────────────────
print("\nV5 sensitivity (impact_plus_amm_fees, bucket rates across detection R):")
v5 = {}
for rr in SENSITIVITY_R:
    po_r = flag_outliers(po.drop(columns=["ratio", "bad"]), rr)
    b = ratio_buckets(venue_spread_backtests(po_r, full_cost))
    v5[str(rr)] = b
    print(f"  R={rr}: " + " | ".join(f"{x['ratio']}:{x['rate']:.0%}" for x in b))

try:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                  text=True, stderr=subprocess.DEVNULL).strip()
except Exception:
    sha = None
payload = {
    "provenance": {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": sha, "script": "experiments/dex_viability.py",
        "params": {"min_liquidity_usd": MIN_LIQUIDITY_USD, "detection_r": DETECTION_R,
                   "amm_fee_bps_per_side": AMM_FEE_BPS_PER_SIDE,
                   "default_impact_bps_per_side": DEFAULT_IMPACT_BPS,
                   "corruption_onset": str(CORRUPTION_ONSET),
                   "protocol": "window=60, entry_z=2, exit_z=0.5, max_holding=1440"},
    },
    "validation": {"v1_pct_flagged": round(float(po.bad.mean()) * 100, 2),
                   "v2_anchor_ratios": v2, "v3_pre_onset_flagged": int(pre.bad.sum()),
                   "v4_spread_median_pre_post": [round(float(s_pre.median()), 1),
                                                  round(float(s_post.median()), 1)],
                   "v5_sensitivity": v5},
    "buckets": {c: ratio_buckets(r) for c, r in results.items()},
    "results": results["impact_plus_amm_fees"],
}
dest = ROOT / "paper" / "dex_viability_data.json"
with open(dest, "w") as f:
    json.dump(payload, f, indent=2)
print(f"\nSaved: {dest}")
