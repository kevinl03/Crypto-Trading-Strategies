#!/usr/bin/env python3
"""
DEX Depth & Execution Cost Analysis — Stat-Arb Research Extension
=================================================================
Joins the 96h pool run with the ~78h depth run to answer questions
the existing CEX-only analysis cannot:

HYPOTHESES
----------
H1: DEX execution costs (slippage + gas) erode cross-DEX spread alpha
    → Compare spread_bps from dex_spreads vs actual price_impact from quotes
H2: Gas fee spikes correlate with spread widening (arbitrage becomes costly)
    → Correlate chain gas with spread magnitude
H3: Perp funding rates predict DEX price dislocations
    → Lead-lag between Hyperliquid funding and DEX pool price movement
H4: Cross-venue (CEX↔DEX) spreads are larger and more persistent than
    intra-CEX spreads
    → Join CEX ticker mid with DEX pool price for overlapping tokens
H5: Execution cost is the binding constraint on DEX stat-arb profitability
    → Build a cost-adjusted spread signal and test whether any pairs survive

OUTPUT
------
Saves figures to statarb/statarb_outputs/plots/depth_*.png
Saves summary CSV to statarb/statarb_outputs/depth_analysis_summary.csv
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ────────────────────────────────────────────────────────────────────
DEX_DIR   = "datasets/statarb-crypto-dex"
DEPTH_DIR = os.path.join(DEX_DIR, "depth")
CEX_DIR   = "datasets/statarb-crypto-research"
OUT_DIR   = "statarb/statarb_outputs/plots"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Cutoff (matches the trimmed notebook) ────────────────────────────────────
CUTOFF = pd.Timestamp("2026-06-30 17:00:00+00:00")

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("Loading data...")

pools    = pd.read_parquet(os.path.join(DEX_DIR, "dex_pools.parquet"))
spreads  = pd.read_parquet(os.path.join(DEX_DIR, "dex_spreads.parquet"))
quotes   = pd.read_parquet(os.path.join(DEPTH_DIR, "dex_quotes.parquet"))
gas      = pd.read_parquet(os.path.join(DEPTH_DIR, "dex_gas.parquet"))
funding  = pd.read_parquet(os.path.join(DEPTH_DIR, "perp_funding.parquet"))

# Parse timestamps and apply cutoff
for df in [pools, spreads, quotes, gas, funding]:
    df["ts"] = pd.to_datetime(df["timestamp"])
    
pools   = pools[pools["ts"] <= CUTOFF].copy()
spreads = spreads[spreads["ts"] <= CUTOFF].copy()
quotes  = quotes[quotes["ts"] <= CUTOFF].copy()
gas     = gas[gas["ts"] <= CUTOFF].copy()
funding = funding[funding["ts"] <= CUTOFF].copy()

print(f"  pools:   {len(pools):>10,} rows | {pools['ts'].min()} → {pools['ts'].max()}")
print(f"  spreads: {len(spreads):>10,} rows | {spreads['ts'].min()} → {spreads['ts'].max()}")
print(f"  quotes:  {len(quotes):>10,} rows | {quotes['ts'].min()} → {quotes['ts'].max()}")
print(f"  gas:     {len(gas):>10,} rows | {gas['ts'].min()} → {gas['ts'].max()}")
print(f"  funding: {len(funding):>10,} rows | {funding['ts'].min()} → {funding['ts'].max()}")

# ══════════════════════════════════════════════════════════════════════════════
# H1: SPREAD SIGNAL vs EXECUTION COST
# Are cross-DEX spreads large enough to survive slippage?
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("H1: Cross-DEX spread vs execution cost (price impact)")
print("="*70)

# Get valid quotes with price impact
valid_quotes = quotes[quotes["price_impact_pct"].notnull()].copy()
valid_quotes["impact_bps"] = valid_quotes["price_impact_pct"] * 100 * 100  # pct -> bps

# Floor timestamps to 1-min for joining
valid_quotes["ts_1min"] = valid_quotes["ts"].dt.floor("1min")
spreads["ts_1min"] = spreads["ts"].dt.floor("1min")

# Per-token, per-minute: spread vs execution cost
# Average price impact across chains/sources for each token
quote_agg = valid_quotes.groupby(["token", "ts_1min"])["impact_bps"].median().reset_index()
spread_agg = spreads[["token", "ts_1min", "spread_bps"]].copy()

merged = spread_agg.merge(quote_agg, on=["token", "ts_1min"], how="inner")
merged["net_spread_bps"] = merged["spread_bps"] - 2 * merged["impact_bps"]  # round-trip

# Compute per-token summary
summary_rows = []
for token in sorted(merged["token"].unique()):
    t = merged[merged["token"] == token]
    pct_profitable = (t["net_spread_bps"] > 0).mean() * 100
    summary_rows.append({
        "token": token,
        "median_spread_bps": t["spread_bps"].median(),
        "mean_spread_bps": t["spread_bps"].mean(),
        "median_impact_bps": t["impact_bps"].median(),
        "mean_impact_bps": t["impact_bps"].mean(),
        "median_net_spread_bps": t["net_spread_bps"].median(),
        "pct_net_positive": pct_profitable,
        "n_obs": len(t),
    })

h1_summary = pd.DataFrame(summary_rows).sort_values("median_net_spread_bps", ascending=False)
print("\nSpread vs Execution Cost (per token):")
print(h1_summary.to_string(index=False, float_format="%.1f"))

# ── Figure: Spread vs Price Impact scatter ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: box plot of spread_bps vs impact_bps per token
tokens_ordered = h1_summary["token"].tolist()
box_data_spread = [merged[merged["token"]==t]["spread_bps"].clip(upper=500) for t in tokens_ordered]
box_data_impact = [merged[merged["token"]==t]["impact_bps"].clip(upper=500) for t in tokens_ordered]

x = np.arange(len(tokens_ordered))
bp1 = axes[0].boxplot(box_data_spread, positions=x-0.2, widths=0.35, patch_artist=True,
                       boxprops=dict(facecolor="steelblue", alpha=0.7), medianprops=dict(color="black"),
                       showfliers=False)
bp2 = axes[0].boxplot(box_data_impact, positions=x+0.2, widths=0.35, patch_artist=True,
                       boxprops=dict(facecolor="coral", alpha=0.7), medianprops=dict(color="black"),
                       showfliers=False)
axes[0].set_xticks(x)
axes[0].set_xticklabels(tokens_ordered, rotation=45, ha="right", fontsize=8)
axes[0].set_ylabel("Basis Points")
axes[0].set_title("Cross-DEX Spread (blue) vs Execution Cost (red)")
axes[0].legend([bp1["boxes"][0], bp2["boxes"][0]], ["Spread (signal)", "Price Impact (cost)"], fontsize=9)
axes[0].set_ylim(0, 300)

# Right: net spread (spread - 2x impact) distribution
for token in tokens_ordered[:6]:  # top 6
    t = merged[merged["token"]==token]
    net = t["net_spread_bps"].clip(-200, 500)
    axes[1].hist(net, bins=50, alpha=0.5, label=token, density=True)
axes[1].axvline(0, color="black", linestyle="--", linewidth=1.5, label="Breakeven")
axes[1].set_xlabel("Net Spread (spread − 2×impact) [bps]")
axes[1].set_ylabel("Density")
axes[1].set_title("Net Spread After Execution Cost")
axes[1].legend(fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "depth_h1_spread_vs_cost.png"), dpi=150)
plt.close(fig)
print(f"  → Saved depth_h1_spread_vs_cost.png")


# ══════════════════════════════════════════════════════════════════════════════
# H2: GAS FEE CORRELATION WITH SPREAD MAGNITUDE
# Do high gas periods widen DEX spreads?
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("H2: Gas fees vs spread magnitude")
print("="*70)

# Ethereum gas (most relevant for arbitrage cost)
eth_gas = gas[gas["chain"] == "ethereum"].copy()
eth_gas["ts_10min"] = eth_gas["ts"].dt.floor("10min")
gas_10min = eth_gas.groupby("ts_10min")["gas_gwei"].median().reset_index()

# Aggregate spreads per 10-min bucket
spreads["ts_10min"] = spreads["ts"].dt.floor("10min")
spread_10min = spreads.groupby("ts_10min").agg(
    median_spread=("spread_bps", "median"),
    mean_spread=("spread_bps", "mean"),
    max_spread=("spread_bps", "max"),
).reset_index()

gas_spread = gas_10min.merge(spread_10min, on="ts_10min", how="inner")

if len(gas_spread) > 10:
    corr_med, p_med = stats.spearmanr(gas_spread["gas_gwei"], gas_spread["median_spread"])
    corr_max, p_max = stats.spearmanr(gas_spread["gas_gwei"], gas_spread["max_spread"])
    print(f"  Spearman(ETH gas, median spread): ρ={corr_med:.3f}, p={p_med:.4f}")
    print(f"  Spearman(ETH gas, max spread):    ρ={corr_max:.3f}, p={p_max:.4f}")
    
    # ── Figure: Gas vs Spread time series ──
    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.plot(gas_spread["ts_10min"], gas_spread["gas_gwei"], color="tab:red", alpha=0.7, linewidth=0.8, label="ETH Gas (gwei)")
    ax1.set_ylabel("ETH Gas (gwei)", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    
    ax2 = ax1.twinx()
    ax2.plot(gas_spread["ts_10min"], gas_spread["median_spread"], color="tab:blue", alpha=0.7, linewidth=0.8, label="Median Spread (bps)")
    ax2.set_ylabel("Median Cross-DEX Spread (bps)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    plt.xticks(rotation=30, ha="right")
    ax1.set_title(f"ETH Gas vs Cross-DEX Spread (Spearman ρ={corr_med:.3f}, p={p_med:.4f})")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "depth_h2_gas_vs_spread.png"), dpi=150)
    plt.close(fig)
    print(f"  → Saved depth_h2_gas_vs_spread.png")

    # Multi-chain gas comparison
    print("\n  Per-chain gas stats:")
    for chain in sorted(gas["chain"].unique()):
        cg = gas[gas["chain"] == chain]
        if cg["gas_gwei"].notnull().any():
            vals = cg["gas_gwei"].dropna()
            print(f"    {chain}: median={vals.median():.4f} gwei, mean={vals.mean():.4f}, std={vals.std():.4f}, n={len(vals)}")


# ══════════════════════════════════════════════════════════════════════════════
# H3: PERP FUNDING → DEX PRICE DISLOCATION
# Does funding rate predict DEX spread movements?
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("H3: Perp funding rate → DEX price dislocation")
print("="*70)

# Build per-token funding panel (Hyperliquid)
funding["ts_10min"] = funding["ts"].dt.floor("10min")
fund_pivot = funding.pivot_table(index="ts_10min", columns="symbol", values="funding_rate", aggfunc="last")

# Build per-token spread panel
spread_pivot = spreads.pivot_table(index="ts_10min", columns="token", values="spread_bps", aggfunc="median")

# Shared tokens between funding and spreads
shared_tokens = sorted(set(fund_pivot.columns) & set(spread_pivot.columns))
print(f"  Shared tokens (funding ∩ spreads): {shared_tokens}")

# Lead-lag correlation: does funding at t predict spread at t+lag?
lags_minutes = [0, 10, 30, 60, 120]  # in units of 10min bins
lead_lag_results = []
for token in shared_tokens:
    if token not in fund_pivot.columns or token not in spread_pivot.columns:
        continue
    fr = fund_pivot[token].dropna()
    sp = spread_pivot[token].dropna()
    # Align
    common_idx = fr.index.intersection(sp.index)
    if len(common_idx) < 50:
        continue
    for lag in lags_minutes:
        lag_bins = lag // 10
        if lag_bins == 0:
            fr_shifted = fr.loc[common_idx]
            sp_shifted = sp.loc[common_idx]
        else:
            fr_shifted = fr.shift(lag_bins).loc[common_idx].dropna()
            sp_shifted = sp.loc[fr_shifted.index]
        if len(fr_shifted) < 30:
            continue
        corr, pval = stats.spearmanr(fr_shifted, sp_shifted)
        lead_lag_results.append({
            "token": token, "lag_min": lag, "spearman_r": corr, "p_value": pval, "n": len(fr_shifted)
        })

if lead_lag_results:
    ll_df = pd.DataFrame(lead_lag_results)
    print("\n  Lead-lag: funding(t) → spread(t + lag)")
    for token in shared_tokens:
        t = ll_df[ll_df["token"] == token]
        if len(t) == 0:
            continue
        best = t.loc[t["spearman_r"].abs().idxmax()]
        sig = "***" if best["p_value"] < 0.001 else "**" if best["p_value"] < 0.01 else "*" if best["p_value"] < 0.05 else ""
        print(f"    {token}: best lag={best['lag_min']}min, ρ={best['spearman_r']:.3f} (p={best['p_value']:.4f}) {sig}")

    # ── Figure: Lead-lag heatmap ──
    pivot_ll = ll_df.pivot_table(index="token", columns="lag_min", values="spearman_r")
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot_ll.values, cmap="RdBu_r", aspect="auto", vmin=-0.3, vmax=0.3)
    ax.set_xticks(range(len(pivot_ll.columns)))
    ax.set_xticklabels([f"{l}min" for l in pivot_ll.columns])
    ax.set_yticks(range(len(pivot_ll.index)))
    ax.set_yticklabels(pivot_ll.index)
    ax.set_xlabel("Lag (funding leads spread)")
    ax.set_title("Spearman ρ: Funding Rate → Cross-DEX Spread")
    for i in range(len(pivot_ll.index)):
        for j in range(len(pivot_ll.columns)):
            v = pivot_ll.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(v) > 0.15 else "black")
    plt.colorbar(im, ax=ax, label="Spearman ρ")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "depth_h3_funding_lead_lag.png"), dpi=150)
    plt.close(fig)
    print(f"  → Saved depth_h3_funding_lead_lag.png")


# ══════════════════════════════════════════════════════════════════════════════
# H4: FUNDING RATE REGIME → SPREAD BEHAVIOR
# Do extreme funding rates coincide with wider/narrower DEX spreads?
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("H4: Funding rate regimes vs DEX spread behavior")
print("="*70)

regime_results = []
for token in shared_tokens:
    fr = fund_pivot.get(token)
    sp = spread_pivot.get(token)
    if fr is None or sp is None:
        continue
    common = fr.index.intersection(sp.index)
    if len(common) < 100:
        continue
    fr_c = fr.loc[common]
    sp_c = sp.loc[common]
    
    # Define regimes: bottom 20%, middle 60%, top 20% of funding
    q20 = fr_c.quantile(0.2)
    q80 = fr_c.quantile(0.8)
    
    neg = sp_c[fr_c <= q20]
    mid = sp_c[(fr_c > q20) & (fr_c < q80)]
    pos = sp_c[fr_c >= q80]
    
    if len(neg) < 10 or len(pos) < 10:
        continue
    
    # Kruskal-Wallis test
    stat, pval = stats.kruskal(neg.dropna(), mid.dropna(), pos.dropna())
    
    regime_results.append({
        "token": token,
        "spread_neg_funding": neg.median(),
        "spread_neutral": mid.median(),
        "spread_pos_funding": pos.median(),
        "kruskal_h": stat,
        "p_value": pval,
    })

if regime_results:
    regime_df = pd.DataFrame(regime_results)
    print("\nMedian spread (bps) by funding regime:")
    print(regime_df.to_string(index=False, float_format="%.2f"))
    
    # ── Figure: Regime comparison ──
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(regime_df))
    w = 0.25
    ax.bar(x - w, regime_df["spread_neg_funding"].clip(upper=2000), w, label="Negative funding (bottom 20%)", color="tab:green", alpha=0.8)
    ax.bar(x, regime_df["spread_neutral"].clip(upper=2000), w, label="Neutral (middle 60%)", color="tab:gray", alpha=0.8)
    ax.bar(x + w, regime_df["spread_pos_funding"].clip(upper=2000), w, label="Positive funding (top 20%)", color="tab:red", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(regime_df["token"], rotation=45, ha="right")
    ax.set_ylabel("Median Cross-DEX Spread (bps)")
    ax.set_title("DEX Spread by Perp Funding Regime")
    ax.legend()
    # Add significance stars
    for i, row in regime_df.iterrows():
        if row["p_value"] < 0.001:
            ax.text(i, max(row["spread_neg_funding"], row["spread_pos_funding"], row["spread_neutral"]) + 10, "***", ha="center", fontsize=10)
        elif row["p_value"] < 0.01:
            ax.text(i, max(row["spread_neg_funding"], row["spread_pos_funding"], row["spread_neutral"]) + 10, "**", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "depth_h4_funding_regime_spread.png"), dpi=150)
    plt.close(fig)
    print(f"  → Saved depth_h4_funding_regime_spread.png")


# ══════════════════════════════════════════════════════════════════════════════
# H5: COST-ADJUSTED PROFITABILITY — WHICH PAIRS SURVIVE?
# Build a full cost model: spread - 2×slippage - gas
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("H5: Cost-adjusted profitability")
print("="*70)

# Gas cost in USD per tx (rough estimate: gas_gwei * 21000 * ETH_price / 1e9)
# Use median ETH price from pools
eth_prices = pools[pools["token"] == "ETH"].groupby("snapshot_idx")["price_usd"].median()
median_eth = eth_prices.median()
print(f"  Median ETH price: ${median_eth:.2f}")

# Median gas per chain
gas_by_chain = gas.groupby("chain")["gas_gwei"].median()
print(f"\n  Median gas by chain:")
for chain, gwei in gas_by_chain.items():
    if pd.notna(gwei):
        # Simple gas cost model: gwei * gas_limit * eth_price / 1e9
        # DEX swap ≈ 150k-300k gas on EVM
        gas_cost_usd = gwei * 200_000 * median_eth / 1e9
        print(f"    {chain}: {gwei:.4f} gwei ≈ ${gas_cost_usd:.4f}/swap")

# For each token: median spread (in USD) vs round-trip cost
print("\n  Per-token profitability at $10k notional:")
prof_rows = []
for token in sorted(merged["token"].unique()):
    t = merged[merged["token"] == token]
    med_spread = t["spread_bps"].median()
    med_impact = t["impact_bps"].median()
    
    # Spread revenue on $10k notional
    revenue_usd = med_spread / 10000 * 10000
    # Slippage cost (round-trip = 2 legs)
    slippage_usd = 2 * med_impact / 10000 * 10000
    # Gas cost (2 swaps, assume ETH chain)
    eth_gwei = gas_by_chain.get("ethereum", 0.5)
    gas_usd = 2 * (eth_gwei * 200_000 * median_eth / 1e9) if pd.notna(eth_gwei) else 0
    
    net_usd = revenue_usd - slippage_usd - gas_usd
    
    prof_rows.append({
        "token": token,
        "spread_bps": med_spread,
        "impact_bps": med_impact,
        "revenue_$10k": revenue_usd,
        "slippage_$10k": slippage_usd,
        "gas_$10k": gas_usd,
        "net_$10k": net_usd,
        "profitable": net_usd > 0,
    })

prof_df = pd.DataFrame(prof_rows).sort_values("net_$10k", ascending=False)
print(prof_df.to_string(index=False, float_format="%.2f"))

n_profitable = prof_df["profitable"].sum()
print(f"\n  {n_profitable}/{len(prof_df)} tokens profitable after costs at $10k notional")

# ── Figure: Waterfall cost breakdown ──
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(prof_df))
ax.bar(x, prof_df["revenue_$10k"], label="Spread Revenue", color="tab:green", alpha=0.8)
ax.bar(x, -prof_df["slippage_$10k"], bottom=prof_df["revenue_$10k"], label="Slippage Cost", color="tab:orange", alpha=0.8)
ax.bar(x, -prof_df["gas_$10k"], bottom=prof_df["revenue_$10k"] - prof_df["slippage_$10k"], label="Gas Cost", color="tab:red", alpha=0.8)
# Net line
ax.scatter(x, prof_df["net_$10k"], color="black", zorder=5, s=40, label="Net P&L")
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xticks(x)
ax.set_xticklabels(prof_df["token"], rotation=45, ha="right")
ax.set_ylabel("USD (on $10k notional)")
ax.set_title("DEX Stat-Arb: Spread Revenue vs Execution Costs")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "depth_h5_cost_waterfall.png"), dpi=150)
plt.close(fig)
print(f"  → Saved depth_h5_cost_waterfall.png")


# ══════════════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY: TIME-OF-DAY PATTERNS
# When are spreads widest / execution cheapest?
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("Supplementary: Time-of-day patterns")
print("="*70)

spreads["hour"] = spreads["ts"].dt.hour
hourly_spread = spreads.groupby("hour")["spread_bps"].agg(["median", "mean", "std"]).reset_index()

gas["hour"] = gas["ts"].dt.hour
eth_hourly_gas = gas[gas["chain"] == "ethereum"].groupby("hour")["gas_gwei"].median().reset_index()

quotes["hour"] = quotes["ts"].dt.hour
valid_q = quotes[quotes["price_impact_pct"].notnull()].copy()
valid_q["impact_bps"] = valid_q["price_impact_pct"] * 100 * 100
hourly_impact = valid_q.groupby("hour")["impact_bps"].median().reset_index()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].bar(hourly_spread["hour"], hourly_spread["median"], color="steelblue", alpha=0.8)
axes[0].set_xlabel("Hour (UTC)")
axes[0].set_ylabel("Median Spread (bps)")
axes[0].set_title("Cross-DEX Spread by Hour")

if len(eth_hourly_gas) > 0:
    axes[1].bar(eth_hourly_gas["hour"], eth_hourly_gas["gas_gwei"], color="coral", alpha=0.8)
    axes[1].set_xlabel("Hour (UTC)")
    axes[1].set_ylabel("Median Gas (gwei)")
    axes[1].set_title("Ethereum Gas by Hour")

axes[2].bar(hourly_impact["hour"], hourly_impact["impact_bps"], color="mediumpurple", alpha=0.8)
axes[2].set_xlabel("Hour (UTC)")
axes[2].set_ylabel("Median Price Impact (bps)")
axes[2].set_title("DEX Execution Cost by Hour")

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "depth_time_of_day.png"), dpi=150)
plt.close(fig)
print(f"  → Saved depth_time_of_day.png")

# Best hours for arbitrage = wide spread + low cost
if len(eth_hourly_gas) > 0:
    tod = hourly_spread.merge(eth_hourly_gas, on="hour", how="left").merge(hourly_impact, on="hour", how="left")
    tod["net_opportunity"] = tod["median"] - 2 * tod["impact_bps"].fillna(0)
    best_hours = tod.nlargest(5, "net_opportunity")
    print("\n  Best hours for DEX arb (spread − 2×impact):")
    for _, row in best_hours.iterrows():
        print(f"    {int(row['hour']):02d}:00 UTC — spread={row['median']:.0f} bps, impact={row['impact_bps']:.0f} bps, net={row['net_opportunity']:.0f} bps")


# ══════════════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY: FUNDING RATE vs MARK PRICE MOVEMENT
# Do extreme funding rates predict price reversals?
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("Supplementary: Funding rate vs subsequent price movement")
print("="*70)

funding_sorted = funding.sort_values(["symbol", "ts"])
price_returns = []
for sym in funding["symbol"].unique():
    sf = funding_sorted[funding_sorted["symbol"] == sym].copy()
    sf["mark_ret_60min"] = sf["mark_px"].pct_change(periods=60).shift(-60)  # forward 60 snapshots ≈ 60min
    sf["mark_ret_30min"] = sf["mark_px"].pct_change(periods=30).shift(-30)
    price_returns.append(sf[["symbol", "ts", "funding_rate", "mark_ret_60min", "mark_ret_30min"]].dropna())

if price_returns:
    pr_df = pd.concat(price_returns)
    
    # Correlation: funding rate vs forward return
    print("\n  Funding rate → forward mark price return:")
    fr_pred_results = []
    for sym in sorted(pr_df["symbol"].unique()):
        s = pr_df[pr_df["symbol"] == sym]
        if len(s) < 50:
            continue
        r30, p30 = stats.spearmanr(s["funding_rate"], s["mark_ret_30min"])
        r60, p60 = stats.spearmanr(s["funding_rate"], s["mark_ret_60min"])
        sig60 = "***" if p60 < 0.001 else "**" if p60 < 0.01 else "*" if p60 < 0.05 else ""
        print(f"    {sym}: 30min ρ={r30:.3f} (p={p30:.3f}), 60min ρ={r60:.3f} (p={p60:.3f}) {sig60}")
        fr_pred_results.append({"symbol": sym, "r_30min": r30, "p_30min": p30, "r_60min": r60, "p_60min": p60})
    
    # ── Figure: Funding vs forward return for BTC and ETH ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, sym in zip(axes, ["BTC", "ETH"]):
        s = pr_df[pr_df["symbol"] == sym]
        if len(s) < 50:
            continue
        ax.scatter(s["funding_rate"] * 10000, s["mark_ret_60min"] * 10000, alpha=0.1, s=3, color="tab:blue")
        # Add regression line
        z = np.polyfit(s["funding_rate"] * 10000, s["mark_ret_60min"] * 10000, 1)
        p = np.poly1d(z)
        xr = np.linspace(s["funding_rate"].min() * 10000, s["funding_rate"].max() * 10000, 100)
        ax.plot(xr, p(xr), color="red", linewidth=2)
        r, pval = stats.spearmanr(s["funding_rate"], s["mark_ret_60min"])
        ax.set_xlabel("Funding Rate (bps)")
        ax.set_ylabel("60min Forward Return (bps)")
        ax.set_title(f"{sym}: Funding → 60min Return (ρ={r:.3f}, p={pval:.3f})")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "depth_funding_vs_return.png"), dpi=150)
    plt.close(fig)
    print(f"  → Saved depth_funding_vs_return.png")


# ══════════════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY: LIQUIDITY vs EXECUTION COST
# Do more liquid pools have lower price impact?
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("Supplementary: Pool liquidity vs execution cost")
print("="*70)

# Join quotes with pool liquidity
quotes_with_liq = valid_quotes.copy()
quotes_with_liq["ts_1min"] = quotes_with_liq["ts"].dt.floor("1min")

# Get best-liquidity pool per token per minute from pools
best_pool = pools.copy()
best_pool["ts_1min"] = best_pool["ts"].dt.floor("1min")
pool_liq = best_pool.groupby(["token", "ts_1min"])["liquidity_usd"].max().reset_index()

ql = quotes_with_liq.merge(pool_liq, on=["token", "ts_1min"], how="inner")

if len(ql) > 100:
    r, p = stats.spearmanr(np.log10(ql["liquidity_usd"].clip(lower=1)), ql["impact_bps"])
    print(f"  Spearman(log10(liquidity), impact_bps): ρ={r:.3f}, p={p:.4f}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for token in sorted(ql["token"].unique()):
        t = ql[ql["token"] == token]
        ax.scatter(t["liquidity_usd"] / 1e6, t["impact_bps"].clip(upper=100),
                   alpha=0.3, s=10, label=token)
    ax.set_xscale("log")
    ax.set_xlabel("Pool Liquidity ($ millions)")
    ax.set_ylabel("Price Impact (bps)")
    ax.set_title(f"Liquidity vs Execution Cost (Spearman ρ={r:.3f})")
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "depth_liquidity_vs_impact.png"), dpi=150)
    plt.close(fig)
    print(f"  → Saved depth_liquidity_vs_impact.png")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("Saving summary")
print("="*70)

summary_path = "statarb/statarb_outputs/depth_analysis_summary.csv"
prof_df.to_csv(summary_path, index=False)
print(f"  → {summary_path}")

h1_path = "statarb/statarb_outputs/depth_spread_vs_cost.csv"
h1_summary.to_csv(h1_path, index=False)
print(f"  → {h1_path}")

if regime_results:
    regime_path = "statarb/statarb_outputs/depth_funding_regimes.csv"
    regime_df.to_csv(regime_path, index=False)
    print(f"  → {regime_path}")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
