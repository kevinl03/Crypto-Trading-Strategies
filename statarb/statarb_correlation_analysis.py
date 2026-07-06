# %% [markdown]
# # Crypto DEX/Perp Statistical Arbitrage: Correlation & Mean-Reversion Analysis
#
# See the accompanying README (statarb_correlation_analysis_README.md) for a full
# explanation of the methodology, assumptions, and how to interpret every output file.
#
# This file uses "# %%" cell markers so it can be run top-to-bottom as a plain
# script (`python statarb_correlation_analysis.py`) OR opened in VS Code /
# Jupyter and run cell-by-cell (VS Code's Python extension renders "# %%" as
# notebook cells automatically).

# %%
# ============================================================================
# CELL 1: CONFIGURATION
# ============================================================================
import os
import warnings
import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe default for headless/script runs; VS Code interactive
                        # window will still show plots inline when run cell-by-cell.
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ---- Working directory: script assumes it is run FROM the directory that ----
# ---- contains the parquet files (same convention as the uploaded dataset). --
WORKDIR = os.getcwd()

# File names (relative to WORKDIR). Adjust if your local layout differs.
PATH_POOLS = os.path.join(WORKDIR, "dex_pools.parquet")
PATH_SPREADS = os.path.join(WORKDIR, "dex_spreads.parquet")
PATH_GAS = os.path.join(WORKDIR, "depth", "dex_gas.parquet")
PATH_QUOTES = os.path.join(WORKDIR, "depth", "dex_quotes.parquet")
PATH_PERP = os.path.join(WORKDIR, "depth", "perp_funding.parquet")

# Output directory (created alongside the input files).
OUTDIR = os.path.join(WORKDIR, "statarb_outputs")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(os.path.join(OUTDIR, "plots"), exist_ok=True)

# ---- Price source -----------------------------------------------------------
# Your files do NOT contain OHLC bars -- they contain periodic point-in-time
# snapshots (one row per asset per collection cycle, "run_id"). Two candidate
# price series can be built from your data:
#
#   "perp"  -> perp_funding.parquet:mark_px   (one row per symbol per snapshot,
#              has a clean explicit `timestamp` column). RECOMMENDED DEFAULT.
#   "pools" -> dex_pools.parquet:price_usd    (many rows per token per snapshot,
#              one per DEX pool; timestamp must be parsed from run_id, and rows
#              are aggregated with a median across pools for the same token +
#              run_id to collapse to one price per token per snapshot).
#
# Whichever you choose is used AS-IS as the "price" at each snapshot -- there
# is no separate open/high/low/close to choose between in this dataset.
PRICE_SOURCE = "perp"  # "perp" or "pools"

# Resample the irregular/near-regular snapshot timestamps onto a fixed grid.
# "1min" matches the apparent native snapshot cadence in the sample data;
# widen this (e.g. "5min") if your real data was collected less frequently,
# or if you want to reduce microstructure noise / bid-ask bounce.
RESAMPLE_FREQ = "1min"

# Drop any asset whose resampled price series is missing more than this
# fraction of points (after resampling) -- avoids polluting results with
# thinly-collected tokens.
MAX_MISSING_FRAC = 0.20

# Minimum number of overlapping, valid observations required to test a pair.
MIN_OBS_PER_PAIR = 200

# Significance threshold used to flag results (does not filter output tables,
# only annotates them -- you keep every row).
ALPHA = 0.05

# Rolling windows (in units of RESAMPLE_FREQ bars).
ROLLING_CORR_WINDOW = 120
ZSCORE_WINDOW = 120

# How many of the strongest correlated pairs to chart in the rolling-correlation
# plot (keeps output readable).
N_TOP_CORR_PAIRS_TO_PLOT = 5

# How many of the most convincingly cointegrated pairs to run the (slower)
# bootstrap half-life confidence interval on, and to generate spread/z-score
# plots for.
N_TOP_COINT_PAIRS_FOR_DETAIL = 10

# Bootstrap settings for OU half-life confidence intervals.
N_BOOTSTRAP = 300
BOOTSTRAP_SEED = 7

print(f"Working directory: {WORKDIR}")
print(f"Outputs will be written to: {OUTDIR}")


# %%
# ============================================================================
# CELL 2: LOAD DATA & BUILD A WIDE PRICE PANEL (timestamp x asset)
# ============================================================================

def _read_parquet_or_raise(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expected file not found: {path}\n"
            f"Update the PATH_* constants in CELL 1 if your files live elsewhere."
        )
    return pd.read_parquet(path)


def build_price_panel_from_perp(path_perp: str) -> pd.DataFrame:
    """
    Build a (timestamp x symbol) price matrix from perp_funding.parquet.
    Uses the `mark_px` column, which is already one observation per symbol
    per snapshot, so no aggregation across rows is needed.
    """
    df = _read_parquet_or_raise(path_perp)
    required = {"symbol", "mark_px", "timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"perp_funding.parquet is missing expected columns: {missing}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "mark_px"])

    # If duplicate (timestamp, symbol) rows exist (e.g. multiple `source`s),
    # average them so the pivot is well-defined.
    grouped = df.groupby(["timestamp", "symbol"], as_index=False)["mark_px"].mean()
    panel = grouped.pivot(index="timestamp", columns="symbol", values="mark_px")
    panel = panel.sort_index()
    return panel


def build_price_panel_from_pools(path_pools: str) -> pd.DataFrame:
    """
    Build a (timestamp x token) price matrix from dex_pools.parquet.
    Each token can have many pool rows per run_id (different DEXs/pairs);
    these are collapsed with a median, which is robust to a single stale or
    manipulated pool skewing the price. Timestamp is parsed from run_id
    since dex_pools.parquet has no explicit timestamp column.
    """
    df = _read_parquet_or_raise(path_pools)
    required = {"run_id", "token", "price_usd"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"dex_pools.parquet is missing expected columns: {missing}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["run_id"], format="%Y%m%d_%H%M%S",
                                      utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "price_usd"])

    grouped = df.groupby(["timestamp", "token"], as_index=False)["price_usd"].median()
    panel = grouped.pivot(index="timestamp", columns="token", values="price_usd")
    panel = panel.sort_index()
    return panel


if PRICE_SOURCE == "perp":
    raw_panel = build_price_panel_from_perp(PATH_PERP)
elif PRICE_SOURCE == "pools":
    raw_panel = build_price_panel_from_pools(PATH_POOLS)
else:
    raise ValueError("PRICE_SOURCE must be 'perp' or 'pools'")

print(f"Raw panel (source={PRICE_SOURCE}): {raw_panel.shape[0]} timestamps x "
      f"{raw_panel.shape[1]} assets")
print("Assets found:", list(raw_panel.columns))

# ---- Resample onto a fixed grid, forward-fill small gaps, then drop assets
# ---- that are still too sparse.
panel = raw_panel.resample(RESAMPLE_FREQ).last()
panel = panel.ffill(limit=5)  # bridge short gaps only; do not fabricate long runs

missing_frac = panel.isna().mean()
keep_assets = missing_frac[missing_frac <= MAX_MISSING_FRAC].index.tolist()
dropped_assets = sorted(set(panel.columns) - set(keep_assets))
if dropped_assets:
    print(f"Dropping assets with >{MAX_MISSING_FRAC:.0%} missing data after "
          f"resampling to {RESAMPLE_FREQ}: {dropped_assets}")
panel = panel[keep_assets].dropna(how="all")

# Final trim: keep only rows where at least 2 assets have data (need pairs).
panel = panel.dropna(thresh=2)

print(f"Final price panel: {panel.shape[0]} timestamps x {panel.shape[1]} assets")
panel.to_parquet(os.path.join(OUTDIR, "price_panel.parquet"))

log_panel = np.log(panel)
returns_panel = log_panel.diff().dropna(how="all")
returns_panel.to_parquet(os.path.join(OUTDIR, "log_returns_panel.parquet"))

print("Saved: price_panel.parquet, log_returns_panel.parquet")


# %%
# ============================================================================
# CELL 3: PAIRWISE CORRELATION OF RETURNS + STATISTICAL SIGNIFICANCE
# ============================================================================
# Correlation is computed on LOG RETURNS, not on price levels. Prices are
# typically non-stationary (they trend), so a naive correlation of price
# levels is prone to "spurious correlation" -- two totally unrelated trending
# series can show high correlation just because both are trending. Returns
# are much closer to stationary, so their correlation is a more honest
# measure of co-movement. Cointegration (Cell 5) is the correct tool for
# relationships between the price LEVELS themselves.

assets = list(returns_panel.columns)
pairs = list(itertools.combinations(assets, 2))

corr_rows = []
for a, b in pairs:
    sub = returns_panel[[a, b]].dropna()
    n_obs = len(sub)
    if n_obs < MIN_OBS_PER_PAIR:
        continue
    pearson_r, pearson_p = stats.pearsonr(sub[a], sub[b])
    spearman_r, spearman_p = stats.spearmanr(sub[a], sub[b])
    corr_rows.append({
        "asset_a": a, "asset_b": b, "n_obs": n_obs,
        "pearson_r": pearson_r, "pearson_p": pearson_p,
        "spearman_r": spearman_r, "spearman_p": spearman_p,
    })

corr_df = pd.DataFrame(corr_rows)

if len(corr_df) > 0:
    # Benjamini-Hochberg FDR correction across all pairs tested simultaneously.
    # With many pairs, some will look "significant" by chance alone (multiple
    # comparisons problem); the q-value tells you the significance level after
    # accounting for that.
    reject, qvals, _, _ = multipletests(corr_df["pearson_p"], alpha=ALPHA, method="fdr_bh")
    corr_df["pearson_q"] = qvals
    corr_df["significant_after_fdr"] = reject

    # 95% CI for Pearson r via Fisher z-transform.
    z = np.arctanh(corr_df["pearson_r"].clip(-0.999999, 0.999999))
    se = 1.0 / np.sqrt(corr_df["n_obs"] - 3)
    z_crit = stats.norm.ppf(1 - ALPHA / 2)
    corr_df["pearson_r_ci_low"] = np.tanh(z - z_crit * se)
    corr_df["pearson_r_ci_high"] = np.tanh(z + z_crit * se)

    corr_df["direction"] = np.where(corr_df["pearson_r"] > 0, "positive", "negative")
    corr_df = corr_df.sort_values("pearson_r", key=lambda s: s.abs(), ascending=False)
else:
    print("No pairs had enough overlapping observations for correlation testing.")

corr_df.to_csv(os.path.join(OUTDIR, "correlation_results.csv"), index=False)
print(f"Saved correlation_results.csv ({len(corr_df)} pairs tested)")
print(corr_df.head(10).to_string(index=False))

# ---- Full correlation matrix + heatmap (for a quick visual overview) -------
full_corr_matrix = returns_panel.corr(method="pearson")
full_corr_matrix.to_csv(os.path.join(OUTDIR, "correlation_matrix_full.csv"))

fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(assets) + 2),
                                 max(5, 0.5 * len(assets) + 2)))
im = ax.imshow(full_corr_matrix.values, vmin=-1, vmax=1, cmap="RdBu_r")
ax.set_xticks(range(len(assets)))
ax.set_xticklabels(assets, rotation=90)
ax.set_yticks(range(len(assets)))
ax.set_yticklabels(assets)
ax.set_title(f"Pearson correlation of log returns ({RESAMPLE_FREQ} bars)")
fig.colorbar(im, ax=ax, label="correlation")
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "plots", "correlation_heatmap.png"), dpi=150)
plt.close(fig)
print("Saved plots/correlation_heatmap.png")


# %%
# ============================================================================
# CELL 4: ROLLING CORRELATION FOR THE STRONGEST PAIRS
# ============================================================================
# A single full-sample correlation number can hide instability (a pair might
# be strongly correlated for the first half of the sample and uncorrelated in
# the second half). Rolling correlation shows whether the relationship is
# STABLE over the 96h window, which matters a lot for whether you'd trust it
# out-of-sample.

top_pairs_for_rolling = corr_df.head(N_TOP_CORR_PAIRS_TO_PLOT)

if len(top_pairs_for_rolling) > 0:
    fig, ax = plt.subplots(figsize=(10, 5))
    for _, row in top_pairs_for_rolling.iterrows():
        a, b = row["asset_a"], row["asset_b"]
        sub = returns_panel[[a, b]].dropna()
        roll_corr = sub[a].rolling(ROLLING_CORR_WINDOW).corr(sub[b])
        ax.plot(roll_corr.index, roll_corr.values,
                label=f"{a}-{b} (full-sample r={row['pearson_r']:.2f})")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Rolling {ROLLING_CORR_WINDOW}-bar correlation, top pairs")
    ax.set_ylabel("correlation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "plots", "rolling_correlation_top_pairs.png"), dpi=150)
    plt.close(fig)
    print("Saved plots/rolling_correlation_top_pairs.png")
else:
    print("Skipping rolling correlation plot: no qualifying pairs.")


# %%
# ============================================================================
# CELL 5: COINTEGRATION SCREENING (relationship between PRICE LEVELS)
# ============================================================================
# Correlation (Cells 3-4) tells you whether returns move together. It does
# NOT tell you whether a stable, tradeable relationship exists between the
# price levels themselves. For statistical arbitrage you generally want
# COINTEGRATION: a linear combination of two (non-stationary) price series
# that is itself stationary / mean-reverting. That linear combination is the
# "spread" you would actually trade.
#
# Method (Engle-Granger, two-step):
#   1. Regress log(price_A) on log(price_B) (OLS) to get a hedge ratio beta.
#   2. Form the residual/spread: log(price_A) - beta * log(price_B).
#   3. Test that spread for a unit root (ADF test). If we can reject a unit
#      root, the spread is stationary -> the pair is cointegrated.
#   4. We also run statsmodels' `coint()`, which implements the same idea
#      with an internally optimized lag length, as a cross-check.
#   5. We try both regression directions (A~B and B~A) since Engle-Granger is
#      not symmetric, and keep whichever direction gives the lower (better)
#      ADF p-value on the residual.

def engle_granger_pair(price_a: pd.Series, price_b: pd.Series):
    """Run Engle-Granger cointegration test in both directions on log prices,
    return the better-fitting direction's hedge ratio, residual series, and
    both formal p-values."""
    log_a = np.log(price_a)
    log_b = np.log(price_b)

    def _fit_direction(y, x):
        X = add_constant(x.values)
        model = OLS(y.values, X).fit()
        alpha, beta = model.params
        resid = y.values - (alpha + beta * x.values)
        resid = pd.Series(resid, index=y.index)
        adf_stat, adf_p, *_ = adfuller(resid, autolag="AIC")
        return {"alpha": alpha, "beta": beta, "resid": resid,
                "adf_stat": adf_stat, "adf_p": adf_p}

    dir_ab = _fit_direction(log_a, log_b)   # log_a = alpha + beta*log_b
    dir_ba = _fit_direction(log_b, log_a)   # log_b = alpha + beta*log_a

    # statsmodels' formal Engle-Granger test (own lag selection), both directions.
    try:
        coint_stat_ab, coint_p_ab, _ = coint(log_a.values, log_b.values, autolag="AIC")
    except Exception:
        coint_p_ab = np.nan
    try:
        coint_stat_ba, coint_p_ba, _ = coint(log_b.values, log_a.values, autolag="AIC")
    except Exception:
        coint_p_ba = np.nan

    if dir_ab["adf_p"] <= dir_ba["adf_p"]:
        chosen = dir_ab
        chosen_direction = f"log({price_a.name}) ~ log({price_b.name})"
        dependent, independent = price_a.name, price_b.name
    else:
        chosen = dir_ba
        chosen_direction = f"log({price_b.name}) ~ log({price_a.name})"
        dependent, independent = price_b.name, price_a.name

    coint_p_best = np.nanmin([coint_p_ab, coint_p_ba])

    return {
        "dependent": dependent,
        "independent": independent,
        "direction": chosen_direction,
        "hedge_ratio": chosen["beta"],
        "intercept": chosen["alpha"],
        "spread": chosen["resid"],
        "adf_stat": chosen["adf_stat"],
        "adf_p": chosen["adf_p"],
        "engle_granger_p_best_direction": coint_p_best,
    }


coint_rows = []
coint_details = {}  # (a,b) -> engle_granger_pair() result, kept for later cells

for a, b in pairs:
    sub = log_panel[[a, b]].dropna()
    if len(sub) < MIN_OBS_PER_PAIR:
        continue
    price_a = np.exp(sub[a])   # back to price space, function takes price levels
    price_b = np.exp(sub[b])
    result = engle_granger_pair(price_a, price_b)
    coint_rows.append({
        "asset_a": a, "asset_b": b, "n_obs": len(sub),
        "direction": result["direction"],
        "hedge_ratio": result["hedge_ratio"],
        "intercept": result["intercept"],
        "adf_stat": result["adf_stat"],
        "adf_p": result["adf_p"],
        "engle_granger_p": result["engle_granger_p_best_direction"],
    })
    coint_details[(a, b)] = result

coint_df = pd.DataFrame(coint_rows)

if len(coint_df) > 0:
    reject_c, qvals_c, _, _ = multipletests(coint_df["adf_p"], alpha=ALPHA, method="fdr_bh")
    coint_df["adf_q"] = qvals_c
    coint_df["cointegrated_after_fdr"] = reject_c
    coint_df = coint_df.sort_values("adf_p", ascending=True)
else:
    print("No pairs had enough overlapping observations for cointegration testing.")

coint_df.to_csv(os.path.join(OUTDIR, "cointegration_results.csv"), index=False)
print(f"Saved cointegration_results.csv ({len(coint_df)} pairs tested)")
print(coint_df.head(10).to_string(index=False))


# %%
# ============================================================================
# CELL 6: OU MEAN-REVERSION PARAMETERS + Z-SCORE SIGNAL FOR COINTEGRATED PAIRS
# ============================================================================
# For each pair's spread (from Cell 5), we:
#   1. Fit a discrete-time AR(1) to the spread: spread_t = c + phi*spread_{t-1} + e_t
#      This is the exact discretization of the continuous-time
#      Ornstein-Uhlenbeck process  d(spread) = theta*(mu - spread)*dt + sigma*dW.
#      phi in (0,1) implies mean reversion; phi close to 1 implies very slow
#      (or no) reversion; phi >= 1 implies no reversion at all.
#   2. Convert phi into OU parameters:
#         theta       = -ln(phi) / dt         (speed of mean reversion)
#         mu          = c / (1 - phi)         (long-run mean of the spread)
#         sigma_eq    = std(e) / sqrt(1-phi^2) (stationary/equilibrium std dev)
#         half_life   = ln(2) / theta          (time for spread to close half
#                                                the gap to its mean)
#      dt is expressed in units of RESAMPLE_FREQ bars, so theta/half_life come
#      out in "number of bars"; we also report half-life converted to hours.
#   3. Compute two z-score series for the spread:
#         rolling z-score:  (spread - rolling_mean) / rolling_std   (adaptive,
#                            standard "pairs trading" signal)
#         OU z-score:        (spread - mu) / sigma_eq                (uses the
#                            fitted long-run mean/vol instead of a rolling
#                            window; more stable but assumes OU params are
#                            constant over the sample)

def fit_ou_from_spread(spread: pd.Series, dt: float = 1.0):
    s = spread.dropna()
    y = s.values[1:]
    x = s.values[:-1]
    X = add_constant(x)
    model = OLS(y, X).fit()
    c, phi = model.params
    resid = y - (c + phi * x)
    resid_std = resid.std(ddof=2)

    is_mean_reverting = 0 < phi < 1
    if is_mean_reverting:
        theta = -np.log(phi) / dt
        mu = c / (1 - phi)
        half_life_bars = np.log(2) / theta
        sigma_eq = resid_std / np.sqrt(1 - phi ** 2)
    else:
        theta = np.nan
        mu = np.nan
        half_life_bars = np.nan
        sigma_eq = np.nan

    # standard error / p-value on phi via the OLS fit (approx test of phi==1,
    # NOT the same as the ADF test's distribution, but a useful supplementary
    # diagnostic reported alongside it).
    phi_se = model.bse[1]
    phi_t = (phi - 1) / phi_se
    phi_p_approx = 2 * (1 - stats.norm.cdf(abs(phi_t)))

    return {
        "phi": phi, "intercept": c, "phi_se": phi_se,
        "phi_p_vs_unit_root_approx": phi_p_approx,
        "theta": theta, "mu": mu, "sigma_eq": sigma_eq,
        "half_life_bars": half_life_bars,
        "resid_std": resid_std,
        "is_mean_reverting": is_mean_reverting,
    }


def bootstrap_half_life_ci(spread: pd.Series, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """Moving block bootstrap on the spread series -> refit AR(1) each time ->
    percentile CI on half-life. Blocks preserve short-range autocorrelation
    structure that a naive i.i.d. resample would destroy."""
    s = spread.dropna().values
    n = len(s)
    if n < 30:
        return np.nan, np.nan
    block_len = max(5, int(round(n ** (1 / 3))))
    n_blocks = int(np.ceil(n / block_len))
    rng = np.random.default_rng(seed)

    half_lives = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block_len, size=n_blocks)
        sample = np.concatenate([s[st:st + block_len] for st in starts])[:n]
        sample_series = pd.Series(sample)
        try:
            fit = fit_ou_from_spread(sample_series)
            if fit["is_mean_reverting"] and np.isfinite(fit["half_life_bars"]):
                half_lives.append(fit["half_life_bars"])
        except Exception:
            continue

    if len(half_lives) < 20:
        return np.nan, np.nan
    lo, hi = np.percentile(half_lives, [2.5, 97.5])
    return lo, hi


bars_per_hour = pd.Timedelta("1h") / pd.Timedelta(RESAMPLE_FREQ)

ou_rows = []
for _, row in coint_df.iterrows():
    a, b = row["asset_a"], row["asset_b"]
    result = coint_details[(a, b)]
    spread = result["spread"]
    ou = fit_ou_from_spread(spread, dt=1.0)

    current_spread = spread.iloc[-1]
    roll_mean = spread.rolling(ZSCORE_WINDOW).mean().iloc[-1]
    roll_std = spread.rolling(ZSCORE_WINDOW).std().iloc[-1]
    rolling_z = (current_spread - roll_mean) / roll_std if roll_std and roll_std > 0 else np.nan
    ou_z = ((current_spread - ou["mu"]) / ou["sigma_eq"]
            if ou["is_mean_reverting"] and ou["sigma_eq"] > 0 else np.nan)

    ou_rows.append({
        "asset_a": a, "asset_b": b,
        "dependent": result["dependent"], "independent": result["independent"],
        "hedge_ratio": result["hedge_ratio"],
        "adf_p": row["adf_p"], "adf_q": row["adf_q"],
        "engle_granger_p": row["engle_granger_p"],
        "phi": ou["phi"], "phi_p_vs_unit_root_approx": ou["phi_p_vs_unit_root_approx"],
        "is_mean_reverting": ou["is_mean_reverting"],
        "theta_per_bar": ou["theta"],
        "half_life_bars": ou["half_life_bars"],
        "half_life_hours": (ou["half_life_bars"] / bars_per_hour
                             if np.isfinite(ou["half_life_bars"]) else np.nan),
        "ou_mu": ou["mu"], "ou_sigma_eq": ou["sigma_eq"],
        "current_spread": current_spread,
        "current_rolling_zscore": rolling_z,
        "current_ou_zscore": ou_z,
    })

ou_df = pd.DataFrame(ou_rows)

# Bootstrap half-life CIs only for the most promising, already-mean-reverting
# pairs (this step is the slowest, so we bound how many pairs get it).
candidates = ou_df[ou_df["is_mean_reverting"]].sort_values("adf_p").head(
    N_TOP_COINT_PAIRS_FOR_DETAIL)

ci_lows, ci_highs = {}, {}
for _, row in candidates.iterrows():
    a, b = row["asset_a"], row["asset_b"]
    spread = coint_details[(a, b)]["spread"]
    lo, hi = bootstrap_half_life_ci(spread)
    ci_lows[(a, b)] = lo / bars_per_hour if np.isfinite(lo) else np.nan
    ci_highs[(a, b)] = hi / bars_per_hour if np.isfinite(hi) else np.nan

ou_df["half_life_hours_ci_low"] = ou_df.apply(
    lambda r: ci_lows.get((r["asset_a"], r["asset_b"]), np.nan), axis=1)
ou_df["half_life_hours_ci_high"] = ou_df.apply(
    lambda r: ci_highs.get((r["asset_a"], r["asset_b"]), np.nan), axis=1)

ou_df.to_csv(os.path.join(OUTDIR, "ou_meanreversion_results.csv"), index=False)
print(f"Saved ou_meanreversion_results.csv ({len(ou_df)} pairs)")
print(ou_df.head(10).to_string(index=False))


# %%
# ============================================================================
# CELL 7: PLOTS FOR THE MOST PROMISING PAIRS (spread + rolling z-score)
# ============================================================================

for _, row in candidates.iterrows():
    a, b = row["asset_a"], row["asset_b"]
    spread = coint_details[(a, b)]["spread"]
    roll_mean = spread.rolling(ZSCORE_WINDOW).mean()
    roll_std = spread.rolling(ZSCORE_WINDOW).std()
    z = (spread - roll_mean) / roll_std

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(spread.index, spread.values, label="spread", color="tab:blue")
    axes[0].plot(roll_mean.index, roll_mean.values, label=f"{ZSCORE_WINDOW}-bar rolling mean",
                 color="tab:orange", linestyle="--")
    axes[0].set_title(f"Spread: log({row['dependent']}) - {row['hedge_ratio']:.3f} * "
                       f"log({row['independent']})  |  half-life ~ "
                       f"{row['half_life_hours']:.2f}h, ADF p={row['adf_p']:.4f}")
    axes[0].legend(fontsize=8)

    axes[1].plot(z.index, z.values, color="tab:green")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].axhline(2, color="red", linestyle="--", linewidth=0.8)
    axes[1].axhline(-2, color="red", linestyle="--", linewidth=0.8)
    axes[1].set_title(f"Rolling {ZSCORE_WINDOW}-bar z-score of spread")

    fig.tight_layout()
    fname = f"spread_zscore_{a}_{b}.png".replace("/", "_")
    fig.savefig(os.path.join(OUTDIR, "plots", fname), dpi=150)
    plt.close(fig)

print(f"Saved {len(candidates)} spread/z-score plots to {os.path.join(OUTDIR, 'plots')}")


# %%
# ============================================================================
# CELL 8: COMBINED SUMMARY TABLE + CONSOLE SUMMARY
# ============================================================================
# One table joining the correlation view (co-movement of returns) with the
# cointegration/OU view (mean-reversion of the price spread) for every pair
# tested in both. This is the single table to scan first.

summary = pd.merge(
    corr_df[["asset_a", "asset_b", "n_obs", "pearson_r", "pearson_p", "pearson_q",
             "significant_after_fdr", "direction"]],
    ou_df[["asset_a", "asset_b", "hedge_ratio", "adf_p", "adf_q",
           "is_mean_reverting", "half_life_hours",
           "half_life_hours_ci_low", "half_life_hours_ci_high",
           "current_rolling_zscore", "current_ou_zscore"]],
    on=["asset_a", "asset_b"], how="outer"
)
summary = summary.sort_values(
    by=["is_mean_reverting", "adf_p"],
    ascending=[False, True],
    na_position="last"
)
summary.to_csv(os.path.join(OUTDIR, "summary_all_pairs.csv"), index=False)

print("\n" + "=" * 78)
print("SUMMARY")
print("=" * 78)
print(f"Price source: {PRICE_SOURCE}  |  Bar size: {RESAMPLE_FREQ}  |  "
      f"Assets analyzed: {len(assets)}  |  Pairs tested: {len(pairs)}")
n_sig_corr = int(corr_df["significant_after_fdr"].sum()) if len(corr_df) else 0
n_coint = int(ou_df["is_mean_reverting"].sum()) if len(ou_df) else 0
print(f"Pairs with FDR-significant return correlation: {n_sig_corr} / {len(corr_df)}")
print(f"Pairs with a mean-reverting (AR1 phi<1) spread: {n_coint} / {len(ou_df)}")
print(f"\nAll outputs written to: {OUTDIR}")
print("  - price_panel.parquet, log_returns_panel.parquet")
print("  - correlation_results.csv, correlation_matrix_full.csv")
print("  - cointegration_results.csv, ou_meanreversion_results.csv")
print("  - summary_all_pairs.csv  <-- start here")
print("  - plots/ (heatmap, rolling correlation, per-pair spread & z-score)")
