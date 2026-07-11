# Statistical Arbitrage Analysis — README

**File:** `statarb_analysis.ipynb`  
**Language:** Python 3.10+  
**Format:** Jupyter Notebook (`.ipynb`)

---

## Overview

This notebook implements a complete statistical arbitrage pipeline for multi-coin OHLCV data stored in a Parquet file. Starting from raw 1-minute candle data across multiple coins and exchanges, it produces:

- Pearson and Spearman return correlations with significance tests
- ADF stationarity tests to confirm I(1) price behaviour
- Engle–Granger pairwise cointegration tests with hedge ratio estimation
- Johansen multivariate cointegration test
- Ornstein–Uhlenbeck parameter estimation (κ, μ, σ, half-life) for each cointegrated pair
- Rolling z-score signals with configurable entry/exit thresholds
- A full suite of diagnostic charts
- All intermediate results written as CSV files to the working directory

---

## Expected Input Data

The notebook expects a single Parquet file with the following schema (matching the Binance-style OHLCV format):

| Column      | Type        | Description                                    |
|-------------|-------------|------------------------------------------------|
| `exchange`  | string      | Exchange name (e.g. `binance`)                 |
| `coin`      | string      | Coin ticker (e.g. `AAVE`, `BTC`)               |
| `timestamp` | int64       | Unix timestamp in milliseconds                  |
| `datetime`  | timestamp   | ISO 8601 datetime string or timestamp object   |
| `open`      | double      | Opening price of the bar                       |
| `high`      | double      | Highest price of the bar                       |
| `low`       | double      | Lowest price of the bar                        |
| `close`     | double      | Closing price of the bar                       |
| `volume`    | double      | Volume traded during the bar                   |

The data is in **long format**: each row is one bar for one coin on one exchange. The notebook pivots this to a wide-format price matrix internally.

---

## Installation

Install required packages with pip:

```bash
pip install pandas pyarrow numpy scipy statsmodels matplotlib seaborn jupyter
```

All packages are standard; no proprietary or unusual dependencies are required.

---

## Usage

1. Place `statarb_analysis.ipynb` in the same directory as your Parquet file.
2. Open the notebook in VS Code (Jupyter extension) or JupyterLab.
3. Edit the **Configuration block** in Cell 0 as needed (see below).
4. Run all cells top to bottom (Kernel → Restart & Run All).

All output CSV files and PNG charts are written to the same directory as the notebook.

---

## Configuration

All user-facing settings are at the top of the notebook in Cell 0 under the `Configuration` heading. No other cell needs to be edited for a standard run.

| Variable         | Default       | Description |
|------------------|---------------|-------------|
| `PARQUET_FILE`   | `"data.parquet"` | Path to your Parquet file. Can be relative (same directory) or absolute. |
| `PRICE_COL`      | `"close"`     | Which OHLC column to use as the price series. Options: `"open"`, `"high"`, `"low"`, `"close"`. See the Price Column Choice section below for guidance. |
| `EXCHANGE_FILTER`| `None`        | Set to a string like `"binance"` to restrict analysis to one exchange. `None` uses all exchanges and labels series as `exchange:coin`. |
| `MIN_CORR`       | `0.6`         | Absolute Pearson correlation threshold for flagging pairs as strongly correlated in the summary output and rolling correlation plot. |
| `COINT_PVALUE`   | `0.05`        | p-value threshold for Engle–Granger cointegration. Pairs with p < this value are treated as cointegrated. |
| `ZSCORE_WINDOW`  | `60`          | Lookback window (in bars) for the rolling mean and standard deviation used in z-score calculation. |
| `ZSCORE_ENTRY`   | `2.0`         | Z-score magnitude at which a position is entered (long or short the spread). |
| `ZSCORE_EXIT`    | `0.5`         | Z-score magnitude below which an open position is flagged for exit. |

### Price Column Choice

`close` is recommended and is the default. It is the most widely used price in quantitative research because it represents the price at which the last trade cleared for that bar, is least susceptible to intrabar noise, and is the price at which most systematic strategies assume execution at the end of a bar.

`(open + close) / 2` (OC midpoint) is a common alternative that averages the bar's start and end; it reduces end-of-bar bias at the cost of slightly less interpretability. To use it, add this line after loading `df_raw` in Cell 1:

```python
df_raw["oc_mid"] = (df_raw["open"] + df_raw["close"]) / 2
```

Then set `PRICE_COL = "oc_mid"` in the configuration.

`(high + low) / 2` (HL midpoint) is sometimes preferred because it captures the full intrabar range. It is less correlated with the close but can be a useful robustness check. Add:

```python
df_raw["hl_mid"] = (df_raw["high"] + df_raw["low"]) / 2
```

---

## Notebook Structure

### Cell 0 — Imports and Configuration
Loads all libraries and defines the configuration constants. No computation occurs here.

### Cell 1 — Load Data
Reads the Parquet file with `pyarrow` and prints schema and sample rows. The `datetime` column is parsed to timezone-naive UTC. If `EXCHANGE_FILTER` is set, the data is filtered here. If multiple exchanges are present and no filter is applied, series are labelled `exchange:coin` so that the same coin on different exchanges is treated as a distinct series.

### Cell 2 — Wide Price Matrix
Pivots from long to wide format using `pivot_table` with `aggfunc="mean"`. If the same `(datetime, series_id)` combination appears more than once (e.g. due to duplicate rows), the prices are averaged. Up to 5 consecutive missing bars are forward-filled; any remaining rows with NaN are dropped. The matrix is saved as `prices_wide.csv`.

### Cell 3 — Log-Returns
Computes `log_returns = ln(P_t / P_{t-1})`. Log-returns are preferred over simple returns in statistical work for three reasons: they are time-additive (the return over two periods is the sum of single-period returns), they are approximately symmetric around zero, and they are better approximated by a normal distribution for short intervals. Descriptive statistics are printed and histograms are saved as `log_return_distributions.png`.

### Cell 4 — ADF Stationarity Tests
Runs the Augmented Dickey-Fuller test on both price levels and log-returns for every series. The null hypothesis is that the series has a unit root (is non-stationary). Cointegration analysis is only valid when price levels are I(1): non-stationary in levels but stationary in first differences. The test uses AIC-selected lag length and a constant-only regression. Results are saved as `adf_tests.csv`.

Expected outcomes:
- Price levels: p > 0.05 (fail to reject H0 — non-stationary, as expected).
- Log-returns: p < 0.05 (reject H0 — stationary, as expected).

If a price series is already stationary (integrated of order 0), it cannot be cointegrated with another series in the standard sense, and the cointegration results for that series should be interpreted with caution.

### Cell 5 — Pearson and Spearman Correlation
Computes pairwise correlation on log-returns using exact p-values from `scipy.stats.pearsonr` and `scipy.stats.spearmanr`. The null hypothesis for each p-value is ρ = 0.

**Pearson correlation** measures the linear association between two return series. It is sensitive to outliers and assumes approximate normality of returns.

**Spearman correlation** measures the monotonic association based on ranks. It is more robust to outliers and non-normality. Comparing Pearson and Spearman for the same pair is informative: if they are similar, the relationship is roughly linear; if Spearman is much stronger, there is a monotonic but nonlinear relationship.

Both correlation matrices, their p-value matrices, and a pairwise summary table (with direction: positive or negative) are saved as CSV files. A heatmap of both is saved as `correlation_heatmaps.png`.

**Direction interpretation:**
- Positive correlation (r > 0): both coins tend to move in the same direction. A pairs trade shorts the outperformer and longs the underperformer, betting on convergence.
- Negative correlation (r < 0): the coins tend to move in opposite directions. A pairs trade would be long both or short both, depending on the specific divergence signal.

Correlation on returns quantifies co-movement speed; it does not by itself imply a mean-reverting spread. Two series can be highly correlated in returns but have a non-stationary spread. Cointegration is the correct test for spread stationarity.

### Cell 6 — Rolling Correlation
Plots the Pearson correlation over a sliding window of `ZSCORE_WINDOW` bars for each pair that meets the `MIN_CORR` threshold. This reveals whether the relationship is stable over time or concentrated in particular regimes. Structural breaks — periods where the rolling correlation drops sharply — can invalidate a pairs strategy that was calibrated on the full sample. The plot is saved as `rolling_correlation.png`.

### Cell 7 — Engle–Granger Cointegration Tests
Tests every pairwise combination of coins for cointegration. The Engle–Granger two-step procedure works as follows:

**Step 1 (OLS regression):** Regress `price_a` on `price_b` using ordinary least squares:

```
price_a = α + β * price_b + ε
```

The coefficient β is the **hedge ratio**: the number of units of coin B required to hedge one unit of coin A. The residual `ε` is the estimated spread.

**Step 2 (ADF on residuals):** Test whether the spread `ε` is stationary using the ADF test with MacKinnon (1994/2010) critical values adjusted for the cointegration regression context (these are more negative than standard ADF critical values because the residuals are by construction a best-fit line).

The `statsmodels.tsa.stattools.coint` function implements both steps. The reported p-value tests H0: no cointegration. Rejecting H0 (p < `COINT_PVALUE`) means the spread is stationary and the pair is a candidate for a mean-reversion strategy.

Important caveats:
- The Engle–Granger test has low power with small samples (fewer than ~200 observations per series).
- The hedge ratio estimated by OLS is not symmetric: regressing A on B gives a different β than regressing B on A. Both directions are tested via `itertools.combinations`, so only one direction per pair is tested. If you suspect the relationship runs the other way, swap the order manually.
- Cointegration is a long-run relationship and can break down over shorter windows. It should be re-estimated periodically in a live strategy.

Results are saved as `cointegration_engle_granger.csv`.

### Cell 8 — Johansen Cointegration Test
The Johansen test is a multivariate generalisation that tests all series simultaneously. Its advantages over the Engle–Granger test are:

1. It tests for the number of cointegrating vectors (not just whether any exist).
2. It has higher power when multiple series are jointly cointegrated.
3. The result is not sensitive to which variable is placed on the left-hand side of the regression.

Two test statistics are reported:
- **Trace statistic**: tests H0 that the rank of the cointegration space is ≤ r, for r = 0, 1, 2, …
- **Max-eigenvalue statistic**: tests H0 that the rank is exactly r versus r + 1.

Critical values at the 90%, 95%, and 99% levels are printed alongside each statistic. The notebook uses `det_order=0` (constant inside the cointegrating vector, appropriate for price series that have no deterministic trend relative to one another) and `k_ar_diff=1` (one lag of differences, equivalent to a VAR(2) in levels).

The cointegrating eigenvectors (loadings of each coin in the cointegrating relationships) are saved as `johansen_eigenvectors.csv`. These can be used to construct multi-asset spread portfolios.

### Cell 9 — Ornstein–Uhlenbeck Parameter Estimation
The OU process is the continuous-time analogue of a mean-reverting AR(1) process. For each cointegrated pair (or all pairs if none are cointegrated), the spread is modelled as:

```
dS_t = κ(μ − S_t) dt + σ dW_t
```

Where:
- `κ` (kappa): speed of mean reversion. A larger κ means the spread reverts to its mean faster.
- `μ` (mu): the long-run equilibrium mean of the spread. In practice, this should be close to zero after removing the OLS intercept from the spread definition.
- `σ` (sigma): the instantaneous volatility of the spread, i.e. how noisy the spread is.
- `half-life = ln(2) / κ`: the expected time (in bars) for a deviation from the mean to decay to half its initial magnitude. This is the most practically useful OU parameter for strategy design.

**Estimation method (discrete-time OLS on AR(1)):**

The continuous OU process, when discretised at interval Δt, yields:

```
S_t = a + b * S_{t-1} + ε_t
```

where `b = exp(−κ Δt)`, `a = μ(1 − b)`, and `Var(ε_t) = σ²(1 − b²) / (2κ)`. OLS is run on this AR(1) and the continuous-time parameters are recovered algebraically:

```
κ = −ln(b) / Δt
μ = a / (1 − b)
σ = std(ε_t) / sqrt(Δt)
half_life = ln(2) / κ
```

`dt=1` is used throughout (dimensionless bar units). If you want the half-life in real time (e.g. minutes), set `dt = bar_duration_in_minutes` in the `fit_ou_params` call.

**Interpreting the AR(1) coefficient:**
- `b ∈ (0, 1)`: the spread is mean-reverting. The closer to 0, the faster it reverts.
- `b ≥ 1`: the spread is explosive (non-stationary). This should not occur for a genuinely cointegrated pair but can arise from estimation error with short samples.
- `b ≤ 0`: the spread oscillates, which is unusual and may indicate a misspecified hedge ratio.

Results are saved as `ou_parameters.csv`.

### Cell 10 — Z-Score Mean Reversion Signals
Constructs a rolling z-score for each pair's spread:

```
z_t = (S_t − mean(S_{t-W:t})) / std(S_{t-W:t})
```

where W = `ZSCORE_WINDOW` bars.

**Signal rules:**
- `z_t < −ZSCORE_ENTRY`: enter long the spread (buy coin A, sell β units of coin B). The spread is abnormally low and expected to revert upward.
- `z_t > +ZSCORE_ENTRY`: enter short the spread (sell coin A, buy β units of coin B). The spread is abnormally high and expected to revert downward.
- `|z_t| < ZSCORE_EXIT`: flag for exit. The spread has reverted close to its mean.

The signal column in the output is a vectorised snapshot: +1 (long), −1 (short), 0 (flat). This is not a full backtest with position carry-over; it shows which bars individually meet the entry or exit criterion. To implement a proper signal with position holding, you would use a stateful loop or a position tracker that holds the signal until the exit condition is met.

For each pair, the spread, rolling mean, rolling standard deviation, z-score, and signal are saved as `zscore_<coin_a>__<coin_b>.csv`. Charts are saved as `zscore_<coin_a>__<coin_b>.png`.

**Relationship between Z-score and OU:**

The z-score approach and OU parameterisation are complementary:
- The OU half-life tells you how quickly the spread is expected to revert, which informs the choice of `ZSCORE_WINDOW`. A reasonable rule of thumb is to set the window to 2–5 times the OU half-life.
- The OU σ tells you the typical spread volatility, which informs the entry threshold. Entering at ±2σ corresponds to events that occur roughly 5% of the time under normality.
- If the half-life is very short (e.g. 2–3 bars), the strategy requires high-frequency execution. If it is very long (e.g. hundreds of bars), the strategy may not generate enough signals to be practical.

### Cell 11 — Signal Summary
Counts long bars, short bars, and flat bars for each pair, and computes the fraction of time the strategy is invested. Also records the OU half-life for reference. Saved as `signal_summary.csv`.

### Cell 12 — Master Results Table
Merges the Engle–Granger results, pairwise correlation stats, and OU parameters into a single CSV (`master_results.csv`) for easy sorting and filtering in Excel or a data tool.

### Cell 13 — Pair Summary Charts
For each top cointegrated pair (up to 5), produces a three-panel chart:
1. Normalised price levels of both coins (each divided by its first observation) to show co-movement visually.
2. Raw spread with rolling mean and ±ZSCORE_ENTRY σ bands.
3. Z-score with entry/exit threshold lines and signal markers.

These are the primary diagnostic charts for evaluating the quality of a pair.

---

## Output Files

| File | Description |
|------|-------------|
| `prices_wide.csv` | Wide-format price matrix: rows are datetimes, columns are series IDs. |
| `adf_tests.csv` | ADF stationarity results for price levels and log-returns. Columns: series, adf_stat, p_value, lags_used, n_obs, crit_1pct, crit_5pct, crit_10pct, stationary. |
| `pearson_correlation.csv` | Pearson correlation matrix (n × n). |
| `pearson_pvalues.csv` | p-values for Pearson correlation under H0: ρ = 0. |
| `spearman_correlation.csv` | Spearman rank correlation matrix. |
| `spearman_pvalues.csv` | p-values for Spearman correlation. |
| `correlation_pairs.csv` | All pairs with Pearson r, Spearman r, both p-values, and direction. |
| `cointegration_engle_granger.csv` | EG cointegration results: coint_stat, p_value, critical values, hedge_ratio, intercept, n_obs, cointegrated flag. |
| `johansen_eigenvectors.csv` | Matrix of Johansen cointegrating vectors (one column per vector). |
| `ou_parameters.csv` | OU parameter estimates per pair: kappa, mu, sigma, half_life, ar1_coef, r_squared. |
| `zscore_<pair>.csv` | Per-bar spread, rolling mean, rolling std, z-score, and signal for one pair. |
| `signal_summary.csv` | Per-pair signal bar counts, fraction invested, and OU half-life. |
| `master_results.csv` | All results merged: correlation + cointegration + OU in one table. |
| `log_return_distributions.png` | Histogram grid of log-return distributions for all series. |
| `correlation_heatmaps.png` | Side-by-side Pearson and Spearman correlation heatmaps. |
| `rolling_correlation.png` | Rolling Pearson correlation over time for strong pairs. |
| `zscore_<pair>.png` | Z-score chart with entry/exit signals for each pair. |
| `pair_summary_<A>_<B>.png` | Three-panel chart (prices, spread, z-score) for top cointegrated pairs. |

---

## Statistical Assumptions and Limitations

**Stationarity requirement.** Engle–Granger and Johansen cointegration tests assume that the individual price series are integrated of order 1 (I(1)). The ADF tests in Cell 4 verify this. If a series is I(0) (already stationary), it cannot be cointegrated in the classical sense.

**Spurious cointegration.** With a small number of observations or a large number of pairs being tested, some pairs will appear cointegrated by chance. Always apply a Bonferroni or Benjamini–Hochberg correction to the p-values when testing many pairs simultaneously. A simple Bonferroni adjustment divides the significance threshold by the number of pairs tested.

**Regime changes.** All tests are run on the full sample. A pair may be cointegrated in one regime and not another. Rolling Engle–Granger tests (run the EG test on a sliding window) can detect breakdowns but are computationally expensive and are not implemented here by default.

**Hedge ratio stability.** The OLS hedge ratio β is estimated once on the full sample. In a live strategy, β should be re-estimated periodically (e.g. monthly or on a rolling window) as the long-run relationship between prices can drift.

**Transaction costs.** The z-score signals do not account for trading fees, slippage, or bid-ask spreads. Cryptocurrency markets can have significant spread costs for altcoins, and the signal frequency implied by a short half-life may be unprofitable after costs.

**Bar size and the OU dt parameter.** The OU half-life is reported in bars. To convert to real time, multiply by the bar duration (e.g. 1 minute for 1-minute OHLCV data). The `dt` parameter in `fit_ou_params` can be set to any positive value; it does not change κ or half-life when `dt=1` (bar units), which is the default.

---

## Common Issues

**`ModuleNotFoundError`**: Run `pip install <package>` for any missing library. The full list is in the Installation section above.

**`Engine 'pyarrow' is not installed`**: Run `pip install pyarrow`.

**`KeyError: 'close'`**: The parquet file uses different column names. Print `df_raw.columns` after loading to inspect and update `PRICE_COL` accordingly.

**Empty cointegrated pairs**: This is normal if the data window is short, the coins move independently, or the significance threshold is strict. Try lowering `COINT_PVALUE` to 0.10 as a diagnostic, and inspect `master_results.csv` sorted by p_value to see the closest pairs.

**Z-score all NaN at the start**: The rolling window requires `ZSCORE_WINDOW` bars before producing a value. The first `ZSCORE_WINDOW` rows of the z-score column will be NaN by design.

**RuntimeWarning about invalid value in divide**: This occurs when the rolling standard deviation is zero (perfectly flat price segment). The notebook guards against this by replacing near-zero standard deviations with NaN, so the z-score will be NaN for those bars.

---

## Extending the Analysis

**Multiple price columns**: To run the analysis on, say, both `close` and `hl_mid`, run the notebook twice with different `PRICE_COL` settings and compare the `master_results.csv` outputs.

**Bonferroni correction**: After running, load `correlation_pairs.csv` and apply:
```python
n_pairs = len(df)
df["pearson_pvalue_bonferroni"] = (df["pearson_pvalue"] * n_pairs).clip(upper=1.0)
```

**Rolling Engle–Granger**: To check cointegration stability, run the `engle_granger_pair` function on rolling subsets of `prices_wide` (e.g. 500-bar rolling windows) and plot the resulting p-values over time.

**Position sizing**: A common approach is to size positions inversely proportional to the spread volatility (`roll_std`), so that each entry has the same expected dollar risk regardless of how wide the spread currently is.

**Live trading integration**: The z-score DataFrames can be updated incrementally by appending new price data and recomputing only the latest row of the rolling statistics. The spread and z-score columns in `zscore_<pair>.csv` provide the historical baseline for live comparison.
