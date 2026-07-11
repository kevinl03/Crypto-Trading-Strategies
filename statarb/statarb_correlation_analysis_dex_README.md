# Crypto DEX/Perp Statistical Arbitrage — Correlation & Mean-Reversion Analysis

This document explains `statarb_correlation_analysis.py`: what it assumes about
your data, exactly what it computes, how to run it, and how to read every
output file. Read the **"Important note about your data"** section first —
it affects how you should interpret everything downstream.

---

## 1. Important note about your data

You described your files as OHLCV data. Based on the schema visible in your
files, that isn't quite right, and the script is built around what your data
actually is, not around OHLC bars. Your dataset is **periodic point-in-time
snapshots**, collected roughly every ~1 minute over 96 hours, across five
files:

| File | Grain | Relevant columns |
|---|---|---|
| `dex_pools.parquet` | one row per **DEX pool** per token per snapshot (`run_id`) | `run_id, token, chain, dex, price_usd` |
| `dex_spreads.parquet` | one row per token per snapshot, summarizing the cross-venue spread | `run_id, token, low_price, high_price, ...` |
| `depth/dex_gas.parquet` | one row per chain per snapshot | `run_id, chain, gas_gwei, timestamp` |
| `depth/dex_quotes.parquet` | one row per (token, chain, source) swap quote per snapshot | `run_id, token, chain, notional_usd, ...` |
| `depth/perp_funding.parquet` | one row per perp **symbol** per snapshot | `symbol, mark_px, funding_rate, timestamp` |

There is no `open`, `high`, `low`, or `close` anywhere in this schema —
each snapshot just records the price at that instant. That's actually fine
for this analysis: at ~1-minute resolution, a single snapshot price is a
perfectly reasonable substitute for a "close." The script treats each
snapshot's price as one bar in a time series and builds returns from
consecutive snapshots.

Two of the five files can be turned into a price time series per asset:

- **`perp_funding.parquet` → `mark_px`** (the script's default). One row per
  symbol per snapshot, with a clean explicit `timestamp` column. This is the
  cleanest and simplest series to build from your data.
- **`dex_pools.parquet` → `price_usd`**. Multiple pool rows per token per
  snapshot (different DEXs / chains / pairs). The script collapses these to
  one price per token per snapshot by taking the **median** across pools,
  which is robust to any single stale or thin pool skewing the number. The
  timestamp for this source is parsed from `run_id` (format
  `YYYYMMDD_HHMMSS`), since `dex_pools.parquet` has no explicit `timestamp`
  column.

`dex_spreads.parquet`, `depth/dex_gas.parquet`, and `depth/dex_quotes.parquet`
are **not** used as price sources by this script — `dex_spreads` is a
derived summary (not a single clean price series), and gas/quote data
describe execution conditions rather than an asset's price. They're
mentioned here so you know the script isn't silently ignoring files it
should be using; a natural extension (not included) would be layering gas
costs or quote slippage on top of the signals this script produces, to see
whether a mean-reversion trade would survive execution costs.

You choose the price source with `PRICE_SOURCE = "perp"` or `"pools"` at the
top of the script. Everything downstream (returns, correlation,
cointegration, OU fitting) works identically regardless of which one you
pick — only Cell 2 (loading) differs.

---

## 2. What the script actually does, cell by cell

The script is a single `.py` file using `# %%` cell markers. Open it in VS
Code with the Python extension and each `# %%` block becomes a runnable
notebook cell (with plots shown inline in the interactive window); or just
run it top-to-bottom as a normal script — both work identically, and it will
produce identical output files either way.

### Cell 1 — Configuration
All tunable parameters live here: file paths, price source, resampling
frequency, rolling window lengths, significance threshold, and bootstrap
settings. Nothing below this cell needs to be edited for normal use.

### Cell 2 — Load data & build the price panel
Reads the chosen source file, pivots it into a wide **(timestamp × asset)**
price matrix, resamples it onto a fixed grid (`RESAMPLE_FREQ`, default
`"1min"`), forward-fills only very short gaps (up to 5 bars — this bridges
momentary missing snapshots without fabricating long stretches of fake
data), and drops any asset still missing more than `MAX_MISSING_FRAC`
(default 20%) of its data after that. Saves `price_panel.parquet` and
`log_returns_panel.parquet`.

Returns are computed as **log returns**: `log(price_t) - log(price_t-1)`.
Log returns are used (rather than simple % returns) because they're
additive over time and better-behaved statistically for the tests used
later.

### Cell 3 — Pairwise correlation of returns + significance
For every pair of assets, computes:
- **Pearson correlation** (linear co-movement) and its p-value.
- **Spearman correlation** (rank-based, robust to non-linear but monotonic
  relationships and to outliers) as a cross-check.
- A **95% confidence interval** for the Pearson correlation, via the Fisher
  z-transform (`arctanh`), which is the standard way to get a CI for a
  correlation coefficient without assuming you know its exact sampling
  distribution in small samples.
- A **Benjamini-Hochberg (FDR) correction** across *all* pairs tested in the
  same run. This matters: if you test 50 pairs at the usual 5% significance
  level, you'd expect ~2-3 "significant" results by pure chance even if
  nothing is really related. The `pearson_q` column and
  `significant_after_fdr` flag account for this — trust `pearson_q < 0.05`
  more than a raw `pearson_p < 0.05` when you're scanning many pairs.

**Why correlation is computed on *returns*, not price levels:** crypto
prices trend, and trending series will show artificially high correlation
with each other even when there's no real relationship (the classic
"spurious regression" problem — e.g. two totally unrelated things that both
happened to rise over 96 hours will look "correlated" in level space).
Returns are much closer to stationary (no persistent trend), so their
correlation is a much more honest measure of whether two assets actually
move together day-to-day. If you want the price-level relationship instead,
that's what cointegration (Cell 5) is for.

Outputs: `correlation_results.csv` (per-pair, sorted by strength),
`correlation_matrix_full.csv` (the full matrix), `plots/correlation_heatmap.png`.

### Cell 4 — Rolling correlation for the strongest pairs
A single full-sample correlation number can hide instability — a pair might
have been correlated for the first half of your window and uncorrelated for
the second half, which the full-sample number won't tell you. This cell
plots a rolling `ROLLING_CORR_WINDOW`-bar correlation for the
`N_TOP_CORR_PAIRS_TO_PLOT` strongest pairs so you can see whether the
relationship is stable over the 96 hours or was a temporary coincidence.

Output: `plots/rolling_correlation_top_pairs.png`.

### Cell 5 — Cointegration screening (Engle-Granger)
This is the core "is there a tradeable relationship" test. Correlation
answers "do returns move together"; cointegration answers "is there a
*stable linear relationship between the price levels* that a spread trade
could exploit."

For each pair (A, B):
1. Regress `log(price_A)` on `log(price_B)` by OLS to get a **hedge ratio**
   (β). Also run the reverse regression (`log(price_B)` on `log(price_A)`),
   since Engle-Granger is not symmetric — which asset you treat as
   "dependent" can change the result.
2. Form the **spread**: `log(price_A) - β·log(price_B)` (the OLS residual).
3. Run an **Augmented Dickey-Fuller (ADF) test** on that spread. The null
   hypothesis is "the spread has a unit root" (i.e., it's a random walk that
   never reverts). A low p-value lets you reject that null — i.e., evidence
   the spread is stationary / mean-reverting, which is what you want for a
   pairs trade.
4. Keep whichever regression direction gives the lower (better) ADF p-value
   on its residual.
5. Also run statsmodels' built-in `coint()` function (both directions,
   minimum p-value kept) as a second, independently-implemented
   cross-check on the same idea.
6. Apply the same **FDR correction** described above (`adf_q` column) since
   many pairs are being tested at once.

Outputs: `cointegration_results.csv`, sorted by ADF p-value (best evidence
of cointegration first).

### Cell 6 — OU (Ornstein-Uhlenbeck) mean-reversion parameters + z-scores
For every pair from Cell 5, the spread series is used to fit a discrete-time
**AR(1)** model:

```
spread_t = c + phi * spread_(t-1) + e_t
```

This is the exact discretization of the continuous-time OU process
`d(spread) = theta * (mu - spread) * dt + sigma * dW`, and `phi` maps onto
OU parameters as:

- `theta = -ln(phi) / dt` — speed of mean reversion (dt = 1 bar here).
- `mu = c / (1 - phi)` — the long-run mean the spread reverts to.
- `sigma_eq = std(residuals) / sqrt(1 - phi²)` — the *stationary* (equilibrium)
  standard deviation of the spread, i.e. how far it typically wanders once
  it's settled into its long-run distribution.
- `half_life = ln(2) / theta` — the time (in bars, and converted to hours)
  for the spread to close half the gap back to its mean after a shock. This
  is usually the single most useful number for deciding whether a
  mean-reversion trade is practically tradeable (a half-life of 20 minutes
  is very different from one of 3 days).

Two z-score series are then computed for the current spread level:

- **Rolling z-score**: `(spread - rolling_mean) / rolling_std` over
  `ZSCORE_WINDOW` bars. This is the standard, adaptive "pairs trading"
  signal — it doesn't assume the OU parameters are constant, just that
  recent history is a reasonable local reference.
- **OU z-score**: `(spread - mu) / sigma_eq`, using the *fitted* long-run
  mean and equilibrium standard deviation instead of a rolling window. More
  stable, but only valid if you believe the OU parameters are roughly
  constant over your whole sample.

**A caveat you should know about `phi < 1`:** in any finite sample, an AR(1)
fit to even a *pure random walk* (no real mean reversion at all) will almost
always come out with an estimated `phi` slightly less than 1, purely from
estimation bias — so `is_mean_reverting = phi < 1` is a *necessary but not
sufficient* condition and will often be `True` even for junk pairs. **Don't
trust that column alone.** The actual statistical confidence in
mean-reversion comes from the ADF test's `adf_p` / `adf_q` (Cell 5), and from
the **bootstrap confidence interval on the half-life** (below) — both are
specifically designed to have the right false-positive rate, unlike a raw
`phi < 1` check.

**Bootstrap half-life confidence interval:** for the `N_TOP_COINT_PAIRS_FOR_DETAIL`
pairs with the strongest ADF evidence, the script runs a **moving block
bootstrap**: it resamples the spread series in contiguous blocks (block
length ≈ n^(1/3), so short-range autocorrelation within a block is
preserved, unlike a naive i.i.d. resample which would destroy it), refits
the AR(1)/OU model on each resample, and collects the 2.5th/97.5th
percentiles of the resulting half-life distribution across `N_BOOTSTRAP`
(default 300) resamples. A narrow CI means the half-life estimate is
reasonably precise; a wide one (or one that includes implausibly large
values) means you shouldn't trust the point estimate much.

Output: `ou_meanreversion_results.csv`.

### Cell 7 — Plots for the most promising pairs
For each of the `N_TOP_COINT_PAIRS_FOR_DETAIL` pairs used in the bootstrap
step, saves a two-panel plot: the raw spread with its rolling mean overlaid,
and the rolling z-score with ±2 reference lines (a conventional, though
arbitrary, entry-signal threshold for pairs trading).

Output: `plots/spread_zscore_<A>_<B>.png` (one per pair).

### Cell 8 — Combined summary table
Merges the correlation view and the cointegration/OU view into one table so
you don't have to cross-reference three CSVs by hand, and prints a short
console summary of how many pairs cleared each bar.

Output: `summary_all_pairs.csv` — **start here.**

---

## 3. Output files reference

All outputs are written to `statarb_outputs/` inside your working directory
(next to the parquet files), which the script creates if it doesn't exist.

| File | Contents |
|---|---|
| `price_panel.parquet` | Resampled (timestamp × asset) price matrix actually used for everything downstream. |
| `log_returns_panel.parquet` | Log returns of the above. |
| `correlation_results.csv` | Per-pair Pearson/Spearman r, p-values, FDR q-values, 95% CI. |
| `correlation_matrix_full.csv` | Full asset × asset Pearson correlation matrix. |
| `cointegration_results.csv` | Per-pair hedge ratio, ADF stat/p-value, Engle-Granger p-value, FDR q-value. |
| `ou_meanreversion_results.csv` | Per-pair OU parameters (theta, mu, sigma_eq, half-life ± bootstrap CI), current z-scores. |
| `summary_all_pairs.csv` | Correlation + cointegration/OU results merged into one table. |
| `plots/correlation_heatmap.png` | Full return-correlation matrix, visually. |
| `plots/rolling_correlation_top_pairs.png` | Rolling correlation over time for the top correlated pairs. |
| `plots/spread_zscore_<A>_<B>.png` | Spread + rolling mean, and rolling z-score, for top cointegrated pairs. |

---

## 4. How to read the results for actual pair selection

A good statistical-arbitrage candidate pair typically has **all** of:

1. A statistically significant correlation in returns (`pearson_q < 0.05` in
   `correlation_results.csv`) — confirms the assets genuinely co-move, not
   just correlated by chance.
2. Strong evidence of cointegration (low `adf_p` / `adf_q` in
   `cointegration_results.csv` — comfortably below 0.05, not just barely
   under it, since with many pairs tested some will clear 0.05 by luck even
   after FDR correction if the correction itself is borderline).
3. A **half-life** (`half_life_hours` in `ou_meanreversion_results.csv`)
   that's short enough to be practically tradeable relative to your holding
   period, with a **bootstrap CI that isn't absurdly wide** — a half-life
   estimate of "6 hours, CI [1h, 400h]" tells you almost nothing useful,
   while "6 hours, CI [4h, 9h]" is a much stronger basis for a trade.
4. A current z-score (rolling or OU) that's meaningfully away from zero —
   this is what would actually trigger a trade signal (conventionally
   |z| > 2 to enter, closing near |z| < 0.5), not just evidence the *pair*
   is good.

None of this constitutes a backtest or a P&L estimate — it only tells you
which relationships exist and how confident to be in them statistically.
Before trading anything, you'd still want to (a) backtest the actual entry/exit
z-score rule out-of-sample, (b) account for the gas costs and slippage/quote
data already sitting in your other two parquet files, and (c) check that a
pair's cointegration holds up on data the script hasn't seen yet, since
Engle-Granger/ADF results on a single 96-hour window can and do overfit to
that particular window.

---

## 5. Usage notes

**Requirements:** `pandas`, `numpy`, `scipy`, `statsmodels`, `matplotlib`,
and a parquet engine (`pyarrow` recommended). Install with:

```bash
pip install pandas numpy scipy statsmodels matplotlib pyarrow
```

**Running it:**
```bash
cd /path/to/folder/containing/your/parquet/files
python statarb_correlation_analysis.py
```
or open the `.py` file in VS Code and run it cell-by-cell via the Python
extension's "Run Cell" links (each `# %%` marker starts a new cell).

**File locations:** the script assumes it is being run *from* the directory
that directly contains `dex_pools.parquet`, `dex_spreads.parquet`, and a
`depth/` subfolder with `dex_gas.parquet`, `dex_quotes.parquet`,
`perp_funding.parquet` — matching the layout implied by your screenshots. If
your real layout differs, edit the `PATH_*` constants in Cell 1.

**Things you'll likely want to tune:**
- `PRICE_SOURCE`: switch between `"perp"` and `"pools"` to compare results
  from the two available price sources — if a pair looks cointegrated under
  both, that's stronger evidence than either alone.
- `RESAMPLE_FREQ`: widen (e.g. `"5min"`) if your real data's snapshot cadence
  is coarser than 1 minute, or to reduce noise.
- `MIN_OBS_PER_PAIR`: raise this if you want to be stricter about excluding
  thinly-covered pairs from the statistical tests.
- `N_TOP_COINT_PAIRS_FOR_DETAIL` / `N_BOOTSTRAP`: the bootstrap step is the
  slowest part of the script (it refits an AR(1) model `N_BOOTSTRAP` times
  per pair); reduce either if runtime becomes an issue with your real
  (larger) dataset, increase for tighter confidence intervals.

**A note on runtime with your real 44 MB dataset:** the synthetic test data
used to validate this script had ~500 snapshots × 7 assets. Your real
dataset spans a full 96 hours — depending on the true snapshot cadence and
number of tokens, this could be several thousand timestamps × 15-20+ assets,
which means more pairs (`n·(n-1)/2` grows quickly) and longer time series per
pair. Correlation and cointegration screening (Cells 3 and 5) scale fine even
at that size; if the bootstrap step (Cell 6) feels slow, lower
`N_TOP_COINT_PAIRS_FOR_DETAIL` and/or `N_BOOTSTRAP` first.
