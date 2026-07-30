# Model improvement experiment results

Scripts: [`run_improve_experiments.py`](../run_improve_experiments.py)  
Notebook protocol updated in [`cex_gbm_new.ipynb`](../cex_gbm_new.ipynb)  
Unified data: LSR + liquidations copied via [`data/_build_cex_unified.py`](../../../stochastic-spread-modeling/data/_build_cex_unified.py)

## 1. Protocol fix (chronological)

| Split | Role | LGBM R² | LGBM DirAcc | Notes |
|---|---|---|---|---|
| Jul19 pre | val (early stop only) | 0.031 | 55.3% | not final skill |
| Jul19 post | final test | **0.058** | 57.4% | naive DirAcc **60.4%** |
| Jul22–28 | forward holdout | **0.063** | 58.5% | naive DirAcc **60.9%** |

Old inverted protocol (Jul22–28 in train, early-stop on Jul19 post) reported **0.068 R²** on that same Jul19 post slice — inflated vs the clean **0.058**.

## 2. Horizon + objective sweep (test = Jul19 post)

- **`zscore_fwd` R² collapses with horizon** (H=2 → 0.058; H=60 → 0.001; H=120 → ~0).
- **`zscore_delta` R² rises with horizon** (H=2 → 0.005; H=60 → 0.049; H=120 → 0.057) and beats naive badly on DirAcc/PnL proxy.
- **Binary classifier on sign** at H=2 fwd: Acc **62.1%** vs naive **60.4%** (small but real edge on direction). At H=120 delta: Acc **58.0%** vs naive **43.1%**.

Best near-term modeling pivot: **longer-horizon delta / sign**, not more H=2 level regression.

## 3. LSR + liquidations (Jul22–28 only, 60/20/20 by snapshot)

| Target | Features | test R² | DirAcc | cls Acc |
|---|---|---|---|---|
| zscore_fwd | spread only | 0.0565 | 58.2% | 61.5% |
| zscore_fwd | + LSR/liqs | 0.0569 | 58.1% | 61.5% |
| zscore_delta | spread only | 0.0090 | 53.1% | 54.7% |
| zscore_delta | + LSR/liqs | 0.0089 | 53.1% | 54.6% |

**No meaningful lift** from LSR/liquidations at H=2 on this week (sparse ~10 min cadence, 4/2 venues).

## 4. Does the validation set choice/order change accuracy?

Jul22–28 held fixed as an untouched yardstick in all five configs, so any movement is attributable to the validation choice alone ([`val_rotation.csv`](val_rotation.csv)).

| Config | val | test | best_iter | test R² | **forward R²** | forward DirAcc |
|---|---|---|---|---|---|---|
| A (current) | Jul19 pre | Jul19 post | 108 | 0.0580 | **0.06331** | 58.46% |
| B (swapped) | Jul19 post | Jul19 pre | 202 | 0.0312 | **0.06291** | 58.32% |
| C | Jul13 tail 20% | Jul19 all | 220 | 0.0525 | **0.06220** | 58.23% |
| D | Jul19 random 20% (non-chrono) | Jul19 rest | 202 | 0.0534 | **0.06291** | 58.32% |
| E | Jul19 first 20% (chrono) | Jul19 rest | 108 | 0.0582 | **0.06331** | 58.46% |

- **Forward R² spread: 0.0011** (0.0622–0.0633), forward DirAcc spread 0.23pp. Validation choice is not load-bearing.
- The validation set enters training *only* through `best_iteration`. B and D both stopped at 202 and produced bit-identical forward metrics; A and E both stopped at 108 and likewise. The metric surface is flat from ~108 to ~220 rounds, so a different stopping point costs nothing.
- The large test-R² swing between A (0.058) and B (0.031) is **not** a protocol effect — it tracks the window, not the role. Jul19 pre-outage (133k rows) is genuinely less predictable than post-outage (633k rows); whichever one is scored as "test" reports its own regime difficulty.
- **Row order within validation is exactly irrelevant**: shuffling val rows gives identical `best_iteration` (108) and forward R² delta of `+0.000000`. LightGBM aggregates the val metric over the whole set, so permutation is a no-op.

## 5. Deferred

See [DEFER_OHLCV.md](DEFER_OHLCV.md) — OHLCV and more same-signal hours stay off until coverage and target justify them.
