# LOGO / Nested Feature-Group Ablation Results

**Verdict: Scenario C.** Under the published LightGBM protocol, microstructure
features do **not** add filtered R² over a pure autoregression on lagged
z-scores/spreads (+ coin/pair identity). The AR baseline is best.

Protocol matches `cex_gbm_new.ipynb`: H=1, W=300, N_LAGS=3, published
`LGBM_PARAMS`, Jul 25–28 test cut (`snapshot_idx >= 3584`). All variants use
a **fixed** boosting-round budget equal to the full-model early-stopping
iterate (**49** rounds). Primary metric: R² on `|ẑ| ≥ 0.5`.

Script: `statarb/run_logo_ablation.py`  
Artifacts: `statarb/outputs_logo/`

## Nested cumulative table (paper Table)

| Variant | # feat. | R² (all) | R² (filt.) | DirAcc (filt.) | Δ R² filt. vs AR |
|---|---:|---:|---:|---:|---:|
| AR baseline (lags + momentum + identity) | 11 | 0.132 | **0.392** | 78.9% | — |
| + ticker | 42 | 0.133 | 0.382 | 78.4% | −0.010 |
| + orderbook | 45 | 0.132 | 0.378 | 78.2% | −0.014 |
| + trade flow | 48 | 0.132 | 0.382 | 78.4% | −0.010 |
| + funding | — | — | — | — | pruned (0 cols) |
| + open interest | — | — | — | — | pruned (0 cols) |
| + cross-venue / full | 62 | 0.133 | 0.380 | 78.1% | −0.012 |

## Classic leave-one-group-out

| Variant | # feat. | R² (all) | R² (filt.) | DirAcc (filt.) |
|---|---:|---:|---:|---:|
| full | 62 | 0.133 | 0.381 | 78.3% |
| − baseline | 53 | 0.014 | **0.124** | 67.9% |
| − ticker | 31 | 0.132 | 0.394 | 79.0% |
| − orderbook | 59 | 0.132 | 0.383 | 78.3% |
| − trades | 59 | 0.132 | 0.381 | 78.3% |
| − cross | 48 | 0.132 | 0.384 | 78.5% |
| − funding / − OI | 62 | (no-op; groups empty) | | |

Dropping the AR baseline destroys performance. Dropping any microstructure
family does not hurt (dropping ticker slightly *helps* filtered R²).

## Notes

- Funding / OI never enter the trained feature matrix under the paper prune
  (`null_pct > 0.4` or `|corr| < 0.005`), matching their absence from the
  published 68-feature booster.
- This rebuild retains **62 / 68** published features (Coinbase bid/ask volume
  lags were pruned here; they appear in the on-disk booster). Conclusions are
  unchanged under either feature set.
- Full-model early-stop best_iter = **49** (paper text cites 74; on-disk
  `statarb_lgbm.txt` has 51 trees). Fixed-round protocol uses 49 for all
  variants.
- Implication for the paper: reframe the dataset contribution as enabling
  future architectures, not as empirically necessary for the current LGBM
  forecast.
