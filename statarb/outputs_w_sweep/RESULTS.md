# ZSCORE_WINDOW (W) sensitivity results

Script: [`../run_zscore_window_sweep.py`](../run_zscore_window_sweep.py)

## Protocol

- Spread-only LightGBM; notebook (`cex_gbm_new.ipynb`) untouched
- **Train:** pre–Jul 25 windows except `jul19_pre`
- **Val:** `jul19_pre` (early stopping only — not scored as skill)
- **Test:** Jul 25–28 (`snapshot_idx ≥ 3584` on jul22–28)
- `HORIZON=1`, `N_LAGS=3`
- Skill metrics below are always **test**

Artifacts:

| Run | `MIN_PERIODS` | Outputs |
|---|---|---|
| Scaled | `≈ 0.3·W` (paper: W=300 → 90) | this dir (`w_sweep_all.csv`, plots) |
| Fixed | `20` for all W | [`../outputs_w_sweep_mp20/`](../outputs_w_sweep_mp20/) |

Plots: [`w_vs_r2_diracc.png`](w_vs_r2_diracc.png), [`w_vs_sample_size.png`](w_vs_sample_size.png)

## Full test results (`MIN_PERIODS=20`)

Primary table for comparing W (stable val size; no early-stop collapse at large W).

| W | R² | DirAcc | R²@\|pred\|>0.5 | DirAcc@\|pred\|>0.5 | PnL proxy | Context vs W=300 |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 0.047 | 55.9% | 0.390 | 78.6% | 0.149 | 0.07× |
| 60 | 0.078 | 58.7% | 0.377 | 77.2% | 0.198 | 0.2× |
| 120 | 0.098 | 60.6% | 0.401 | 78.8% | 0.228 | 0.4× |
| 180 | 0.111 | 61.9% | 0.416 | 79.8% | 0.247 | 0.6× |
| 240 | 0.118 | 62.8% | 0.421 | 80.6% | 0.259 | 0.8× |
| **300 (paper)** | **0.124** | **63.1%** | **0.419** | **80.7%** | **0.266** | **1×** |
| 400 | 0.129 | 63.5% | 0.423 | 81.5% | 0.274 | 1.3× |
| 560 | 0.136 | 63.8% | 0.423 | 81.9% | 0.282 | 1.9× |
| 720 | 0.141 | 64.0% | 0.420 | 81.9% | 0.287 | 2.4× |
| 960 | 0.142 | 64.2% | 0.412 | 82.3% | 0.290 | 3.2× |
| 1280 | 0.146 | 64.4% | 0.405 | 82.7% | 0.297 | 4.3× |

Source: [`../outputs_w_sweep_mp20/w_sweep.csv`](../outputs_w_sweep_mp20/w_sweep.csv)

### Cost / benefit vs paper W=300

| W | Context | Δ DirAcc | Δ R² |
|---:|---:|---:|---:|
| 560 | ~1.9× | +0.7pp | +0.012 |
| 720 | ~2.4× | +0.9pp | +0.017 |
| 1280 | ~4.3× | +1.3pp | +0.022 |

Lifts flatten past ~560–720. Scaled `0.3·W` runs match mid-W closely; at large W that regime shrinks val badly (e.g. ~11k rows at W=1280), so prefer the fixed-mp=20 table for high-W reads.

## Decision

**Keep W=300** (`MIN_PERIODS=90`).

Moving to ~720 buys only ~1pp DirAcc for ~2×+ lookback — small enough to be holdout noise, while live pays longer warmup and slower adaptation. No change to the production window or paper campaigns.
