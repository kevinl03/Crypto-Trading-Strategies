# Why LightGBM over Random Forest (tabular baseline)

**Purpose:** Justify keeping **LightGBM** as the production / paper-trading head after fitting a classical **Random Forest** peer on the same Jul-25 protocol.  
**Role of RF:** Literature / classical ML baseline (parallel to the LSTM sequence peer) — not a live replacement.  
**Evidence:** `statarb/outputs_rf/` · runner `statarb/run_rf_zscore_baseline.py` · branch `feat/rf-zscore-baseline`.

---

## Verdict (paste-ready)

On the shared LOGO Jul 25–28 feature matrix, RF and LightGBM learn **nearly the same ranking signal** (`corr(pred_RF, pred_LGBM) ≈ 0.98`). LightGBM remains the deployed choice because, at the paper confidence gate `|pred| ≥ 0.9`, it delivers **~2× more trades** at **similar DirAcc**, **higher R²**, and **higher total `pnl_proxy` mass**, with a live paper-trading path RF never had. RF’s slightly higher own-gate DirAcc is a **selection artifact** of a compressed `|pred|` scale, not a skill win on matched rows.

---

## Shared protocol

| Knob | Value |
|---|---|
| Target | \(y_t = z_{t+1}\) of cross-exchange `spread_bps` |
| Z window | \(W{=}300\), `min_periods{=}90`, \(H{=}1\), `N_LAGS{=}3` |
| Features | Same LOGO cache as LGBM paper tables (62/68 published cols; six Coinbase volume lags pruned in cache) |
| Split | Train pre Jul 25 · test Jul 25–28 (no shuffle) |
| Metrics | DirAcc · R² · mean `pnl_proxy = sign(pred) × y` |
| Naive | \(\hat z \leftarrow z_t\) on identical rows |
| RF fit | `n_estimators=400`, `max_depth=20`, `min_samples_leaf=200`, `max_features=sqrt`, `max_samples=1e6`, `n_jobs=8` on full ~4.95M train rows (~5 min CPU) |

---

## 1. Headline numbers (own-gate — what *not* to over-read)

| Model | Filter | n | Fire rate | DirAcc | R² | mean pnl | ≈ total pnl (`mean×n`) |
|---|---|---:|---:|---:|---:|---:|---:|
| LightGBM | all | 1,680,426 | 100% | 62.8% | **0.133** | 0.269 | — |
| Random Forest | all | 1,680,426 | 100% | 62.9% | 0.126 | 0.267 | — |
| LightGBM | `|pred|≥0.5` | 273,183 | 16.3% | 78.4% | **0.384** | 0.758 | — |
| Random Forest | `|pred|≥0.5` | 216,627 | 12.9% | 79.2% | 0.380 | 0.805 | — |
| LightGBM | `|pred|≥0.9` | **66,513** | **4.0%** | 85.2% | **0.525** | 1.154 | **~77k** |
| Random Forest | `|pred|≥0.9` | 33,038 | 2.0% | **86.0%** | 0.478 | **1.299** | ~43k |
| Naive \(z_t\) | `|z_t|≥0.9` | 602,453 | — | 69.5% | −0.523 | 0.545 | — |

Source: `statarb/outputs_rf/METRICS.md` (full-panel RF; production `statarb/outputs/statarb_lgbm.txt` scored on the same test rows).

**Reading guide:** RF’s τ=0.9 DirAcc looks “best” only because it clears the absolute gate half as often. Prefer §2–§3 for architecture claims.

---

## 2. Matched-row check (fair skill)

Score every model on the **same row mask** — removes fire-rate confounding.

| Mask | Model | n | DirAcc | R² | mean pnl |
|---|---|---:|---:|---:|---:|
| LGBM `|pred|≥0.9` | LightGBM | 66,513 | 85.2% | **0.525** | 1.154 |
| LGBM `|pred|≥0.9` | Random Forest | 66,513 | **85.2%** | 0.489 | 1.154 |
| LGBM `|pred|≥0.9` | Naive \(z_t\) | 66,513 | 85.3% | 0.464 | 1.119 |
| RF `|pred|≥0.9` | Random Forest | 33,038 | 86.0% | 0.478 | 1.299 |
| RF `|pred|≥0.9` | LightGBM | 33,038 | 86.0% | **0.521** | 1.299 |
| RF `|pred|≥0.9` | Naive \(z_t\) | 33,038 | **87.8%** | 0.523 | 1.300 |

**Takeaways**

1. On LGBM’s high-confidence book, RF and LGBM have **identical DirAcc**; LGBM wins **R²**.
2. On RF’s own `|pred|≥0.9` slice, **naive persistence DirAcc is highest** — that gate is picking easy continuation / extreme-\(z\) regimes, not proving RF superiority.
3. Predictions agree strongly: `corr(RF, LGBM) ≈ 0.979` (magnitudes `corr(|RF|,|LGBM|) ≈ 0.957`).

---

## 3. Why RF `|pred|` is compressed (and why that matters)

| Quantity (test) | RF | LightGBM |
|---|---:|---:|
| mean `|pred|` | 0.245 | 0.275 |
| p99 `|pred|` | 1.02 | 1.26 |
| fraction `|pred|≥0.9` | **2.0%** | **4.0%** |

**Mechanism:** RF predicts a **mean of leaf means** across hundreds of trees (`min_samples_leaf=200` already smooths each leaf). Averaging shrinks extremes toward the center (**under-dispersion**). LightGBM **boosts residuals**, so later trees keep pushing mass into the tails. Same directional ranking, smaller RF magnitudes → the paper’s absolute τ=0.9 is a **stricter percentile** for RF.

So “RF DirAcc 86% vs LGBM 85% at τ=0.9” ≠ better model; it means “RF rarely says 0.9.”

---

## 4. Why we still choose LightGBM

| Axis | Winner | Evidence |
|---|---|---|
| **Trade volume at paper τ=0.9** | **LGBM** | ~2× fire rate / n (67k vs 33k) at similar DirAcc |
| **Calibration / R² on gated & all rows** | **LGBM** | Higher R² own-gate and matched-row; all-rows R² 0.133 vs 0.126 |
| **Total proxy throughput** | **LGBM** | `mean×n` at τ=0.9 ≈ 77k vs ≈ 43k |
| **Matched-row DirAcc** | Tie | Identical on LGBM’s τ=0.9 mask |
| **Live deployment** | **LGBM** | `experiments/paper_trade_lgbm.py`, Aug 4–7 campaign, τ tables — RF has no live book |
| **Ops / artifact** | **LGBM** | ~1.2 MB text booster, native categoricals, fast retrain loop; RF joblib ~180 MB, heavier RAM fit |
| **Classical baseline honesty** | RF useful | Shows GBDT is not “winning by being the only tree model” — bagging peer tracks the same signal |

**Decision rule used:** Prefer the head that maximizes **deployable high-confidence throughput** (n × quality) under a fixed absolute abstention gate, not the model that maximizes DirAcc on its own sparsest slice.

---

## 5. What RF *is* good for in the paper

- Classical tabular peer next to LSTM (deep) and mechanical \(z_t\) (non-learned).
- Sanity check that the 60+ feature matrix carries signal under bagging as well as boosting.
- Cautionary example for **τ reporting**: always show fire rate / n and a matched-row column when models differ in `|pred|` scale.

---

## 6. Caveats (keep in Methods / footnote)

1. LOGO cache is **62 features**; production booster lists **68** (six Coinbase volume lags missing in cache — filled as NaN for LGBM score). Not a perfect feature-parity stress test, but both models see the same frame.
2. RF used `max_samples=1e6` bootstrap draws for RAM/wall-clock on ~5M rows; still full-panel `X` for fitting.
3. Validation (Aug 4–7) headline in the paper remains the **live LGBM book** (`tau09_w300_report`); RF was not re-traded live.
4. Own-gate RF τ=0.9 mean pnl can look high; pair it with n and matched-row tables (§2).

---

## Artifacts

| Path | Contents |
|---|---|
| `statarb/run_rf_zscore_baseline.py` | Train / score / mem budget check |
| `statarb/outputs_rf/METRICS.md` | Comparison table |
| `statarb/outputs_rf/metrics_test.json` | Machine-readable suite |
| `statarb/outputs_rf/rf_model.joblib` | Fitted RF (~180 MB) |
| `statarb/outputs/statarb_lgbm.txt` | Production LGBM head |

---

## Paper one-liner

> A sklearn Random Forest on the same Jul-25 tabular matrix tracks LightGBM predictions almost one-to-one, but LightGBM’s residual boosting yields a less compressed confidence scale—roughly doubling `|pred|≥0.9` trade count at matched DirAcc with higher R²—so we retain LightGBM as the live trading head and treat RF as the classical bagging baseline.
