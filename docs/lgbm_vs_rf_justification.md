# Why LightGBM over Random Forest (tabular baseline)

**Purpose:** Justify keeping **LightGBM** as the production / paper-trading head after fitting a classical **Random Forest** peer on the paper **test** split.  
**Role of RF:** Literature / classical bagging baseline (tree-family control) — not a live replacement.  
**Evidence:** `statarb/outputs_rf/` · runner `statarb/run_rf_zscore_baseline.py` · branch `feat/rf-zscore-baseline`.  
**Split vocabulary:** matches Experimental Setup (`sec:splits`) — dates stated once; elsewhere use **train / test / validation** only.

| Set | Definition (paper) | Role for RF |
|---|---|---|
| **Training** | June → mid-July 2026, before the Jul 25 cut (~2.9M feature rows in paper text; LOGO cache may be larger) | Fit RF here only |
| **Test** | July 25–28, 2026 (~1.7M rows; CEX API–collected, chronological) | **Primary RF vs LGBM architecture table** |
| **Validation** | ~72h live paper campaign Aug 4–7, 2026 (`max_open=50`) | Live LGBM book / τ table; RF not live-traded |

Paper gate: \(\tau{=}0.9\) (chosen on **validation** closes; Table 5 in the paper is validation-only). \(W{=}300\) retained (diminishing returns past that under \(\tau{=}0.9\)).

---

## Verdict (paste-ready)

On the shared LOGO **test** feature matrix, RF and LightGBM learn **nearly the same ranking signal** (`corr(pred_RF, pred_LGBM) ≈ 0.98`). LightGBM remains the deployed choice because, at the paper confidence gate `|pred| ≥ 0.9`, it delivers **~2× more trades** at **similar DirAcc**, **higher R²**, and **higher total `pnl_proxy` mass**, and it alone has the **validation** live path. RF’s slightly higher own-gate DirAcc is a **selection artifact** of a compressed `|pred|` scale, not a skill win on matched rows.

---

## Shared protocol

| Knob | Value |
|---|---|
| Target | \(y_t = z_{t+1}\) of cross-exchange `spread_bps` |
| Z window | \(W{=}300\), `min_periods{=}90`, \(H{=}1\), `N_LAGS{=}3` |
| Features | Same LOGO cache as LGBM paper tables (62/68 published cols; six Coinbase volume lags pruned in cache) |
| Split | Train / test / validation as `sec:splits` (no shuffle) |
| Metrics | DirAcc · R² · mean `pnl_proxy = sign(pred) × y` |
| Naive | \(\hat z \leftarrow z_t\) on identical rows |
| RF fit | `n_estimators=400`, `max_depth=20`, `min_samples_leaf=200`, `max_features=sqrt`, `max_samples=1e6`, `n_jobs=8` on full LOGO train panel (~5 min CPU) |

---

## 1. Test-set headline numbers (own-gate — what *not* to over-read)

Architecture compare lives on the **test** set. Do not label this “offline Campaign A/B” or mix Jul 31 into the RF table — those framings were dropped from the paper.

| Model | Filter | n | Fire rate | DirAcc | R² | mean pnl | ≈ total pnl (`mean×n`) |
|---|---|---:|---:|---:|---:|---:|---:|
| LightGBM | all | 1,680,426 | 100% | 62.8% | **0.133** | 0.269 | — |
| Random Forest | all | 1,680,426 | 100% | 62.9% | 0.126 | 0.267 | — |
| LightGBM | `|pred|≥0.5` | 273,183 | 16.3% | 78.4% | **0.384** | 0.758 | — |
| Random Forest | `|pred|≥0.5` | 216,627 | 12.9% | 79.2% | 0.380 | 0.805 | — |
| LightGBM | `|pred|≥0.9` | **66,513** | **4.0%** | 85.2% | **0.525** | 1.154 | **~77k** |
| Random Forest | `|pred|≥0.9` | 33,038 | 2.0% | **86.0%** | 0.478 | **1.299** | ~43k |
| Naive \(z_t\) | `|z_t|≥0.9` | 602,453 | — | 69.5% | −0.523 | 0.545 | — |

Source: `statarb/outputs_rf/METRICS.md` (full-panel RF; production `statarb/outputs/statarb_lgbm.txt` on the same **test** rows).

Paper LGBM test reference (published `tab:test`): all-rows R² 0.133 / DirAcc 62.8%; `|ẑ|≥0.9` R² 0.535 / DirAcc 85.3%. Our LGBM re-score on the LOGO cache is within rounding / 62-vs-68 feature fill (85.2% / 0.525).

**Reading guide:** RF’s τ=0.9 DirAcc looks “best” only because it clears the absolute gate half as often. Prefer §2–§3 for architecture claims.

---

## 2. Matched-row check (fair skill)

Score every model on the **same test-row mask** — removes fire-rate confounding.

| Mask | Model | n | DirAcc | R² | mean pnl |
|---|---|---:|---:|---:|---:|
| LGBM `|pred|≥0.9` | LightGBM | 66,513 | 85.2% | **0.525** | 1.154 |
| LGBM `|pred|≥0.9` | Random Forest | 66,513 | **85.2%** | 0.489 | 1.154 |
| LGBM `|pred|≥0.9` | Naive \(z_t\) | 66,513 | 85.3% | 0.464 | 1.119 |
| RF `|pred|≥0.9` | Random Forest | 33,038 | 86.0% | 0.478 | 1.299 |
| RF `|pred|≥0.9` | LightGBM | 33,038 | 86.0% | **0.521** | 1.299 |
| RF `|pred|≥0.9` | Naive \(z_t\) | 33,038 | **87.8%** | 0.523 | 1.300 |

**Takeaways**

1. On LGBM’s high-confidence **test** book, RF and LGBM have **identical DirAcc**; LGBM wins **R²**.
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
| **Trade volume at paper τ=0.9 (test)** | **LGBM** | ~2× fire rate / n (67k vs 33k) at similar DirAcc |
| **Calibration / R² on gated & all rows** | **LGBM** | Higher R² own-gate and matched-row; all-rows R² 0.133 vs 0.126 |
| **Total proxy throughput (test)** | **LGBM** | `mean×n` at τ=0.9 ≈ 77k vs ≈ 43k |
| **Matched-row DirAcc** | Tie | Identical on LGBM’s τ=0.9 mask |
| **Validation / live deployment** | **LGBM** | Only LGBM has the Aug 4–7 validation book and paper τ table; RF has no live book |
| **Ops / artifact** | **LGBM** | ~1.2 MB text booster, native categoricals, fast retrain loop; RF joblib ~180 MB, heavier RAM fit |
| **Classical baseline honesty** | RF useful | Shows GBDT is not “winning by being the only tree model” — bagging peer tracks the same signal |

**Decision rule used:** Prefer the head that maximizes **deployable high-confidence throughput** (n × quality) under a fixed absolute abstention gate, not the model that maximizes DirAcc on its own sparsest slice.

---

## 5. Validation set (paper) — what RF does *not* claim

Paper **validation** = Aug 4–7 ~72h live campaign. The paper’s τ sensitivity table is **validation-only** (\(\tau \in \{0.5, 0.75, 0.9, 1.0\}\) with \(n\)); headline validation at \(\tau{=}0.9\): \(n{=}12{,}795\), DirAcc 86.7%, R² 0.599, mean pnl \(+1.37\).

| Model | Set | Filter | n | DirAcc | R² | mean pnl |
|---|---|---|---:|---:|---:|---:|
| LightGBM (live book) | validation | `|ẑ|≥0.9` | 12,795 | 86.7% | 0.599 | +1.37 |
| Random Forest | validation | — | — | — | — | *not live-scored* |

Mechanical baseline DirAcc / mean-PnL figures in Results are also **validation-panel** capacity-matched replays — keep RF out of that live mechanical story.

---

## 6. What RF *is* good for in the paper

- Classical **bagging** peer under the same train→test feature matrix (tree-family control next to mechanical \(z_t\)).
- Sanity check that the 60+ feature matrix carries signal under bagging as well as boosting.
- Cautionary example for **τ reporting**: always show fire rate / \(n\) and a matched-row column when models differ in `|pred|` scale.
- Optional deep peer (LSTM) remains a separate architecture track; do not conflate with RF.

---

## 7. Where to integrate in the paper (aligned with current main)

Keep the **primary claim** unchanged: learned forecast beats **mechanical** `|z_t|` rules on the **validation** panel. Add RF only as a **within-family tabular control** on the **test** set.

| Paper place | Insert |
|---|---|
| **§Methodology (`sec:lgbm`)** | After LGBM hyperparams: one paragraph — sklearn RF on the same feature matrix / train–test split as classical bagging peer; production remains LGBM; note averaging → compressed `\|pred\|`. |
| **§Experimental Setup (`Baseline Definitions`)** | New subsubsection **Tree ensemble baseline (Random Forest)** — scored on **test**; same metrics; report own-gate **and** matched-row. |
| **§Results — Test-set evaluation** | Extend `tab:test` or add a small LGBM vs RF vs naive table (all-rows + τ=0.9, with **n**). One sentence: own-gate DirAcc can favor RF via fire rate; matched-row DirAcc ties, LGBM R² higher. |
| **§Ablation / Model Tuning** | Best home for depth: **Boosting vs bagging** — fire-rate + matched-row; do **not** put RF into the validation-only τ table (Table 5). |
| **§Discussion / limitations** | RF is a **test**-set peer, not validation-live; LOGO 62/68 feature caveat if those numbers are cited. |
| **Do not** | Relabel RF results as “offline / Campaign A–B / Jul 31”; do not replace validation mechanical baseline figures with RF. |

**Narrative ladder**

```text
Mechanical |z_t| on validation  →  “need a learned forecast?”     (primary live claim)
Naive z_t on model rows         →  “more than persistence?”
RF (bagging) on test            →  “just any tree?”               (new)
LightGBM (boosting)             →  deployed head + validation
```

---

## 8. Caveats (Methods / footnote)

1. LOGO cache is **62 features**; production booster lists **68** (six Coinbase volume lags missing in cache — filled as NaN for LGBM score). Both models see the same frame.
2. RF used `max_samples=1e6` bootstrap draws for RAM/wall-clock; still full-panel `X` for fitting.
3. **Validation** headline remains the live LGBM book; RF was not re-traded live.
4. Own-gate RF τ=0.9 mean pnl can look high; pair it with \(n\) and matched-row tables (§2).
5. Paper train-size text (~2.9M) may differ from LOGO cache row count (~4.95M) depending on which windows were pooled — always cite the LOGO cache when reporting these RF numbers.

---

## Artifacts

| Path | Contents |
|---|---|
| `statarb/run_rf_zscore_baseline.py` | Train / score / mem budget check |
| `statarb/outputs_rf/METRICS.md` | Test-set comparison table |
| `statarb/outputs_rf/metrics_test.json` | Machine-readable suite |
| `statarb/outputs_rf/encoder.joblib` | OrdinalEncoder + medians (tracked) |
| `statarb/outputs_rf/rf_model.joblib` | Fitted RF (~180 MB; gitignored) |
| `statarb/outputs/statarb_lgbm.txt` | Production LGBM head |

---

## Paper one-liner

> On the paper test set, a sklearn Random Forest tracks LightGBM predictions almost one-to-one, but LightGBM’s residual boosting yields a less compressed confidence scale—roughly doubling `|pred|≥0.9` trade count at matched DirAcc with higher R²—so we retain LightGBM as the live validation head and treat RF as the classical bagging baseline.
