# Tree baselines for cross-exchange next-\(z\) forecasting (RF vs LightGBM)

**Purpose:** Evidence notes for the paper — how to use Random Forest **without** turning the story into a weak “LightGBM beats RF” bake-off.  
**Role of RF:** Classical bagging control on the **test** set (sanity that signal is not booster-specific).  
**Reported model:** LightGBM (live **validation** path).  
**Strategic framing (reviewer-aware):** Emphasize the **problem setting**; deemphasize the particular learner. Position the work as a **simple but effective baseline** for a setting that has received little attention — not as an architecture paper whose main claim is GBDT ≫ RF.

**Split vocabulary:** Experimental Setup (`sec:splits`) — dates once; elsewhere **train / test / validation** only.

| Set | Definition (paper) | Role for RF |
|---|---|---|
| **Training** | June → mid-July 2026, before the Jul 25 cut (~2.9M feature rows in paper text; LOGO cache may be larger) | Fit RF here only |
| **Test** | July 25–28, 2026 (~1.7M rows; CEX API–collected, chronological) | RF vs LGBM peer table |
| **Validation** | ~72h live paper campaign Aug 4–7, 2026 (`max_open=50`) | Primary live claim (LGBM vs mechanical); RF not live-traded |

Paper gate: \(\tau{=}0.9\) (chosen on **validation** closes; τ table is validation-only). \(W{=}300\) retained (diminishing returns past that under \(\tau{=}0.9\)).

**Research framing only:** Do **not** motivate LightGBM / trees with CPU-live inference, artifact size, or ops deployability. Those are engineering footnotes at best — not paper claims.

---

## Framing advice (read before citing RF in the paper)

Reviewers can dismiss a thin RF comparison as “insufficient baselines” *or* as a weak bake-off. Do **not** lead with method novelty of GBDT, and do **not** promise deferred DNN/LSTM comparisons. Lead with:

1. **Understudied setting.** Same-asset **cross-exchange** spreads at ~1-minute cadence with **live L2 / trade-flow** features that candle APIs cannot reconstruct; mechanical \(|z_t|\) rules dominate prior crypto pairs work and miss venue/regime structure.
2. **Hard evaluation.** Chronological train/test, live validation under real collection latency, capacity-matched mechanical peers, selective \(\tau\) gate, z-unit settlement.
3. **Intentional tree/forest baseline class.** Prior supervised spread work often uses DNN/LSTM; we instead use **tabular tree ensembles** (LightGBM production head; Random Forest bagging control) suited to engineered lags and native categoricals — a different inductive bias, not a “future bake-off.”
4. **Simple effective baseline.** A gated tree regressor already beats matched persistence on validation; RF on the test set tracks the same ranking → lift is largely **formulation + setting + features + gate**.
5. **Test-set RF is enough.** Both test and validation are OOS relative to training. LightGBM’s \(\tau{=}0.9\) skill is similar on the two sets → a separate RF validation campaign is unnecessary for this control.

Use RF to show **robustness of the tree baseline class** (bagging ≈ boosting on ranking). Avoid claiming “we introduce a superior ML method” or “we will compare to LSTMs later.”

---

## Verdict (paste-ready)

**Problem-first:** Live next-snapshot forecasting of cross-exchange \(z_{t+1}\) with microstructure that is not recoverable from OHLCV is a sparsely studied setting; a simple confidence-gated tree regressor is already an effective baseline against mechanical \(|z_t|\) rules under live validation.

**Method-second (test set):** On the shared LOGO test matrix, RF and LightGBM learn nearly the same ranking (`corr ≈ 0.98`). LightGBM remains the reported model for better \(\tau{=}0.9\) **throughput / R²** and because it is the model under live validation. RF’s slightly higher own-gate DirAcc is a **selection artifact** of compressed `|pred|`, not a skill win on matched rows — so do not market “LGBM wins the tree bake-off” as a contribution.

**Why no RF on validation:** Test and validation are both held-out; LGBM filtered skill aligns across them; the forest peer on test is a sufficient tree-family control.

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
| RF fit | `n_estimators=400`, `max_depth=20`, `min_samples_leaf=200`, `max_features=sqrt`, `max_samples=1e6`, `n_jobs=8` on full LOGO train panel |

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

## 4. Why LightGBM is the reported model (research reasons)

Keep these as the paper rationale for preferring LightGBM over RF — **not** CPU/ops.

| Axis | Winner | Evidence |
|---|---|---|
| **Trade volume at paper τ=0.9 (test)** | **LGBM** | ~2× fire rate / n (67k vs 33k) at similar DirAcc |
| **Calibration / R² on gated & all rows** | **LGBM** | Higher R² own-gate and matched-row; all-rows R² 0.133 vs 0.126 |
| **Total proxy throughput (test)** | **LGBM** | `mean×n` at τ=0.9 ≈ 77k vs ≈ 43k |
| **Matched-row DirAcc** | Tie | Identical on LGBM’s τ=0.9 mask |
| **Validation / live evaluation** | **LGBM** | Only LGBM has the Aug 4–7 validation book and paper τ table |
| **What RF shows** | Shared signal | Bagging tracks boosting → lift is not “GBDT magic” |

**Paper claim hierarchy:** (1) setting + live microstructure + next-\(z\) formulation vs mechanical rules on **validation**; (2) test-set RF control → simple tree baseline is robust; (3) LGBM chosen for gate throughput / R² and as the validation model — with test RF sufficient because test≈validation LGBM skill.

---

## 5. Validation set (paper) — primary evidence (not RF)

Paper **validation** = Aug 4–7 ~72h live campaign. The paper’s τ sensitivity table is **validation-only**; headline at \(\tau{=}0.9\): \(n{=}12{,}795\), DirAcc 86.7%, R² 0.599, mean pnl \(+1.37\).

| Model | Set | Filter | n | DirAcc | R² | mean pnl |
|---|---|---|---:|---:|---:|---:|
| LightGBM (live book) | validation | `|ẑ|≥0.9` | 12,795 | 86.7% | 0.599 | +1.37 |
| LightGBM (LOGO test) | test | `|ẑ|≥0.9` | 66,513 | 85.2% | 0.525 | 1.154 |
| Random Forest | validation | — | — | — | — | *not scored* |

Mechanical baseline DirAcc / mean-PnL figures in Results are **validation-panel** capacity-matched replays — the right peer for the contribution. Keep RF out of that story.

**Sufficiency argument:** validation DirAcc/R² under \(\tau{=}0.9\) sit close to the filtered test regime → OOS behavior is consistent; RF on test is an adequate tree-family control without a second live campaign.

---

## 6. What RF is good for (and what it is not)

**Good for**

- Showing the **problem + features + gate** carry signal under more than one tree inductive bias.
- A short Results / Ablation control so reviewers see a classical bagging peer.
- Teaching τ reporting hygiene (fire rate / \(n\) + matched rows when `|pred|` scales differ).

**Not good for**

- Carrying the paper’s novelty (“our method beats RF”).
- Substituting for stronger future peers (calibrated RF, HistGBM, linear, sequence models) if a reviewer demands a fuller bake-off — answer by restating setting-first contributions and releasing data for others to beat the baseline.
- Motivating model choice via CPU-live / ops arguments in the research narrative.

---

## 7. Where to integrate (aligned with current main + reviewer advice)

| Paper place | Insert |
|---|---|
| **Abstract / Intro contributions** | Lead with **cross-exchange live next-\(z\)** setting + selective gate + live validation vs mechanical peers. Mention LightGBM as the **simple tree baseline**, not the invention. |
| **§Methodology** | Trees as tabular inductive bias; RF bagging control scored on test; LGBM for validation. |
| **§Experimental Setup** | RF under Baseline Definitions as classical bagging control on **test**; note test↔validation similarity → no RF live campaign. |
| **§Results — Test** | Compact LGBM vs RF vs naive table with **n**; favor LGBM on throughput / R²; sufficiency sentence for skipping RF validation. |
| **§Ablation** | Prefer **boosting vs bagging** as a short robustness note, not a methods shootout. Do **not** put RF into the validation-only τ table. |
| **§Discussion** | If asked “why not more models?”: contribution is the setting + protocol + released panel; tree baseline is intentionally simple; community can extend on the dataset release. |
| **Do not** | Lead Results with RF; inflate “LGBM ≫ RF”; use offline/Campaign/Jul 31 labels; cite CPU-live as research motivation. |

**Narrative ladder**

```text
Understudied setting (cross-ex live L2 next-z)
  → mechanical |z_t| fails / underuses structure     (validation, primary)
  → simple gated tree baseline works                 (LGBM live)
  → RF on test ≈ same ranking                        (robustness, secondary)
  → LGBM preferred: ~2× gate n, higher R²            (research, not ops)
  → release data/code so others beat this baseline
```

---

## 8. Caveats (Methods / footnote)

1. LOGO cache is **62 features**; production booster lists **68** (six Coinbase volume lags missing — NaN-filled for LGBM). Same frame for both.
2. RF used `max_samples=1e6` bootstrap draws; full-panel `X` still held for fit.
3. **Validation** headline is live LGBM only; RF not re-traded live (by design — see sufficiency argument).
4. Own-gate RF τ=0.9 mean pnl can look high; always pair with \(n\) and matched-row tables (§2).
5. Paper train-size text (~2.9M) may differ from LOGO cache (~4.95M) — cite LOGO when reporting these RF numbers.
6. A single RF peer will not satisfy every reviewer; treat it as **robustness**, and keep the contribution **setting-first**.

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

## Paper one-liners

**Contribution (preferred):**

> We study next-snapshot forecasting of same-asset cross-exchange z-scores under live multi-venue microstructure—a setting poorly covered by mechanical \(|z_t|\) pairs rules—and show that a simple confidence-gated *tabular tree* baseline (LightGBM; Random Forest as a test-set bagging control), rather than the DNN/LSTM architectures common in prior spread work, already improves selective direction and z-unit settlement versus matched persistence in live validation, with public code and data for stronger models to beat.

**RF footnote (secondary):**

> On the test matrix, a Random Forest tracks LightGBM’s ranking almost one-to-one; we report LightGBM for gate throughput and R² and evaluate it under live validation. Because filtered LightGBM skill is similar on the chronological test set and the live validation campaign—both out-of-sample relative to training—a separate Random Forest validation run is unnecessary for this tree-family control.
