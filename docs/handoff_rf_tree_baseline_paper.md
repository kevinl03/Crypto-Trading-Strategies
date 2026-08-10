# Handoff: RF tree baseline + paper LaTeX (merge-ready)

**Branch:** `feat/rf-zscore-baseline`  
**Purpose:** Guide a new chat (or merge review) on (1) how Random Forest results were produced, (2) how the paper LaTeX/PDF was built and framed, (3) what to keep vs drop at merge.  
**Related evidence doc:** [`docs/lgbm_vs_rf_justification.md`](lgbm_vs_rf_justification.md)  
**PDF checkpoint:** [`paper/Baseline-gradient-boosting-cross-market-spread-prediction.pdf`](../paper/Baseline-gradient-boosting-cross-market-spread-prediction.pdf)  
**Older campaign framing (not RF-specific):** [`docs/handoff_paper_campaign_framing.md`](handoff_paper_campaign_framing.md)

---

## What this branch delivered

1. **Random Forest z-score baseline** on the same LOGO train/test feature matrix as LightGBM.
2. **Paper framing:** setting-first, tree-based ML as a simple baseline (not DNN/LSTM bake-off); RF as a classical control on the **test** set; LightGBM = reported / validation model.
3. **Results table** `tab:rf_test` + prose preferring LightGBM (gate \(n\), R², boosting vs leaf-averaging tails).
4. **Abstract / conclusion** polish: next-step gate \(|\hat{z}_{t+1}|\geq 0.9\), plain results (DirAcc, R², Sharpe), GitHub vs Hugging Face release split.

**Not in this branch as paper content:** LSTM offline peer (artifacts exist under `statarb/outputs_lstm_size_matched/`; optional later; see decision note below).

---

## Split vocabulary (must use)

| Paper name | Meaning |
|---|---|
| **Training** | Before Jul 25 cut (June → mid-July 2026) |
| **Test** | Jul 25–28 chronological holdout (CEX API–collected) |
| **Validation** | Aug 4–7 ~72h live paper campaign |

Do **not** say offline / Campaign A–B / Jul 31 for RF. RF is scored on **test** only. Live mechanical baselines stay on **validation**.

---

## Reviewer framing (authoritative)

Emphasize **problem setting** (same-asset cross-exchange, live L2 microstructure, next-\(z\) vs mechanical \(|z_t|\)).  
Deemphasize “we invent a better booster.”  
Contribution = **simple but effective tree baseline** for an understudied setting + live eval + data/code release.  
RF shows ranking skill is not unique to LightGBM; prefer LightGBM for more trades at absolute \(\tau{=}0.9\) and higher R² (leaf averaging compresses RF tails; boosting keeps more mass in tails).

**Do not** motivate trees/LGBM with CPU-live / ops in the research narrative.  
**Do not** digress into “why we skipped RF on validation.” Just say RF was compared on the same OOS test set as LightGBM.

---

## Running RF results

### Prerequisites

- Python venv with sklearn, lightgbm, joblib, pandas, etc.
- LOGO feature cache (preferred):

```text
statarb/outputs_logo/cache/df_train_logo.parquet
statarb/outputs_logo/cache/df_test_logo.parquet
```

- Production LightGBM booster for side-by-side scoring:

```text
statarb/outputs/statarb_lgbm.txt
```

### Commands

```powershell
cd C:\Users\Kev\repos\stochastic-spread-modeling-analysis

# Smoke (plumbing)
.\.venv\Scripts\python.exe statarb\run_rf_zscore_baseline.py --smoke --out-dir statarb\outputs_rf_smoke

# Full fit used for paper numbers (~5 min CPU; n_jobs=8, max_samples=1e6)
.\.venv\Scripts\python.exe statarb\run_rf_zscore_baseline.py --out-dir statarb\outputs_rf
```

Script: [`statarb/run_rf_zscore_baseline.py`](../statarb/run_rf_zscore_baseline.py)

### Outputs (paper-facing)

| Path | Role | Git |
|---|---|---|
| `statarb/outputs_rf/METRICS.md` | Human table for `tab:rf_test` | tracked |
| `statarb/outputs_rf/metrics_test.json` | Machine-readable | tracked |
| `statarb/outputs_rf/encoder.joblib` | OrdinalEncoder + medians | tracked |
| `statarb/outputs_rf/rf_model.joblib` | Fitted RF (~180 MB) | **gitignored** |

### Headline numbers in the paper (`tab:rf_test`, LOGO 62-feat re-score)

| Model | Filter | n | DirAcc | R² | Mean pnl |
|---|---|---:|---:|---:|---:|
| LightGBM | all | 1,680,426 | 62.8% | **0.133** | 0.269 |
| Random Forest | all | 1,680,426 | 62.9% | 0.126 | 0.267 |
| LightGBM | \(\|\hat{z}\|\geq 0.9\) | **66,513** | 85.2% | **0.525** | 1.154 |
| Random Forest | \(\|\hat{z}\|\geq 0.9\) | 33,038 | 86.0% | 0.478 | 1.299 |

**Caveat:** Production LGBM `tab:test` uses 68 features → \(\tau{=}0.9\) is 85.3% / 0.535. Footnote in Results explains the LOGO 62-col gap.

**Matched-row nuance:** On LightGBM’s \(|\hat{z}|\geq 0.9\) mask, DirAcc ties at 85.2%; LGBM wins R² (0.525 vs 0.489). Prefer LGBM for ~2× fire rate + R² + total `mean×n`.

### Primary live claim (unchanged; not RF)

Validation \(\tau{=}0.9\): \(n{=}12{,}795\), DirAcc 86.7%, R² 0.599, mean pnl +1.37, per-trade Sharpe 1.08; hourly portfolio Sharpe 1.90; +15.4 pp DirAcc vs capacity-matched mechanical persistence.

---

## Building the LaTeX / PDF

### Tectonic location (Windows)

Often **not** on PATH. Use:

```powershell
C:\Users\Kev\AppData\Local\tectonic\tectonic.exe --version
# Tectonic 0.17.0
```

Session-only PATH:

```powershell
$env:Path += ";C:\Users\Kev\AppData\Local\tectonic"
```

### Build + Baseline checkpoint

```powershell
cd C:\Users\Kev\repos\stochastic-spread-modeling-analysis\paper
& "C:\Users\Kev\AppData\Local\tectonic\tectonic.exe" gradient-boosting-cross-market-spread-prediction.tex
Copy-Item -Force gradient-boosting-cross-market-spread-prediction.pdf Baseline-gradient-boosting-cross-market-spread-prediction.pdf
```

Main entry: `paper/gradient-boosting-cross-market-spread-prediction.tex`  
Sections: `paper/sections/*.tex`  
Bib: `paper/references.bib` (includes `breiman2001random`, DNN/LSTM peer keys)

### Macros for release URLs

Defined in the main `.tex`:

- `\coderef` → GitHub `kevinl03/stochastic-spread-modeling`
- `\datasetref` → Hugging Face `SFU-fintech-AI/statarb-crypto-research`

Anonymous/review mode may withhold URLs (see `SUBMISSION_GUIDE.md`).

---

## Where RF / tree framing lives in LaTeX

| File | What changed |
|---|---|
| `paper/sections/abstract.tex` | Setting → tree-based ML vs DNN/LSTM (cited) → RF control on same OOS test → \(\|\hat{z}_{t+1}\|\geq 0.9\) → DirAcc/R²/Sharpe → mechanical +15.4 pp → RF ranking / LGBM preferred → GitHub + HF release |
| `paper/sections/introduction.tex` | Tree baseline + RF as classical control baseline on test |
| `paper/sections/related_work.tex` | Tree ensembles vs DNN/LSTM; RF control; no CPU-deployable claim |
| `paper/sections/methodology.tex` | Why trees for tabular panel; RF fit; LGBM for validation |
| `paper/sections/experimental_setup.tex` | RF subsection: same OOS test set as LGBM |
| `paper/sections/results.tex` | `tab:rf_test` + leaf-averaging / boosting-tails sentence + prefer LGBM |
| `paper/sections/conclusion.tex` | RF on same test set; LGBM reported for R² / ~2× trades |

Style notes from late edits:

- No em dashes in abstract.
- Prefer “Random Forest baseline as a control,” not “bagging control.”
- Avoid “panel” jargon in abstract; “intentional” / “simple tree-ensemble” dropped in favor of **tree-based ML model**.
- Do not cite figures in the abstract.

---

## Optional LSTM (not merged into paper yet)

Artifacts: `statarb/outputs_lstm_size_matched/`, docs `lgbm_vs_lstm_pros_cons.md`, `results_lstm_lgbm_consolidated.md`.

Would help “insufficient baselines” if added as a **secondary offline peer**, but protocol differs (\(\tau{=}0.5\), subsampled test panel, different features, no live LSTM). Decision deferred: current merge can ship RF + mechanical only.

---

## Merge checklist

- [ ] Commit any dirty tree (often `abstract.tex` + `Baseline-…pdf` after last polish).
- [ ] Confirm `rf_model.joblib` stays gitignored.
- [ ] Skim abstract for GitHub vs Hugging Face wording.
- [ ] Rebuild Baseline PDF once more after final abstract touch.
- [ ] Merge `feat/rf-zscore-baseline` → `main` (or open PR).
- [ ] After merge: update `paper/versions/README.md` checkpoint blurb if needed.

### Suggested PR summary bullets

- Add RF z-score baseline runner + test-set metrics vs LightGBM / naive.
- Frame paper as setting-first tree baseline; RF control on same OOS test set; prefer LGBM for gate throughput / R².
- Abstract/results polish: next-step confidence filter, plain DirAcc/R²/Sharpe, release split (GitHub code, HF data).

---

## Chat / transcript pointer

Primary conversation for this work: agent transcript `62ca8567-3d31-4676-a5b0-ed6c0b78c268` (RF vs LGBM baseline / paper framing).
