# Handoff: Cut paper from ~9 pages → ICAIF 8 pages

**Branch when written:** `feat/rf-zscore-baseline` (merge into `main` for context refresh)  
**Companion handoff (RF / LaTeX / results):** [`docs/handoff_rf_tree_baseline_paper.md`](handoff_rf_tree_baseline_paper.md)  
**PDF checkpoint:** `paper/Baseline-gradient-boosting-cross-market-spread-prediction.pdf`  
**Target:** ACM ICAIF `sigconf`, **8 pages total** incl. figures + references  
**Goal of next chat:** execute lowest-risk cuts, rebuild, confirm page count ≤ 8

---

## Protect (do not cut for space)

| Keep | Why |
|---|---|
| Abstract framing | Setting + tree baseline + results + release split |
| Intro problem-setting / live L2 novelty | Reviewer “setting-first” ask |
| `tab:baselines` + +15.4 pp DirAcc claim | Primary live contribution |
| `tab:rf_test` + prefer-LGBM paragraph | Classical control; leaf-averaging / boosting tails |
| Short τ=0.9 / Sharpe justification | Abstract gate story (table can stay slim) |

---

## Lowest priority → cut first (Pass A, ~1 page)

| # | Action | ~Savings | Risk |
|---|---|---|---|
| 1 | **Delete fig7a + fig7b** (baseline DirAcc / mean-pnl bars). Keep `tab:baselines`. Optional: one combined `fig7_baseline_diracc_pnl.png` if a visual is still wanted. | 0.35–0.55 p | Lowest |
| 2 | **Delete `w_vs_r2_diracc` figure.** Keep 2–3 sentences on plateau / why \(w{=}300\). | 0.25–0.35 p | Lowest |
| 3 | **Delete feature-importance figure.** Keep a short sentence naming top features. | 0.20–0.30 p | Lowest |

**Do Pass A → rebuild → check page count before cutting prose.**

### Files likely touched in Pass A

- `paper/sections/results.tex` — remove `\includegraphics` for fig7a/fig7b (and captions/labels)
- `paper/sections/ablation.tex` — remove \(w\)-sensitivity and/or importance figures
- Rebuild Baseline PDF after cuts

---

## Pass B if still over (low–medium risk)

| # | Action | ~Savings |
|---|---|---|
| 4 | Collapse Methodology feature `\itemize` blocks into one short “feature families” paragraph | 0.25–0.40 p |
| 5 | Deduplicate mechanical baseline definitions (Results vs Experimental Setup) | 0.10–0.15 p |
| 6 | Trim Related Work method-recap paragraph; keep classical pairs + Fischer latency + DNN/LSTM contrast | 0.10–0.20 p |
| 7 | Shrink FE-vs-raw ablation to Δ DirAcc / Δ R² + short table or prose | 0.15–0.25 p |

---

## Pass C only if needed

| # | Action | Notes |
|---|---|---|
| 8 | Replace Discussion `tab:bps` with 1–2 sentence z≠bps caveat | Honesty kept; less page |
| 9 | Shrink pipeline figure width (~0.85\\columnwidth) | Prefer not deleting |
| 10 | Fold `tab:test` into prose / RF footnote | Medium risk |
| — | Do **not** cut Ethics just for space without checking venue norms | Small savings |

---

## Build / verify page count

```powershell
cd C:\Users\Kev\repos\stochastic-spread-modeling-analysis\paper
& "C:\Users\Kev\AppData\Local\tectonic\tectonic.exe" gradient-boosting-cross-market-spread-prediction.tex
Copy-Item -Force gradient-boosting-cross-market-spread-prediction.pdf Baseline-gradient-boosting-cross-market-spread-prediction.pdf
# Confirm page count in PDF viewer or:
# (Get-Content ... won't work easily); open PDF — target ≤ 8 pages
```

Tectonic binary (if not on PATH): `C:\Users\Kev\AppData\Local\tectonic\tectonic.exe`

---

## Framing reminders (while cutting)

- Setting-first; method = simple effective tree baseline
- RF = same OOS **test** set as LightGBM (control), not a bake-off
- No CPU-ops motivation; no “why no RF validation” digression
- No em dashes in polished abstract/conclusion prose
- Code on **GitHub** (`\coderef`); dataset on **Hugging Face** (`\datasetref`)
- LSTM offline peer exists but was **left out** of the paper (protocol mismatch); do not add during page cut unless asked

---

## Suggested next-chat prompt

> Read `docs/handoff_page_reduction_8pages.md` and `docs/handoff_rf_tree_baseline_paper.md`. Execute Pass A (drop fig7a/7b, w-sensitivity fig, feature-importance fig), rebuild Baseline PDF, report page count. If still >8 pages, continue with Pass B.
