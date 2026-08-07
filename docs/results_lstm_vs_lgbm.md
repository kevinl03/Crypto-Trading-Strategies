# Results: LSTM vs LightGBM \(z_{t+1}\) Forecast

**Date:** 2026-08-07  
**Branch:** `feat/lstm-zscore-baseline`  
**LSTM artifacts:** [`statarb/outputs_lstm/`](../statarb/outputs_lstm/) (`metrics.json`, `METRICS.md`, `feature_schema.json`, `statarb_lstm.pt`)  
**LGBM offline protocol:** [`statarb/cex_gbm_new.ipynb`](../statarb/cex_gbm_new.ipynb) → `statarb/outputs/statarb_lgbm.txt`  
**LGBM live peer (same protocol family):** Jul 31 campaign ([`docs/baseline_lgbm_vs_mechanical_z.md`](baseline_lgbm_vs_mechanical_z.md), [`docs/handoff_paper_campaign_framing.md`](handoff_paper_campaign_framing.md))  
**Ops / cost note:** [`docs/lstm_vs_lgbm_resource_practicality.md`](lstm_vs_lgbm_resource_practicality.md)

---

## 1. Claim (careful)

On the **same Jul-25 holdout protocol** (train &lt; 2026-07-25; test = Jul 25–28; \(y=z_{t+1}\); W=300 / min_periods=90; trade when `|pred|>0.5`), the LSTM research run reports **higher filtered DirAcc / R² / mean `pnl_proxy`** than the documented LightGBM offline test numbers — but this is **not** a matched-row, full-test, capacity-matched bakeoff.

Use LSTM as a **same-target research baseline**. Prefer LightGBM for **live** practicality (see resource note). Do **not** claim “LSTM beats LGBM in production” from these tables alone.

---

## 2. Shared experimental contract

| Knob | Value |
|---|---|
| Split | Train = jun13…jul22_24 (`snapshot_idx < 3584`); Test = jul25_28 (`≥ 3584`) |
| Target | \(y_t = z_{t+1}\) of cross-exchange `spread_bps` |
| Z definition | Rolling W=300, `min_periods=90`, group `(window_id, coin, pair)` |
| Entry filter | `|pred| > 0.5` (LSTM offline) / `|pred| ≥ 0.5` (LGBM live docs) |
| Direction / PnL | `sign(pred)`, `pnl_proxy = sign(pred) × z_{t+1}` |
| Universe | 23 volatile coins × 6 venues |

**Where they diverge (material for comparison):**

| | LSTM (this run) | LGBM (production stack) |
|---|---|---|
| Features | LSTM-native sequences (18 channels × `SEQ_LEN=64`) | 68 tabular lag / cross-exchange columns |
| Test coverage | Stride 4 → then **150k** chronological subsample | Full offline test panel (handoff `eval_results.csv`) |
| Train coverage | Stride 8 → **250k** subsample; early-stop on chronological val | Full pooled train; early-stop historically on test (notebook caveat) |
| Hardware | GPU train (~9 min end-to-end capped) | CPU train (minutes; no sequence tensor) |
| Live path | None yet | `experiments/paper_trade_lgbm.py` + campaigns |

---

## 3. Headline metrics (filtered `|pred|>0.5`)

Primary columns match campaign reporting: **DirAcc, R², mean `pnl_proxy`**, plus Sharpe where defined.

### 3.1 Offline holdout

| Model | Slice | n | DirAcc | R² | mean `pnl_proxy` | Sharpe (per-trade) | Sharpe A (closed hourly) |
|---|---|---:|---:|---:|---:|---:|---:|
| **LSTM** | filtered | 36,105 | **82.4%** | **0.492** | **+0.839** | **0.86** | 3.88 |
| LSTM | all (subsample) | 150,000 | 67.4% | 0.233 | +0.363 | 0.38 | 4.80 |
| LSTM naive \(z_t\) | `|z_t|>0.5` | 92,412 | 70.5% | −0.281 | +0.483 | 0.48 | 4.55 |
| **LGBM offline** | filtered | — | **78.4%** | **0.383** | — | — | — |
| LGBM offline | all | — | 62.8% | 0.133 | — | — | — |

Sources: LSTM → `statarb/outputs_lstm/metrics.json`. LGBM offline → handoff `eval_results.csv` summary in [`docs/handoff_paper_campaign_framing.md`](handoff_paper_campaign_framing.md) § offline model (R²/DirAcc only; no offline mean-pnl table in that summary).

### 3.2 LGBM live peer (same protocol family; different evaluation surface)

Jul 31 live paper session (not Jul 25 parquet holdout):

| Model | Slice | n | DirAcc | R² | mean `pnl_proxy` | Hourly Sharpe **B** |
|---|---|---:|---:|---:|---:|---:|
| LGBM live | `|pred|≥0.5` settled | 7,973 | **76.9%** | **0.347** (on 8,775 scored entries) | **+0.746** | **2.41** |
| Naive on LGBM entry rows | same rows | 8,775 | 77.0% | 0.173 | — | — |
| Mechanical persistence | `|z|≥0.5`, `max_open=50` | 8,550 | 69.3% | — | +0.437 | 2.63 |

Sources: [`docs/baseline_lgbm_vs_mechanical_z.md`](baseline_lgbm_vs_mechanical_z.md), [`docs/results_jul31_live_metrics_lit.md`](results_jul31_live_metrics_lit.md).

---

## 4. Interpretation

**What supports LSTM looking strong offline**

- On its capped Jul25 test subsample, filtered LSTM beats its own `|z_t|>0.5` naive on DirAcc (+11.9 pp), R² (0.49 vs −0.28), and mean `pnl_proxy` (+0.84 vs +0.48).
- Filtered R² / DirAcc also sit above the **documented LGBM offline filtered** numbers (0.49 / 82.4% vs 0.38 / 78.4%).

**Why this is not a clean “LSTM wins” verdict**

1. **Different row sets.** LSTM scores 150k stride-subsampled sequences; LGBM offline scores the full test panel. Selection differs.
2. **Different features.** Sequence microstructure vs 68-lag booster — skill is not isolated from representation.
3. **Filter selectivity.** LSTM fires on 36k / 150k ≈ 24% of scored rows; LGBM live fires on a different density. Higher filtered metrics can partly be **harder selection**, not pure skill.
4. **Naive construction differs.** LSTM naive uses `|z_t|>0.5` (mechanical-style entry). LGBM live naive is scored on **the same rows the model entered** (`|pred|≥0.5`). On Jul31, that matched-row naive DirAcc ≈ model DirAcc (77.0% vs 76.7%) — the honest DirAcc bar LGBM already cleared mainly on R²/pnl.
5. **No live LSTM book.** LGBM has capacity-matched Sharpe B and mechanical peers; LSTM Sharpe A here is offline closed-only pseudo-hourly.
6. **Ops.** Even if offline numbers favor future LSTM tuning, live practicality still favors LGBM ([resource note](lstm_vs_lgbm_resource_practicality.md)).

**Practical takeaway**

| Axis | Winner today |
|---|---|
| Offline filtered DirAcc / R² (imperfect compare) | LSTM run (with caveats above) |
| Live validated trading + mechanical peers | **LGBM** |
| Train / serve practicality | **LGBM** |
| Literature ablation (Tsoku-style learned z forecast) | LSTM has a seat at the table |

---

## 5. How these results were verified

Verification is layered: **reproducibility of the LSTM numbers**, **protocol parity checks**, and **cross-check against published LGBM artifacts** (not a second blind retrain of LGBM in this pass).

### 5.1 LSTM — regenerate / inspect

1. **Code path:** `statarb/run_lstm_zscore.py` → `lstm_zscore_lib.evaluate_model_and_naive`  
   - `pnl_proxy = sign(pred) * y` with `y = z_{t+1}`  
   - `DirAcc = mean(sign(pred)==sign(y))` excluding exact zeros  
   - `R² = sklearn.r2_score(y, pred)`  
   - Filtered mask: `abs(pred) > 0.5`
2. **Data cut:** `WINDOWS` + `snapshot_idx` 3584 split identical to `cex_gbm_new.ipynb`.
3. **Leakage guards:** sequences built only inside `window_id`; target via `shift(-1)` within group; scalers fit on train mask only; early-stop on chronological **val**, not test.
4. **Artifact check:** open `statarb/outputs_lstm/metrics.json` and confirm it matches the runner stdout headline block from the 2026-08-07 GPU run (~9 min, early-stop epoch 20, best val_rmse 0.91118).
5. **Sanity vs naive:** on the same scored subsample, naive persistence under `|z_t|>0.5` is weaker on DirAcc/R²/pnl — so LSTM is not accidentally scoring shuffled labels.
6. **Re-run command (deterministic seed 42):**

```bash
cd statarb
../.venv/Scripts/python.exe -u run_lstm_zscore.py --stride 8 --test-stride 4 --batch-size 512 --max-epochs 25
```

Expect filtered headline metrics within noise of `metrics.json` (GPU nondeterminism can cause small drift unless `cudnn.deterministic` is forced).

### 5.2 LGBM — verify against existing sources of truth

1. **Offline R²/DirAcc:** numbers in §3.1 are taken from the handoff summary of `eval_results.csv` for the Jul25-split booster (`statarb/outputs/statarb_lgbm.txt`), not re-inferred here.
2. **Live DirAcc / pnl / Sharpe B:** taken from Jul31 session JSON/docs (`summary.json`, `portfolio_sharpe_report.json`, `metrics_report.csv`) as compiled in the baseline / lit results docs.
3. **Protocol match:** Jul31 live model is the same H=1 / W=300 / 68-feat stack as the Jul25 offline notebook — appropriate as a **protocol-family** peer, not as the same parquet test rows as LSTM.

### 5.3 What would make the comparison tighter (future)

1. Score **LGBM and LSTM on identical Jul25 test indices** (no LSTM stride/subsample, or apply the same subsample to both).
2. Report **matched-row naive** for LSTM: evaluate \(z_t\) on the rows where `|pred_LSTM|>0.5` (same construction as LGBM live).
3. Add offline mean `pnl_proxy` + Sharpe A for LGBM on that shared panel.
4. Only then claim a head-to-head skill ranking; keep ops conclusion separate.

### 5.4 Red flags we already account for

| Risk | Mitigation / status |
|---|---|
| Filter inflation of DirAcc | Always show naive / mechanical peers; don’t lead with filtered DirAcc alone |
| Train–test leakage via early stop on test | LSTM uses val carve; older LGBM notebook warned about test early-stop |
| Subsample cherry-picking | Documented; chronological keep (train tail / test head) — still not full panel |
| Confusing Sharpe A vs live Sharpe B | Explicit labels in metrics notes |
| GPU process crash after export | Exit code noisy; artifacts on disk were complete (`statarb_lstm.pt`, metrics written) |

---

## 6. File index

| Path | Role |
|---|---|
| `statarb/outputs_lstm/metrics.json` | LSTM machine-readable results |
| `statarb/outputs_lstm/METRICS.md` | LSTM dual-metric writeup |
| `statarb/outputs_lstm/feature_schema.json` | Channels, scalers, maps |
| `docs/handoff_paper_campaign_framing.md` | LGBM offline + live tables |
| `docs/baseline_lgbm_vs_mechanical_z.md` | LGBM vs mechanical live |
| `docs/lstm_vs_lgbm_resource_practicality.md` | Time/RAM/live practicality |
