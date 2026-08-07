# Results: LSTM vs LightGBM \(z_{t+1}\) Forecast

**Date:** 2026-08-07  
**Branch:** `feat/lstm-zscore-baseline`  
**LSTM (size-matched):** [`statarb/outputs_lstm_size_matched/`](../statarb/outputs_lstm_size_matched/)  
**LGBM same-window offline:** [`statarb/outputs_lgbm_offline_jul25/`](../statarb/outputs_lgbm_offline_jul25/) (`score_lgbm_offline_pnl_sharpe.py`)  
**LGBM booster:** [`statarb/outputs/statarb_lgbm.txt`](../statarb/outputs/statarb_lgbm.txt)  
**LGBM live peer:** Jul 31 campaign ([`docs/baseline_lgbm_vs_mechanical_z.md`](baseline_lgbm_vs_mechanical_z.md))  
**Pros/cons (paper):** [`docs/lgbm_vs_lstm_pros_cons.md`](lgbm_vs_lstm_pros_cons.md)  
**Ops / cost note:** [`docs/lstm_vs_lgbm_resource_practicality.md`](lstm_vs_lgbm_resource_practicality.md)

---

## 1. Claim (careful)

On the **same Jul 25–28 calendar holdout** (\(y=z_{t+1}\); W=300 / min_periods=90; `|pred|>0.5`), a **size-matched LSTM** (`hidden=160` ≈ LGBM 1.24 MB) reports slightly higher filtered DirAcc / R² / mean `pnl_proxy` than LightGBM scored on the full holdout — but panels are **not matched-row** (LSTM stride/subsample vs LGBM full valid-target rows).

Use LSTM as a **literature-aligned research baseline**. Prefer LightGBM for **live** practicality. Do **not** claim “LSTM beats LGBM in production” from these tables alone.

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

### 3.1 Same-window Jul 25–28 offline (primary)

| Model | Slice | n | DirAcc | R² | mean `pnl_proxy` | Sharpe (per-trade) | Sharpe A |
|---|---|---:|---:|---:|---:|---:|---:|
| **LGBM** | filtered | 263,464 | 78.7% | 0.389 | +0.765 | 0.73 | 3.90 |
| LGBM | all | 1,680,081 | 63.1% | 0.132 | +0.268 | 0.26 | 6.33 |
| Naive `|z_t|>0.5` (full panel) | filtered | 1,043,242 | 68.3% | −0.395 | +0.428 | 0.41 | 5.72 |
| **LSTM size-matched** | filtered | 37,601 | **81.6%** | **0.470** | **+0.816** | **0.83** | 3.93 |
| LSTM size-matched | all (subsample) | 150,000 | 67.4% | 0.229 | +0.363 | 0.38 | 4.77 |

Sources: `statarb/outputs_lgbm_offline_jul25/metrics.json`, `statarb/outputs_lstm_size_matched/metrics.json`. Same metric code (`evaluate_model_and_naive` / equivalent). LSTM test panel is stride/subsampled; LGBM is the full valid-target holdout.

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

- Same calendar window: size-matched LSTM filtered DirAcc / R² / mean pnl (81.6% / 0.47 / +0.82) sit slightly above full-panel LGBM (78.7% / 0.39 / +0.77).
- Both beat mechanical-style `|z_t|>0.5` naive on filtered DirAcc/R²/pnl (LGBM clearly; LSTM on its own subsample).

**Why this is not a clean “LSTM wins” verdict**

1. **Different row sets.** LSTM scores 150k stride-subsampled sequences; LGBM scores the full valid-target panel (~1.68M).
2. **Different features.** Sequence microstructure vs 68-lag booster — skill is not isolated from representation.
3. **Filter selectivity.** LSTM fires on ~25% of its subsample; LGBM fires on ~16% of the full panel — densities differ.
4. **No live LSTM book.** LGBM has Jul31 Sharpe B + mechanical peers; offline Sharpe A ≠ live Sharpe B.
5. **Ops.** Live practicality still favors LGBM ([resource note](lstm_vs_lgbm_resource_practicality.md)).

**Practical takeaway**

| Axis | Winner today |
|---|---|
| Same-window offline filtered skill (imperfect panels) | LSTM slight |
| Live validated trading + mechanical peers | **LGBM** |
| Train / serve practicality | **LGBM** |
| Literature ablation (Tsoku-style learned z forecast) | LSTM |

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

### 5.2 LGBM — same-window offline scorer

1. **Command:** `statarb/score_lgbm_offline_pnl_sharpe.py` → `statarb/outputs_lgbm_offline_jul25/`.
2. **Model:** existing `statarb/outputs/statarb_lgbm.txt` (not retrained); features rebuilt for Jul25–28 with the notebook FE path.
3. **Live DirAcc / pnl / Sharpe B:** still from Jul31 docs (§3.2) as a protocol-family peer, not the same parquet rows.

### 5.3 What would make the comparison tighter (future)

1. Score **LGBM and LSTM on identical Jul25 test indices** (drop LSTM subsample, or subsample LGBM the same way).
2. Report **matched-row naive** for LSTM: \(z_t\) on rows where `|pred_LSTM|>0.5`.
3. Only then claim a strict head-to-head skill ranking; keep ops conclusion separate.

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
| `statarb/outputs_lgbm_offline_jul25/metrics.json` | Same-window LGBM offline |
| `statarb/outputs_lstm_size_matched/metrics.json` | Size-matched LSTM |
| `statarb/outputs_lstm_size_matched/model_size_report.json` | MB vs LGBM |
| `docs/lgbm_vs_lstm_pros_cons.md` | Paper pros/cons |
| `docs/baseline_lgbm_vs_mechanical_z.md` | LGBM vs mechanical live |
| `docs/lstm_vs_lgbm_resource_practicality.md` | Time/RAM/live practicality |
