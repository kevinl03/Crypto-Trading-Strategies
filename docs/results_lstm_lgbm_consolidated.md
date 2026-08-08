# Consolidated Results: LightGBM vs LSTM (\(z_{t+1}\))

**Date:** 2026-08-07 · **Branch:** `feat/lstm-zscore-baseline`  
**Primary paper metrics:** DirAcc · R² · mean `pnl_proxy` (`sign(pred) × z_{t+1}`, z-units)  
**Protocol:** H=1, W=300 / min_periods=90, trade when `|pred| > 0.5` (live docs use `≥ 0.5`)

---

## What to use where

| Paper section | Use this |
|---|---|
| **Abstract / Intro claim** | Learned z-forecast + abstention beats mechanical `|z|` peers in live paper trading; size-matched LSTM is a competitive deep baseline offline |
| **Methods** | Shared task + filter; LGBM = production tabular; LSTM = literature-aligned sequence peer (size-matched ~1.24 MB) |
| **Results — main** | §1 Live LGBM vs mechanical · §2 Offline same-window LGBM vs LSTM |
| **Results — sensitivity** | §3 LGBM τ ablation (justify τ=0.5) |
| **Discussion** | §4 Verdict + paste-ready paragraph |
| **Appendix** | Pros/cons, ops, full τ CSV, bps diagnostic |

---

## Shared contract

| Knob | Value |
|---|---|
| Target | `y = z_{t+1}` of cross-exchange `spread_bps` |
| Split (offline) | Train &lt; Jul 25 · Test = Jul 25–28 (`snapshot_idx` cut 3584) |
| Filter | `|pred| > 0.5` |
| LGBM | 68 tabular lag / microstructure features · CPU booster |
| LSTM | 18 channels × `SEQ_LEN=64` · `hidden=160` · ~1.29 MB fp32 ≈ LGBM 1.24 MB |
| Caveat | Offline LSTM test panel is stride/subsampled (150k); LGBM uses full valid-target rows — same calendar window, not matched indices |

---

## 1. Live — LightGBM vs mechanical (primary trading evidence)

**Session:** Jul 31 paper campaign · `max_open=50` · settle `pnl_proxy = direction × z_{t+1}`  
**Source:** [`docs/baseline_lgbm_vs_mechanical_z.md`](baseline_lgbm_vs_mechanical_z.md)

| Strategy | n closed | DirAcc | mean pnl_proxy | Hourly Sharpe B |
|---|---:|---:|---:|---:|
| **LightGBM** `|pred|≥0.5` | 7,973 | **76.9%** | **+0.746** | 2.41 |
| Mechanical persistence `|z|≥0.5` | 8,550 | 69.3% | +0.437 | 2.63 |

Filtered R²: LGBM **0.35** vs matched-row naive **0.17**.  
**Lift vs persistence:** DirAcc **+7.6 pp** · mean pnl_proxy **~+71%**.

→ **Use as the main Results trading table.** Sharpe B is portfolio context, not the verdict.

---

## 2. Offline — same-window Jul 25–28 (architecture compare)

**Sources:** [`statarb/outputs_lgbm_offline_jul25/`](../statarb/outputs_lgbm_offline_jul25/) · [`statarb/outputs_lstm_size_matched/`](../statarb/outputs_lstm_size_matched/)

### 2.1 Filtered `|pred| > 0.5` (headline)

| Model | n | DirAcc | R² | mean pnl_proxy | Sharpe/trade | Sharpe A |
|---|---:|---:|---:|---:|---:|---:|
| **LGBM** (full holdout) | 263,464 | 78.7% | 0.389 | +0.765 | 0.73 | 3.90 |
| **LSTM** size-matched (subsample) | 37,601 | **81.6%** | **0.470** | **+0.816** | **0.83** | 3.93 |
| Naive `|z_t| > 0.5` (same 150k LSTM panel) | 92,412 | 70.5% | −0.281 | +0.483 | 0.48 | 4.55 |
| Naive `|z_t| > 0.5` (full LGBM panel) | 1,043,242 | 68.3% | −0.395 | +0.428 | 0.41 | 5.72 |

LSTM filtered `n` (37.6k) is smaller than naive on the same panel (92.4k) because filters differ: `|pred|>0.5` vs `|z_t|>0.5` — see [`statarb/outputs_lstm_size_matched/METRICS.md`](../statarb/outputs_lstm_size_matched/METRICS.md).

### 2.2 All rows (context)

| Model | n | DirAcc | R² | mean pnl_proxy |
|---|---:|---:|---:|---:|
| LGBM | 1,680,081 | 63.1% | 0.132 | +0.268 |
| LSTM size-matched | 150,000 | 67.4% | 0.229 | +0.363 |

**Takeaway:** On the shared holdout protocol, size-matched LSTM is **competitive / slightly ahead** on filtered z-metrics. Both beat `|z|` naive on DirAcc, R², and mean pnl_proxy. LightGBM remains the **deployed** model (live path, CPU, full-panel scoring).

---

## 3. LGBM filter sensitivity (justify τ = 0.5)

**Source:** [`docs/lgbm_pred_tau_filter_ablation.md`](lgbm_pred_tau_filter_ablation.md)  
**Volume metric:** `total pnl_proxy = mean_pnl_proxy × n` (no fire-rate penalty)

| τ | Role | n | DirAcc | R² | mean pnl_z | total pnl_z |
|---:|---|---:|---:|---:|---:|---:|
| 0.10 | Max total pnl | 1.18M | 66.5% | 0.176 | +0.361 | **+426k** |
| **0.50** | **Protocol** | 263k | **78.7%** | **0.389** | **+0.765** | +202k |
| 1.00 | High confidence | 43k | 86.4% | 0.563 | +1.271 | +54k |

Abstention works (higher τ → better per-trade skill). Total z-proxy mass peaks at low τ. Keep **τ=0.5** for live parity and clear lift vs `|z|>0.5` naive — not because it maximizes total pnl.

---

## 4. Verdict (paper framing)

| Axis | Winner | Note |
|---|---|---|
| Live trading vs mechanical peers | **LGBM** | Only model with live book + Sharpe B |
| Offline filtered z-skill (Jul25) | LSTM slight | Caveat: subsampled panel |
| Production / ops | **LGBM** | CPU, ~1.2 MB, no sequence RAM wall |
| Literature deep baseline | LSTM | Size-matched Tsoku-style peer |
| Filter τ choice | **0.5** | Sensitivity supports it |

**Do not claim:** LSTM beats LGBM in live production · offline Sharpe A = live Sharpe B · matched-row identity of offline panels.

### Paste-ready Discussion

> We evaluate learned next-step z-forecasts for cross-exchange spreads under a common protocol (`y = z_{t+1}`, trade when `|pred| > 0.5`). In live paper trading, LightGBM improves on capacity-matched mechanical `|z|`-threshold persistence on DirAcc, mean z-settled pnl_proxy, and filtered R². Offline on Jul 25–28, a size-matched LSTM is competitive with LightGBM on conditional z-metrics (with a thinner `|pred|>0.5` book than `|z|>0.5` naive on the same panel) and serves as a literature-aligned deep baseline, while LightGBM remains the practical deployment model. An abstention sweep shows that raising τ improves per-trade skill while reducing total pnl_proxy (`mean × n`); τ=0.5 is retained for live protocol continuity and its lift versus mechanical peers.

---

## 5. Artifact index

| Path | Role |
|---|---|
| [`docs/baseline_lgbm_vs_mechanical_z.md`](baseline_lgbm_vs_mechanical_z.md) | Live LGBM vs mechanical |
| [`docs/lgbm_pred_tau_filter_ablation.md`](lgbm_pred_tau_filter_ablation.md) | τ sensitivity |
| [`docs/lgbm_vs_lstm_pros_cons.md`](lgbm_vs_lstm_pros_cons.md) | Pros/cons for Discussion |
| [`docs/lstm_vs_lgbm_resource_practicality.md`](lstm_vs_lgbm_resource_practicality.md) | Ops / cost |
| `statarb/outputs_lgbm_offline_jul25/` | Same-window LGBM metrics |
| `statarb/outputs_lstm_size_matched/` | Size-matched LSTM metrics |
| `statarb/outputs_lgbm_tau_sweep_jul25/` | Full τ grid (csv/json) |
| `statarb/run_lstm_zscore.py` | Train/eval CLI |
| `statarb/sweep_lgbm_pred_tau.py` | τ sweep runner |
