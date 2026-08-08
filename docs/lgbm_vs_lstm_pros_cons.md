# LightGBM vs LSTM — Pros / Cons for the Research Paper

**Purpose:** Decision-ready comparison for Methods / Discussion when positioning our cross-exchange \(z_{t+1}\) forecaster against a deep sequence baseline.  
**Shared task:** predict rolling z of cross-exchange `spread_bps` one snapshot ahead; trade when `|pred| > 0.5`; `pnl_proxy = sign(pred) × z_{t+1}`.  
**Evidence anchors:** Jul 25–28 holdout (same calendar window); size-matched LSTM (`hidden=160` ≈ LGBM 1.24 MB); live LGBM Jul 31 campaign; ops note in [`lstm_vs_lgbm_resource_practicality.md`](lstm_vs_lgbm_resource_practicality.md).

---

## 0. Same-window snapshot (Jul 25–28, filtered `|pred|>0.5`)

| Model | n | DirAcc | R² | mean `pnl_proxy` | Sharpe / trade | Sharpe A |
|---|---:|---:|---:|---:|---:|---:|
| **LGBM** (full holdout) | 263,464 | 78.7% | 0.389 | +0.765 | 0.73 | 3.90 |
| **LSTM** size-matched (subsampled holdout) | 37,601 | 81.6% | 0.470 | +0.816 | 0.83 | 3.93 |

Sources: `statarb/outputs_lgbm_offline_jul25/`, `statarb/outputs_lstm_size_matched/`.  
Caveat: LSTM test panel was stride/subsampled (150k all-rows); LGBM used the full valid-target panel. Same window and metric code; not yet matched-row identical indices.

**Paper one-liner:** Under a shared Jul 25–28 protocol, a size-matched LSTM is competitive with (slightly above) LightGBM on filtered DirAcc / R² / mean pnl, while LightGBM remains far preferable for live deployment and literature-honest production claims.

---

## 1. Pros — LightGBM

| Pro | Why it matters in *this* paper |
|---|---|
| **Production-proven live path** | Already runs in `paper_trade_lgbm.py` with multi-hour campaigns, mechanical peers, and Sharpe B (Jul 31: DirAcc 76.9%, mean pnl +0.75, Sharpe B 2.41). |
| **Ops / systems fit** | CPU-only booster; small artifact (~1.2 MB); per-snap state is lags + rolling z, not 64-step tensors. Matches ~110 s collector cadence without GPU. |
| **Tabular microstructure inductive bias** | Cross-exchange lags, BA, imbalance, flow, momentum/accel map cleanly to tree splits — strong for heterogeneous, partially missing venue features. |
| **Fast experiment loop** | Retrain in minutes on CPU after parquet load; no sequence materialization (avoids multi‑GB 3D tensors). |
| **Interpretability for Methods** | Feature importance / gain tables; easy ablations (drop OB, FR/OI, etc.) already used in the campaign narrative. |
| **Filter + dual metrics already framed** | `|pred|≥0.5` as abstention aligns with Han & Li’s “ML as filter” story without needing an LSTM head. |
| **Honest baseline story** | Matched-row naive on live entries shows DirAcc ≈ persistence; edge argued via R² / pnl vs mechanical — a careful claim reviewers can audit. |
| **Native categoricals** | `coin` / `pair` handled inside LightGBM without separate embedding tables. |

---

## 2. Cons — LightGBM

| Con | Why it matters |
|---|---|
| **No explicit sequence model** | Relies on hand-rolled lags (`N_LAGS=3`) + engineered momentum; cannot freely learn long temporal filters beyond what features encode. |
| **Feature engineering burden** | Representation is the product of notebook FE; results are entangled with that recipe (harder to claim “architecture” wins). |
| **Early-stop / leakage hygiene** | Research notebook historically early-stopped on test (documented caveat); needs careful wording vs LSTM’s val carve. |
| **Live DirAcc vs naive is thin** | On Jul 31 filtered rows, DirAcc ≈ matched naive (~77%); absolute directional claim is weak without R²/pnl/mechanical peers. |
| **Less aligned with “deep pairs” lit headline** | Tsoku & Makatjane (2026) lead with DNN/LSTM spread forecasts; trees need an extra sentence (“GBDT remains SOTA-practical for tabular microstructure”). |

---

## 3. Pros — LSTM

| Pro | Why it matters in *this* paper |
|---|---|
| **Literature alignment** | Closest architectural peer to Tsoku & Makatjane (learned forecast of standardized spread/z); Sheng/Ma-style dual error+trading reporting; Han & Li for PyTorch LSTM + abstention analogy. |
| **Sequence-native representation** | Consumes multivariate history (`SEQ_LEN=64`) of spread state + pair-leg micro without forcing the 68-column lag table. |
| **Competitive offline filtered skill** | On Jul 25–28 (size-matched), filtered DirAcc / R² / mean pnl / Sharpe-per-trade sit slightly above LGBM’s full-holdout filtered numbers (table §0). |
| **Clear lift vs `|z_t|>0.5` naive (own panel)** | Large R² / DirAcc gap vs mechanical-style naive on the LSTM subsample — useful as an internal persistence control. |
| **Size can be matched to LGBM** | `hidden=160` ≈ 1.29 MB fp32 vs 1.24 MB booster — allows a capacity-controlled architecture comparison, not “bigger net wins.” |
| **Val-based early stopping** | Chronological val carve avoids stopping on the final holdout (cleaner than the older GBM notebook habit). |

---

## 4. Cons — LSTM

| Con | Why it matters |
|---|---|
| **Live impracticality (primary)** | Needs per-series history buffers, PyTorch runtime, optional GPU; no live paper trader yet. Retrain/prep dominated by sequence build / RAM ([resource note](lstm_vs_lgbm_resource_practicality.md)). |
| **Memory / data engineering cost** | Full dense tensors are multi‑GB; production runs required stride + caps (250k/150k). Cost is front-loaded in FE, not just “training epochs.” |
| **Evaluation not yet matched-row to LGBM** | Subsampled test indices ≠ full LGBM panel; cannot claim strict pairwise superiority until identical rows are scored. |
| **No live Sharpe B / mechanical campaign** | Offline Sharpe A ≠ live portfolio Sharpe B with open MTM and `max_open=50`. |
| **Opacity** | Harder to explain *which* micro channels drive trades; less natural feature-importance story than trees. |
| **Hyperparameter / nondeterminism surface** | GPU training, seq length, hidden size, stride interact; small metric drift across runs unless fully deterministic. |
| **Overclaim risk** | Higher filtered metrics can partly reflect **selection density** (LSTM fires on fewer rows in the subsample) — must show naive/mechanical controls. |
| **Task may not need long memory** | Decision grain is one snapshot ahead with strong lag-1 persistence; trees + expanded lags may capture most of the signal LSTMs are credited for. |

---

## 5. Axis-by-axis (for a paper table)

| Axis | Prefer | One-line reason |
|---|---|---|
| Live paper trading / systems | **LGBM** | Deployed, CPU, small state |
| Offline filtered proxy skill (current evidence) | **LSTM (slight)** | Higher DirAcc/R²/pnl on Jul25 protocol (caveat: subsample) |
| Capacity-controlled architecture compare | Tie / both | Size-matched LSTM ≈ LGBM MB |
| Literature “deep forecast of spread/z” | **LSTM** | Tsoku-style peer |
| Literature “ML filter on arb” | Either | `|pred|>0.5` plays Han-like role for both |
| Interpretability / ablations | **LGBM** | Importance + FE toggles |
| Retrain frequency / researcher loop | **LGBM** | No sequence RAM wall |
| Honest dual metrics (error + trading) | Both | Same suite: R², DirAcc, pnl, Sharpe |

---

## 6. Suggested Discussion framing (paste-ready)

> We compare gradient-boosted trees and a size-matched LSTM under a common cross-exchange \(z_{t+1}\) protocol (Jul 25 holdout, `|pred|>0.5`). The LSTM is competitive on offline filtered DirAcc, R², and mean pnl_proxy, consistent with deep sequence models used for cointegrated-spread forecasting (Tsoku & Makatjane, 2026). However, LightGBM remains the practical production model: it admits a CPU live paper trader, avoids multi‑GB sequence materialization, and already supports capacity-matched mechanical baselines and live Sharpe reporting. We therefore treat the LSTM as a **literature-aligned research baseline**, not as a replacement for the deployed tabular forecaster. Claims of superiority are limited by non-identical test-row sampling until a matched-index evaluation is reported.

---

## 7. What not to claim

- Do **not** claim LSTM “beats LGBM in live trading” — no LSTM live book.
- Do **not** equate offline Sharpe A with Jul 31 Sharpe B.
- Do **not** lead with filtered DirAcc alone without naive/mechanical peers (selection inflation).
- Do **not** imply equal test panels until stride/subsample is removed or mirrored on LGBM.

---

## 8. Related artifacts

| Artifact | Role |
|---|---|
| `statarb/outputs_lgbm_offline_jul25/` | Same-window LGBM pnl/Sharpe/R²/DirAcc |
| `statarb/outputs_lstm_size_matched/` | Size-matched LSTM + `model_size_report.json` |
| `docs/results_lstm_lgbm_consolidated.md` | Consolidated results entrypoint |
| `docs/lstm_vs_lgbm_resource_practicality.md` | Time/RAM/live practicality |
| `docs/baseline_lgbm_vs_mechanical_z.md` | Live LGBM vs mechanical |
| Tsoku & Makatjane (2026); Han & Li (2024); Sheng & Ma (2022) | Lit peers for LSTM / dual metrics / filter framing |
