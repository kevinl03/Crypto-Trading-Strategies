# Engineering Prompt: LSTM Z-Score Forecaster (CEX Cross-Exchange Spreads)

Execute this prompt **piecewise**. Complete one phase, stop for review, then continue only when asked (or when the parent plan says to finish all phases).

**Primary artifact:** [`statarb/cex_lstm_zscore.ipynb`](../../statarb/cex_lstm_zscore.ipynb)  
**Reference (loader / split only — do NOT copy LightGBM feature columns):** [`statarb/cex_gbm_new.ipynb`](../../statarb/cex_gbm_new.ipynb)  
**Export dir:** `statarb/outputs_lstm/`

---

## 1. Goal and non-goals

### Goal

Train a **PyTorch LSTM** that predicts the next rolling z-score of cross-exchange `spread_bps`:

\[
z_t = \frac{\mathrm{spread\_bps}_t - \mu_t}{\sigma_t},\quad
\mu_t,\sigma_t = \mathrm{rolling}(W{=}300,\ \mathrm{min\_periods}{=}90)
\]

\[
y_t = z_{t+H},\quad H = 1
\]

Evaluate on the **July 25–28 holdout** with the same trade filter used for LGBM campaigns:

- Enter when `|pred| > 0.5`
- Direction = `sign(pred)`
- Settlement proxy: `pnl_proxy = sign(pred) * z_{t+1}`

Fair comparison with LGBM is on **data + split + target + filter**, not feature-column identity.

### Non-goals (v1)

- Do **not** force the 68 LightGBM columns / lag-tabular recipe.
- Do **not** build a live paper trader (`paper_trade_lstm.py`) in this pass.
- Do **not** early-stop on the test set.
- Do **not** bridge sequences across `window_id` gaps.
- Do **not** load OHLCV for v1.

---

## 2. Data contract

### Source

| Key | Value |
|---|---|
| HF repo | `SFU-fintech-AI/statarb-crypto-research` |
| Config revision | `c5c695d3cec28db8801fe6de173b3c21f3803436` |
| Parquet revision (jul22–28) | `main` |
| Local-first roots | `C:/Users/Kev/repos/stochastic-spread-modeling/data/cex_unified`, `../data/cex_unified`, `../../stochastic-spread-modeling/data/cex_unified`, `./cex_unified` |

Reuse the GBM notebook’s local-or-HF loader pattern (`dataset_config` vs `parquet_dir`), `window_id` tagging, and error-row drop (`error IS NOT NULL`).

### Tables (v1)

| Table | Load | Parsed fields |
|---|---|---|
| `spread_matrix` | yes | expand `pairwise_spreads` → `exchange_a, exchange_b, spread_bps, p1, p2` |
| `ticker` | yes | `mid, spread_bps, bid_volume, ask_volume` |
| `orderbook` | yes | `imbalance` |
| `trades` | yes | `buy_sell_ratio, total_volume` |
| `funding_rate` | optional | `funding_rate` — include only if coverage ok after Phase 2 diagnostics |
| `open_interest` | optional | `oi_amount` — same |
| `ohlcv` | **no** | — |

### Universe

- Venues: `binance, bybit, okx, coinbase, kraken, mexc`
- Coins: volatile set used by collector (`scripts/data.py` `VOLATILE_COINS`)
- Grain after expand: `(window_id, snapshot_idx, coin, pair)` where `pair = exchange_a + "__" + exchange_b`

### `WINDOWS` (must match GBM Jul-25 protocol)

Train (`role=train`):

1. `jun13` — root / HF configs without prefix  
2. `jun22` — `test/` / `test_*`  
3. `jul13` — `validation/` / `validation_*`  
4. `jul19_pre` — `validation_jul19-22/pre_outage` / `jul19_22_pre_outage_*`  
5. `jul19_post` — `validation_jul19-22/post_outage` / `jul19_22_post_outage_*`  
6. `jul22_24` — `validation_jul22-28`, `load_mode=parquet_dir`, **`snapshot_idx_max=3584`** (`snapshot_idx < 3584`)

Test (`role=test`):

7. `jul25_28` — same `validation_jul22-28`, **`snapshot_idx_min=3584`** (`snapshot_idx >= 3584`)

`snapshot_idx` is **local to each window**; never compare across windows.

### Val carve (improvement vs GBM notebook)

From the **pooled train** panel, carve chronological validation for early stopping:

- Prefer: last contiguous train window slice, or last ~15–20% of train samples ordered by `(window_id order, snapshot_idx)`.
- Fit scalers **only** on remaining train.
- Never use test for early stopping or scaler fit.

---

## 3. Target and trading policy

| Constant | Value |
|---|---|
| `HORIZON` | `1` |
| `ZSCORE_WINDOW` | `300` |
| `MIN_PERIODS` | `90` |
| `ENTRY_TAU` | `0.5` (strict `>` for filter metrics to match research notebook; `>=` ok if documented) |
| Target | `groupby([window_id, coin, pair]).zscore.shift(-HORIZON)` |

Trading metrics (offline proxy, no fees in pnl):

- `trade_mask = abs(pred) > ENTRY_TAU`
- `direction = sign(pred)` (map 0 → skip)
- `pnl_proxy = direction * y_true` on traded rows
- Report DirAcc and mean `pnl_proxy` on traded rows

Naive baseline (matched rows):

- `pred_naive = z_t` (current zscore at decision time) as forecast of `z_{t+1}`
- Same `|pred_naive| > 0.5` filter for filtered comparison

---

## 4. Literature constraints / defaults

Adapt, do not clone markets:

| Paper | Steal | Avoid |
|---|---|---|
| Tsoku & Makatjane (2026) | MSE regression of standardized spread/z; dual forecast+trading metrics; Adam | Daily Yahoo, Johansen basket, 10/90 percentile signals |
| Han & Li (2024) | PyTorch LSTM stack; abstention analogy for `|pred|>0.5` | 3-class trend-of-z head |
| Sheng & Ma (2022) / mis-cited “Shen” in notes | 2 LSTM layers, Adam ~1e-3, sequence of spread-state metric, dual R²+trading table | CSI300 RP target / equity R²~0.92 claims |

Locked architecture defaults:

```text
ZScoreLSTM:
  inputs: X (B, T, C), coin_id (B,), pair_id (B,)
  coin_emb = Embedding(n_coins, 8)
  pair_emb = Embedding(n_pairs, 8)
  h = LSTM(input_size=C, hidden=128, num_layers=2, dropout=0.2, batch_first=True)
  out = Linear(hidden + 8 + 8, 1)   # last timestep + embeddings
  loss = MSE
  opt = Adam(lr=1e-3, weight_decay=1e-5)
  SEQ_LEN = 64
  batch_size = 256
  max_epochs = 50
  patience = 8 on val RMSE
  seed = 42
```

---

## 5. Feature channel spec (LSTM-native)

Build one supervised example per decision time \(t\) for each `(window_id, coin, pair)`:

- History window: snapshots `[t-SEQ_LEN+1, …, t]` **within the same `window_id`**
- Require: valid `zscore` at \(t\), valid `target` at \(t\), and full `SEQ_LEN` history after warmup

### Channel groups (v1)

Order channels stably; persist names in `feature_schema.json`.

1. **Spread state (2):** `spread_bps`, `zscore`
2. **Leg A ticker (4):** `mid_a`, `ba_bps_a`, `bid_vol_a`, `ask_vol_a`
3. **Leg B ticker (4):** `mid_b`, `ba_bps_b`, `bid_vol_b`, `ask_vol_b`
4. **Leg A/B orderbook (2):** `imb_a`, `imb_b`
5. **Leg A/B trades (4):** `bsr_a`, `vol_a`, `bsr_b`, `vol_b`
6. **Light cross-venue (3, optional if cheap):** `cross_mid_std`, `cross_ba_std`, `net_ob_pressure`  
   Computed from available venues at snap \(t\) for that coin; same values repeated across the sequence step (no future data).

**Transforms (fit on train only):**

- `log1p` on volume channels (`bid_vol_*`, `ask_vol_*`, `vol_*`)
- Optional winsorize 1%/99% on `spread_bps`, `ba_bps_*`
- `StandardScaler` per channel on flattened train sequence values (or on decision-time rows — pick one, document it)
- Relative mid: prefer `log(mid_a) - log(mid_b)` as an extra channel **or** replace raw mids if scale dominates; if used, name it `log_mid_diff` and drop raw mids **or** keep both and document

**v1.1 optional:** append `funding_a/b`, `oi_a/b` only if Phase 2 null rate is acceptable (e.g. <40% on train decision rows).

**Exclude:** LGBM lag columns, funding/OI if sparse, OHLCV, any feature that uses \(t+1\) or later.

Tensor shapes:

```text
X:        float32 (N, SEQ_LEN, C)
y:        float32 (N,)
coin_id:  int64   (N,)
pair_id:  int64   (N,)
z_now:    float32 (N,)   # for naive baseline
meta:     window_id, snapshot_idx, coin, pair (for debugging / exports)
```

---

## 6. Model / train hyperparameters

| Hyperparam | Default |
|---|---|
| `SEQ_LEN` | 64 |
| `hidden_size` | 128 |
| `num_layers` | 2 |
| `dropout` | 0.2 |
| `emb_dim` | 8 |
| `lr` | 1e-3 |
| `weight_decay` | 1e-5 |
| `batch_size` | 256 |
| `max_epochs` | 50 |
| `patience` | 8 |
| `num_workers` | 0 on Windows |
| Device | CUDA if available else CPU |

Training loop requirements:

- Shuffle train batches; do **not** shuffle val/test
- Track train/val MSE + RMSE each epoch
- Save best checkpoint by **val RMSE**
- After train: reload best weights before test eval

---

## 7. Eval suite

**Primary reporting slice = filtered** (`|pred| > 0.5`), consistent with LGBM campaign tables:

| Metric | Definition |
|---|---|
| **DirAcc** | `mean(sign(pred) == sign(y))` on filtered rows (exclude exact zeros) |
| **R²** | `sklearn.r2_score(y, pred)` on filtered rows |
| **mean pnl_proxy** | `mean(sign(pred) * z_{t+1})` on filtered rows |
| **Sharpe (per-trade)** | `mean(pnl_proxy) / std(pnl_proxy)` on filtered trades, Rf=0 |
| **Sharpe A (closed hourly)** | sum filtered `pnl_proxy` by `(window_id, snapshot_idx//33)`, then mean/std (offline closed-only; **not** live Sharpe B with open MTM) |

Also report the same four metrics for naive persistence (`z_t → z_{t+1}`, filter `|z_t| > 0.5`) on matched rows. Keep full-set MAE/RMSE/R²/DirAcc as secondary diagnostics.

Write:

- Notebook printed **headline filtered** table first
- `statarb/outputs_lstm/metrics.json` (includes `headline_filtered`)
- `statarb/outputs_lstm/METRICS.md` leading with DirAcc / R² / mean pnl_proxy / Sharpe

Compare narratively to LGBM offline/live numbers if available; do not require beating LGBM in v1.

---

## 8. Notebook section map

Create [`statarb/cex_lstm_zscore.ipynb`](../../statarb/cex_lstm_zscore.ipynb) with:

| § | Title | Phase |
|---|---|---|
| 0 | Title / overview | 1 |
| 1 | Imports & Config | 1 |
| 2 | Data Loading (`WINDOWS`, loader, train/test pools) | 1 |
| 3 | Spread backbone + z + target | 2 |
| 4–6 | Ticker / OB / Trades parse + leg join | 2 |
| 7 | Optional FR/OI coverage check | 2 |
| 8 | Panel panel + sequence builder | 2 |
| 8.5 | Diagnostics; finalize channels | 2 |
| 9 | Train/val/test tensors + scalers | 3 |
| 10 | `ZScoreLSTM` + train loop | 3 |
| 11 | Evaluate (+ naive + `|pred|>0.5`) | 4 |
| 12 | Export `.pt` + `feature_schema.json` | 5 |
| 13 | Write `METRICS.md` | 5 |

---

## 9. Acceptance criteria

- [ ] Notebook loads HF/local data with Jul25 `3584` cut
- [ ] Sequences never cross `window_id`
- [ ] Target is \(z_{t+1}\) with W=300, min_periods=90
- [ ] LSTM trains with early stop on **val**
- [ ] Test metrics reported for all + `|pred|>0.5`
- [ ] Naive persistence baseline on matched rows
- [ ] Artifacts in `statarb/outputs_lstm/`: `statarb_lstm.pt`, `feature_schema.json`, `metrics.json`, `METRICS.md`
- [ ] Channel list documented; no claim of LGBM column parity

---

## 10. Piecewise execution rule

| Phase | Stop when |
|---|---|
| **0** | This prompt file exists and is complete |
| **1** | Notebook §§0–2 run; train/test table counts printed |
| **2** | Sequences built; shapes + target stats printed |
| **3** | Best val checkpoint saved; loss curves shown |
| **4** | Test metric tables printed + `metrics.json` |
| **5** | Full export + `METRICS.md` |

If implementing in one session end-to-end, still mark phases done in order and keep section checkpoints as print assertions in the notebook.

---

## Implementation notes for agents

1. Prefer copying loader code from `cex_gbm_new.ipynb` §2 verbatim (tables, cuts, payload parse), then diverge at feature construction.
2. Payload parsing via `itertuples` is slow but matches the existing pipeline; keep it unless a vectorized parser already exists in-repo.
3. On Windows, keep `DataLoader(num_workers=0)`.
4. Memory: build sequences window-by-window or coin-by-coin if pooling explodes RAM; float32 everywhere.
5. Seed everything: `random`, `numpy`, `torch`, and `torch.cuda` if present.
6. Do not edit the plan file; do not commit unless asked.
