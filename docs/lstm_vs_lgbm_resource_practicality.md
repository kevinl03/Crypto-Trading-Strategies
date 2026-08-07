# LSTM vs LightGBM — Training / Live Resource Practicality

**Purpose:** Compare wall-clock and memory cost of the LSTM z-score forecaster against the production LightGBM stack, framed for **offline retrain practicality** and **live paper/market execution**.  
**Branch artifacts:** `statarb/cex_lstm_zscore.ipynb`, `statarb/lstm_zscore_lib.py`, `statarb/run_lstm_zscore.py`  
**LGBM baseline:** `statarb/cex_gbm_new.ipynb` + `experiments/paper_trade_lgbm.py`

This note is about **ops cost**, not offline predictive skill. Skill comparison uses the same Jul-25 split, \(z_{t+1}\) target, and `|pred|>0.5` filter (see `statarb/outputs_lstm/METRICS.md` once training finishes).

---

## Bottom line (practical)

For this cross-exchange, ~1–2 minute snapshot grain:

| Question | Answer |
|---|---|
| Is LSTM practical as a drop-in live replacement for LGBM? | **No — materially worse ops profile** |
| Where does the cost hurt most? | Offline **sequence materialization / RAM**, optional **GPU**, and live **history buffer** of length `SEQ_LEN` |
| Where are they similar? | Same collector, same z warmup (`MIN_PERIODS=90` ≈ 2.5–3h), same settle H=1 |
| Sensible role for LSTM here | Research ablation / literature peer (Tsoku-style), **not** default production head |

LightGBM remains the practical production forecaster for live CEX paper trading in this repo.

---

## Measured offline costs (this repo)

### Protocol held constant

- Split: train &lt; Jul 25 / test Jul 25–28 (`snapshot_idx` cut 3584)
- Target: \(z_{t+1}\), `ZSCORE_WINDOW=300`, `MIN_PERIODS=90`
- Universe: 23 coins × 6 venues
- Data: local `cex_unified` / HF `SFU-fintech-AI/statarb-crypto-research`

### LightGBM (current production path)

| Resource | Typical / observed |
|---|---|
| Hardware | **CPU only** (no GPU required) |
| Feature grain | One **tabular row** per `(snapshot, coin, pair)` with fixed lags (`N_LAGS=3`) + cross-exchange block → **68** booster features |
| Train wall-clock | Notebook train on the same windows is typically **on the order of minutes** (boosting + early stop; no sequence tensor build) |
| Peak RAM | Dominated by **pooled parquet panels** + sparse lag matrix — large, but **no** `(N, T, C)` 3D tensor |
| Artifact size | Small text booster (`statarb/outputs/statarb_lgbm.txt`) |
| Retrain cadence | Feasible on a laptop/workstation CPU after a data pull |

### LSTM (v1 in this branch — measured 2026-08-07)

Hardware: RTX 4060 8 GB + CPU RAM. Config used for the timed GPU run:

- `SEQ_LEN=64`, `hidden=128`, `layers=2`, `batch=512`
- `train_stride=8`, `test_stride=4`
- Cap after build: **250k train / 150k test** sequences (full dense build is much larger)
- `max_epochs=25`, `patience=6`

| Stage | Observed | Notes |
|---|---|---|
| Phase 1 load (train+test tables) | **~6–8 min** | Same parquet parse path as GBM; payload `itertuples` bound |
| Phase 2 panel + sequence build | **~5–10+ min** | Built `(594k, 64, 18)` train / `(420k, 64, 18)` test **before** subsample |
| Phase 3 GPU epochs | **~20–40 s / epoch** (order-of-magnitude) | After subsample to 250k; early-stop often finishes in &lt;15–25 epochs |
| End-to-end (GPU, capped) | **~15–35 min** | Most time is **data prep**, not matmul |
| CPU-only (earlier attempt) | **~8+ min / epoch** at ~950k seqs | Impractical without aggressive subsample |
| Prior OOM | Scaler fit on full flatten ≈ **35 GB** float64 | Mitigated by chunked scaler + caps; dense stride-1 still risky |

**RAM shape (order of magnitude, float32):**

\[
\mathrm{bytes} \approx N \times \mathrm{SEQ\_LEN} \times C \times 4
\]

| Tensor | Approx size |
|---|---|
| Full train @ stride 1 (~4.7M × 64 × 18) | **~22 GB** (failed path / not used) |
| Train @ stride 8 before cap (~595k × 64 × 18) | **~2.7 GB** |
| After cap (250k × 64 × 18) | **~1.2 GB** |
| Test after cap (150k × 64 × 18) | **~0.7 GB** |

LGBM never materializes the 3D history tensor; that is the main offline RAM delta.

---

## Live execution practicality

Shared live constraints (both models):

- Collector cadence ≈ **110 s/snapshot** with `--slow-every 1` ([`docs/paper_trading_lgbm.md`](paper_trading_lgbm.md))
- Warmup: `snapshot_idx >= MIN_PERIODS (+ lags)` → **~2.5–3 h** before first valid z
- Filter: `|pred| > 0.5`, settle at `t+H`

### Per-snapshot inference

| | LightGBM (live today) | LSTM (would need) |
|---|---|---|
| Input | Current lag-tabular row (~68 feats) | Last **64** aligned snaps × ~18 channels **per** `(coin, pair)` |
| Predict API | `booster.predict` on CPU, ms-scale per batch of pairs | Forward LSTM on CPU or GPU; still need history assemble |
| State to keep | Rolling z buffers + last 3 lags (small) | Rolling z **plus** full `SEQ_LEN` multivariate ring buffer for every active series |
| Concurrent series | Up to ~23 × 15 ≈ **345** pair series | Same 345 series, each with a 64×C tensor → larger hot state |
| Dependency | CPU-only trader process already proven (`paper_trade_lgbm.py`) | Extra: PyTorch runtime; GPU optional but heavy for a paper trader |
| Failure mode | Missing column / cat mismatch | History gap / window_id reset / partial venue nulls in the sequence window |

At ~110 s between snapshots, **raw predict latency is not the bottleneck for either model**. The practical gap is:

1. **Memory & complexity of live feature state** (64-step tensors vs 3 lags)
2. **Ops surface** (Torch + optional CUDA vs a small booster file)
3. **Retrain / experiment loop speed** (sequence build vs tabular lags)

### Offline retrain for “keep model fresh”

| | LGBM | LSTM |
|---|---|---|
| Laptop CPU retrain | Practical | Painful without subsample + long waits |
| GPU workstation | Unused | Helps epochs only; **prep still CPU-bound** |
| Risk of OOM on full history | Low–moderate (panels) | **High** if stride=1 / no caps |
| Good fit for frequent retrain | Yes | Poor |

---

## Why LSTM looks less practical here (even if accuracy were similar)

1. **Task horizon is one snapshot ahead** on already-lagged microstructure. A short lag table + trees already matches the decision grain; a 64-step RNN adds capacity mostly where we pay RAM/time.
2. **Cost is front-loaded in data engineering**, not in “neural net training romance.” GPU does not fix parquet parse + sequence construction.
3. **Live stack is already LGBM-shaped.** Production path, paper campaigns, and mechanical baselines assume a cheap tabular predict. LSTM needs a second trader path and heavier state.
4. **Literature peers (Tsoku / Sheng / Han) run at coarser or different tasks.** Daily cointegration z or index RP is not the same as minute cross-CEX microstructure with hundreds of parallel series.

---

## Framing for paper / decision memo

Recommended wording:

> We implement an LSTM \(z_{t+1}\) forecaster as a **same-protocol research baseline** (Jul-25 split, `|pred|>0.5`). Offline, sequence construction and memory dominate cost: full-resolution tensors are multi‑GB to tens of GB, and even a GPU-accelerated capped run remains prep-heavy versus LightGBM’s tabular lag recipe. For **live** paper trading at ~110 s snapshots, LightGBM needs only a small lag state and a CPU booster, whereas LSTM requires per-series history tensors and a deep-learning runtime. Given comparable experimental intent (learned spread forecast vs mechanical z), **LightGBM is the practical production choice**; LSTM is retained as a literature-aligned ablation, not as the default live head.

---

## Numbers to refresh after the current GPU run finishes

Update this table from `statarb/outputs_lstm/METRICS.md` / the runner log:

| Field | Value |
|---|---|
| End-to-end wall-clock (GPU run) | _fill_ |
| Epochs until early stop | _fill_ |
| Peak RAM (if measured) | _fill_ |
| Filtered DirAcc / R² / mean pnl_proxy / Sharpe | _fill_ |

---

## Related files

- LSTM prompt: [`docs/prompts/lstm_zscore_engineering_prompt.md`](prompts/lstm_zscore_engineering_prompt.md)
- LGBM live notes: [`docs/paper_trading_lgbm.md`](paper_trading_lgbm.md)
- LGBM vs mechanical: [`docs/baseline_lgbm_vs_mechanical_z.md`](baseline_lgbm_vs_mechanical_z.md)
