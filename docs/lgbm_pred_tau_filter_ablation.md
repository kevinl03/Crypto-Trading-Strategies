# LightGBM `|pred| > τ` Filter Ablation

**Date:** 2026-08-07  
**Branch:** `feat/lstm-zscore-baseline`  
**Purpose:** Sensitivity of offline LGBM forecast / trading proxies to the abstention threshold τ.  
**Runner:** [`statarb/sweep_lgbm_pred_tau.py`](../statarb/sweep_lgbm_pred_tau.py)  
**Artifacts:** [`statarb/outputs_lgbm_tau_sweep_jul25/`](../statarb/outputs_lgbm_tau_sweep_jul25/) (`TAU_SWEEP.md`, `tau_sweep.json`, `tau_sweep.csv`)  
**Model:** [`statarb/outputs/statarb_lgbm.txt`](../statarb/outputs/statarb_lgbm.txt) (not retrained)  
**Related:** [`docs/results_lstm_vs_lgbm.md`](results_lstm_vs_lgbm.md), [`docs/handoff_paper_campaign_framing.md`](handoff_paper_campaign_framing.md)

---

## 1. Setup

| Knob | Value |
|---|---|
| Holdout | Jul 25–28 (`snapshot_idx ≥ 3584`) |
| Rows scored | 1,679,736 (valid \(z_{t+1}\) + next `spread_bps`) |
| Target | \(y = z_{t+1}\) of cross-exchange `spread_bps` |
| Filter | trade iff \(\lvert\hat{y}\rvert > \tau\) |
| Direction | \(\operatorname{sign}(\hat{y})\) |
| τ grid | 0.10, 0.25, 0.35, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 1.00, 1.10, 1.25, 1.50, 1.75, 2.00 |

Live campaigns use **τ = 0.5** (`|pred| ≥ 0.5` in session docs). This ablation is **offline** on the same calendar holdout as the LGBM vs LSTM compare; it does not re-run the Jul 31 live book.

---

## 2. Definitions

| Term | Definition |
|---|---|
| **Fire rate** | \(n_{\text{traded}} / n_{\text{all}}\) — fraction of scored rows with \(\lvert\hat{y}\rvert > \tau\) |
| **DirAcc** | \(\mathbb{P}(\operatorname{sign}(\hat{y})=\operatorname{sign}(z_{t+1}))\) on traded rows |
| **R²** | sklearn \(R^2(z_{t+1},\hat{y})\) on traded rows |
| **mean pnl_proxy** | \(\operatorname{mean}(\operatorname{sign}(\hat{y})\cdot z_{t+1})\) — z-units, not $ PnL |
| **Sharpe / trade** | \(\operatorname{mean}(\text{pnl\_proxy}) / \operatorname{std}(\text{pnl\_proxy})\), Rf=0 |
| **Sharpe A** | mean/std of **summed** filtered pnl_proxy per `(window_id, snapshot_idx//33)` (offline closed-only; not live Sharpe B) |
| **mean gross bps** | \(\operatorname{mean}(\operatorname{sign}(\hat{y})\cdot\Delta\text{spread\_bps})\), \(\Delta s = s_{t+1}-s_t\) |
| **win_bps** | fraction of traded rows with gross bps > 0 (not the same as DirAcc) |
| **quality×coverage score** | \(\text{mean pnl\_proxy}\times\sqrt{\text{fire rate}}\) |

**Fire rate vs DirAcc:** fire rate is *coverage* (how often we trade). DirAcc is *conditional sign accuracy on \(z_{t+1}\)*. Raising τ usually raises DirAcc and cuts fire rate.

---

## 3. Ranking rule (recommended τ)

There is no single globally optimal τ. Conditional quality (DirAcc, R², mean pnl_proxy, Sharpe/trade) rises with τ, while coverage and Sharpe A fall.

**Default recommendation rule used here:**

\[
\tau^\star = \arg\max_{\tau:\;\text{fire}\ge 5\%,\; n\ge 500}
\;\underbrace{\text{mean pnl\_proxy}\times\sqrt{\text{fire rate}}}_{\text{quality×coverage score}}
\]

This prefers higher z-proxy skill without collapsing to a handful of ultra-confident rows. Single-metric argmaxes are reported separately for transparency.

---

## 4. Results

### 4.1 Recommended operating point (score rule)

Under the rule above: **τ = 0.25**

| Metric | Value |
|---|---:|
| n | 658,383 |
| fire rate | 39.2% |
| DirAcc | 71.6% |
| R² | 0.260 |
| mean pnl_proxy | +0.515 |
| Sharpe / trade | 0.504 |
| Sharpe A | 5.21 |
| mean gross bps | −0.63 |

Nearby, **τ = 0.35** is almost tied on the composite score (0.320 vs 0.323) with better DirAcc/R² (74.7% / 0.314) and still 27% fire.

### 4.2 Campaign default (τ = 0.50)

| Metric | Value |
|---|---:|
| n | 263,424 |
| fire rate | 15.7% |
| DirAcc | 78.7% |
| R² | 0.389 |
| mean pnl_proxy | +0.765 |
| Sharpe / trade | 0.728 |
| Sharpe A | 3.90 |
| mean gross bps | −0.84 |
| quality×coverage score | 0.303 (~6% below peak at τ=0.25) |

### 4.3 High-confidence contrast (τ = 1.00)

| Metric | Value |
|---|---:|
| n | 42,519 |
| fire rate | 2.5% |
| DirAcc | 86.4% |
| R² | 0.563 |
| mean pnl_proxy | +1.271 |
| Sharpe / trade | 1.088 |
| Sharpe A | 1.94 |
| mean gross bps | −1.36 |

### 4.4 Optima by single metric (\(n \ge 500\))

| Criterion | Best τ | Comment |
|---|---:|---|
| quality×coverage score | **0.25** | Balanced default under §3 |
| DirAcc / R² / mean pnl_proxy / Sharpe/trade | **2.00** | Tiny book (n=1,401; 0.1% fire) |
| Sharpe A | **0.10** | Max coverage; weakest conditional skill |
| mean gross bps | **0.10** | Least negative bps; still &lt; 0 |

### 4.5 Full sweep

| τ | n | fire | DirAcc | R² | mean pnl_z | Sharpe/tr | Sharpe A | score | mean gross bps | win_bps |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 1,180,549 | 0.703 | 0.6648 | 0.1761 | +0.3605 | 0.356 | 6.152 | 0.3023 | −0.4481 | 0.366 |
| 0.25 | 658,383 | 0.392 | 0.7160 | 0.2601 | +0.5152 | 0.504 | 5.209 | **0.3225** | −0.6261 | 0.367 |
| 0.35 | 454,279 | 0.270 | 0.7465 | 0.3137 | +0.6144 | 0.595 | 4.661 | 0.3195 | −0.7145 | 0.366 |
| **0.50** | 263,424 | 0.157 | 0.7868 | 0.3890 | +0.7653 | 0.728 | 3.900 | 0.3031 | −0.8379 | 0.365 |
| 0.60 | 184,213 | 0.110 | 0.8093 | 0.4326 | +0.8652 | 0.810 | 3.462 | 0.2865 | −0.9141 | 0.365 |
| 0.70 | 127,922 | 0.076 | 0.8285 | 0.4726 | +0.9668 | 0.891 | 3.040 | 0.2668 | −1.0047 | 0.365 |
| 0.75 | 106,094 | 0.063 | 0.8362 | 0.4900 | +1.0180 | 0.927 | 2.817 | 0.2558 | −1.0339 | 0.366 |
| 0.80 | 88,411 | 0.053 | 0.8433 | 0.5063 | +1.0679 | 0.962 | 2.607 | 0.2450 | −1.0885 | 0.366 |
| 0.90 | 61,456 | 0.037 | 0.8531 | 0.5351 | +1.1645 | 1.023 | 2.237 | 0.2227 | −1.2379 | 0.363 |
| 1.00 | 42,519 | 0.025 | 0.8635 | 0.5633 | +1.2709 | 1.088 | 1.944 | 0.2022 | −1.3595 | 0.362 |
| 1.10 | 29,320 | 0.017 | 0.8708 | 0.5850 | +1.3790 | 1.144 | 1.676 | 0.1822 | −1.5130 | 0.362 |
| 1.25 | 16,827 | 0.010 | 0.8764 | 0.6086 | +1.5548 | 1.233 | 1.352 | 0.1556 | −1.8322 | 0.361 |
| 1.50 | 7,113 | 0.004 | 0.8756 | 0.6099 | +1.8457 | 1.326 | 0.961 | 0.1201 | −2.6708 | 0.355 |
| 1.75 | 3,130 | 0.002 | 0.8792 | 0.6246 | +2.1168 | 1.429 | 0.771 | 0.0914 | −3.4335 | 0.366 |
| 2.00 | 1,401 | 0.001 | 0.8822 | 0.6390 | +2.3993 | 1.541 | 0.635 | 0.0693 | −4.8707 | 0.368 |

### 4.6 Naive persistence refs (same holdout)

| Filter | n | DirAcc | R² | mean pnl_proxy |
|---|---:|---:|---:|---:|
| \(\lvert z_t\rvert > 0.5\) | 1,043,054 | 68.3% | −0.395 | +0.428 |
| \(\lvert z_t\rvert > 1.0\) | 506,051 | 70.1% | −0.562 | +0.583 |

LGBM remains well above naive on R² and mean pnl_proxy at both matched-style thresholds; DirAcc lift grows as τ rises (e.g. τ=1.0: 86.4% vs naive 70.1%).

---

## 5. Interpretation for the paper

1. **`|pred|>τ` is an abstention knob** (Han-style filter): higher τ → better conditional forecast/trading *quality*, lower *coverage*.
2. **Composite-score optimum is τ ≈ 0.25–0.35** on this holdout; **τ = 0.50** (live protocol) is only ~6% below the peak score while delivering materially higher DirAcc/R² than τ=0.25.
3. **Do not pick τ by DirAcc alone** — that pushes τ→2 with a vanishing book and collapsing Sharpe A.
4. **τ does not fix economics of Δspread:** mean gross bps is negative at every grid point; win_bps stays ~36%. Positive pnl_proxy ≠ positive H=1 bps return.
5. **Protocol continuity:** keep **τ = 0.5** as the primary live/offline compare threshold unless a new live campaign is re-run at another τ. Report this sweep as a **sensitivity / ablation figure**.

### Suggested paper wording

> We treat \(\lvert\hat{y}\rvert>\tau\) as an abstention threshold. On the Jul 25–28 LightGBM holdout, raising τ monotonically improves filtered DirAcc, R², and mean z-settled pnl_proxy while reducing fire rate and closed hourly Sharpe A. A quality–coverage score \(\text{mean pnl\_proxy}\sqrt{\text{fire}}\) peaks near τ=0.25–0.35; the campaign default τ=0.5 remains close on that score with stronger conditional skill. We retain τ=0.5 for protocol comparability with live paper sessions and report the full τ sweep as a sensitivity analysis. Separately, mean H=1 spread-bps returns stay negative across τ, so the filter improves z-proxy selectivity rather than flipping Δspread P&L.

---

## 6. Reproduce

```bash
cd statarb
../.venv/Scripts/python.exe -u sweep_lgbm_pred_tau.py
```

Writes `statarb/outputs_lgbm_tau_sweep_jul25/{TAU_SWEEP.md,tau_sweep.json,tau_sweep.csv}`.

---

## 7. What this ablation does *not* do

- Retrain the booster (predictions only; threshold post-hoc).
- Re-score the Jul 31 live book or capacity-matched mechanical peers at each τ.
- Optimize for dollar PnL / fees (bps column is diagnostic only).
- Sweep LSTM τ (LGBM-only here; LSTM can reuse the same grid later).
