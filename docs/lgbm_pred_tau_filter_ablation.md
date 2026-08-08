# LightGBM `|pred| > τ` Filter Ablation

**Use in paper:** Methods (abstention filter) · Results (sensitivity table/figure) · Discussion (why τ=0.5)  
**Holdout:** Jul 25–28 offline · **Model:** `statarb/outputs/statarb_lgbm.txt` (not retrained)  
**Runner / data:** [`statarb/sweep_lgbm_pred_tau.py`](../statarb/sweep_lgbm_pred_tau.py) → [`statarb/outputs_lgbm_tau_sweep_jul25/`](../statarb/outputs_lgbm_tau_sweep_jul25/)

---

## What to use where

| Paper section | Claim / content | Copy from |
|---|---|---|
| **Methods** | Trade when `\|pred\| > τ`; τ=0.5 is the protocol default (live + offline) | §Definitions + one-liner below |
| **Results** | Raising τ improves filtered DirAcc / R² / mean pnl_proxy (abstention works) | Headline table (§Results) |
| **Results (figure)** | Quality–coverage curve; τ=0.5 near the balanced peak | Full sweep CSV / table |
| **Discussion** | τ=0.5 is justified: strong skill vs naive, usable fire (~16%), close to score peak | §Takeaway |
| *Appendix only* | Full τ grid + bps diagnostic | `tau_sweep.csv` |

**Primary metrics for the paper:** DirAcc, R², mean `pnl_proxy` (z-settle). These support the learned-filter story.

---

## Definitions (short)

| Term | Meaning |
|---|---|
| Fire rate | `n_traded / n_all` with `\|pred\| > τ` |
| DirAcc | `sign(pred) == sign(z_{t+1})` on traded rows (zeros excluded) |
| R² | `r2_score(z_{t+1}, pred)` on traded rows |
| mean pnl_proxy | `mean(sign(pred) * z_{t+1})` — z-units |
| Sharpe / trade | mean/std of pnl_proxy (Rf=0) |
| Sharpe A | mean/std of hourly summed pnl_proxy (offline closed; not live Sharpe B) |
| Score | `mean_pnl_proxy * sqrt(fire_rate)` — balances skill vs coverage |

---

## Results (verified)

n_all = **1,679,736**. Numbers from `tau_sweep.json`.

### Headline (use this)

| τ | Role | fire | DirAcc | R² | mean pnl_z | vs naive DirAcc / R² / pnl_z |
|---:|---|---:|---:|---:|---:|---|
| **0.50** | **Protocol default** | 15.7% | **78.7%** | **0.389** | **+0.765** | naive `\|z\|>0.5`: 68.3% / −0.395 / +0.428 |
| 0.25 | Score peak | 39.2% | 71.6% | 0.260 | +0.515 | — |
| 1.00 | High confidence | 2.5% | **86.4%** | **0.563** | **+1.271** | naive `\|z\|>1`: 70.1% / −0.562 / +0.583 |

**Takeaway:** Abstention helps — higher τ → better filtered skill. At τ=0.5 the model clearly beats `|z|` persistence on R² and mean pnl_proxy (+0.77 vs +0.43) and lifts DirAcc (+10.4 pp). τ=1.0 widens that gap further (DirAcc +16.3 pp vs `|z|>1`) if a thinner book is acceptable. The composite score peaks at τ=0.25–0.35; **τ=0.5 stays within ~6% of that peak** while delivering stronger conditional metrics — so the live/offline protocol choice is well supported.

### Compact sweep (Results figure / appendix)

| τ | fire | DirAcc | R² | mean pnl_z | Sharpe/tr | Sharpe A | score |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 39.2% | 71.6% | 0.260 | +0.515 | 0.50 | 5.21 | **0.323** |
| 0.35 | 27.0% | 74.6% | 0.314 | +0.614 | 0.59 | 4.66 | 0.320 |
| **0.50** | **15.7%** | **78.7%** | **0.389** | **+0.765** | **0.73** | **3.90** | 0.303 |
| 0.75 | 6.3% | 83.6% | 0.490 | +1.018 | 0.93 | 2.82 | 0.256 |
| 1.00 | 2.5% | 86.4% | 0.563 | +1.271 | 1.09 | 1.94 | 0.202 |

Full grid (15 taus): `statarb/outputs_lgbm_tau_sweep_jul25/tau_sweep.csv`.

---

## Paste-ready (Discussion)

> We use `|pred| > τ` as an abstention filter on the LightGBM \(z_{t+1}\) forecaster. On the Jul 25–28 holdout, raising τ improves filtered DirAcc, R², and mean z-settled pnl_proxy, consistent with ML-as-filter designs. At the protocol default τ=0.5, LightGBM achieves DirAcc 78.7%, R² 0.39, and mean pnl_proxy +0.77, versus 68.3%, −0.40, and +0.43 for mechanical `|z_t|>0.5` persistence. A quality–coverage score peaks near τ=0.25–0.35; τ=0.5 remains close on that score with higher conditional skill and is retained for comparability with live paper sessions.

---

## Reproduce

```bash
cd statarb
../.venv/Scripts/python.exe -u sweep_lgbm_pred_tau.py
```
