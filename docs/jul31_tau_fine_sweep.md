# Jul 31 Fine `|pred| ≥ τ` Sweep (τ ∈ [0.75, 1.0])

**Session:** `data/paper_trading/July31st_8_hr`  
**Model:** production 68-feat LightGBM (`statarb/outputs/statarb_lgbm.txt`), W=300, H=1  
**Filter:** post-hoc on scored signal rows with forward z (`z_fwd`)  
**Sharpe/trade:** `mean(pnl) / std(pnl)` with `pnl = sign(pred) × z_fwd`, Rf = 0  

Reproduce:

```bash
.venv/Scripts/python.exe scripts/sweep_live_session_tau.py \
  data/paper_trading/July31st_8_hr \
  --taus 0.75 0.8 0.85 0.9 0.95 1.0 \
  --out data/paper_trading/July31st_8_hr/tau_sweep_live_fine.json
```

---

## Signal filter (`|pred| ≥ τ` → `z_fwd`)

| τ | n | DirAcc | R² | mean pnl | Sharpe/trade | naive DirAcc | naive R² |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.75 | 3,609 | 82.88% | 0.466 | +1.038 | 0.915 | 82.68% | 0.345 |
| 0.80 | 2,982 | 83.57% | 0.482 | +1.095 | 0.953 | 82.96% | 0.371 |
| 0.85 | 2,452 | 84.54% | 0.497 | +1.156 | 0.993 | 83.65% | 0.400 |
| **0.90** | **2,026** | **84.90%** | **0.514** | **+1.195** | **1.024** | 84.35% | 0.424 |
| 0.95 | 1,708 | 84.95% | 0.516 | +1.223 | 1.026 | 84.72% | 0.430 |
| 1.00 | 1,426 | 85.27% | 0.534 | +1.265 | 1.072 | 84.92% | 0.441 |

**Sharpe ≥ 1 first occurs at τ = 0.90** (not τ = 1.0). At τ = 1.0, Sharpe/trade ≈ 1.07.

---

## Live closed trades (post-hoc `|pred| ≥ τ`)

Book was filled live at τ = 0.5; tighter τ here is a subset of those closes.

| τ | n closed | DirAcc | mean pnl | Sharpe/trade | Hourly Sharpe A |
|---:|---:|---:|---:|---:|---:|
| 0.75 | 3,262 | 82.99% | +1.034 | 0.914 | 2.23 |
| 0.80 | 2,688 | 83.78% | +1.096 | 0.957 | 2.17 |
| 0.85 | 2,216 | 84.66% | +1.152 | 0.989 | 2.12 |
| **0.90** | **1,826** | **84.94%** | **+1.188** | **1.015** | 2.04 |
| 0.95 | 1,542 | 84.82% | +1.204 | 1.007 | 1.98 |
| 1.00 | 1,283 | 85.19% | +1.249 | 1.056 | 1.99 |

---

## Takeaway

- Raising τ from 0.75 → 1.0 lifts DirAcc (~83% → 85%), R² (0.47 → 0.53), and Sharpe/trade (0.91 → 1.07), while cutting n by ~60%.
- **τ ≈ 0.90** is where per-trade Sharpe first crosses 1.0; it is also near the DirAcc elbow (gains flatten after 0.85–0.90).
- Model R² stays well above matched-row naive R² at every τ in this band; DirAcc is only slightly above naive.

Artifacts: `data/paper_trading/July31st_8_hr/tau_sweep_live_fine.json`, `docs/jul31_tau_fine_sweep.csv`.
