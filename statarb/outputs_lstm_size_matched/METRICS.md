# LSTM Z-Score Metrics

Protocol: predict z_(t+1) with W=300, min_periods=90, SEQ_LEN=64.
**Primary slice:** `|pred| > 0.5` (same filter axis as LGBM campaigns).
Split: train < Jul 25 / test Jul 25-28 (`snapshot_idx` cut 3584).

Features are **LSTM-native** (pair-leg microstructure sequences), not the 68 LGBM columns.

## Headline (filtered) — compare to LGBM

| Model | n | DirAcc | R2 | mean pnl_proxy | Sharpe (per-trade) | Sharpe A (closed hourly) |
|---|---:|---:|---:|---:|---:|---:|
| LSTM | 37601 | 0.8158559612776256 | 0.4697818177777725 | 0.815693115434002 | 0.834517884549893 | 3.9336042118956773 |
| Naive z_t | 92412 | 0.704572999177596 | -0.2813243158016343 | 0.4827753293485979 | 0.48252890179985 | 4.545413271384821 |

Definitions (aligned with live LGBM reporting):
- `pnl_proxy = sign(pred) * z_(t+1)`
- DirAcc / R2 / mean pnl_proxy reported on **filtered** rows
- `sharpe_per_trade = mean(pnl_proxy) / std(pnl_proxy)` on filtered trades (Rf=0)
- `sharpe_closed_hourly_A` = mean/std of **summed** filtered pnl_proxy per `(window_id, snapshot_idx//33)` bucket (closed-only; offline proxy of Sharpe A). **Not** live portfolio Sharpe B with open MTM.

## Full slices

### LSTM
- All: n=150000, DirAcc=67.42%, R2=0.2288, mean_pnl_proxy=0.3628, sharpe_per_trade=0.3773, sharpe_closed_hourly_A=4.7738
- Filtered: n=37601, DirAcc=81.59%, R2=0.4698, mean_pnl_proxy=0.8157, sharpe_per_trade=0.8345, sharpe_closed_hourly_A=3.9336

### Naive z_t -> z_(t+1)
- All: n=150000, DirAcc=69.29%, R2=-0.2064, mean_pnl_proxy=0.3397, sharpe_per_trade=0.3502, sharpe_closed_hourly_A=4.6235
- Filtered: n=92412, DirAcc=70.46%, R2=-0.2813, mean_pnl_proxy=0.4828, sharpe_per_trade=0.4825, sharpe_closed_hourly_A=4.5454

## Literature adaptations

- **Tsoku & Makatjane (2026):** MSE regression of standardized spread/z; report forecast + trading metrics.
- **Han & Li (2024):** PyTorch LSTM; `|pred|>0.5` as trade abstention filter (not their 3-class trend head).
- **Sheng & Ma (2022)** (repo notes mis-cite as Shen): 2-layer LSTM + Adam; dual error/trading table.

## Artifacts

- `statarb_lstm.pt`
- `feature_schema.json`
- `metrics.json`
