# Random Forest vs LightGBM (paper protocol)

Classical tabular baseline on the **same LOGO / Jul-25 feature frames** as LightGBM.
RF is a literature peer — not a live replacement. Production head remains LightGBM.

## Protocol

- Target: `y = z_{t+1}` · W=300 · H=1 · N_LAGS=3
- Split: train pre Jul 25 · test Jul 25–28 (LOGO cache)
- Features: 62 tabular cols (published intersection on cache)
- Missing vs 68-feat booster: ['tk_bid_volume_lag1_coinbase', 'tk_bid_volume_lag2_coinbase', 'tk_bid_volume_lag3_coinbase', 'tk_ask_volume_lag1_coinbase', 'tk_ask_volume_lag2_coinbase', 'tk_ask_volume_lag3_coinbase']
- Gate: `|pred| >= tau` (paper headline τ=0.9); also report all-rows and τ=0.5
- Naive peer: `ẑ ← z_t` on identical rows

## Comparison table

| Model | Set | Filter | n | DirAcc | R² | mean pnl_proxy |
|---|---|---|---:|---:|---:|---:|
| LightGBM | test | all | 1680426 | 62.8% | 0.133 | 0.269 |
| LightGBM | test | |ẑ|≥0.5 | 273183 | 78.4% | 0.384 | 0.758 |
| LightGBM | test | |ẑ|≥0.9 | 66513 | 85.2% | 0.525 | 1.154 |
| Random Forest | test | all | 1680426 | 62.9% | 0.126 | 0.267 |
| Random Forest | test | |ẑ|≥0.5 | 216627 | 79.2% | 0.380 | 0.805 |
| Random Forest | test | |ẑ|≥0.9 | 33038 | 86.0% | 0.478 | 1.299 |
| Naive z_t | test | all | 1680081 | 67.0% | -0.287 | 0.302 |
| Naive z_t | test | |ẑ|≥0.5 | 1043242 | 68.3% | -0.395 | 0.428 |
| Naive z_t | test | |ẑ|≥0.9 | 602453 | 69.5% | -0.523 | 0.545 |

### Paper reference cells (LGBM validation live book)

| Model | Set | Filter | n | DirAcc | R² | mean pnl_proxy |
|---|---|---|---:|---:|---:|---:|
| LightGBM (paper / tau09 report) | validation | `|ẑ|≥0.9` | 12795 | 86.7% | 0.599 | 1.372 |

## Resources

- Backend: `sklearn.ensemble.RandomForestRegressor` (CPU)
- Fit wall-clock: 0.0 s
- Predict (test) wall-clock: 5.51 s
- Train rows used: 4947685 / available 4947685
- Test rows: 1680426
- RF params: `{"max_depth": 20, "max_features": "sqrt", "max_samples": 1000000, "min_samples_leaf": 200, "n_estimators": 400, "n_jobs": 8, "random_state": 42}`
- Peak note: CPU only. Defaults cap train rows / n_jobs / max_samples after a 32 GiB box hit ~13 GiB private with full 4.7M X and n_jobs=-1. Use --mem-check-only first.

## Artifacts

- `rf_model.joblib` — fitted RandomForestRegressor
- `encoder.joblib` — OrdinalEncoder for coin/pair (+ medians)
- `metrics_test.json` — full metric block
- `METRICS.md` — this file
