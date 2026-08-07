# Jul 25-28 holdout: LGBM vs size-matched LSTM

Same window, target \(z_{t+1}\), filter `|pred|>0.5`, metric code (`evaluate_model_and_naive`).

| Model | n | DirAcc | R2 | mean pnl_proxy | Sharpe/trade | Sharpe A |
|---|---:|---:|---:|---:|---:|---:|
| **LGBM filtered** | 263464 | 0.7868 | 0.3890 | 0.7653 | 0.7277 | 3.9013 |
| LGBM all | 1680081 | 0.6307 | 0.1317 | 0.2675 | 0.2648 | 6.3305 |
| Naive `|z_t|>0.5` | 1043242 | 0.6832 | -0.3948 | 0.4279 | 0.4148 | 5.7200 |
| **LSTM size-matched filtered** | 37601 | 0.8159 | 0.4698 | 0.8157 | 0.8345 | 3.9336 |
| LSTM size-matched all | 150000 | 0.6742 | 0.2288 | 0.3628 | 0.3773 | 4.7738 |

Note: LSTM scored a stride/subsampled test panel (n=150k all); LGBM scored the full Jul25-28 rows with valid target. Both use the same calendar holdout and definitions.
