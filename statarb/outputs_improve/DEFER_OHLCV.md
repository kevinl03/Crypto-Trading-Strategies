# OHLCV deferred (plan item)

Do **not** enable OHLCV in the GBM pipeline yet.

## Why
- Local 1m OHLCV only covers Jun 13–16 (9 venues) and partially Jun 22–24 (kraken+gateio).
- Jul 13, Jul 19–22, and Jul 22–28 have **no** OHLCV in `cex_unified`.
- Turning the toggle on would inject structured NaNs on most train/test rows and add nothing to the holdouts we care about.

## Also deferred
- Collecting more hours of the *same* signal set (diminishing returns; persistence already explains ~91% of R²).
- Longer LightGBM boosting / hyperparameter retunes (already flat within 0.0005 R²).

## Revisit when
- OHLCV is backfilled for Jul19+ on the same venues as the mid-based spread features, **and**
- paper/backtest both use the same price definition (mids vs 1m closes).
