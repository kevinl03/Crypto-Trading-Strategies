# 8h live LGBM paper-trading session — 2026-07-30

**Window:** `2026-07-30T07:56:05Z` → `2026-07-30T15:56:20Z` (8.0h, unattended)
**Model:** `statarb/outputs_ob_fix/statarb_lgbm.txt` (LightGBM, 73 features, `coin`/`pair` native categoricals)
**Data:** live collector run `data/statarb/20260730_075605` (23 volatile coins × 6 venues)
**Entry rule:** open when `|pred| >= 0.5`, settle after `HORIZON = 2` snapshots against realized z-score
**Target:** z-score of `spread_bps` at `t+2` (identical to training target)

## Session totals

| | |
|---|---|
| Collector snapshots | 264 (~110s each, not the assumed 60s) |
| Predictions scored | 80,040 |
| Paper bets closed | 4,901 (19 open at cutoff) |
| Settled DirAcc | **73.1%** |
| Mean PnL proxy (z-units) | **0.678** |

## Metrics vs persistence baseline

Scored on the last 49,310 retained signals, comparing model `pred_t` against realized `z_{t+2}`, with the naive
`z_t → z_{t+2}` baseline on **identical rows** ([`metrics_report.csv`](metrics_report.csv)).

| Rows | Model R² | Model DirAcc | Naive R² | Naive DirAcc | Model edge |
|---|---|---|---|---|---|
| All predictions (49,310) | +0.054 | 55.3% | −0.547 | **62.3%** | **−7.0pp** |
| Entries `\|pred\| >= 0.5` (5,054) | +0.248 | **76.5%** | −0.045 | 75.4% | **+1.1pp** |

## Interpretation

**Live R² matches offline expectations.** The full-population R² of **0.054** is consistent with the offline
Jul 22–28 forward holdout (**0.063**). The model behaves in production the way the backtest said it would.

**The high entry R² is mostly a selection artifact, not skill.** Filtering to `|pred| >= 0.5` selects high-|z|
rows where target std rises from 1.09 → 1.29, inflating the R² denominator. The naive baseline gets the same
free lift on those rows (−0.547 → −0.045) without any model involved.

**The 76.5% DirAcc is mostly the filter.** Naive persistence scores 75.4% on the exact same rows, so the model's
genuine directional increment is about **1.1pp**. On the full population the model is *worse* at direction than
persistence (55.3% vs 62.3%).

**Mechanism.** On entry rows `corr(pred, z_t) = 0.66` and the model's sign agrees with the current z-score's sign
**80.9%** of the time. When it fires, it largely restates "z is far from zero and will stay on that side."

This reproduces the offline finding that ~91% of test R² is recoverable from lag features alone, with the full
microstructure stack adding ~0.006 R² while hurting DirAcc.

## Caveats

- `pnl_proxy` is `sign(pred) × realized_z`, in **z-units, not dollars**. No fees, no slippage, no fill modeling.
- No position sizing or capital constraint; one open bet per `(coin, pair)`, cap 50 concurrent.
- `signals.jsonl` retains the most recent 50,000 signals, so metrics cover 49,310 of the 80,040 predictions.
- Single 8h window on one market regime. Not sufficient to establish live edge.

## Follow-ups

1. Add fee/slippage-aware dollar PnL instead of a z-unit proxy.
2. Persist a per-snapshot naive baseline column in the live loop so the increment is visible in real time.
3. Re-run against the longer-horizon `zscore_delta` / sign targets, which the offline sweep favored over H=2 level regression.
4. Sweep `entry_tau` — the current 0.5 threshold was picked a priori, not tuned.
