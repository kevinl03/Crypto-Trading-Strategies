# Live paper trading for the LGBM spread model

Tracked as #57.

Related to #12 (rule-based OU/z-score paper trader). This is a separate system: #12's
`experiments/paper_trader.py` runs threshold rules on one pair, while the StatArb GBM needs
cross-venue features across all coins on a shared snapshot grid.

## What was built

`experiments/paper_trade_lgbm.py` tails a running `collect_statarb_data` JSONL run, rebuilds the
notebook feature pipeline (rolling z-score, lags, long-to-wide per venue, cross-exchange
dispersion/rank/imbalance), and predicts with the trained booster. Support scripts:
`scripts/check_paper_health.py` (health probe), `scripts/watch_paper_8h.ps1` (restart watchdog),
`scripts/report_paper_session.py` (post-session scoring vs persistence baseline).

Docs: `docs/paper_trading_lgbm.md`. Branch: `experiment/alt-prediction-target`.

## First session: 2026-07-30, 8h unattended

Model `statarb/outputs_ob_fix/statarb_lgbm.txt` (73 features), 23 volatile coins x 6 venues,
entry at `|pred| >= 0.5`, horizon 2 snapshots.

| | |
|---|---|
| Snapshots / predictions | 264 / 80,040 |
| Settled bets | 4,901 |
| Settled DirAcc | 73.1% |
| Mean PnL proxy (z-units) | 0.678 |

Scored against the naive `z_t -> z_{t+2}` baseline on identical rows:

| Rows | Model R2 | Model DirAcc | Naive R2 | Naive DirAcc | Model edge |
|---|---|---|---|---|---|
| All (49,310) | +0.054 | 55.3% | -0.547 | 62.3% | **-7.0pp** |
| Entries `\|pred\| >= 0.5` (5,054) | +0.248 | 76.5% | -0.045 | 75.4% | **+1.1pp** |

## Findings

1. **Live R2 matches offline.** Full-population 0.054 vs 0.063 on the Jul 22-28 holdout. No live/offline gap.
2. **The headline entry numbers are mostly selection bias.** Thresholding on `|pred|` selects high-|z| rows
   where target std rises 1.09 -> 1.29, inflating R2 for any predictor. Naive persistence gets the same lift.
3. **Genuine directional edge is ~1.1pp**, not the 76.5% headline. On the full population the model is
   *worse* than persistence at direction (55.3% vs 62.3%).
4. **Mechanism:** on entry rows `corr(pred, z_t) = 0.66` and the model's sign matches the current z-score's
   sign 80.9% of the time. It largely restates "z is far from zero and will stay there."

This matches the offline result that persistence recovers ~91% of test R2.

## Caveats

- `pnl_proxy` is `sign(pred) * realized_z` in z-units. Not dollars; no fees, slippage, or fill modeling.
- No position sizing or capital constraint. One bet per `(coin, pair)`, cap 50 concurrent.
- `signals.jsonl` retains the last 50k signals, so metrics cover 49,310 of 80,040 predictions.
- One 8h window, one regime. Not enough to establish live edge.

## Next steps

- [ ] Fee/slippage-aware dollar PnL instead of the z-unit proxy (overlaps #30).
- [ ] Log the naive baseline per snapshot in the live loop so the increment is visible in real time.
- [ ] Re-run against the `zscore_delta` / sign targets the offline sweep favored over H=2 level regression.
- [ ] Sweep `entry_tau`; 0.5 was chosen a priori, never tuned.
- [ ] Multi-day session across regimes before treating any of this as a live edge.
