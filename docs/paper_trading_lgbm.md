# Live LightGBM paper trading

`experiments/paper_trader.py` runs OU/z-score rules on a single asset pair over WebSocket feeds. It cannot
drive the trained StatArb GBM, which needs cross-venue features over all coins at a shared snapshot grain.
`experiments/paper_trade_lgbm.py` fills that gap.

## How it works

The trader tails a running `collect_statarb_data` JSONL run rather than opening its own exchange
connections, so collection and inference stay decoupled and a trader restart never loses market data.

1. **Ingest** — byte-offset tail of `spread_matrix`, `ticker`, `orderbook`, `trades` JSONL files.
2. **Rebuild features** — the same transforms as current `statarb/cex_gbm_new.ipynb`: rolling z-score
   (`ZSCORE_WINDOW=300`, `MIN_PERIODS=90`), three lags per signal, long→wide pivot per exchange, and the
   cross-exchange / momentum / accel block. Venue set is `TOP_EXCHANGES` (6 venues).
3. **Predict** — `model.predict()` on the booster feature list (68 for `statarb/outputs/statarb_lgbm.txt`), with `coin`/`pair` restored to the booster's
   training categorical vocabularies via `pandas_categorical`.
4. **Trade** — open a paper bet when `|pred| >= entry_tau`, one per `(coin, pair)`, settled after
   `HORIZON=1` snapshots against the realized z-score.

Feature parity matters more than speed here: the booster silently produces garbage if a column is missing or
a categorical is encoded differently, so `prepare_X` reindexes to `model.feature_name()` rather than trusting
column order.

## Running a session

```powershell
# 1. start the collector (writes data/statarb/<run_id>/)
python -m experiments.collect_statarb_data --assets volatile --interval 60 --slow-every 1 --hours 8 --skip-ohlcv

# 2. point the trader at that run
python -m experiments.paper_trade_lgbm `
    --model statarb/outputs/statarb_lgbm.txt `
    --run-dir data/statarb/<run_id> `
    --hours 8 --entry-tau 0.5 --poll-sec 20 `
    --output-dir data/paper_trading/<session>
```

Omit `--run-dir` to auto-detect the newest collector run.

Outputs per session: `config.json`, `signals.jsonl` (+ `signals_001.jsonl` … when a shard
hits 50k lines), same pattern for `trades.jsonl`, `summary.json`. Shards are append-only so
older rows are never truncated.

## Supporting scripts

| Script | Purpose |
|---|---|
| `scripts/check_paper_health.py` | One-shot health probe; exit 0 ok / 1 warn / 2 hard fail. Writes `health_latest.json`. |
| `scripts/watch_paper_8h.ps1` | Watchdog that restarts collector or trader if either dies before the deadline. |
| `scripts/report_paper_session.py` | Post-session scoring, including the naive persistence baseline on identical rows. |

## Evaluating results honestly

`report_paper_session.py` always scores the naive `z_t → z_{t+HORIZON}` baseline on the **same rows** as the
model. This is load-bearing: thresholding on `|pred|` selects high-|z| rows with larger target variance, which
inflates R² for *any* predictor. Without the matched baseline an entry-filtered R² looks far better than the
model's real contribution.

## Known limitations

- `pnl_proxy` is `sign(pred) × realized_z` in z-units. It is not dollar PnL and ignores fees, slippage, and fills.
- Collector cadence with `--slow-every 1` is ~110s per snapshot, not the nominal 60s. Size warmup expectations
  accordingly: `MIN_PERIODS=90` needs ~90 snapshots (~2.5–3h) before predictions start.
- Predictions only begin once `snapshot_idx >= MIN_PERIODS + N_LAGS`; earlier snapshots are skipped permanently
  so the scoring cursor can advance.
