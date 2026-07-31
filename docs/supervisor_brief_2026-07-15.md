# Project Status Brief — Cross-Venue Crypto Stat-Arb with ML Signals

**Date:** July 15, 2026 &nbsp;|&nbsp; **Team:** Kirill Litvinov, Tania &nbsp;|&nbsp; **Supervisor meeting**

---

## Research Question

> Can gradient-boosted tree models conditioning on microstructure features (order-book state, funding rate) achieve higher out-of-sample risk-adjusted returns than OU and z-score mean-reversion baselines on 1-minute cross-venue spreads in high-volatility CEX crypto pairs, under identical transaction-cost and execution assumptions?

## Baselines

| Strategy | Mechanism | Entry / Exit |
|---|---|---|
| **Z-score** (model-free) | Rolling window mean & std of cross-venue spread; signal = $(s_t - \mu_t)/\sigma_t$ | Enter \|z\| ≥ 2, exit \|z\| < 0.5, max-hold stop |
| **OU process** (model-based) | Estimates mean-reversion speed $\theta$ and half-life $t_{1/2} = \ln 2/\theta$ via rolling OLS on AR(1) discretization | Same thresholds; adds dynamics — *when* to expect reversion |
| **Controls** | Buy-and-hold (−426 bps), random-entry (−26,295 bps median, 1000 sims) | — |

Key finding: OU wins on fast spreads ($t_{1/2}$ ≈ 1.5–2 min); z-score wins when half-life is long relative to window — together they form a non-trivial bar for ML.

## Data

- **Live collectors** capturing 1-min granularity from multiple CEXs: **top-of-book quotes, funding rates, trades** — fields unavailable in backfilled data.
- **~168 hours** (~10,080 observations per instrument-venue pair) across high-volatility assets (WIF, PEPE, CRV, DOGE, SOL, BONK, FLOKI, SEI). Published to Hugging Face.

  > Note: this is the combined target across collection windows, not one continuous
  > run — actual windows so far are train (~65h), test (~60h), and an initial
  > validation window (~63h wall-clock, 1,502 snapshots after collector
  > interruptions ate into the fixed budget). A second validation run is in
  > progress as of July 16.

- Quote data enables **stale-print detection** — we verified some venue pairs show 26× inflated spread std from stale prints vs live quotes, making execution-realistic evaluation possible.

### Signals collected (the data edge)

Our live collector (`experiments/collect_statarb_data.py`) snapshots **9 concurrent
signal families** every ~60–90s, across 23 assets × 12 CEX venues. This is the
core non-backfillable advantage: public exchange APIs serve historical OHLCV
candles, but **not** historical quotes, order books, or operational status —
so every one of these fields only exists because we collected it live, and
cannot be reconstructed retroactively for any past window.

| Signal | What it captures | Why it matters |
|---|---|---|
| `ticker` | Best bid/ask, last price | Core quote-mid spread signal; the executable instrument |
| `orderbook` (L20) | Top-20 levels per side | Verifies executability — separates genuine dislocations from stale prints |
| `trades` | Recent prints (last 50) | Print-vs-quote comparison; trade-count as a liquidity proxy |
| `ohlcv` | 1-min OHLCV (backfillable, kept for continuity) | Cross-checks against the live quote series |
| `funding_rate` | Perp funding rate | Regime/control feature; distinguishes funding-driven basis from spot LOP mispricing |
| `open_interest` | Perp OI | Liquidity/activity regime proxy (currently under-populated for spot-only pairs — known pipeline gap being fixed) |
| `withdrawal_status` | Per-network deposit/withdraw enabled flags, fees, min/max limits | Tests whether trapped capital (disabled withdrawals) explains persistent dislocations |
| `exchange_status` | Venue operational status (ok/maintenance) | Rare-event filter for venue outages |
| `spread_matrix` | Precomputed pairwise cross-venue spread | Convenience signal derived from ticker, not new information |

Not yet collected but available via CCXT and under consideration: `fetchLiquidations`
(forced-liquidation events, a proxy for cascade risk), live volume-tiered
`fetchTradingFees` (we currently use a static fee table), and `fetchBidsAsks`
(a lighter-weight batch endpoint relevant to increasing polling frequency on
high-liquidity pairs without hitting rate limits).

## ML Model (Tania)

- **XGBoost** on a feature matrix: rows = minute-level observations, columns = engineered features (prices, order-book, funding rate).
- Chronological **train / validation / test** splits (no shuffle — prevents look-ahead bias).

## Pipeline

```
CEX Venues ──► Live Collectors (1-min) ──► Quotes / Order Book / Funding Rate ──┐
             ► Backfill API ──► Historical OHLCV ──────────────────────────────┤
                                                                               ▼
                                                          HuggingFace Dataset (~168h)
                                                                               │
                                            ┌───────────────────────────────────┤
                                            ▼                                  ▼
                                   Cross-venue spread              Feature matrix (ML)
                                     │           │                        │
                                  Z-score       OU                    XGBoost
                                     │           │                        │
                                     └─────► Unified backtest (same fees/fills) ◄──┘
                                                        │
                                              Paper trading engine
                                           (live order book + funding)
```

## Next Steps

| Priority | Action | Purpose |
|---|---|---|
| 1 | Scale live dataset to **≥ 2× current** (~336 h) | Increase N and regime coverage |
| 2 | Add **months of backfilled 1-min OHLCV** | Breadth study: regime robustness on price-only features |
| 3 | **Ablation**: price-only vs price+book+funding features | Isolates marginal value of microstructure data |
| 4 | Add **linear baseline** (logistic/ridge on same features) | Separates feature value from tree nonlinearity |
| 5 | **Paper trading** the winning strategy live | End-to-end confirmation with real fills |

## Questions for Supervisor

1. **Thesis framing** — is the two-hypothesis structure (H1: trees beat baselines; H2: improvement attributable to microstructure features) well-scoped?
2. **Baseline sufficiency** — OU + z-score + controls adequate, or should we add a linear ML baseline?
3. **Data strategy** — pursue both breadth (months OHLCV) and depth (weeks of live book/funding) as complementary studies, or prioritize one?
