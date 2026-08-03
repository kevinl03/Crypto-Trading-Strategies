# Dataset Signal Uniqueness Analysis

Which signals in our CEX stat-arb dataset are **exclusive to live collection** versus reconstructable from freely available historical data? This analysis informs how we frame the Hugging Face dataset's novelty.

## Signal Classification

### Tier 1 — Truly Irreproducible

These signals have **zero historical API endpoints** on any exchange. If you weren't recording them live, the data is gone forever.

| Signal | What We Collect | API Evidence |
|--------|----------------|--------------|
| **L2 Order Book** | `imbalance`, `bid_depth`, `ask_depth`, `bid_vwap`, `ask_vwap`, `slippage_bps` across 6 CEXs at 30s cadence | Binance has no historical depth replay endpoint. A whitelisted-only bulk download exists for Futures `T_DEPTH`, but it's gated, incomplete (`S_DEPTH` covers only BTCUSDT), and requires special access. No equivalent on OKX, Bybit, Coinbase, Kraken, or MEXC. Paid vendors (Tardis, Kaiko) charge thousands/month. |
| **Withdrawal/Deposit Status** | Per-coin, per-chain `depositEnable`/`withdrawEnable` flags via `fetchCurrencies()` → `GET /sapi/v1/capital/config/getall` | Returns current state only. No exchange offers a historical endpoint to query past enable/disable state. This data simply does not exist anywhere once the moment passes. |
| **Exchange Operational Status** | `status` (ok / maintenance / error) via `fetchStatus()` → `GET /sapi/v1/system/status` | Returns current state only. Binance publishes retrospective uptime blog posts but has no API for historical maintenance windows. |

### Tier 2 — Practically Irreproducible (Shallow Rolling Windows)

These signals have limited historical endpoints with **30-day rolling windows**. Our archive extending beyond that window is exclusive.

| Signal | What We Collect | API Evidence |
|--------|----------------|--------------|
| **Open Interest** | `oi_amount`, `oi_value` per exchange per perp symbol | `GET /futures/data/openInterestHist` — Binance docs: *"Only the data of the latest 30 days is available."* Minimum period granularity is 5m. |
| **Long/Short Ratio** | `long_short_ratio` (aggregate account positioning) per exchange | `GET /futures/data/globalLongShortAccountRatio` — **30-day max history** confirmed. Requests for older timestamps return errors. |
| **Liquidations (USDT-M)** | Side, price, contracts, quote value from public forced-liquidation feed | `GET /fapi/v1/forceOrders` is user-specific and geo-blocked. Public WebSocket `!forceOrder@arr` is real-time only. Historical CSV snapshots at data.binance.vision were removed for USDT-M; only partially restored for COIN-M. |

### Tier 3 — Reconstructable With Effort

These signals can be rebuilt from publicly available historical data sources, but our dataset provides them pre-synchronized across 6 exchanges at 30-second cadence.

| Signal | What We Collect | API Evidence |
|--------|----------------|--------------|
| **Trade Flow / Buy-Sell Ratio** | `buy_sell_ratio`, `buy_volume`, `sell_volume`, `total_volume` per exchange | Binance provides bulk historical `aggTrades` downloads at data.binance.vision with `isBuyerMaker` field going back years. REST `GET /api/v3/aggTrades` supports `startTime`/`endTime` pagination. **However**, coverage varies across other exchanges and synchronizing 6 venues at 30s cadence from raw trade tapes is significant engineering effort. |
| **Funding Rate** | `funding_rate` per exchange per perp symbol | `GET /fapi/v1/fundingRate` supports `startTime`/`endTime` pagination with limit 1000/request. **Full history back to contract listing** — no 30-day cap. Binance also provides bulk monthly `fundingRate` files. Similar endpoints exist on Bybit and OKX. |
| **Top-of-Book Volumes** | `bid_volume`, `ask_volume` from ticker REST | Partially approximable from OHLCV high-low range, but true BBO queue sizes require live ticker snapshots. No bulk historical ticker REST data exists, though 24h stats (volume, VWAP) are embedded in OHLCV. |
| **OHLCV / Spread / Z-scores** | Candle data, cross-venue spreads, rolling z-scores, OU parameters | Freely available everywhere. OHLCV from all exchanges via REST with full pagination. Spread/z-score/OU signals are pure computations on price series. |

## Usage in Production Models

| Signal | In Production LGBM (73 features)? | Notes |
|--------|-----------------------------------|-------|
| L2 Orderbook (`imbalance`) | **Yes** | Used as `ob_imbalance_lag{1,2}_{exchange}` |
| Top-of-Book Volumes (`bid_volume`, `ask_volume`) | **Yes** | Used as `tk_bid_volume_lag{1,2}_{exchange}` |
| Trade Flow (`buy_sell_ratio`, `total_volume`) | **Yes** | Used as `tr_buy_sell_ratio_lag{1,2}_{exchange}` |
| Funding Rate | **Yes** | Used as `fr_funding_rate_lag{1,2}_{exchange}` |
| Open Interest | **Yes** | Used as `oi_oi_amount_lag{1,2}_{exchange}` |
| **Withdrawal/Deposit Status** | **No** — legacy `build_features.py` only | Not in LGBM pipeline; untapped |
| **Exchange Status** | **No** — legacy only | Not in LGBM pipeline; untapped |
| **Long/Short Ratio** | **No** — ablation only (`run_improve_experiments.py`, jul22-28) | Not in production model |
| **Liquidations** | **No** — ablation only | Not in production model |
| OHLCV | **No** — explicitly disabled (`LOAD_TABLES["ohlcv"] = False`) | Incomplete cross-window coverage |

## Implications for Hugging Face Dataset Framing

### Headline Differentiator

The dataset captures **synchronized multi-exchange microstructure** — L2 orderbook imbalance, top-of-book volumes, and trade flow across 6 CEXs at 30-second cadence. This is the core signal class that drives our production model and cannot be obtained from standard historical data providers.

### Untapped Research Value

Four signal types are collected but not yet exploited in any production model:
- **Withdrawal/deposit status** (Tier 1) — when withdrawals are disabled on one exchange, arbitrage becomes one-directional. This is a real alpha signal with zero historical record anywhere.
- **LSR** (Tier 2) — aggregate positioning data with only 30-day API history.
- **Liquidations** (Tier 2) — cascade/forced-selling proxy with unreliable historical access.
- **Exchange status** (Tier 1) — maintenance events create dislocations.

These represent exclusive research opportunities for the community.

### Key Message

> Standard crypto datasets provide OHLCV candles. This dataset captures what happens *between* the candles — the microstructure, order flow, and operational state that drive short-term price dynamics and are lost the moment they occur.

## Verification Methodology

Signal classifications were verified against actual exchange API documentation (August 2026):

- **Binance**: REST API docs, `binance-public-data` GitHub repo, data.binance.vision bulk downloads
- **CCXT**: Unified API mappings for `fetchCurrencies`, `fetchStatus`, `fetchOrderBook`, `fetchTrades`, `fetchLongShortRatioHistory`
- **DexScreener**: API docs confirm snapshot-only with zero historical endpoints (excluded from dataset — not used in models)
- **StackOverflow/GitHub Issues**: Community confirmation that no exchange provides historical L2 depth replay
