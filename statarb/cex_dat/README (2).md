---
license: cc-by-4.0
pretty_name: Cross-Exchange Crypto Stat-Arb Signals (Train + Test)
tags:
- finance
- cryptocurrency
- market-microstructure
- statistical-arbitrage
- order-book
- high-frequency
size_categories:
- 1M<n<10M
configs:
- config_name: ohlcv
  data_files: ohlcv.parquet
- config_name: ticker
  data_files: ticker.parquet
- config_name: orderbook
  data_files: orderbook.parquet
- config_name: trades
  data_files: trades.parquet
- config_name: spread_matrix
  data_files: spread_matrix.parquet
- config_name: funding_rate
  data_files: funding_rate.parquet
- config_name: open_interest
  data_files: open_interest.parquet
- config_name: withdrawal_status
  data_files: withdrawal_status.parquet
- config_name: exchange_status
  data_files: exchange_status.parquet
- config_name: test_ticker
  data_files: test/ticker.parquet
- config_name: test_orderbook
  data_files: test/orderbook.parquet
- config_name: test_trades
  data_files: test/trades.parquet
- config_name: test_spread_matrix
  data_files: test/spread_matrix.parquet
- config_name: test_funding_rate
  data_files: test/funding_rate.parquet
- config_name: test_open_interest
  data_files: test/open_interest.parquet
- config_name: test_withdrawal_status
  data_files: test/withdrawal_status.parquet
- config_name: test_exchange_status
  data_files: test/exchange_status.parquet
- config_name: test_ohlcv_live
  data_files: test/ohlcv_live.parquet
- config_name: validation_ticker
  data_files: validation/ticker.parquet
- config_name: validation_orderbook
  data_files: validation/orderbook.parquet
- config_name: validation_trades
  data_files: validation/trades.parquet
- config_name: validation_spread_matrix
  data_files: validation/spread_matrix.parquet
- config_name: validation_funding_rate
  data_files: validation/funding_rate.parquet
- config_name: validation_open_interest
  data_files: validation/open_interest.parquet
- config_name: validation_withdrawal_status
  data_files: validation/withdrawal_status.parquet
- config_name: validation_exchange_status
  data_files: validation/exchange_status.parquet
---

# Cross-Exchange Crypto Statistical-Arbitrage Signals

Multi-signal cross-exchange snapshots for cryptocurrency statistical-arbitrage
research, collected via [ccxt](https://github.com/ccxt/ccxt) REST APIs across 12
exchanges and 23 volatile assets at a ~60s cadence.

This repo holds chronologically separated **train** and **test** splits for
time-series evaluation. A **validation** split will be added after a deliberate
collection break.

## Unified schema

Every signal subset shares the same columns:

| Column | Type | Notes |
|---|---|---|
| `run_id` | string | source run id |
| `ts` | string | UTC ISO-8601 collection time |
| `snapshot_idx` | int64 | monotonic snapshot counter |
| `exchange` | string | venue (null for `spread_matrix`) |
| `coin` | string | asset |
| `market` | string | ccxt unified symbol |
| `symbol` | string | perp symbol (funding/OI) |
| `error` | string | non-null if that fetch failed this snapshot |
| `payload` | string | full original JSON record (lossless) |

```python
from datasets import load_dataset
train = load_dataset("SFU-fintech-AI/statarb-crypto-research", "spread_matrix", split="train")
test  = load_dataset("SFU-fintech-AI/statarb-crypto-research", "test_spread_matrix", split="train")
```

---

## Train split (root-level Parquet files)

First contiguous collection window before a host outage interrupted the machine.

- **Window:** 2026-06-13 23:48 UTC → 2026-06-16 19:34 UTC (~2.83 days)
- **Run id:** `20260613_234819`
- **Snapshots:** 1–3909
- **Rows:** ~3.18M
- **Assets (23):** BTC, ETH, SOL, DOGE, XRP, ADA, AVAX, CRV, LDO, UNI, AAVE, ARB, OP, PEPE, WIF, BONK, FLOKI, SHIB, WLD, SEI, SUI, TIA, ENA
- **Exchanges (12):** Binance, Kraken, KuCoin, Bybit, OKX, Gate.io, MEXC, HTX, Bitget, Crypto.com, Coinbase, Phemex

| Subset | Rows | Size |
|---|---|---|
| `ticker` | 1,051,180 | 75 MB |
| `orderbook` | 793,329 | 253 MB |
| `trades` | 1,051,285 | 245 MB |
| `spread_matrix` | 89,907 | 61 MB |
| `funding_rate` | 75,070 | 1.3 MB |
| `open_interest` | 57,476 | 1.0 MB |
| `withdrawal_status` | 62,123 | 0.3 MB |
| `exchange_status` | 1,954 | <0.1 MB |

### Train OHLCV — `ohlcv` subset

True 1-minute OHLCV backfilled for the train window (9 deep-history venues).

- **Rows:** ~808k  •  **Venues:** binance, bitget, bybit, coinbase, cryptocom, htx, kucoin, mexc, okx
- **Not covered:** kraken, gateio, phemex (history too shallow for this window)

---

## Test split (`test/*.parquet`)

Out-of-sample window collected after a deliberate break from train. Volatile
asset universe matches train exactly.

- **Window:** 2026-06-22 03:10 UTC → 2026-06-24 15:10 UTC (~60 h)
- **Run id:** `20260622_031058`
- **Snapshots:** 1–3457
- **Rows:** ~2.81M
- **Assets / exchanges:** same 23 coins × 12 venues as train

| Subset | File | Rows | Size |
|---|---|---|---|
| `test_ticker` | `test/ticker.parquet` | 929,011 | 66 MB |
| `test_orderbook` | `test/orderbook.parquet` | 701,342 | 223 MB |
| `test_trades` | `test/trades.parquet` | 929,384 | 215 MB |
| `test_spread_matrix` | `test/spread_matrix.parquet` | 79,511 | 55 MB |
| `test_funding_rate` | `test/funding_rate.parquet` | 66,423 | 1.2 MB |
| `test_open_interest` | `test/open_interest.parquet` | 50,856 | 0.9 MB |
| `test_withdrawal_status` | `test/withdrawal_status.parquet` | 54,876 | 0.3 MB |
| `test_exchange_status` | `test/exchange_status.parquet` | 1,730 | <0.1 MB |

### Test live OHLCV — `test_ohlcv_live` subset

1-minute OHLCV for **Kraken** and **Gate.io** captured live during the test
window (these venues cannot be backfilled). Includes a ~72-minute gap from host
sleep (2026-06-24 15:11–16:23 UTC) that was patched via historical API backfill
(rows tagged `source=backfill` in the file).

- **File:** `test/ohlcv_live.parquet`
- **Rows:** ~466k  •  **Venues:** kraken, gateio  •  **Phemex:** ticker-only (API limitation)

Columns: `run_id, exchange, coin, symbol, timestamp, datetime, open, high, low, close, volume, source, fetched_at`.

---

## Validation split (`validation/*.parquet`) — ⚠️ in progress, partial

A third chronologically-separated window, collected starting **2026-07-13
05:06 UTC**, intended as a held-out validation split distinct from both train
and test. **The collection run is still in progress at the time of this
upload** — the files below cover only snapshots 1–1420 of the run so far.
This section and the data will be updated in place once collection completes;
check `provenance.generated_utc` in each file's origin or the commit history
for the freshest state.

- **Window (partial):** 2026-07-13 05:06 UTC → 2026-07-15 02:19 UTC (~45 h elapsed)
- **Run id:** `20260713_050603`
- **Snapshots:** 1–1420 (partial; target ~2,600+)
- **Rows (partial):** ~1.12M
- **Assets / exchanges:** same 23 coins × 12 venues as train/test
- **Known gaps:** the collection process was interrupted and resumed three
  times (host sleep, terminal/session exits); cumulative gaps total roughly
  6 hours. Gaps show up as missing `snapshot_idx` values rather than corrupted
  rows — no partial/malformed records were produced.

| Subset | File | Rows (partial) |
|---|---|---|
| `validation_ticker` | `validation/ticker.parquet` | 371,986 |
| `validation_orderbook` | `validation/orderbook.parquet` | 280,841 |
| `validation_trades` | `validation/trades.parquet` | 369,864 |
| `validation_spread_matrix` | `validation/spread_matrix.parquet` | 31,847 |
| `validation_funding_rate` | `validation/funding_rate.parquet` | 26,278 |
| `validation_open_interest` | `validation/open_interest.parquet` | 20,162 |
| `validation_withdrawal_status` | `validation/withdrawal_status.parquet` | 22,076 |
| `validation_exchange_status` | `validation/exchange_status.parquet` | 700 |

Schema is identical to train/test (verified column-for-column). A backfilled
1-minute OHLCV subset (`validation_ohlcv`, matching the train `ohlcv` config)
will be added once the run completes.

---

## Provenance & limitations

- Collected with `experiments/collect_statarb_data.py`, converted via
  `experiments/export_run_to_parquet.py` (zstd).
- ~2% of test `orderbook` rows are `error` placeholders for venues without L2
  for a given pair — filter on `error IS NULL`.
- Stablecoin data from an accidental resume is published separately at
  `SFU-fintech-AI/statarb-crypto-stablecoins`.
