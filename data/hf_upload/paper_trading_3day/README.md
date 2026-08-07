---
license: mit
tags:
  - crypto
  - paper-trading
  - lightgbm
pretty_name: StatArb LightGBM ~3-day paper-trading campaign
---

# LightGBM ~3-day paper-trading campaign (Aug 4–7)

Live paper run extending the shorter Jul 30 / Jul 31 ~8h sessions. Folder: `paper_trading_3day/` in
[`SFU-fintech-AI/statarb-crypto-research`](https://huggingface.co/datasets/SFU-fintech-AI/statarb-crypto-research).

## Session

- **`aug4_lgbm_3day/`** — Campaign C (~2026-08-04 → 2026-08-07 UTC)
  - Model: 68-feature LightGBM (`statarb_lgbm.txt`), H=1, z-window 300, `|pred|≥0.5`, `max_open=50`
  - **1,013,380** scored predictions · **50,690** closed trades
  - Directional accuracy **79.0%** · mean `pnl_proxy` **+0.837**
  - Trade span ≈ **72.9 hours** (~3 days)
  - Collector run id (local): `20260804_062334`

## Files

| Path | Description |
|---|---|
| `trades.jsonl`, `trades_001.jsonl` | Settled bets (`pnl_proxy = direction × exit_z`) |
| `signals_aug04_0842Z.jsonl` … | Scored predictions; named by first-record UTC time (`augDD_HHMMZ`, military 24h) |
| `summary.json` / `config.json` / `session_config.json` | Session stats and knobs |
| `portfolio_sharpe_report.json` | Hourly / snapshot Sharpe variants (z-proxy) |
| `sim_persistence_hold_report.json` | H=1 vs persistence-hold ablation |
| `dashboard.json` | Orchestrator health snapshot at end of run |
| `METRICS.md` | Paper-facing metrics writeup for this campaign |

## Caveats

- PnL is a **z-score proxy**, not dollars (no fees / slippage / fills)
- Live window is longer than the 8h campaigns but still short vs multi-month literature Sharpes
- Do not annualize the hourly Sharpe as a primary claim
