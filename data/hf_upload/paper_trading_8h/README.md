---
license: mit
tags:
  - crypto
  - paper-trading
  - lightgbm
pretty_name: StatArb LightGBM 8h paper-trading sessions
---

# LightGBM ~8h paper-trading sessions (Jul 30 + Jul 31)

Live paper runs used in the research paper. Folder: `paper_trading_8h/` in
[`SFU-fintech-AI/statarb-crypto-research`](https://huggingface.co/datasets/SFU-fintech-AI/statarb-crypto-research).

## Sessions

- **`july31_lgbm_8h/`** — primary paper session (~2026-08-01 UTC). 7,973 closed trades, 76.9% DirAcc. Includes `trades.jsonl` + `signals*.jsonl`.
- **`july30_lgbm_8h/`** — companion session (2026-07-30). 4,901 closed trades, 73.1% DirAcc. `trades.jsonl` only (no signal shards on disk).

## Files

- `trades.jsonl` — settled bets (`pnl_proxy = direction × exit_z`)
- `signals*.jsonl` — scored predictions (**Jul 31 only**)
- `summary.json` / `config.json` / `metrics_report.csv` — session stats and model vs persistence baseline
- Jul 31 extras: portfolio Sharpe + mechanical-z baseline reports

## Caveats

- PnL is a **z-score proxy**, not dollars (no fees / slippage / fills)
- Short live windows; not capital-normalized or annualized
