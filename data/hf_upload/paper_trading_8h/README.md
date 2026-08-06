---
license: mit
task_categories:
  - time-series-forecasting
  - other
tags:
  - crypto
  - statistical-arbitrage
  - paper-trading
  - lightgbm
  - cross-exchange-spread
pretty_name: StatArb LightGBM 8h paper-trading sessions
---

# Paper trading sessions (~8h) — LightGBM live campaign

Two live paper-trading runs used for the research paper writeup. Uploaded under
`paper_trading_8h/` in [`SFU-fintech-AI/statarb-crypto-research`](https://huggingface.co/datasets/SFU-fintech-AI/statarb-crypto-research).

| Folder | Original local path | Window (UTC) | Closed trades | Settled DirAcc |
|---|---|---|---|---|
| `july30_lgbm_8h/` | `data/paper_trading/lgbm_8h_20260730` | 2026-07-30 ~8h | 4,901 | 73.1% |
| `july31_lgbm_8h/` | `data/paper_trading/July31st_8_hr` | 2026-08-01 ~live weekend window | 7,973 | 76.9% |

## Contents (per session)

- `config.json` — entry threshold, horizon, z-window, start time
- `summary.json` — closed/open counts, DirAcc, mean PnL proxy
- `trades.jsonl` — settled paper bets (`pnl_proxy = direction × exit_z`)
- `signals*.jsonl` — scored predictions (Jul 31; Jul 30 retains recent shard in report only)
- `metrics_report.csv` — model vs persistence baseline on matched rows
- Jul 31 also includes `portfolio_sharpe_report.json` and baseline comparison reports

## Important caveats

- PnL is a **z-score proxy**, not dollar P&L. No fees, slippage, or fill modeling.
- Short live windows; not capital-normalized or annualized.
- Primary paper session is **Jul 31**; Jul 30 is companion context.

## Upload from this repo

Requires `HF_TOKEN` (write access to the dataset repo):

```bash
export HF_TOKEN=hf_...
python -m experiments.upload_paper_trading_to_hf
```

By default the script opens a **Hugging Face pull request** (does not commit to `main`).
Pass `--direct-to-main` only if you intentionally want a direct Hub commit.
