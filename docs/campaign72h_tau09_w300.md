# 72h Campaign — W=300, τ=0.9 vs protocol τ=0.5

**Session:** Aug 4–7 live paper campaign (`5day_Aug4_2026` / HF PR `#4` `paper_trading_3day/aug4_lgbm_3day`)  
**Model:** production 68-feat LightGBM, **W=300**, H=1  
**Live entry:** τ=0.5 (`max_open=50`) · **This note:** post-hoc `|pred| ≥ τ` on the **50,690** settled closes  

Settlement: `pnl_proxy = direction × exit_z`.  
R² below is `r2_score(exit_z, pred)` on the filtered closes.

---

## Headline (τ = 0.9 @ W = 300)

| Metric | τ = 0.5 (live) | **τ = 0.9** | Change |
|---|---:|---:|---|
| Closed trades | 50,690 | **12,795** | **−74.8%** (keep 25.2%) |
| DirAcc | 79.0% | **86.7%** | +7.7 pp |
| R² (pred vs exit_z) | 0.439 | **0.599** | +0.160 |
| Mean pnl_proxy | +0.837 | **+1.372** | +64% |
| Sharpe / trade | 0.746 | **1.077** | crosses ≥ 1 |
| Hourly Sharpe A | 4.38 | 1.83 | lower (fewer fills / hour) |
| Total pnl_proxy | +42,447 | +17,558 | −58.6% z-mass |
| Hours with pnl > 0 | 74/74 | 74/74 | unchanged |

### Trade count / risk framing

Raising the filter from **τ=0.5 → τ=0.9** cuts the book to about **one quarter** of live closes (**−75% trades**, not ~50%). That is intentional abstention: fewer bets, higher per-trade signal quality (DirAcc, R², mean pnl, Sharpe/trade ≥ 1), and less aggregate z-proxy exposure (total pnl mass down ~59%). Hourly Sharpe A falls because the hourly P&L series is thinner—not because per-trade risk-adjusted quality worsens.

---

## Nearby τ reference (same closes)

| τ | n | % of τ=0.5 | DirAcc | R² | mean pnl | Sharpe/trade |
|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 50,690 | 100% | 79.0% | 0.439 | +0.837 | 0.746 |
| 0.75 | 20,866 | 41% (−59%) | 84.5% | 0.546 | +1.166 | 0.954 |
| **0.90** | **12,795** | **25% (−75%)** | **86.7%** | **0.599** | **+1.372** | **1.077** |
| 1.00 | 9,390 | 19% (−81%) | 88.0% | 0.635 | +1.511 | 1.167 |

---

## Reproduce

Artifacts: `data/paper_trading/5day_Aug4_2026/tau09_w300_report.{json,csv}`  
Source trades: HF dataset PR [`#4`](https://huggingface.co/datasets/SFU-fintech-AI/statarb-crypto-research/discussions/4) (`refs/pr/4`; not yet on `main` at time of this note).

```bash
# After trades*.jsonl are local under data/paper_trading/5day_Aug4_2026/
.venv/Scripts/python.exe -c "..."  # see tau09_w300_report.json generation in session
```

**Caveat:** live capacity fill used τ=0.5. Post-hoc τ=0.9 is a high-confidence **subset** of that book, not a full re-sim with `max_open=50` under τ=0.9 entry (which would admit some lower-|pred| slots differently). Directionally the quality lift matches Jul31.
