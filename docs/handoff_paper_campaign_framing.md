# Handoff: Paper campaign framing & results compilation

**Branch:** `experiment/paper-trading-live-testing`  
**Purpose of this doc:** Guide a *new* chat that will revise an older draft paper using this campaign’s data, figures, and ablation narrative.  
**Do not treat this as the paper itself** — it is framing guidance + a source-of-truth index.

---

## Goals for the next conversation (two frames)

### Frame 1 — Problem, unique value, literature contrast
Explain *what is new* relative to typical crypto / equities / daily-horizon literature:
- minute-scale CEX microstructure
- high-volatility asset universe
- previous literature uses the z-score of the spread as a mechanical rule-based signal instead of training a model to predict the z-score itself or predict the direction and magnitude of spread mean reversion
- confidence filter `|pred| ≥ 0.5` as *trade selection*, with ablation vs unfiltered
- live-collected features that are hard to backfill (esp. L2 orderbook)
- LightGBM regression to predict the direction of magnitude of the z-score of the spread between different coins cross-exchange

### Frame 2 — Results, methodology, dataset
Document:
- data construction (HF / `cex_unified` windows, Jul 25 split)
- data ingestion and processing pipeline 
- LightGBM hyperparam tuning + boosting 
- two live paper trading campaigns (weekday Jul 30 vs weekend Jul 31)
- metrics to include are Sharpe ratio, win rate, directional accuracy, RMSA, MAE and R^2
- limitations (collector crash, warmup, proxy PnL, smallish data size/only one month or so of historical data, cadence ≈110s)

---

## Source-of-truth paths (this branch)

| Artifact | Path |
|---|---|
| Jul 31 live session (rename) | `data/paper_trading/July31st_8_hr/` |
| Jul 30 live session | `data/paper_trading/lgbm_8h_20260730/` |
| Offline model (Jul 25 split) | `statarb/outputs/statarb_lgbm.txt` (+ `eval_results.csv`, `feature_importance.csv`) |
| Training notebook | `statarb/cex_gbm_new.ipynb` |
| Archived Jul 30 training recipe | `statarb/cex_gbm_new_live_8h_july30.ipynb` (H=2 / older protocol) |
| Live trader | `experiments/paper_trade_lgbm.py` |
| Ablation report script | `scripts/report_paper_session.py` |
| Figure generator | `scripts/generate_paper_campaign_figures.py` |
| Figures | `docs/paper_campaign_figures/*.png` |
| Prior writeups | `docs/tickets/lgbm_paper_trading.md`, `docs/paper_trading_lgbm.md` |

Regenerate figures:
```powershell
cd <analysis-worktree>
.\.venv\Scripts\python.exe scripts/generate_paper_campaign_figures.py
.\.venv\Scripts\python.exe scripts/report_paper_session.py data/paper_trading/July31st_8_hr
```

---

## Frame 1 — Unique value proposition (draft talking points)

### 1. High-frequency crypto, centralized exchanges, ~1-minute grain
- We operate on **CEX** multi-venue snapshots (not daily bars / not single-venue retail candles only).
- Nominal collector interval **60s**; with `--slow-every 1` observed wall cadence **~110s/snap** (important for “hours of data” conversions).
- Universe: **23 high-volatility** assets (`VOLATILE_COINS`: BTC, ETH, SOL, memes, DeFi majors, etc.) × up to **6 venues** (binance, bybit, okx, coinbase, kraken, mexc).

### 2. Mean-reversion / OU-style z-score target (not binary classification)
- Cross-exchange **spread** in bps → rolling z-score (`ZSCORE_WINDOW`, `MIN_PERIODS`).
- Supervised target: **`z_{t+H}`** via `groupby(coin,pair).zscore.shift(-HORIZON)`.
- Jul 31 model / notebook: **`HORIZON = 1`** (predict next snapshot’s z).  
  Jul 30 live campaign model: **`HORIZON = 2`** (different protocol — do not pool naively).

**Baseline (must stay crystal clear in the paper):**  
we compare:
- **Model:** `pred → z_{t+H}`
- **Executed trades:** observations where `|pred|>= 0.5`  

Without this comparison, `|pred|≥0.5` DirAcc looks falsely strong because filtering selects high-|z| / high-persistence states.

### 3. Confidence filter as trading policy
- Paper bets only when `|pred| ≥ 0.5` (entry tau).
- Ablation reports **all predictions** vs **entries only**.
- Interpretation for the paper:
  - Filter can raise R² / DirAcc by **selection**.
  - Significance of the *model* is the **gap vs persistence on the same filtered rows**, not the filtered DirAcc alone.

### 4. Scale vs daily literature (careful wording)
- Offline pooled training matrices are large (order **~2.9M** train feature rows after build; many coins×pairs per snap).
- Converting “rows → years of daily S&P” is a **rhetorical analogy**, not a literal sample-size equivalence. Prefer:
  - state number of **snapshots / wall hours of live collection windows**, and
  - number of **coin×pair×time** supervised rows,
  - and note overlapping cross-section dependence (do not claim i.i.d. “10+ years”).

Rough live-collection intuition: multiple multi-day CEX windows (Jun–Jul 2026) at ~1-min grain dwarf a single daily equity series in *observation count*, but dependence structure differs.

### 5. Live / hard-to-backfill features (dataset edge for HF release)
| Signal | In `cex_gbm_new` training? | Historical backfill? |
|---|---|---|
| OHLCV | Off (`LOAD_TABLES['ohlcv']=False`) | Easy |
| funding_rate | On | Often backfillable |
| open_interest | On | Partial / venue-dependent |
| ticker (mid, BA, sizes) | On | Hard (need live snaps) |
| L2 **orderbook imbalance** | On | **Hard** — core live edge |
| trades (flow / volume) | On | Hard |
| spread_matrix | On (derived) | Hard (needs aligned ticker) |
| liquidations / LSR | Collected live; **not** in this GBM recipe | Limited history |

**Market-friction angle:** orderbook **imbalance** (bid- vs ask-heavy) as a continuous regressor for short-horizon z movement — microstructure friction, not a binary classifier.

### 6. LightGBM on large tabular microstructure
- Single multi-output-style tabular GBM with `coin` / `pair` categoricals.
- Lag features + cross-exchange dispersion/ranks + momentum / `zscore_accel`.
- Correlation screen in notebook (`corr_target`; lag inter-correlation heatmap — “covariance/correlation” story for redundancy of lagged z).

---

## Frame 2 — Methodology & results (source numbers)

### Offline training (`cex_gbm_new` → `statarb/outputs/statarb_lgbm.txt`)

**Split:** train = all windows **before 2026-07-25**; test = **Jul 25–28** (`snapshot_idx` cut **3584** on jul22–28 run).

**Protocol constants:**
- `HORIZON=1`, `ZSCORE_WINDOW=300`, `N_LAGS=3`, `MIN_PERIODS=90`
- `NUM_BOOST_ROUND=2500`, `EARLY_STOPPING=250` (best iteration printed **74** in run logs)
- Leaves 255, lr 0.1, `min_child_samples=200`, feature/bagging fractions 0.6/0.7, L1/L2 0.1

**Booster:** **68 features** (imbalance-only OB; no `slippage_bps`).

**Offline `eval_results.csv`:**
| set | R² | DirAcc |
|---|---:|---:|
| test (all) | 0.133 | 0.628 |
| test `\|pred\|>0.5` | 0.383 | 0.784 |
| train (all) | 0.128 | 0.618 |
| train `\|pred\|>0.5` | 0.447 | 0.815 |

### Live campaign A — Jul 30 weekday (Wed→Thu window)
- Model: `outputs_ob_fix` (**73** feats), **H=2**, `ZSCORE_WINDOW=120`
- Collector ~8h; report in `lgbm_8h_20260730/metrics_report.csv`

| set | n | R² | DirAcc |
|---|---:|---:|---:|
| model \| all | 49,310 | **0.054** | **0.553** |
| naive \| all | 49,310 | −0.547 | 0.623 |
| model \| \|pred\|≥0.5 | 5,054 | 0.248 | 0.765 |
| naive \| \|pred\|≥0.5 | 5,054 | −0.045 | 0.754 |

### Live campaign B — Jul 31 weekend (Fri→Sat; folder `July31st_8_hr`)
- Model: `statarb/outputs/statarb_lgbm.txt` (**68** feats), **H=1**, `ZSCORE_WINDOW=300`
- Planned 12h; **useful live data ~8h** (collector `UnboundLocalError` ~snap 264); preds active ~5h after warmup (~90 snaps)
- `metrics_report.csv` (horizon=1):

| set | n | R² | DirAcc |
|---|---:|---:|---:|
| model \| all | 58,995 | **0.104** | **0.609** |
| naive \| all | 58,995 | −0.338 | 0.656 |
| model \| \|pred\|≥0.5 | 8,775 | **0.347** | 0.767 |
| naive \| \|pred\|≥0.5 | 8,775 | 0.174 | **0.770** |

### How to phrase the “improvement” claims (aligned to user intent, with honesty)

**Unfiltered live lift (Jul30 → Jul31 protocols — not a pure ceteris-paribus A/B):**
- Model R² **0.054 → 0.104** (~2× on all live preds).
- Model DirAcc **0.553 → 0.609** (~+5.6 pp absolute; ~10% relative if framed as (0.609−0.553)/0.553).

**Filtered “headline” R²:**
- Offline test filter R² **~0.38**; Jul31 live filter R² **~0.35** (not literally 0.40 — don’t oversell).
- On Jul31 filtered rows, **naive DirAcc ≈ model DirAcc (0.770 vs 0.767)** → filter DirAcc is mostly persistence/selection; model still beats naive on **R²** (0.347 vs 0.174).

**Ablation message for significance:**
1. Show all-pred metrics (weak absolute R², but model ≫ naive R²).
2. Show filtered metrics (high R² / DirAcc).
3. Show **model − trade selections** on matched rows (Fig 6) so reviewers see what is not explained by persistence.

---

## Figures (already generated)

| File | Story |
|---|---|
| `docs/paper_campaign_figures/fig1_ablation_r2_diracc.png` | Model vs naive × filter × campaign |
| `docs/paper_campaign_figures/fig2_pred_vs_realized_jul31.png` | Pred vs realized z scatter |
| `docs/paper_campaign_figures/fig3_filter_r2_lift.png` | Filter lift on model R² |
| `docs/paper_campaign_figures/fig4_feature_importance_top20.png` | GBM importances |
| `docs/paper_campaign_figures/fig5_cum_pnl_proxy_jul31.png` | Cumulative z-unit proxy (not $ PnL) |
| `docs/paper_campaign_figures/fig6_model_minus_naive_diracc.png` | Selection-bias / edge check |

---

## Strategy summary (one paragraph)

We predict the **next-snapshot z-score** of cross-exchange crypto spreads using LightGBM on lagged spread/z, multi-venue ticker microstructure, using the ticker, orderbook imbalance, trade history, funding rate, and open interest data, on a set of  multi-window live CEX snapshots. At runtime we paper-trade only high-confidence predictions (`|pred|≥0.5`), settling against realized future z, and we always score a **persistence baseline on identical rows** so confidence filtering cannot be mistaken for alpha.

---

## Known limitations (must appear in paper revisions)

1. Jul31 collector died early; machine sleep left trader idle on frozen data — quote **useful** hours, not planned 12h.
2. `pnl_proxy = sign(pred) × realized_Δz` — not fees/slippage/$ PnL.
3. Cadence ≠ 60s under full signal load.
4. Warmup: `MIN_PERIODS=90` ⇒ ~2.5–3h before preds on Jul31 protocol.
6. Cross-sectional rows are dependent; do not claim IID daily-equity year-equivalence without caveats. (we never claimed iid - this is stock market data and claiming iid would be frankly ridiculous)
7. Orderbook coverage in the 68-feature model is thinner than the 73-feature Jul30 model (mostly Coinbase imbalance lags + aggregates).

---

## Checklist for the next chat

1. Open this branch; read this handoff + `metrics_report.csv` for both sessions.  
2. Pull figures from `docs/paper_campaign_figures/`.  
3. Rewrite paper sections along **Frame 1** then **Frame 2**.  
4. Keep baseline + ablation front-and-center.  
5. Soften any “40% R² / 10 years of S&P” claims into precise, defensible wording.  
6. Optionally re-run `report_paper_session.py` / figure script after any metric definition changes.  
7. If needed, add Jul30 `signals.jsonl` analysis (folder may lack full signals — trades + metrics exist).

---

## Suggested commit note for maintainers

This handoff + figures + generator script belong on `experiment/paper-trading-live-testing` beside `data/paper_trading/July31st_8_hr` and the Jul30 session folder.
