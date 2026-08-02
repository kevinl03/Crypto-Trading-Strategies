# Handoff: Paper campaign framing & results compilation

**Planning branch:** `paper-planning` (this doc)  
**Live-campaign data branch (historical):** `experiment/paper-trading-live-testing`  
**ICAIF template / paper source branch:** `paper/icaif26-template-update`  
**Purpose of this doc:** Guide a *new* chat that will rewrite the paper using this campaign’s data, figures, and ablation narrative under the current framing.  
**Do not treat this as the paper itself** — it is framing guidance + a source-of-truth index.

---

## Current paper framing (authoritative)

| Item | Value |
|---|---|
| **Working title (new paper)** | Gradient Boosting for High-Frequency Cryptocurrency Cross-Market Spread Prediction in Volatility-Driven Pairs Trading |
| **New LaTeX (to create)** | e.g. `paper/gradient-boosting-cross-market-spread-prediction.tex` on `paper/icaif26-template-update` — **does not exist yet** |
| **Old reference draft** | `paper/stochastic-cross-venue-ohlcv-trading.tex` — OU/OHLCV content; keep under this name |
| **Venue** | ICAIF '26 (ACM sigconf), Milan, Italy — double-blind |
| **Hard limit** | **8 pages total** incl. figures + references; no appendices / supplementary materials |
| **Deadline** | August 9, 2026, 23:59 AOE |
| **Template** | ACM `acmart` v2.19, `\documentclass[sigconf,anonymous,review]{acmart}` — see `paper/SUBMISSION_GUIDE.md` |
| **Clean sample (start here)** | `paper/acm-sample/sigconf-sample.tex` — official ACM sample (isolated from the paper build) |

### Status of the `.tex` files (important)
- `stochastic-cross-venue-ohlcv-trading.tex` is the **old OU / OHLCV paper**. Filename and title match that content — keep it as **reference only**.
- The **new gradient-boosting (LightGBM) ICAIF paper** will be a **separate new `.tex`** created during regeneration (suggested name: `gradient-boosting-cross-market-spread-prediction.tex`).
- **Regenerate** from `paper/acm-sample/sigconf-sample.tex` + this handoff + campaign metrics/figures — do **not** overwrite the stochastic reference draft in place. The `acm-sample/` folder is isolated and does **not** affect compiling either paper.

### What this paper *is*
- **Machine learning:** LightGBM = **gradient boosting** (tree ensemble). Predictions are deterministic given fixed features/model; training can be made reproducible with seeds.
- **Task:** regress / predict the **forward rolling z-score** of the **same-asset cross-exchange spread** (not Geometric Brownian Motion; not OU SDE trading).
- **Features:** ticker mid/BA, L2 orderbook imbalance, trade flow, funding, OI, lagged spread/z — **OHLCV is off** (`LOAD_TABLES['ohlcv']=False`).
- **Policy:** trade only when `|pred| ≥ 0.5`; always compare to a **persistence / rule-based z-score baseline** on matched rows.

### What this paper is *not* (deprecated framing)
- Not “stochastic spread modeling” via Ornstein–Uhlenbeck parameter trading.
- Not an OHLCV-candle primary pipeline (filename/title must not imply OHLCV).
- Not **Geometric Brownian Motion** — do not abbreviate the method as “GBM” in the paper title/body without saying **gradient boosting** / LightGBM (finance audiences read GBM as the SDE). Prefer “gradient boosting” or “LightGBM” in prose.

### Related branches
- **Regenerate** a **new** ACM-compliant GBT draft on `paper/icaif26-template-update` (start from `paper/acm-sample/sigconf-sample.tex`; leave `stochastic-cross-venue-ohlcv-trading.tex` untouched as the old OU reference).
- Keep campaign metrics/figures paths below as the empirical backbone.

---

## Goals for the next conversation (two frames)

### Frame 1 — Problem, unique value, literature contrast
Explain *what is new* relative to typical crypto / equities / daily-horizon literature:
- minute-scale CEX microstructure
- high-volatility asset universe
- previous literature uses the z-score of the spread as a mechanical rule-based signal instead of training a **gradient-boosting** model to predict the future z-score (direction *and* magnitude of short-horizon spread dynamics)
- confidence filter `|pred| ≥ 0.5` as *trade selection*, with ablation vs unfiltered
- live-collected features that are hard to backfill (esp. L2 orderbook)
- LightGBM regression on cross-exchange same-asset spreads (not single-venue cointegrated *different*-asset pairs)

### Frame 2 — Results, methodology, dataset
Document:
- data construction (HF / `cex_unified` windows, Jul 25 split)
- data ingestion and processing pipeline 
- LightGBM hyperparam tuning + boosting 
- two live paper trading campaigns (weekday Jul 30 vs weekend Jul 31)
- metrics to include are Sharpe ratio, win rate, directional accuracy, RMSE, MAE and R²
- limitations (collector crash, warmup, proxy PnL, smallish data size/only one month or so of historical data, cadence ≈110s)
- stay inside the **8-page** ICAIF limit (no appendix dump)

---

## Source-of-truth paths (campaign data; see also template branch)

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

### 2. Mean-reversion / z-score target (not binary classification)
- Cross-exchange **spread** in bps → rolling z-score (`ZSCORE_WINDOW`, `MIN_PERIODS`).
- Supervised target: **`z_{t+H}`** via `groupby(coin,pair).zscore.shift(-HORIZON)`.
- Jul 31 model / notebook: **`HORIZON = 1`** (predict next snapshot’s z).  
  Jul 30 live campaign model: **`HORIZON = 2`** (different protocol — do not pool naively).

**Baseline (must stay crystal clear in the paper):**  
we compare:
- **Model:** `pred → z_{t+H}`
- **Executed trades:** observations where `|pred|>= 0.5`  

Without this comparison, `|pred|≥0.5` DirAcc looks falsely strong because filtering selects high-|z| / high-persistence states.

**Additional Baseline:**
we also tested a plain rule-based z-score signal as a trade criterion, and the results are stored in the repo. we found that our model had comparable sharpe ratio, better dir_acc, and much better proxy pnl. 

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

### 6. LightGBM (gradient boosting) on large tabular microstructure
- Single multi-output-style tabular **gradient-boosting** model with `coin` / `pair` categoricals (say “gradient boosting / LightGBM” in the paper — not bare “GBM”).
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

## Literature Review — Key Papers

**Direct HFT Crypto Pairs / Spread Trading**

1. Fischer, Krauss & Deinert (2019) — Statistical Arbitrage in Cryptocurrency Markets
MDPI: https://www.mdpi.com/1911-8074/12/1/31
Trained a random forest on 40 coins at 1-minute granularity to predict which coin outperforms the cross-sectional median; bought top-3, shorted bottom-3, held for 120 minutes. Found 7.1 bps/day after 15 bps/half-turn costs and Sharpe 2.55, but crucially showed alpha drops from 20.5 bps to 3.8 bps with a 1-minute execution delay and disappears entirely by minute 5. Gap: predicts cross-sectional rank rather than the z-score of a pairwise spread; no orderbook, funding rate, or OI features; not cross-exchange.

2. Fil & Kristoufek (2020) — Pairs Trading in Cryptocurrency Markets
Semantic Scholar: https://www.semanticscholar.org/paper/Pairs-Trading-in-Cryptocurrency-Markets-Fil-Kristoufek/5d311f01c4b1bb7981c05eb5af28ff12d306d7bc
Applied distance and cointegration methods at 5-minute, 1-hour, and daily frequencies on 26 Binance coins; found 11.61% monthly return at 5-minute frequency vs. –0.07% at daily, but results were highly sensitive to transaction cost assumptions. Gap: purely rule-based z-score thresholds, no model predicting z-score; single-exchange only; no microstructure features.

3. Tadi & Kortchemski (2021) — Evaluation of Dynamic Cointegration-Based Pairs Trading in Cryptocurrency Markets
Emerald: https://www.emerald.com/insight/content/doi/10.1108/SEF-06-2020-0235/full/html
Used Engle-Granger and Johansen cointegration with OU half-life calibration on BitMEX 1-minute data; z-score entry/exit with basket trading yielding Sharpe 7.94. Reported total P&L in XBT, number of trades, and OU half-life. Gap: fully rule-based entry/exit on z-score threshold; no predictive model; single exchange (BitMEX perpetuals only).

4. Ko, Lin, Do et al. (2023) — Pairs Trading in Cryptocurrency Markets: A Comparative Study of Statistical Methods
Taylor & Francis: https://www.tandfonline.com/doi/full/10.1080/10293523.2023.2268386
Most comprehensive multi-method benchmark: compared six pair-selection approaches (cointegration, correlation, distance, Hurst exponent, SDE residual, fluctuation behaviour) at 1-min, 5-min, and 60-min on 30 Binance coins in Q1 2022; distance method yielded 208–236% gross total return across frequencies. Reported 11 criteria including total return, Sharpe, MDD, number of trades, average trade duration, and win rate. Gap: rule-based z-score signals only; single exchange; no predictive model; Q1 2022 was unusually volatile, inflating gross returns.

5. Tadi & Witzany (2025) — Copula-Based Trading of Cointegrated Cryptocurrency Pairs
Springer: https://link.springer.com/article/10.1186/s40854-024-00702-7
Tested copula-based signal on 5-minute Binance data across 20 pairs and 104 monthly trading cycles (2021–2023); achieved 35–37% annualized returns and Sharpe ~0.95 after costs, benchmarked against cointegration baseline, return-based copula, and buy-and-hold. Most rigorous cost-inclusive multi-year crypto pairs paper in the set. Gap: rule-based entry/exit; single exchange; no model predicting future spread value or direction.

6. Palazzi (2025) — Trading Games: Beating Passive Strategies in the Bullish Crypto Market
Wiley: https://onlinelibrary.wiley.com/doi/full/10.1002/fut.70018
Cointegration Z-score pairs trading with optimized parameter sweep, adaptive trailing stop-loss and volatility filtering on 10 major cryptos from 2019–2024; 37 of 90 pairs found cointegrated; consistently outperformed passive buy-and-hold with low market exposure. Reported Sharpe, cumulative return, MDD by consensus mechanism. Gap: rule-based; single exchange; no ML model.

7. Tsoku & Makatjane (2026) — Deep Learning-Based Pairs Trading: Real-Time Forecasting of Co-Integrated Cryptocurrency Pairs
Frontiers: https://www.frontiersin.org/journals/applied-mathematics-and-statistics/articles/10.3389/fams.2026.1749337/full
The only crypto paper that trains a model (DNN+LSTM ensemble) to directly forecast spread dynamics on cointegrated crypto pairs (BNB, ETH, LTC, XRP, USDT); uses dynamic Johansen tests, reports RMSE/MAE/MAPE, signal accuracy, and 99% prediction intervals. Closest existing paper to the present work. Gap: lower frequency (not minute-level); single exchange; no orderbook/funding/OI features; no cross-exchange spread; does not report directional accuracy, Sharpe, win rate, or profit-per-trade as primary metrics.

**Adjacent: ML Models Trained to Predict Spread Value or Direction**

8. Shen et al. (2022) — Stock Index Spot–Futures Arbitrage Prediction Using Machine Learning Models
PubMed/MDPI: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9601484/
Best methodological analogue for dual-metric reporting: compared LASSO, XGBoost, BPNN, LSTM on predicting the CSI 300 spot-futures arbitrage spread interval; LSTM achieved R² 92.09%, MAPE 0.70%, RMSE 0.00813, and 58.18% arbitrage trading return; also broke down performance by bull/bear regime. Gap: equity index futures spread (slow-moving, structured), not crypto; R² of 92% is not a realistic target for noisy cross-exchange crypto spreads; no microstructure features.

9. Sarmento et al. (2024) — Machine Learning-Enhanced Pairs Trading
MDPI Forecasting: https://www.mdpi.com/2571-9394/6/2/24
Most methodologically similar paper in terms of frequency and ML approach: applied BiLSTM with attention, Transformer, N-BEATS, N-HiTS, CNN, and TCN to 1-minute Brazilian stock price ratios (price ratio = equivalent of spread); found hybrid reversion+ML yields highest profit-per-trade, with model abstention (skip trades when predicted magnitude is small) further improving quality. Gap: equities not crypto; single exchange (same stock, ON vs PN share class); no orderbook, funding, OI, or cross-venue features; does not report R², Sharpe, or directional accuracy as primary metrics.

10. Liou, Liu & Cheng (2024) — Price Spread Prediction in HFT Pairs Trading Using Deep Learning
ScienceDirect: https://www.sciencedirect.com/science/article/abs/pii/S1057521924007257
Trained deep learning models with XGBoost feature selection on Taiwan intraday LOB data to predict the relationship between price spread and boundaries; model entry/exit signals improved win rate and stable profits vs. rule-based baseline. Gap: equities; single exchange; no R², Sharpe, or directional accuracy reported as headline metrics; no cross-exchange features.

11. Perrone et al. (2026) — Pairs Trading with Time-Series Deep Learning Models
ScienceDirect: https://www.sciencedirect.com/science/article/pii/S2405918826000024
Reformulated statistical arbitrage as a panel-level spread residual prediction task; compared LSTM, Informer, Autoformer, iTransformer, Scaleformer, Chronos, AdaBoost; transformer-based models dominated on Sharpe and AUC; AUC used as threshold-independent directional quality metric. Gap: daily/equity data; no crypto; no microstructure features; Sharpe improvements driven by model choice more than frequency.

12. Han & Li (2024) — LSTM for Arbitrage Spread Optimization
PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC11784865/
Used LSTM not to replace the rule-based signal but as a trade filter — skip predicted-unprofitable trades before execution; significantly outperformed unfiltered CSI 300 strategy in net returns. Most useful framing for a hybrid rule+ML system. Gap: Chinese equity futures; no crypto; filter-only use (not direct spread prediction); single exchange.

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

We use **gradient boosting (LightGBM)** to predict the **next-snapshot z-score** of same-asset **cross-exchange** crypto spreads from lagged spread/z and live CEX microstructure (ticker, orderbook imbalance, trades, funding, open interest — **not OHLCV**). At runtime we paper-trade only high-confidence predictions (`|pred|≥0.5`), settle against realized future z, and always score a **persistence / rule-based z-score baseline on identical rows** so confidence filtering cannot be mistaken for alpha.

---

## Known limitations (must appear in paper revisions)

1. Jul31 collector died early; machine sleep left trader idle on frozen data — quote **useful** hours, not planned 12h.
2. `pnl_proxy = sign(pred) × realized_Δz` — not fees/slippage/$ PnL.
3. Cadence ≠ 60s under full signal load.
4. Warmup: `MIN_PERIODS=90` ⇒ ~2.5–3h before preds on Jul31 protocol.
6. Cross-sectional rows are dependent; do not claim IID daily-equity year-equivalence without caveats. (we never claimed iid - this is stock market data and claiming iid would be frankly ridiculous)
7. Orderbook coverage in the 68-feature model is thinner than the 73-feature Jul30 model (mostly Coinbase imbalance lags + aggregates).
8. **Page budget:** ICAIF '26 is 8 pages total — prioritize Frame 1 novelty + ablation honesty over exhaustive tables.

---

## Checklist for the next chat

1. Open `paper-planning`; read this handoff + `metrics_report.csv` for both sessions.  
2. Pull figures from `docs/paper_campaign_figures/`.  
3. On `paper/icaif26-template-update`, **create a new** GBT `.tex` from `paper/acm-sample/sigconf-sample.tex` + `SUBMISSION_GUIDE.md` using Frame 1/2 and this handoff — leave `stochastic-cross-venue-ohlcv-trading.tex` as the old OU/OHLCV reference.  
4. Keep gradient-boosting / LightGBM wording; avoid OU/OHLCV/“stochastic GBM” titles.  
5. Keep baseline + ablation front-and-center.  
6. Soften any “40% R² / 10 years of S&P” claims into precise, defensible wording.  
7. Fit **≤ 8 pages**; double-blind (`anonymous,review`); no appendices.  
8. Optionally re-run `report_paper_session.py` / figure script after any metric definition changes.  
9. If needed, add Jul30 `signals.jsonl` analysis (folder may lack full signals — trades + metrics exist).

---

## Portfolio Sharpe Ratios — Jul 31 Live Session

*Source branch: `analysis/jul31-sharpe-ratio` · Doc: `docs/results_jul31_live_metrics_lit.md`*

### Sharpe definition

Formula: `Sharpe = mean(x) / std(x)`, Rf = 0.

**Mark-to-market (MTM):** value each still-open bet at current z using the same proxy as settlement (`direction × z_t`). Hourly P&L = change in equity across clock-hour boundaries — lit practice of counting unrealized inventory (cf. Tadi & Kortchemski).

**Portfolio equity at time t:**
```
equity(t) = Σ(closed by t) pnl_proxy + Σ(still open at t)(direction × z_t)
```

### Sharpe variants (Jul 31, `|pred|≥0.5` filtered trades only)

| ID | Series definition | Sharpe | Use in paper |
|---|---|---:|---|
| **A** | Sum of **closed** `pnl_proxy` per clock hour | **2.35** | Ablation |
| **B** | Hourly **equity Δ** including **open MTM** | **2.41** | **Headline** |
| **C** | B ÷ fixed capital `max_open = 50` | **2.41** | Same (constant capital cancels) |
| **C2** | B ÷ live `n_open` each hour | **2.57** | Sensitivity (strongest hourly) |
| **D** | Per-**snapshot** equity Δ + open MTM (n=171, ~109s bars) | **1.70** | Finer-bar robustness |

**Recommended paper headline:** Hourly portfolio Sharpe = **2.41** (variant B, open MTM included).
Report A = 2.35 as closed-only ablation and D = 1.70 as finer-bar robustness.

Per-trade Sharpe (reference only): 0.67 = mean/std of individual trade `pnl_proxy` (n=7,973).

### Hourly buckets — all positive

| Hour end (UTC) | Realized cum. | Open MTM | n_open | Hourly equity Δ |
|---|---:|---:|---:|---:|
| 06:00 | 254.4 | +50.8 | 50 | +305.2 |
| 07:00 | 1043.9 | +23.0 | 40 | +761.7 |
| 08:00 | 2257.9 | +64.4 | 50 | +1255.5 |
| 09:00 | 3572.7 | +71.8 | 50 | +1322.2 |
| 10:00 | 4953.1 | +61.7 | 50 | +1370.3 |
| 11:00 | 5949.7 | 0.0 | 0 | +934.8 |

**Every observed clock hour had positive portfolio P&L.** Total settled `pnl_proxy` ≈ +5,950 z-units over 6 hours.

### Non-claims (critical for judges)

- Do **not** annualize as "Sharpe ≈ 220" as a primary result (n = 6 hours).
- Do **not** claim fee-net profitability — these are gross z-proxy PnL.
- Do **not** claim "beats Tadi 7.94" — different metric construction (multi-month capital returns vs 6h gross z-proxy).
- Naive persistence baseline used for **R²/DirAcc only**, not Sharpe.

### Literature Sharpe comparison (qualitative, definition-aware)

| Paper | Their Sharpe | Comparability |
|---|---|---|
| Tadi & Kortchemski (2021) | 7.94 (portfolio, multi-month, Rf=0) | Closest *construction*; sample/units differ |
| Tadi & Witzany (2025) | ~0.95 (ann., fee-aware, multi-year) | Realistic longer-sample band; fee-net live may land nearer here |
| Fischer et al. (2019) | 2.55 (ann., after 15 bps) | Comparable regime (HF crypto) but alpha dies at 5-min delay |
| Fil & Kristoufek (2020) | N/A (return % reported) | Extreme fee sensitivity matches our proxy warning |

**Framing:** Our **2.41 hourly portfolio Sharpe** is a strong live risk-adjusted result under our proxy definition. Literature Sharpes in the 0.95–7.94 band are mostly longer-sample, capital-return Sharpes. The contribution is occupying the gap: **minute-scale crypto + learned z forecast + dual forecast/trading metrics + live execution**, not a numerical dominance claim.

### Paste-ready Sharpe results paragraph

> In a live paper-trading session on 2026-08-01 UTC (`July31st_8_hr`), a LightGBM model trained to forecast the next-snapshot cross-exchange spread z-score opened positions only when |ẑ| ≥ 0.5. Over 7,973 settled bets, directional accuracy was 76.9% and the coefficient of determination between predictions and realized exit z-scores was 0.35. Forming an hourly portfolio P&L series in z-units—including mark-to-market valuation of open bets as direction × z_t—yields an hourly Sharpe ratio of 2.41 (2.35 using closed trades only; Rf = 0). Every observed clock hour had positive portfolio P&L. These figures are gross of fees and are not annualized; they characterize risk-adjusted stability of the live book under a z-score settlement proxy.

---

## Suggested commit note for maintainers

Keep this handoff on `paper-planning`. Campaign figures/scripts/data stay beside `data/paper_trading/July31st_8_hr` and the Jul30 session folder (historically on `experiment/paper-trading-live-testing`). On `paper/icaif26-template-update`: start from `paper/acm-sample/sigconf-sample.tex` into a **new** GBT `.tex`; keep `stochastic-cross-venue-ohlcv-trading.tex` as the old OU reference.
