# ~72h / ~3-day Live Campaign — Metrics for Paper Revision

**Session:** `data/paper_trading/5day_Aug4_2026`  
**Collector run:** `data/statarb/20260804_062334`  
**Branch:** `feature/3day-paper-trading`  
**Model:** `statarb/outputs/statarb_lgbm.txt` (68 feats, H=1, z-window 300, `|pred|≥0.5`, `max_open=50`)

This note compiles **what the paper can claim** from the longer live run (vs the prior ~8h Jul 30/31 campaigns), plus how to answer reviewer feedback. Settlement is the z-unit proxy (`direction × z_exit`), not dollar P&L.

---

## 1. Do we have all the data?

**Yes — full paper-trading artifacts are available.**

| Artifact | Status | Notes |
|---|---|---|
| `trades.jsonl` + `trades_001.jsonl` | ✅ ~20 MB | **50,690** closed trades |
| `signals.jsonl` … `signals_020.jsonl` | ✅ ~211 MB | **1,013,380** scored predictions |
| `summary.json` | ✅ | End-of-session trader stats |
| `portfolio_sharpe_report.json` | ✅ | Mid-run (~60h / 41k closes); see §3 |
| `sim_persistence_hold_report.json` | ✅ | Hold ablation on ~815k signals mid-run |
| Collector `spread_matrix` / `ticker` / `orderbook` / … | ✅ | Continuous through session end (~snap 3030); multi-day JSONL |

**Timeline (UTC) — verified from `trades*.jsonl`:**
- First trade: `2026-08-04T08:42:13Z` (after ~90-snap z warmup)
- Last exit: `2026-08-07T09:36:28Z`
- **Trade span ≈ 72.9 hours (~3.0 days)** of active paper trading  
- `config.json` `started_at`: `2026-08-04T15:40:14Z` (trader config write; collector/warmup preceded trading)

**Caveats on completeness:**
- 50 orphan opens at session end never appear in `trades.jsonl` (same as Jul 31).
- Persistence sim was run on a mid-session signal cut (~40.9k live closes / ~815k signals), not the final 50.7k — directionally unchanged; re-run if you need exact final-cut ablation.

---

## 2. Headline metrics the paper can use (H=1 live book)

Replace / extend the Jul 31 ~8h tables with a **Campaign C (~72h / ~3-day)** block:

| Metric | Jul 31 (~8h / ~6 active h) | **Aug 4–7 (~72.9h trade span)** |
|---|---|---|
| Predictions scored | ~80k (session-dependent) | **1,013,380** |
| Closed trades (`|pred|≥0.5`, cap 50) | 7,973 | **50,690** |
| Directional accuracy | 76.9% | **78.99%** |
| Mean `pnl_proxy` (= `direction × z_exit`) | +0.75 | **+0.837** |
| Median \|pred\| on entries | — | **0.693** |
| Coins / exchange-pairs traded | — | **23 / 15** (verified) |
| Closed-only hourly Sharpe (z-proxy) | 2.35 (A) / **2.41** (B w/ MTM) | **4.38** (A full) / **4.10** (mid A/B) |
| Hours with positive hourly z-proxy PnL | 6/6 | **74/74** (verified; no hour gaps) |

**Settlement definition (unchanged, publishable as proxy):**
\[
\mathrm{pnl\_proxy} = \mathrm{direction} \times z_{\mathrm{exit}},\quad
\mathrm{direction} = \mathrm{sign}(\hat{z}),\quad H=1.
\]
Matches the training target (gross z-unit settlement, not dollars).

**Suggested paper one-liner (literature-safe):**
> In a ~73h live paper session (50,690 settled bets; 1.01M scored predictions; 23 coins × 15 venue-pairs), the LightGBM H=1 policy achieved 79.0% directional accuracy and mean z-unit settlement +0.84. Closed-only hourly portfolio Sharpe was 4.38 (Rf=0; every one of 74 clock hours positive). These are **gross z-proxy** diagnostics of live book stability—not annualized strategy Sharpes and not fee-net P&L—so they are not numerically comparable to longer-sample literature Sharpes (e.g. Tadi & Kortchemski’s 7.94 with Rf=0 on ~1y BitMEX pairs P&L).

---

## 3. Portfolio Sharpe (what to report)

**Verified recompute** (sum `pnl_proxy` by UTC exit hour over all 50,690 closes): **74/74 hours > 0**, closed-only hourly Sharpe **A = 4.3845**. Mid-run artifact `portfolio_sharpe_report.json` (41,017 closes / 60 hours) matches A/B/C ≈ **4.10**, D ≈ **1.19**.

| Variant | Definition | Value |
|---|---|---|
| **A** Closed-only hourly | Sum of closed `pnl_proxy` per UTC hour | **4.10** (mid) / **4.38** (full 74h) |
| **B** Hourly equity + open MTM | Δ equity with `direction × z_t` on opens | **4.10** (mid-run report) |
| **C** B / `max_open=50` | Capital-normalized in slot units | **4.10** (mid) |
| **D** Per-snapshot equity | Finer bar (~89s) | **1.19** (mid) |

**Publishing guidance (extend Jul 31 lit notes in `docs/results_jul31_live_metrics_lit.md`):**
- Prefer headline **A = 4.38** (full book, closed-only) plus mid **B ≈ 4.10** if discussing open MTM.
- **Do not annualize** (`4.38 × √(24×365)` is not a claim).
- **Do not** equate to Tadi & Kortchemski’s ~1y annualized pairs Sharpe (7.94, Rf=0), Fil & Kristoufek’s cost-sensitive backtests, or Tadi & Witzany’s ~2y fee-aware annualized Sharpes (~0.93–0.97).
- Use **74/74** as short-window *stability* evidence (scales Jul 31’s 6/6), not as a multi-month win-rate claim.

---

## 4. Persistence hold ablation (buy/sell vs buy/hold/sell)

Offline sim on session signals (`scripts/paper_trading_3day/sim_persistence_hold.py`):

**Rule:** enter when `|pred|≥0.5`; stay open while `sign(pred)==position` and `|pred|≥τ`; min hold 1 snap; `max_open=50`.

| Policy | n closed | Mean hold (snaps) | DirAcc | Mean pnl_proxy | Hourly Sharpe (closed) | Per-trade Sharpe |
|---|---:|---:|---:|---:|---:|---:|
| **LIVE H=1** | 40,932* | 1.0 | **0.790** | **+0.838** | 4.03 | **0.76** |
| H=1 sim (\|pred\| rank fill) | 108,927 | 1.0 | 0.815 | +0.852 | 3.73 | 0.80 |
| **Persist max_hold=20** | 33,295 | **3.3** (p50=2, p90=7) | 0.669 | +0.395 | 4.17 | 0.39 |
| Persist no max | 32,484 | 3.3 | 0.662 | +0.369 | 3.85 | 0.37 |

\*Sim cut mid-run; final live book is 50,690 — same qualitative ranking.

### Insight for the paper
- Holding **does not help** forecast-aligned metrics: DirAcc −12pp, mean z-proxy halves, per-trade Sharpe halves.
- Median hold is only **2 snaps** — the `|pred|≥τ` gate dies quickly; you do not get long “ride” trades.
- Hourly Sharpe is roughly flat (~4.0–4.2) because fewer, longer trades still pack positive hours — not evidence the overlay improves the learned one-step forecast.
- Correct framing: **ablation showing H=1 is the right settlement for this trained target**; persistence hold is a trading overlay that degrades signal quality.

**Suggested ablation sentence:**
> Extending holds while `sign(pred)` agrees and `|pred|≥0.5` (mean hold 3.3 snapshots) reduces directional accuracy from 79% to 67% and mean z-proxy from +0.84 to +0.40. We therefore retain H=1 settlement as the primary evaluation of the learned one-step forecast.

---

## 5. Literature framing (how to read 4.38 / 74/74 / 79%)

Same discipline as Jul 31 (`docs/results_jul31_live_metrics_lit.md` §6): compare **definitions**, not raw numbers.  
Numeric headlines below were checked against the PDFs in `literature/` (abstracts in `literature/README.md`; Sharpe/delay figures from paper bodies where needed).

| Dimension | Campaign C (this run) | Typical HF crypto pairs lit |
|---|---|---|
| Horizon of evidence | **~73 live hours** (74 clock hours with closes) | Weeks to ~1–2 years in these peers |
| PnL unit | Gross **z-proxy** (`dir × z_exit`) | Currency / strategy return % |
| Costs | No fees / slippage / fills | Often fee-sensitive or fee-inclusive |
| Aggregation | Hourly book P&L (variant A/B) | Daily/monthly / annualized returns |
| Annualized? | **No** (primary) | Usually yes when Sharpe is quoted |
| Signal type | **ML predicts** \(z_{t+1}\) + `\|pred\|≥0.5` | Mostly mechanical cointegration / distance / z rules (ML peers differ in task) |

**What Campaign C adds vs Jul 31 (2.41 / 6/6):** longer live path — DirAcc holds (~77% → ~79%), hourly Sharpe stays strong under the *same proxy*, and every clock hour remains green over **12×** more hours. That answers “was the 8h window a fluke?” — it does **not** license “we beat Tadi 7.94.”

| Paper | Verified headline (from PDF) | How Campaign C should be used |
|---|---|---|
| **Tadi & Kortchemski (2021)** | Minute BitMEX data ~2018-09 → 2019-10; scenario-2 **Sharpe 7.94** with **Rf=0** on strategy P&L (they also discuss realized vs unrealized P&L for MDD). Abstract: dynamic cointegration / OU half-life basket pairs beats buy-and-hold. | Closest *construction* cousin (portfolio/strategy P&L Sharpe, Rf=0, longer sample). Cite methodologically; **do not** rank our hourly z-proxy 4.38 vs their 7.94. |
| **Tadi & Witzany (2025)** | Copula cointegrated pairs; Table VI optimal cases ≈ **35–37% annualized return**, **Sharpe ≈ 0.93–0.97** (EG 0.97 / KSS 0.93 at \(\alpha_1=0.10\)); sample ~2021-01 → 2023-01; footnotes: fees included in reported calcs. Abstract: beats buy-and-hold on profitability and risk-adjusted returns. | Longer-sample, fee-aware annualized band. Our 4.38 is still short-window gross z-proxy. |
| **Fil & Kristoufek (2020)** | Distance + cointegration on 26 Binance coins at 5m / 1h / daily. Abstract: strategies **underperform classical benchmarks** overall, but results are **sensitive to parameters, transaction costs, and execution windows**; daily distance −0.07% monthly vs **+11.61% monthly at 5-minute**. | Use for cost/execution sensitivity — not as a Sharpe peer number. Keep gross z-proxy vs fee-net economics separate. |
| **Fischer, Krauss & Deinert (2019)** | RF predicts whether a coin beats the **cross-sectional median over the next 120 min**; top-3/flop-3, reverse after 120 min; OOS **7.1 bps/day after 15 bps half-turn costs**. Body Table 3: round-trip alpha falls with execution delay and **vanishes by minute \(t+5\)**. | Delay / execution peer, **not** a spread-z forecast peer. Our H=1 hold is ~1–2 min; report a delayed-entry ablation before claiming robustness to their setting. |
| **Ko, Lin, Do et al. (2023)** | Comparative study (distance / corr / coint / SDR / GA / NSGA-II) on 26 Binance coins at 1/5/60 min over **79 days** (2018-01-11 → 03-31). Abstract: **NSGA-II best at 2.84% average return**; SDR 1.63%; correlation −0.48%. | Method bake-off peer for pair selection — **not** a “huge return” benchmark (older notes’ 208–236% figure is **not** in this paper). |
| **Tsoku & Makatjane (2026)** | Dynamic Johansen cointegration + **Dynamic Weighted Ensemble of DNN and LSTM** to **forecast spread dynamics** on BNB/ETH/LTC/XRP/USDT. Abstract: only dynamically coherent pairs suit mean-reversion; ensemble best predictive accuracy. | Closest *task family* (learned spread forecast). Differentiator: minute CEX cross-exchange microstructure, explicit \(z_{t+1}\) target, `\|pred\|≥0.5` policy, live paper campaigns. |

**DO NOT CLAIM:** annualized Sharpe; fee-net profitability; “beats 7.94”; that 74/74 generalizes beyond this ~3-day window; that Ko reports triple-digit returns.

**Framing sentence for the paper:**
> Relative to mechanical high-frequency crypto pairs-trading studies, Campaign C’s contribution is a **learned** next-snapshot z forecast evaluated **live** over ~3 days with dual forecast/trading metrics (DirAcc, mean z-proxy, hourly portfolio Sharpe under a z-settlement proxy)—not numerical dominance on annualized net strategy Sharpe.

---

## 6. Mapping to reviewer feedback

Feedback appears aimed at the GBM paper (`paper/gradient-boosting-cross-market-spread-prediction.tex` + `paper/sections/*`), not only the older OU draft.

| # | Feedback | What to do with this ~72h material |
|---|---|---|
| **1** | More sophisticated baselines (Sect. 2.2 peers) | Keep capacity-matched **mechanical persistence / mean-reversion** on Jul 31; **re-run the same baselines on Campaign C signals** for a 3-day table. Optionally add Fischer-style delayed-execution baseline (see #5). |
| **2** | Sensitivity on \(w\) (z-window) | Still needed offline (train/eval with \(w\in\{120,200,300,400\}\`). Live campaign used \(w=300\); this ~72h run does **not** replace a \(w\) sweep — flag as required offline experiment. |
| **3** | Motivate via crypto-market challenges | Use live evidence: non-backfillable LOB/trades, cross-venue latency, 24/7 continuity over ~3 days, minute-scale same-asset dislocations. |
| **4** | Novelty beyond problem setting | Emphasize **method**: learned \(\hat{z}_{t+1}\) + confidence filter + dual metrics + **live** microstructure panel; Table `tab:positioning` already helps — strengthen “what the model does that rules cannot” (nonlinear OB×spread interactions), not only cross-exchange. |
| **5** | Differentiation from Fischer [2] / 1-min delay | Acknowledge their harder delayed-execution setting. Add: (a) we are **spread z forecast**, not cross-sectional rank; (b) report a **delayed-entry ablation** (act on \(\hat{z}_t\) at \(t+1\)) on Campaign C; (c) do **not** claim we beat their delay experiment until that ablation exists. |
| **6** | Remove dataset/repo links (double-blind) | Submission build already uses `\anonsubmission` / withheld URLs in `gradient-boosting-cross-market-spread-prediction.tex`. Ensure camera-ready/public build is separate; strip any remaining identifying `\url` / email in anonymous PDF. |

---

## 7. Recommended paper edits (Campaign C)

1. **Experimental setup:** add Campaign C (~2026-08-04 → 08-07, ~72.9h trade span, same model as Jul 31).
2. **Results table:** DirAcc / mean pnl_proxy / n_closed / hourly Sharpe A=4.38 / 74/74 green hours; note scale-up from 8h → ~3 days.
3. **Baselines:** regenerate mechanical persistence + mean-reversion under `max_open=50` on Campaign C signal panel (script path already used for Jul 31).
4. **Ablation:** short subsection on persistence hold (this note §4).
5. **Limitations / lit framing:** z-proxy not dollars; no annualization; no “beats Tadi 7.94”; orphan inventory at session end.
6. **Double-blind:** verify anonymous build has no HF/GitHub URLs.

---

## 8. Files to cite in the revision

| Path | Use |
|---|---|
| `data/paper_trading/5day_Aug4_2026/summary.json` | Headline counts |
| `data/paper_trading/5day_Aug4_2026/trades*.jsonl` | Per-trade DirAcc / pnl_proxy; full-book A + 74/74 |
| `data/paper_trading/5day_Aug4_2026/signals*.jsonl` | Baseline replay / R² |
| `data/paper_trading/5day_Aug4_2026/portfolio_sharpe_report.json` | Mid-run Sharpe A–D |
| `data/paper_trading/5day_Aug4_2026/sim_persistence_hold_report.json` | Hold ablation |
| `data/statarb/20260804_062334/` | Underlying live microstructure |
| `scripts/paper_trading_3day/sim_persistence_hold.py` | Reproduce hold ablation |
| `scripts/portfolio_sharpe_paper_session.py` | Reproduce Sharpe |
| `docs/results_jul31_live_metrics_lit.md` | Lit comparison template |

---

## 9. Bottom line

- **74/74 and A=4.38 are real** (recomputed from all 50,690 closes; no hour gaps).  
- **Frame vs lit:** same as Jul 31 — gross z-proxy live-book stability, not capital/fee/annualized Sharpe race vs Tadi 7.94. Campaign C mainly shows the 8h result was not a fluke.  
- **Holding:** ablation still worse on DirAcc / z-proxy — keep H=1.
