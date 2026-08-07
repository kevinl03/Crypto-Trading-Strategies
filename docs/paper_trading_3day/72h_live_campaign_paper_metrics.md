# ~72h / ~3-day Live Campaign — Metrics for Paper Revision

**Session:** `data/paper_trading/5day_Aug4_2026`  
**Collector run:** `data/statarb/20260804_062334`  
**Branch:** `feature/5day-paper-trading` (rebased onto `origin/main` as of 2026-08-07)  
**Model:** `statarb/outputs/statarb_lgbm.txt` (68 feats, H=1, z-window 300, `|pred|≥0.5`, `max_open=50`)

This note compiles **what the paper can claim** from the longer live run (vs the prior ~8h Jul 30/31 campaigns), plus how to answer reviewer feedback. Settlement is the z-unit proxy (`direction × z_exit`), not dollar P&L.

---

## 1. Do we have all the data?

**Yes — full paper-trading artifacts are available** (restored from `pre_restart_20260807_164107` after an accidental archive-on-restart).

| Artifact | Status | Notes |
|---|---|---|
| `trades.jsonl` + `trades_001.jsonl` | ✅ ~20 MB | **50,690** closed trades |
| `signals.jsonl` … `signals_020.jsonl` | ✅ ~211 MB | **1,013,380** scored predictions |
| `summary.json` / `dashboard.json` | ✅ | Final snapshot at death |
| `portfolio_sharpe_report.json` | ✅ | Mid-run (~60h mark); see §3 |
| `sim_persistence_hold_report.json` | ✅ | Hold ablation on ~815k signals mid-run |
| Collector `spread_matrix` / `ticker` / `orderbook` / … | ✅ | Continuous through death (~snap 3030); multi-day JSONL |

**Timeline (UTC):**
- First trade: `2026-08-04T08:42:13Z` (after ~90-snap z warmup)
- Last trade / death: `2026-08-07T09:36:28Z`
- **Trade span ≈ 72.9 hours (~3.0 days)** of active paper trading  
- Wall-clock orchestrator uptime before death ≈ 65h (session started `2026-08-04T16:33:46Z`; shorter than trade span because collector warmup preceded the resumed paper session)

**Caveats on completeness:**
- 50 orphan opens at kill never appear in `trades.jsonl` (same as Jul 31).
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
| Coins / exchange-pairs traded | — | **23 / 15** |
| Closed-only hourly Sharpe (z-proxy) | 2.35 (A) / **2.41** (B w/ MTM) | **~4.10–4.38** (see §3) |
| Hours with positive hourly z-proxy PnL | 6/6 | **74/74** |

**Settlement definition (unchanged, publishable as proxy):**
\[
\mathrm{pnl\_proxy} = \mathrm{direction} \times z_{\mathrm{exit}},\quad
\mathrm{direction} = \mathrm{sign}(\hat{z}),\quad H=1.
\]
Matches the training target (gross z-unit settlement, not dollars).

**Suggested paper one-liner:**
> In a ~72h (~3-day) live paper session (50,690 settled bets; 1.01M scored predictions), the LightGBM policy maintained directional accuracy of 79.0% and mean z-unit settlement of +0.84, with every observed clock hour producing positive portfolio z-proxy P&L (closed-only hourly Sharpe ≈ 4.1).

---

## 3. Portfolio Sharpe (what to report)

From `portfolio_sharpe_report.json` (computed on ~41k closes / 60 hourly bars mid-run) and a full-book closed-only recompute on all 50,690 trades:

| Variant | Definition | Value |
|---|---|---|
| **A** Closed-only hourly | Sum of closed `pnl_proxy` per UTC hour | **~4.10** (mid) / **~4.38** (full 74h) |
| **B** Hourly equity + open MTM | Δ equity with `direction × z_t` on opens | **~4.10** (mid-run report) |
| **C** B / `max_open=50` | Capital-normalized in slot units | **~4.10** |
| **D** Per-snapshot equity | Finer bar (~89s) | **~1.19** |

**Publishing guidance (same as Jul 31 lit notes):**
- Headline: **hourly portfolio Sharpe ≈ 4.1** under z-proxy, Rf=0, ~3-day live window.
- **Do not annualize** as a primary claim (`4.1 × √(24×365)` is nonsense for a 3-day sample).
- **Do not** equate to Tadi & Kortchemski’s multi-month capital Sharpe (~7.94).
- Prefer reporting **A and B**; note D is more conservative at snapshot frequency.

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

## 5. Mapping to reviewer feedback

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

## 6. Recommended paper edits (Campaign C)

1. **Experimental setup:** add Campaign C (~2026-08-04 → 08-07, ~72.9h trade span, same model as Jul 31).
2. **Results table:** DirAcc / mean pnl_proxy / n_closed / hourly Sharpe for Campaign C; note scale-up from 8h → ~3 days.
3. **Baselines:** regenerate mechanical persistence + mean-reversion under `max_open=50` on Campaign C signal panel (script path already used for Jul 31).
4. **Ablation:** short subsection on persistence hold (this note §4).
5. **Limitations:** Sharpe not annualized; orphan inventory at kill; z-proxy settlement (not dollars).
6. **Double-blind:** verify anonymous build has no HF/GitHub URLs.

---

## 7. Files to cite in the revision

| Path | Use |
|---|---|
| `data/paper_trading/5day_Aug4_2026/summary.json` | Headline counts |
| `data/paper_trading/5day_Aug4_2026/trades*.jsonl` | Per-trade DirAcc / pnl_proxy |
| `data/paper_trading/5day_Aug4_2026/signals*.jsonl` | Baseline replay / R² |
| `data/paper_trading/5day_Aug4_2026/portfolio_sharpe_report.json` | Sharpe A–D |
| `data/paper_trading/5day_Aug4_2026/sim_persistence_hold_report.json` | Hold ablation |
| `data/statarb/20260804_062334/` | Underlying live microstructure |
| `scripts/paper_trading_3day/sim_persistence_hold.py` | Reproduce hold ablation |
| `scripts/portfolio_sharpe_paper_session.py` | Reproduce Sharpe |

---

## 8. Bottom line

- **Data:** complete enough for a Campaign C results section.  
- **Forecast / proxy trading:** strong and **more stable** than the 8h window (DirAcc ~79%, hourly z-Sharpe ~4.1, 74/74 green hours).  
- **Holding:** ablation shows **worse** DirAcc / z-proxy — keep H=1; report as negative ablation.  
- **Rebase:** `feature/5day-paper-trading` successfully rebased onto `origin/main`.
