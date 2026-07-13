---
description: 'Self-audit the cross-venue stat-arb research (paper + repo + data) against ACM ICAIF acceptance standards. Produces a claims ledger, ranked red-flag list, pre-improvement research questions, and a prioritized revision plan. Run before making any paper edits.'
---

# ICAIF '26 Research-Readiness Self-Audit

You are acting as three personas simultaneously:
1. A **senior quantitative researcher** who verifies every number against code and data.
2. A **skeptical ICAIF meta-reviewer** trained on the 2025 accepted-papers distribution.
3. A **research strategist** deciding what to include, exclude, or reframe to maximize acceptance probability.

Your job is NOT to edit the paper yet. It is to produce audit artifacts (Section 8) that make every subsequent edit evidence-driven.

---

## 0. Level-0 Prompt (the original high-level ask — expand it, do not just answer it)

> Self-analyze our data conclusions and transform the information presentation for top-tier research readiness (target: ACM ICAIF '26). Audit every aspect: research claims, data format, conclusions, inference methods, and inclusion/exclusion of data. Be conservative: include critical findings that demonstrate novelty; exclude irrelevant parts, or reframe them as the rationale for strategy pivots (stablecoins → volatile assets, OU vs. z-score, DEX vs. CEX, asset-count requirements).

**Expansion protocol (mandatory):** For every premise in this prompt and every section below, work two levels deep:
- **Level 1 — Decompose:** Restate the premise as concrete, testable sub-claims about THIS repository (cite file paths).
- **Level 2 — Interrogate:** For each sub-claim, answer four questions: (a) What is the evidence? (b) What is its provenance (which script, which data file, which commit)? (c) What is the strongest counter-hypothesis a hostile reviewer would raise? (d) What experiment or check would settle it?
- Anything that cannot be traced to code + data gets marked **UNVERIFIED** in the claims ledger. Never invent or assume numbers.

---

## 1. Venue Ground Truth (ICAIF '26 — verified from CFP, 2026-07)

| Constraint | Value | Current paper status |
|---|---|---|
| Deadline | **August 2, 2026** (submission via CMT) | ~3 weeks away |
| Length | **8 pages TOTAL** incl. figures AND references, sigconf 2-col | Must be measured; over-length = desk reject |
| Supplementary | **None accepted** — paper must be self-contained | Degradation/ablation tables must fit in 8 pp |
| Review | **Double-blind**; use `sigconf` + `anonymous` class option | VIOLATED: authors listed, `\setcopyright{none}`, no anonymous flag |
| Venue metadata | Milan, Italy, Nov 14–17 2026 | WRONG in tex: says "Brooklyn, NY, USA" |
| Anonymity leakage | Repo README names authors + "Submitted to ACM ICAIF 2026" | Any repo link in the paper breaks blindness; plan an anonymized artifact |
| Self-citation | Third person only | Check "Research Journey" subsection voice |
| Topic fit | "Blockchain and cryptocurrency", "Trading", "Market microstructure", "Financial time series" all listed | In scope — but see AI-relevance gap (§3.D) |

**2025 accepted-papers signal** (from the full list): the distribution is dominated by LLMs, RL, deep learning, GNNs, and generative models. Closest accepted neighbors to this work: *Attention Factors for Statistical Arbitrage*; *Deep Mean-Reversion: Physics-Informed Contrastive Pairs Trading*; *ISEPT: Image-Based Pair Trading*; *DeltaLag: Learning Dynamic Lead-Lag Patterns*; *Graph Learning for FX Statistical Arbitrage*; *GNNs for Uniswap v3*; *Is BTC Enough?* — i.e., stat-arb, lead-lag, and crypto are welcome, but essentially every accepted paper carries a substantive learning/AI component or a novel formal framework. A pure OU + z-score empirical comparison is at risk of "insufficient AI relevance." Treat this as the #1 strategic question (§3.D).

---

## 2. Evidence Inventory (audit against these; cite paths in every finding)

| Artifact | Path | Role |
|---|---|---|
| Paper source | [paper/stochastic-cross-venue-ohlcv-trading.tex](../../paper/stochastic-cross-venue-ohlcv-trading.tex) | Every claim originates here |
| Figures pipeline | [paper/generate_figures.py](../../paper/generate_figures.py) | Reads `data/historical/backtest_results.json` + parquet — verify figures regenerate from committed data |
| Canonical backtest | [experiments/backtest_historical.py](../../experiments/backtest_historical.py) | Produced paper numbers? Which code path (Python vs C++)? |
| C++ engine | [cpp/signal_engine.cpp](../../cpp/signal_engine.cpp) | Known divergences vs Python (see PR review) |
| PR #28 review | [reviews/28/28_REVIEW.md](../../reviews/28/28_REVIEW.md) | Documents Python/C++ result divergence — unresolved? |
| Internal handoff | [docs/internal/STATARB_RESEARCH_HANDOFF.md](../../docs/internal/STATARB_RESEARCH_HANDOFF.md) | Contains numbers that CONTRADICT the paper (see §3.A) |
| Historical data | `data/historical/{CRV,WIF,PEPE,DOGE,SOL}/` + `backtest_results.json` | 30-day 1-min OHLCV parquet |
| Live paper trading | `data/paper_trading/live_campaign_wif/`, `live_campaign_pepe/`, `*_ou/`, `*_zscore/` | EXISTS but unused by paper — see §3.A red flag #3 |
| Stablecoin run | `data/statarb/20260602_190651/` (120 snapshots) | Basis of the 25.9× fee-to-spread claim |
| Robustness | [experiments/robustness_test.py](../../experiments/robustness_test.py) + [robustness_results.txt](../../robustness_results.txt) | Results file is EMPTY — run or remove the claim surface |
| ML pipeline (unused in paper) | [experiments/train_spread_model.py](../../experiments/train_spread_model.py), [experiments/build_features.py](../../experiments/build_features.py) | 63-feature LightGBM spread predictor — candidate AI-relevance fix |
| Ablation/optimization | [experiments/optimize_strategy.py](../../experiments/optimize_strategy.py), [experiments/optimize_v2.py](../../experiments/optimize_v2.py) | Provenance for Table `ablation_k`? |
| **HF dataset (main)** | `datasets/statarb-crypto-research/` → hub `SFU-fintech-AI/statarb-crypto-research` | 23 assets × 12 venues, ~60s snapshots; **orderbook.parquet (793k/701k rows L2) + trades.parquet** + ticker/spread_matrix/funding/OI; chronologically split train (Jun 13–16) / test (Jun 22–24) + backfilled 1-min OHLCV. Independent collection instrument vs `data/historical/` |
| HF dataset (DEX) | `datasets/statarb-crypto-dex/` | dex_pools/dex_spreads + depth — supports the DEX include/exclude decision (§F) |
| HF dataset (stablecoins) | `datasets/statarb-crypto-stablecoins/` | Published stablecoin evidence for the 25.9× claim |

---

## 3. Audit Dimensions (run the Level-1/Level-2 protocol on each)

### A. Claims-integrity audit — every quantitative claim traced to code + data

Build a **claims ledger** row for every number in the abstract, tables 1–7, and conclusion. Seeded Level-2 findings that MUST be resolved (verified against the repo on 2026-07-12):

1. **100% win rates** (WIF bin–mexc, 1,341 trades; all CRV OU pair-models). No real strategy prints 100% over 1,341 trades. Counter-hypothesis: the exit rule ("close when spread crosses μ") combined with close-price fills guarantees a win by construction whenever reversion eventually occurs, i.e., win rate is an artifact of no max-holding constraint + frictionless fills, not edge. Determine which. If artifact: report win rate under a max-holding constraint and label the idealization explicitly.
2. **Paper vs. handoff contradictions.** Handoff doc: WIF best = binance–cryptocom **z-score**, +49,544 bps, Sharpe **0.93**; "18/20 CRV, 13/20 WIF pair-models profitable." Paper: WIF best = binance–mexc **OU**, +59,045 bps, Sharpe **2.46**; "10/10" for every profitable asset. Which run is canonical? Identify the exact `backtest_results.json` generation command/params, re-run, and reconcile. If the discrepancy is due to a code change between June 2 and paper writing, document it.
3. **Live paper trading contradicts the backtest and the paper's own "future work" claim.** `data/paper_trading/live_campaign_wif/WIF_binance_cryptocom_ou/summary.json`: 70,781 s runtime (~19.7 h), 7,832 ticks, **0 trades**, while the backtest implies ~45 trades/day for WIF. The paper says live validation is future work — but the repo contains it, and it (superficially) refutes the signal frequency. Level-2: is this a config difference (`vol_filter_mult=0.8`, `exit_z=0.0`, `max_holding_sec=300`, pair = binance–cryptocom not binance–mexc, 1 s ticks vs 1-min candles) or a genuine no-signal regime? Decide: report it honestly (strengthens the friction narrative and honesty contribution) or justify exclusion in writing. A reviewer who finds this repo post-acceptance is a worst-case scenario.
4. **Python/C++ engine divergence (PR #28, unresolved?).** Static-prefix vs rolling-window OU estimation; z-score window includes vs excludes current bar; mean-reversion guard only in C++. Paper numbers depend on which binary was present at run time. Fix the divergence, state the convention in the paper (window, bar inclusion, re-estimation cadence), and regenerate all results from one engine with a pinned seed/config dump.
5. **Empty robustness artifact.** [robustness_results.txt](../../robustness_results.txt) is empty. Either run [experiments/robustness_test.py](../../experiments/robustness_test.py) (ADF tests, rolling stability) and integrate, or remove any implied claim.
6. **Sharpe and P&L denomination.** "+59,045 bps" ≈ 590% monthly — per what capital? Sum of per-trade bps on unit notional is not a return on capital; capital is fragmented across two venues and both leg inventories. Define: return basis, annualization convention (√252? √(minutes)?), and whether Sharpe uses daily aggregation. Reviewers compute this immediately.

### B. Data audit — format, coverage, and what OHLCV can and cannot support

1. **Stale-print hypothesis (the central threat to validity).** A 1-min OHLCV close on a thin venue is the *last trade*, possibly minutes old, with an untradeable wide bid-ask around it. The "38-minute Crypto.com lag" may be absence-of-trades, not tradeable mispricing — the classic stale-quote arbitrage illusion (Makarov & Schoar address this with order-book data). **You HAVE the instrument to settle this**: `datasets/statarb-crypto-research/orderbook.parquet` (L2, 793k train + 701k test rows, incl. Crypto.com and MEXC). Level-2 checks: (a) volume/trade-count per candle on the slow venue during "lag" episodes (use `trades.parquet`); (b) forward-fill statistics — how often is close[t] == close[t-1]?; (c) at each OU entry signal reconstructed on the HF window, compute executable edge = signal minus (half-spread_i + half-spread_j) from the L2 books at that snapshot — what fraction of signals survive?; (d) the live campaign's bid/ask ticks in `ticks.jsonl` as corroboration; (e) does backtest profit survive if entries execute at L2-implied bid/ask rather than close? This single analysis decides whether the headline result is real — and if it survives, order-book-verified executability becomes a headline strength no prior OHLCV-only crypto arb paper has.
2. **Lag measured by the wrong instrument.** The 38-min/1.5-min "latency" is inferred from OU half-life, but half-life ≠ lead-lag. The paper cites DTW and multireference-alignment lead-lag papers (ICAIF '23) — reviewers will ask why lag wasn't measured directly. Compute cross-correlation lead-lag profiles (and optionally DTW) per pair; report both.
3. **Excluded exchanges and survivorship.** Gate.io/Kraken/Coinbase/Phemex excluded for API depth; document that exclusion is data-driven, not results-driven. State timezone/UTC alignment, gap handling, and `merge_asof` tolerance (figures use 1-min tolerance — is the backtest identical?).
4. **Stablecoin negative result: 120 snapshots (~2 h) vs "definitively unviable."** The claim is likely true but the evidence base is thin for the word "definitive." Either soften the language or extend with the 30-day framework applied to one stablecoin pair.
5. **Selection circularity.** Assets were *selected by* spread std from a 48-h pilot, then the *finding* is that spread std separates profitability. With n=5 assets the "sharp threshold at 15 bps" is 3 points above and 2 below a line that equals the fee — close to tautological. Options: (a) reframe as a *fee-floor screening criterion* validated out-of-sample, (b) test the gray zone on the HF universe — **23 assets × 12 venues already collected** (`statarb-crypto-research`), including mid-volatility assets (BONK, FLOKI, SEI, SUI, TIA, ENA, ARB, OP...) that populate the 12–25 bps region the 5-asset study skips, (c) present per-pair (n=50) rather than per-asset (n=5) with proper dependence caveats. Option (b) converts the weakest claim into a genuinely tested threshold — likely the highest-value experiment available.
6. **DEX data exists (`data/dex/`, datasets/statarb-crypto-dex) but is absent from the paper.** Decide: exclude entirely (cleanest for 8 pages) or one sentence in the journey framing. Do not leave dangling references.

### C. Inference-method audit — statistics a quant reviewer will recompute

1. **"Walk-forward validation" is mislabeled.** Splitting 30 days into three independent 10-day in-sample runs is subsample stability, not walk-forward. Real walk-forward: calibrate on window *t*, trade on *t+1*, roll. Rename the current table AND add true out-of-sample walk-forward (the codebase already supports rolling estimation).
2. **No statistical significance anywhere.** Add bootstrap CIs on Sharpe/net P&L (block bootstrap over days), and control for testing 50 pair-model combinations (e.g., White's reality check / Benjamini–Hochberg, or at minimum report the multiplicity honestly).
3. **OU estimation specification is under-documented.** MLE vs OLS-on-AR(1)? Window length? Re-estimation cadence? `dt` convention? The `b >= 0` non-mean-reverting guard? These currently differ between engines (§A.4) — the paper must state one spec.
4. **Feasibility of the short leg.** "Sell on exchange i" requires pre-positioned inventory or margin on the expensive venue; the paper's fee model ignores this. Either model inventory constraints (both-legs-funded, halving deployable capital) or state the assumption explicitly in the friction table.
5. **Fee model provenance.** "Actual fee schedule via CCXT" — snapshot date, VIP tier assumed, maker/taker choice. Pin these in [scripts/fees.py](../../scripts/fees.py) and cite.

### D. Novelty & AI-relevance positioning (the strategic decision)

State, in one sentence each, the top-3 candidate novelty claims, then pick ONE to lead with:
- N1: First systematic *trading* evaluation of same-asset cross-venue spreads (extends Makarov–Schoar from documentation to strategy).
- N2: Venue-specific price latency as persistent, asset-specific microstructure (extends ICAIF lead-lag literature cross-venue).
- N3: The σ_spread > fee viability threshold + honest degradation template + strong negative results (stablecoins, DOGE/SOL).

Then resolve the **AI-relevance gap** — choose one and commit:
1. **Elevate the dormant ML pipeline** ([experiments/train_spread_model.py](../../experiments/train_spread_model.py), 63 features, LightGBM, walk-forward CV): predict spread reversion / filter OU signals; OU and z-score become baselines. Highest acceptance leverage; verify the pipeline actually produces positive results before committing.
2. **Reframe as rigorous empirical microstructure + benchmark**: the multi-signal dataset is ALREADY published with chronological train/test splits (`SFU-fintech-AI/statarb-crypto-research`: OHLCV + L2 orderbook + trades + funding + OI, 23 assets × 12 venues) — position it as the community artifact with the OU study as the reference baseline (datasets do get accepted, cf. FinDER/FinAgentBench — but those are LLM-era; risk remains).
3. **Add RL threshold optimization** (cited future work) — highest effort, lowest feasibility in 3 weeks. Likely reject this option.
Document the decision and its rationale in the revision plan.

### E. Presentation & artifact audit

1. Anonymize: `\documentclass[sigconf,anonymous]{acmart}`, strip authors, fix conference metadata (Milan, not Brooklyn), remove `nonacm` confusion, ensure no self-revealing repo links; prepare an anonymized code/data artifact (e.g., Anonymous GitHub / Zenodo) if referenced. **The HF org name `SFU-fintech-AI` leaks author affiliation** — any dataset citation in the submitted PDF needs an anonymized mirror or neutral handle.
2. Fix CCS concepts: "Human-centered computing~Collaborative and social computing" is a copy-paste error; the significance-100 entry should go.
3. Measure page count in sigconf 2-col; identify what to cut if >8 pp (candidates: per-asset subsections → table; Rust/engineering content stays out; journey subsection → 1 paragraph in intro).
4. Figures: confirm all four regenerate from committed data via [paper/generate_figures.py](../../paper/generate_figures.py); check legibility at column width; annotate the threshold plot with asset names.
5. Reproducibility statement: pinned requirements, one-command result regeneration, config dump per results JSON (git SHA, engine used, params).
6. Every table needs its generating script named in the audit ledger (not necessarily in the paper).

### F. Inclusion/exclusion & narrative framing (the user's "conservative" strategy)

Decision rules — apply to each candidate content block:
- **Include as headline** iff it survives §A–§C audits AND supports the chosen novelty claim N*.
- **Reframe as journey/motivation (≤1 short paragraph)** iff it explains a pivot that a reviewer would otherwise ask about: stablecoin failure → volatile assets (KEEP, it motivates the fee-floor threshold); A* pathfinding origins (compress to one clause); DEX exploration (cut or one clause); asset-count rationale (KEEP as selection protocol).
- **Exclude** iff it neither survives audit nor explains a pivot: C++/Rust engineering, collector bandwidth tooling, Excel exports, funding-arb monitor.
- **Negative results are assets, not liabilities** at ICAIF — but only when framed with adequate evidence (§B.4): stablecoins (25.9× fee ratio), DOGE/SOL (efficient venues). Keep both, with honest evidence-strength language.
- The live zero-trade campaign (§A.3) is the hardest call: if the stale-print analysis (§B.1) shows the backtest edge is partly illusory, the honest paper is "OHLCV backtests overstate cross-venue arb; here is the degradation chain backtest → frictions → live," which is arguably a STRONGER, more ICAIF-worthy contribution than +590%/month. Evaluate this inversion seriously.

---

## 4. Reviewer simulation

After completing §3, write three independent ICAIF reviews (R1 microstructure expert, R2 ML methods expert, R3 practitioner), each with: summary, 3 strengths, 5 weaknesses, questions to authors, and a score on {reject, weak reject, borderline, weak accept, accept}. Be adversarial: R1 leads with stale prints and short-leg feasibility; R2 leads with AI relevance, multiplicity, and the mislabeled walk-forward; R3 leads with 100% win rate and capital denomination. Then write the meta-review with the single decision-critical issue.

---

## 5. Output artifacts (create these files; do not edit the paper in this run)

| File | Content |
|---|---|
| `reviews/self_audit/CLAIMS_LEDGER.md` | Table: claim → paper location → generating script/data → status {VERIFIED, UNVERIFIED, CONTRADICTED, ARTIFACT-SUSPECT} → action |
| `reviews/self_audit/RED_FLAGS.md` | Ranked by (reviewer severity × fix effort); each with counter-hypothesis and settling experiment |
| `reviews/self_audit/RESEARCH_QUESTIONS.md` | The question backlog from §6, extended with everything new you find |
| `reviews/self_audit/REVIEWER_SIM.md` | §4 output |
| `reviews/self_audit/REVISION_PLAN.md` | Prioritized plan with effort estimates vs. the Aug 2 deadline; includes the D-decision (AI-relevance path) and F-decisions (include/reframe/exclude table) |

---

## 6. Seeded research-question backlog (answer BEFORE improving the paper)

**Tier 1 — decide acceptance (answer first):**
1. Is the Crypto.com/MEXC "lag" tradeable liquidity or stale prints? Settle with L2 evidence: at each reconstructed entry signal on the HF windows, executable edge = signal − (half-spreads from `orderbook.parquet`) — what fraction survives, per venue? Corroborate with `trades.parquet` activity and live `ticks.jsonl`.
2. Which engine + config produced `backtest_results.json`, and do Python and C++ paths now agree bar-for-bar? Can every paper number be regenerated with one command?
3. Why did 19.7 h of live WIF trading produce 0 trades vs. ~37 expected — config, pair choice, or regime? What do the recorded signals in `signals.jsonl` show?
4. Is the 100% win rate an artifact of the exit-at-μ rule with unbounded holding? What is the win rate and P&L with max-holding 30/60/120 min and bid-ask fills?
5. Which AI-relevance path (D.1 ML / D.2 benchmark / D.3 RL) is feasible by Aug 2, and does the LightGBM pipeline produce a positive, honest result on committed data?
6. **Cross-instrument replication:** do spread std, latency ranking, and pair-level profitability replicate on the HF train window (Jun 13–16, independent snapshot collector) and hold on the HF test window (Jun 22–24) — i.e., does the paper's edge survive on data it has never touched, at 23-asset breadth?

**Tier 2 — survive review:**
6. What are block-bootstrap CIs on Sharpe/net for the 6 headline pair-models, and which survive multiplicity control across all 50?
7. What does direct cross-correlation lead-lag estimation give per pair, and does it match the OU half-life story?
8. What is Sharpe/P&L denominated on realistic two-venue capital with inventory constraints (short-leg feasibility)?
9. Does true walk-forward (calibrate t, trade t+1) preserve the edge? At what re-estimation cadence?
10. Do the handoff-vs-paper number contradictions (§A.2) trace to a code change, parameter change, or data change?

**Tier 3 — polish:**
11. Actual page count in sigconf; what gets cut?
12. Are all anonymity leaks closed (tex metadata, repo README, figure paths, acknowledgments)?
13. Do all figures regenerate from committed data on a clean clone?
14. Is the stablecoin evidence base (120 snapshots) defensible for "definitive," or should language soften / data extend?
15. What snapshot date / fee tier does [scripts/fees.py](../../scripts/fees.py) encode, and is it cited?

Extend this list with every new question your Level-2 interrogation raises; nothing gets marked resolved without a file-path citation.
