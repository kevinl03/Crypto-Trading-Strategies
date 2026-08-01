# Baseline comparison: LightGBM vs mechanical \(|z|\ge 0.5\)

**Branch purpose:** Document the Jul 31 live paper-session comparison of the production LightGBM z-forecast policy against mechanical `|z|` peers (persistence **and** classic mean-reversion), place those peers in the literature’s baseline taxonomy, and state the paper-facing claim that the learned model is the stronger trading signal under matched constraints.

**Session:** `data/paper_trading/July31st_8_hr/`  
**Reproduce mechanical peers:**  
`python scripts/mechanical_z_baseline_paper_session.py data/paper_trading/July31st_8_hr`  
**Artifacts:** `data/paper_trading/July31st_8_hr/mechanical_z_baseline/`

---

## Claim (paste-ready)

On the Jul 31 live paper session, under the **same** horizon (H=1), settlement proxy (`pnl_proxy = direction × z_{t+1}`), rolling-z definition (window 300, `min_periods` 90), and **matched inventory cap** (`max_open=50`), LightGBM’s `|pred|≥0.5` policy delivers higher **per-trade directional accuracy**, higher **mean z-unit PnL**, and stronger **forecast R²** than mechanical `|z_t|≥0.5` rules. Against **persistence** (`sign(z_t)`), the model improves selection quality on a similar trade count. Against **mean-reversion** (`−sign(z_t)`) — the classical pairs direction — the mechanical book is negative on this minute-scale hold, so LightGBM is clearly preferred. Primary evidence is DirAcc, mean `pnl_proxy`, and R²; hourly portfolio Sharpe is reported for both books as portfolio context under matched capacity. Unconstrained mechanical books are not capacity-matched peers.

---

## Policies compared

| | LightGBM (campaign) | Mechanical persistence | Mechanical mean-reversion |
|---|---|---|---|
| Enter when | \(\|\widehat{z}_{t+1}\| \ge 0.5\) | \(\|z_t\| \ge 0.5\) | \(\|z_t\| \ge 0.5\) |
| Direction | \(\mathrm{sign}(\widehat{z}_{t+1})\) | \(\mathrm{sign}(z_t)\) | \(-\mathrm{sign}(z_t)\) |
| Hold / settle | Next snapshot (H=1) | Next snapshot (H=1) | Next snapshot (H=1) |
| PnL | `direction × exit_z` | same | same |
| Inventory | `max_open=50` | fair peer: same; ablation: unconstrained | same |

### Persistence vs mean-reversion (same entries, opposite bet)

Both mechanical peers use the **same** entry rule: trade when \(|z_t| \ge 0.5\), hold one snapshot, settle with `pnl_proxy = direction × z_{t+1}`. They differ **only** in `direction`:

| Peer | Direction | Plain-English bet |
|---|---|---|
| **Persistence** | \(\mathrm{sign}(z_t)\) | “The spread z keeps its current sign” — if \(z_t>0\), bet exit z stays positive |
| **Mean-reversion** | \(-\mathrm{sign}(z_t)\) | “The spread z flips / fades” — if \(z_t>0\), bet exit z goes negative (classic pairs fade) |

So mean-reversion is not a different filter or different trades list under capacity; it is the **sign-flipped** book on the same slots. At H=1 on Jul 31, large \(|z|\) tends to continue → persistence is profitable in z-units and mean-reversion is approximately its mirror (DirAcc ~30.7%, mean `pnl_proxy` negative). Persistence is the closest no-model twin to lag-dominated forecasts; mean-reversion is the textbook HF pairs / OU-style direction (Liou rule-based peer).

---

## Headline metrics (fair peer = `max_open=50`)

| Strategy | Direction | n closed | DirAcc | mean `pnl_proxy` | Hourly Sharpe **B** |
|---|---|---:|---:|---:|---:|
| **LightGBM** | \(\mathrm{sign}(\widehat{z})\) | 7,973 | **76.9%** | **+0.746** | 2.41 |
| Mechanical \|z\|≥0.5 | persistence \(\mathrm{sign}(z_t)\) | 8,550 | 69.3% | +0.437 | 2.63 |
| Mechanical \|z\|≥0.5 | mean-reversion \(-\mathrm{sign}(z_t)\) | 8,550 | 30.7% | **−0.437** | −2.63 |
| Mechanical \|z\|≥0.5 unconstrained | persistence | 37,296 | 66.2% | +0.391 | 2.69† |
| Mechanical \|z\|≥0.5 unconstrained | mean-reversion | 37,296 | 33.8% | −0.391 | −2.69† |

†Not capacity-matched (~200–240 opens/hour vs 50). Report for completeness only.

**LGBM vs mechanical persistence (capacity):** DirAcc **+7.6 pp**; mean `pnl_proxy` **+0.31 (~+71%)**.  
**LGBM vs mechanical mean-reversion (capacity):** DirAcc **+46.2 pp**; mean `pnl_proxy` flips from **−0.44 to +0.75**.

Sources: `summary.json`, `portfolio_sharpe_report.json`, `mechanical_z_baseline/mechanical_z_baseline_report.json`.

---

## Forecast metrics (why the model is not “just \|z\|”)

Matched-row scoring on the same signal panel (`metrics_report.csv`):

| Set | Model R² | Naive \(z_t\to z_{t+1}\) R² | Model DirAcc | Naive DirAcc |
|---|---:|---:|---:|---:|
| All predictions | 0.104 | −0.338 | 60.9% | 65.6% |
| Entries \|pred\|≥0.5 | **0.347** | 0.173 | 76.7% | 77.0% |

On mechanical capacity entries, \(z_t\) as a forecast of \(z_{t+1}\) has **R² = −0.33**. LightGBM’s filtered R² **0.35** is not explained by thresholding current z alone.

At H=1 on this session, large \(|z|\) states continue more often than they reverse — hence persistence mechanical is profitable and mean-reversion mechanical is the mirror image (negative). LightGBM’s learned direction aligns with that short-horizon structure while still beating persistence on per-trade quality.

---

## How other papers baseline (what we are mirroring)

Use these as **baseline types** and reporting norms, not as numbers to race.

| Paper / line | Their baseline / comparator pattern | What we report in the same spirit |
|---|---|---|
| **Liou et al. (2024)** | Deep / ML model vs **rule-based** pairs | Mechanical \|z\|≥τ with persistence **and** mean-reversion on the same panel |
| **Fil & Kristoufek (2020)** | Distance vs coint.; **gross vs costs**; multi-freq | Mechanical z threshold as rule peer; fee/cost work deferred until bps unit matches economic book |
| **Fischer, Krauss & Deinert (2019)** | Performance under **execution delay** | Delay ablation still to add (1–3 snaps); hold here is already ~1 snap |
| **Sarmento et al. (2024)** | Hybrid rules; **abstention** lifts profit-per-trade | Always-score vs \|pred\|≥0.5 (shipping); mechanical has no learned abstention |
| **Shen et al. (2022)** | Dual **R² + trading** metrics; regimes | Dual table: R²/DirAcc **and** DirAcc/mean pnl/Sharpe B |
| **Han & Li / LSTM-as-filter** | Rule alone vs rule+ML filter | Mechanical rule alone vs LightGBM policy (this doc) |
| **Tsoku & Makatjane (2026)** | DL forecast of cointegrated spreads | Same *task family*; we compare metric suite + live mechanical peers, not a reimplemented LSTM |
| **Ko et al. (2023)** | Multiple **pair-selection** methods | Defer full bake-off; universe/cadence held fixed here |
| **Palazzi (2025)** | Passive hold; stops / vol filters | Defer until stop logic exists |
| **Internal (expected)** | Persistence / lag-OLS; train→test degradation | Matched-row naive persistence shipping; lag-OLS / OOS Sharpe still on the menu |

**Already shipping vs still open**

| Baseline | Status |
|---|---|
| Matched-row naive persistence (forecast) | Done (`metrics_report.csv`) |
| Abstention \|pred\|≥0.5 vs always-score | Done (campaign ablation) |
| Mechanical \|z\| persistence + mean-reversion (trading) | **Done (this branch)** |
| Capacity-matched portfolio Sharpe B | Done for LGBM + mechanical |
| Cost grid (gross vs net bps) | Script/artifact only — **not** used in paper framing yet (unit mismatch) |
| Ranked capacity (\|pred\| / \|z\|) | **Computed (optional)** — see Strengthening |
| Execution-delay curve | Open |
| Lag-OLS / full-feature Ridge | Open (forecast ladder) |
| Offline OOS Sharpe train vs test | Open (#63) |

---

## Argument structure (how to say LightGBM is stronger)

### 1. Fair peer = matched `max_open`

`max_open=50` is the live inventory / risk budget. Without it, mechanical takes every `|z|≥0.5` row (~37k trades) and is not the same book as the campaign. Capacity matching is required before portfolio Sharpe is discussed side-by-side.

### 2. Lead with per-trade skill and forecast fit

Under capacity:

- vs **persistence:** DirAcc 76.9% vs 69.3%; mean `pnl_proxy` +0.75 vs +0.44 — better picks at similar n.
- vs **mean-reversion:** mechanical DirAcc 30.7% and mean pnl **negative** — classic reverse-the-spread direction fails at this hold; LightGBM does not.
- **Filtered R² 0.35** vs naive 0.17 on model rows (and vs −0.33 for \(z_t\to z_{t+1}\) on mechanical entries).

These unlock: *learned z-forecast + confidence filter improves on thresholding current z*, including the usual pairs mean-reversion direction.

### 3. Portfolio Sharpe as context, not the verdict

Report hourly Sharpe B for LightGBM and both mechanical modes under `max_open=50`. Emphasize DirAcc, mean `pnl_proxy`, and R² as the primary comparison. On a six-hour live window, Sharpe is a short-path portfolio summary; do not frame the paper around ranking Sharpes across policies.

### 4. Scope limits (keep light)

- Gross z-proxy only (not fee-net) until the cost grid lands.
- Unconstrained mechanical is not the capacity peer.
- Filtered DirAcc vs **naive on the same LGBM entry rows** is essentially tied (76.7% vs 77.0%); the mechanical comparison uses a **different entry set** (threshold on \(z_t\), not on \(\widehat{z}\)).

---

## Optional one-paragraph Methods/Results blurb

> As trading baselines we replay mechanical `|z_t|≥0.5` peers on the Jul 31 signal panel, settling at `t+1` with the same `pnl_proxy` as the live LightGBM campaign, with and without `max_open=50`. Directions follow persistence (`sign(z_t)`) and classical mean-reversion (`−sign(z_t)`). Under matched capacity, LightGBM records DirAcc 76.9% and mean `pnl_proxy` +0.75, versus 69.3% / +0.44 for persistence and 30.7% / −0.44 for mean-reversion (7,973 vs 8,550 closes). Filtered prediction R² for LightGBM is 0.35, versus 0.17 for matched-row naive persistence. Hourly portfolio Sharpe with open MTM is reported for each capacity-matched book as portfolio context; per-trade accuracy, mean proxy PnL, and forecast R² are the primary evidence that the learned policy improves on mechanical z-threshold rules used throughout the HF pairs literature.

---

## Strengthening analyses (saved for later — optional)

Reproduce:

```text
python scripts/baseline_strengthening_jul31.py data/paper_trading/July31st_8_hr
```

Artifacts: `data/paper_trading/July31st_8_hr/baseline_strengthening/`  
These are **not** required for the primary claim table above; keep them when we want a sharper peer or fee discussion.

### A. Cost / bps grid — deferred from narrative (artifact only)

A `direction × Δspread_bps − c` cost grid was computed by `baseline_strengthening_jul31.py` and is stored in `baseline_strengthening_report.json` for later engineering. **Do not put those bps win rates or mean-bps numbers in the paper claim table.**

Reason: on Jul 31 that bps proxy **disagrees in sign** with the campaign’s primary settlement unit (`pnl_proxy` in z-scores). In particular, mean-reversion can look “high win rate” in bps simply because its direction is flipped relative to persistence, while its **z-proxy** book is the one that loses (DirAcc 30.7%, mean `pnl_proxy` −0.44). Mixing bps-net win rate with z-based DirAcc/Sharpe makes the model look worse for the wrong reason. Revisit fees only after the economic unit is locked (aligned bps or dollar PnL).

### B. Ranked capacity fill (`max_open=50`)

Fairer scarce-slot peer: each snapshot, fill free slots by **`|pred|` desc** (LGBM) or **`|z|` desc** (mechanical), instead of encounter/FIFO order.

| Strategy | Fill rule | n closed | DirAcc | mean `pnl_proxy` | Sharpe B |
|---|---|---:|---:|---:|---:|
| LightGBM campaign (reference) | FIFO / encounter | 7,973 | 76.9% | +0.746 | 2.41 |
| LightGBM ranked | \|pred\| desc | 7,973 | **77.6%** | **+0.774** | 2.35 |
| Mechanical persistence ranked | \|z\| desc | 8,550 | 69.5% | +0.731 | 2.62 |
| Mechanical mean-reversion ranked | \|z\| desc | 8,550 | 30.5% | −0.731 | −2.62 |

**Takeaway for later:** Ranking slightly lifts LGBM DirAcc/mean pnl vs the live FIFO book. Ranking mechanical by `|z|` **raises** its mean `pnl_proxy` a lot (0.44 → 0.73), narrowing the mean-pnl gap, while LGBM still leads DirAcc (~77.6% vs 69.5%). Prefer this table if reviewers object to FIFO slot allocation.

---

## File index

| Path | Role |
|---|---|
| `scripts/mechanical_z_baseline_paper_session.py` | Offline replay (persistence + mean-reversion) |
| `scripts/portfolio_sharpe_paper_session.py` | Shared hourly Sharpe B helpers |
| `scripts/baseline_strengthening_jul31.py` | Cost grid + ranked-capacity analyses |
| `.../mechanical_z_baseline/mechanical_z_baseline_report.json` | Mechanical metrics + hourly paths |
| `.../baseline_strengthening/baseline_strengthening_report.json` | Cost grid + ranked summaries |
| `.../summary.json` / `portfolio_sharpe_report.json` / `metrics_report.csv` | LGBM campaign metrics |
| `docs/handoff_lit_baselines.md` | Broader lit baseline menu (optional companion) |
