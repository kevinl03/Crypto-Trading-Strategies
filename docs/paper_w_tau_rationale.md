# Choosing \(W\) and \(\tau\): Paper Integration Note

**Use in paper:** Methods (target + abstention) · Ablation / sensitivity · Discussion (why not \(\tau{=}0.5\))  
**Primary evidence:** Jul~31 live fine \(\tau\) sweep · offline \(W\) sweep · 72h post-hoc \(\tau{=}0.9\)  
**Branch artifacts:** [`docs/jul31_tau_fine_sweep.md`](jul31_tau_fine_sweep.md), [`docs/campaign72h_tau09_w300.md`](campaign72h_tau09_w300.md)

---

## 1. What \(\widehat{z}\) / `|pred|` means (units = rolling \(\sigma\))

The regression target is the next-snapshot **z-score** of the cross-exchange spread in basis points:

\[
z_t = \frac{\mathrm{spread\_bps}_t - \mu_t^{(W)}}{\sigma_t^{(W)}},
\qquad
\mu_t^{(W)},\sigma_t^{(W)}
=
\mathrm{rolling\,mean/std}\bigl(\mathrm{spread\_bps};\, W\bigr).
\]

LightGBM outputs \(\widehat{z}_{t+1}\) (written `pred` in logs). Because \(z\) is already standardized by the trailing window standard deviation \(\sigma^{(W)}\):

- \(\widehat{z} = 0\) means “next spread at the rolling mean,”
- \(|\widehat{z}| = 1\) means “next spread one rolling \(\sigma\) away from that mean,”
- the confidence rule \(|\widehat{z}| \geq \tau\) therefore means: **only trade when the model forecasts a move of at least \(\tau\) rolling standard deviations.**

So \(\tau\) is not an arbitrary score cutoff; it is a **magnitude gate in the same units as the target**. Raising \(\tau\) keeps only larger predicted dislocations and abstains on weak forecasts.

---

## 2. Why \(W = 300\) (context vs. readiness)

\(W\) is the lookback that defines \((\mu^{(W)},\sigma^{(W)})\) and therefore the entire z-label the model is trained to predict.

| Effect of **larger** \(W\) | Tradeoff |
|---|---|
| Longer market context for mean/vol of the spread | Richer, smoother z-scale |
| More snapshots before a valid \(z\) exists (`min_periods`) | **Forced warmup** — the model cannot act until the window is populated |
| Slower adaptation when regime/vol shifts | Live book pays longer cold-start and stickier \(\sigma\) |

Offline \(W\)-sensitivity (spread-only LGBM, Jul~25 holdout; fixed `min_periods=20`) shows test skill rising with \(W\) into a **plateau** around \(560\)–\(720\), with only modest lifts past the paper default (\(W{=}300\): R² ≈ 0.124, DirAcc ≈ 63.1%; \(W{=}720\): +~1pp DirAcc, +0.017 R² for ~2.4× lookback). On the Jul~31 collector, a \(W\times\tau\) grid likewise **did not** favor larger \(W\): at matched \(\tau\), \(W{=}300\) matched or beat \(400/560/720\) on DirAcc and R².

**Decision:** keep **\(W = 300\)** (`min_periods = 90` in production) — enough context for a stable z-scale without paying unnecessary warmup or sluggish \(\sigma\) for a small holdout gain.

---

## 3. Why \(\tau = 0.9\) (confidence gate; improvement over \(0.5\))

### Protocol default \(\tau = 0.5\)
Early live campaigns used \(|\widehat{z}| \geq 0.5\): a permissive gate that fires on ~10–15% of scored rows and maximizes **coverage** / total z-proxy mass. It was chosen for live parity, not because it maximized per-trade SNR.

### Fine sweep on Jul~31 (signal rows, \(W{=}300\))

Sharpe/trade \(= \mathrm{mean}(\mathrm{pnl})/\mathrm{std}(\mathrm{pnl})\), \(\mathrm{pnl}=\mathrm{sign}(\widehat{z})\cdot z_{t+1}\), Rf\(=0\).

| \(\tau\) | \(n\) | DirAcc | R² | mean pnl | **Sharpe/trade** |
|---:|---:|---:|---:|---:|---:|
| 0.75 | 3,609 | 82.9% | 0.466 | +1.04 | 0.915 |
| 0.80 | 2,982 | 83.6% | 0.482 | +1.10 | 0.953 |
| 0.85 | 2,452 | 84.5% | 0.497 | +1.16 | 0.993 |
| **0.90** | **2,026** | **84.9%** | **0.514** | **+1.20** | **1.024** |
| 0.95 | 1,708 | 85.0% | 0.516 | +1.22 | 1.026 |
| 1.00 | 1,426 | 85.3% | 0.534 | +1.26 | 1.072 |

**\(\tau = 0.9\) is the first grid point where per-trade Sharpe reaches \(\geq 1\)** (crosses between 0.85 and 0.90).  
*(Clarification for drafting: this is **per-trade** Sharpe on settled z-proxy PnL—not per-snapshot portfolio Sharpe, which is already \(>1\) at lower \(\tau\).)*

Relative to the initial live threshold \(\tau{=}0.5\) (Jul~31: \(n{=}8{,}775\) signal entries, DirAcc 76.7%, R² 0.347, Sharpe/trade 0.67):

- \(\tau{=}0.9\) keeps **~23%** of those entries (\(n{=}2{,}026\)) — a large abstention cut,
- DirAcc rises to **84.9%**, R² to **0.514**, Sharpe/trade to **1.02**.

Within the high-confidence band, moving from \(\tau{=}0.80\) (\(n{\approx}3{,}000\)) to \(\tau{=}0.90\) (\(n{\approx}2{,}000\)) drops about **one third** of remaining trades while DirAcc/R²/mean pnl keep rising and Sharpe first clears 1. Fewer trades → less aggregate z-exposure and a **higher hit / PnL-per-trade ratio**, i.e. lower incidence of weak or losing z-proxy bets per fill—not a claim of fewer absolute losses in dollar space.

### 72h confirmation (post-hoc on live closes, \(W{=}300\))

| | \(\tau{=}0.5\) (live) | \(\tau{=}0.9\) |
|---|---:|---:|
| \(n\) closed | 50,690 | 12,795 (**−75%**) |
| DirAcc | 79.0% | **86.7%** |
| R² | 0.439 | **0.599** |
| mean pnl | +0.837 | **+1.372** |
| Sharpe/trade | 0.746 | **1.077** |

Same pattern: stricter \(|\widehat{z}|\) gate → fewer trades, higher per-trade quality, Sharpe/trade \(\geq 1\).

---

## 4. Paste-ready prose (Discussion / Ablation)

> The model predicts the next-snapshot z-score \(\widehat{z}_{t+1}\) of cross-exchange `spread_bps`. Because \(z_t = (\mathrm{spread}_t-\mu_t^{(W)})/\sigma_t^{(W)}\), predictions are already in units of the trailing window standard deviation: \(|\widehat{z}|\geq\tau\) means we trade only when the forecasted dislocation is at least \(\tau\) rolling \(\sigma\). The window length \(W\) sets that scale. Larger \(W\) supplies more market context for \((\mu,\sigma)\) but lengthens warmup and slows adaptation; a sensitivity sweep shows diminishing returns past \(W{\approx}300\)--\(560\), so we retain \(W{=}300\) for production.
>
> The live campaigns initially used a permissive confidence gate \(\tau{=}0.5\). A fine sweep on Jul~31 shows that raising \(\tau\) monotonically improves filtered DirAcc, R², and mean z-settled PnL while reducing fire rate. **Per-trade** Sharpe (mean/std of \(\mathrm{sign}(\widehat{z})\,z_{t+1}\)) first exceeds one at \(\tau{=}0.9\) (Sharpe \(1.02\) at \(n{=}2{,}026\) vs \(0.67\) at \(\tau{=}0.5\), \(n{=}8{,}775\)). Relative to \(\tau{=}0.5\), this abstains on roughly three quarters of entries; within the high-\(\tau\) band, moving from \(\sim\)3k to \(\sim\)2k trades still improves the PnL-per-trade ratio. The same \(\tau{=}0.9\) filter on the \(\sim\)72h campaign lifts DirAcc from 79.0\% to 86.7\% and R² from 0.44 to 0.60 while cutting closed trades by \(\sim\)75\%. We therefore treat \(\tau{=}0.9\) (with \(W{=}300\)) as the confidence operating point that first delivers per-trade Sharpe \(\geq 1\), improving on the original \(\tau{=}0.5\) protocol at the cost of coverage.

---

## 5. Paste-ready LaTeX sketch

```latex
\subsection{Selecting \(W\) and the confidence threshold \(\tau\)}
\label{sec:w_tau_choice}

The target is the next-snapshot z-score
\(z_t=(\mathrm{spread\_bps}_t-\mu_t^{(W)})/\sigma_t^{(W)}\).
Model outputs \(\hat{z}_{t+1}\) are therefore in units of the trailing
standard deviation \(\sigma^{(W)}\): the rule \(|\hat{z}|\geq\tau\) admits
only forecasts of at least \(\tau\) rolling \(\sigma\).

Increasing \(W\) enriches the context used to form \((\mu,\sigma)\) but
delays the first valid \(z\) (warmup) and slows adaptation.
Offline and Jul~31 \(W\) grids show diminishing returns beyond
\(W{=}300\); we keep \(W{=}300\).

Live campaigns began at \(\tau{=}0.5\). On Jul~31, per-trade Sharpe
\(\mathrm{mean}(\mathrm{pnl})/\mathrm{std}(\mathrm{pnl})\) with
\(\mathrm{pnl}=\mathrm{sign}(\hat{z})\,z_{t+1}\) first exceeds one at
\(\tau{=}0.9\) (Table~\ref{tab:tau_fine}): DirAcc \(84.9\%\), R$^2$ \(0.514\),
Sharpe \(1.02\), versus \(76.7\%\), \(0.347\), and \(0.67\) at \(\tau{=}0.5\).
This retains about one quarter of \(\tau{=}0.5\) entries. On the
\(\sim\)72h campaign, the same post-hoc gate yields DirAcc \(86.7\%\) and
R$^2$ \(0.599\) on \(12{,}795\) of \(50{,}690\) closes.
```

Suggested table numbers: copy from [`jul31_tau_fine_sweep.md`](jul31_tau_fine_sweep.md) (signal-filter block).

---

## 6. Drafting cautions (do not overclaim)

| Say | Avoid |
|---|---|
| Per-**trade** Sharpe first \(\geq 1\) at \(\tau{=}0.9\) | “Per-snapshot Sharpe = 1 at \(\tau{=}0.9\)” (false; snapshot book Sharpe is already \(>1\) earlier) |
| Fewer trades → higher per-trade quality / less z-proxy mass | “50% fewer” without stating the baseline (\(\tau{=}0.5\) cut is ~75%; \(0.8{\to}0.9\) is ~⅓) |
| Improvement over initial \(\tau{=}0.5\) protocol | That \(\tau{=}0.9\) was used in the original live fill (it was post-hoc / recommended operating point) |
| Gross z-proxy | Fee-net or dollar PnL |
