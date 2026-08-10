# LOGO / Nested Feature-Group Ablation Results

**Verdict: Scenario C.** AR baseline (lags + momentum + identity) carries
filtered R²; microstructure adds ~0.01 or less (often slightly negative).

Protocol: H=1, W=300, N_LAGS=3, Jul 25–28 offline test, filter **τ=0.9**
(paper confidence gate). Fixed rounds = **49**.

Script: `statarb/run_logo_ablation.py` · Artifacts: `statarb/outputs_logo/`

## Nested cumulative (τ=0.9)

| Variant | # feat. | R² (all) | R² (filt.) | DirAcc (filt.) | Δ R² filt. vs AR |
|---|---:|---:|---:|---:|---:|
| AR baseline | 11 | 0.132 | 0.534 | 85.5% | +0.000 |
| +ticker | 42 | 0.133 | 0.522 | 85.1% | -0.012 |
| +orderbook | 45 | 0.132 | 0.519 | 85.2% | -0.015 |
| +trades | 48 | 0.132 | 0.522 | 85.1% | -0.012 |
| +funding (pruned) | 48 | 0.132 | 0.522 | 85.1% | -0.012 |
| +oi (pruned) | 48 | 0.132 | 0.522 | 85.1% | -0.012 |
| +cross | 62 | 0.133 | 0.524 | 85.1% | -0.010 |
| full (all surviving) | 62 | 0.133 | 0.524 | 85.1% | -0.010 |

## Classic leave-one-group-out

| Variant | # feat. | R² (all) | R² (filt.) | DirAcc (filt.) |
|---|---:|---:|---:|---:|
| full | 62 | 0.133 | 0.528 | 85.3% |
| −baseline | 53 | 0.014 | 0.018 | 76.8% |
| −ticker | 31 | 0.132 | 0.542 | 85.9% |
| −orderbook | 59 | 0.132 | 0.526 | 85.3% |
| −trades | 59 | 0.132 | 0.525 | 85.3% |
| −funding (pruned) | 62 | 0.133 | 0.528 | 85.3% |
| −oi (pruned) | 62 | 0.133 | 0.528 | 85.3% |
| −cross | 48 | 0.132 | 0.526 | 85.3% |

## Takeaway

- **Important:** AR baseline (lags + momentum + coin/pair identity).
- **Not important:** ticker / OB / trades / cross — ~0.01 R² or worse vs AR.
- Funding / OI prune to 0 columns.
- Fee-aware / bps-net checks are separate extras (see `docs/ablation_results.md`).
