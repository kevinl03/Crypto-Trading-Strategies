# Ablation results (consolidated)

Branch: **`experiment/ablation-results`** (on current `main`).

**Paper gate:** `|pred| ≥ τ=0.9` (consistent with the rest of the paper).  
**Offline holdout:** Jul 25–28 (same protocol as the paper offline tables).

## 1. LOGO / nested feature groups — main result

**Important:** AR baseline = lags + momentum + coin/pair identity (11 feats).  
**Not important:** microstructure (ticker/OB/trades/cross) — about **0.01 R²** and slightly *negative* vs AR. Funding/OI prune to 0 cols.

| Variant | # feat. | R² (all) | R² (filt. τ=0.9) | DirAcc (filt.) | Δ vs AR |
|---|---:|---:|---:|---:|---:|
| AR baseline (lags + momentum + identity) | 11 | 0.132 | **0.534** | 85.5% | — |
| + ticker | 42 | 0.133 | 0.522 | 85.1% | −0.012 |
| + orderbook | 45 | 0.132 | 0.519 | 85.2% | −0.015 |
| + trade flow | 48 | 0.132 | 0.522 | 85.1% | −0.012 |
| + funding / + OI | — | — | — | — | pruned |
| + cross / full | 62 | 0.133 | 0.524 | 85.1% | −0.010 |

Classic LOGO: drop baseline → filt. R² 0.528→0.018; drop microstructure → no hurt.

Artifacts: `statarb/run_logo_ablation.py`, `statarb/outputs_logo/`

## 2–3. Fee gate + bps-net (extras)

Not feature-importance claims — cost / target checks:

- **Fee gate (Jul30/Jul31 live):** no `|pred|` slice clears 16 bps RT (mean gross −0.71 / −1.25 bps).
- **bps-net retrain (Jul25–28):** no positive mean net@16 from ΔS / fee-net targets.

See `docs/fee_aware_gate_summary.md`, `statarb/outputs_bps_net/RESULTS.md`.
