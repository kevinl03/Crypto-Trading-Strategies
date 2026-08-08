# Ablation results (consolidated)

Single experiment branch: **`experiment/ablation-results`**.

Three related checks under the published LightGBM protocol (H=1, W=300,
N_LAGS=3, Jul 25–28 test unless noted). Together they answer: *do microstructure
features help?* and *does any of this clear fees?*

| Study | Question | Verdict | Artifacts |
|---|---|---|---|
| **LOGO / nested feature groups** | Do ticker/OB/trades/funding/OI/cross add filtered R² beyond AR lags? | **Scenario C** — AR baseline best (~0.392 filt. R²); microstructure flat/slightly worse; funding & OI prune to 0 cols | `statarb/run_logo_ablation.py`, `statarb/outputs_logo/` |
| **Fee-aware profit gate** | On live Jul30/Jul31 paper trades, does any `\|pred\|` slice clear RT fees? | **No** — mean gross ≈ −0.7 bps; higher `\|pred\|` worse; pair RT fees ≈ 105 bps | `scripts/fee_aware_trade_gate.py`, `docs/fee_aware_gate_summary.md`, `data/paper_trading/*/fee_aware_gate_report.json` |
| **bps-net retrain** | Does training on ΔS / (ΔS−fee) instead of forward z create fee-clearing trades? | **No** — gross bps nearly unpredictable (R²≈0.003); no variant has positive mean net@16 | `statarb/run_bps_net_retrain.py`, `statarb/outputs_bps_net/` |

## Feature importance (bps retrain dumps)

- **zscore_fwd**: still AR-dominated (`zscore_lag1/2/3` ≈ 64% gain).
- **bps_gross / bps_net_flat16** (`best_iter=1`): rank shifts toward `spread_bps_lag1`, OB imbalance, cross mid — **unstable** (single tree).
- **bps_net_pair_fee**: `pair` ≈ 60% gain (learns fee identity, not alpha); **0 trades**.

Funding / OI never appear (pruned before train).

## Bottom line

Short-horizon z prediction is real; **tradable edge after costs is not**, and
retargeting or confidence gating does not fix that at H=1 taker economics.
Next research levers: longer hold / horizon, maker routes, cheaper venues — not
another H=1 feature or target swap alone.
