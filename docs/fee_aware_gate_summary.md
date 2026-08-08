# Fee-aware profit gate — campaign recheck

Calibrates `E[gross_bps | |pred|]` on settled LightGBM paper trades and asks
whether any confidence slice clears round-trip fees (default **16 bps** = 4×4).
Pair-specific fees use `scripts/fees.py` taker schedules (4-leg RT).

## July31st_8_hr

- n trades: **7973**
- mean gross bps: **-0.709**
- mean pair RT fee bps: **104.98**
- mean net (pair fees): **-105.692**
- any |pred| slice clears 16 bps: **False**

### Cumulative from top |pred|

| top frac | n | abs_pred min | mean gross | net@16 | clears@16 |
|---:|---:|---:|---:|---:|:---:|
| 100% | 7973 | 0.500 | -0.709 | -16.709 | no |
| 50% | 3987 | 0.698 | -0.717 | -16.717 | no |
| 25% | 1994 | 0.878 | -0.852 | -16.852 | no |
| 10% | 798 | 1.130 | -1.009 | -17.009 | no |
| 5% | 399 | 1.291 | -1.589 | -17.589 | no |
| 2% | 160 | 1.535 | -1.902 | -17.902 | no |
| 1% | 80 | 1.677 | -1.970 | -17.970 | no |

### Fee-clearing gates (in-sample mean net > 0)

- fee=0.0: clears with loosest `|pred|≥2.718` (n=5, mean net=0.304)
- fee=5.0: **no clearing subset**
- fee=10.0: **no clearing subset**
- fee=15.0: **no clearing subset**
- fee=16.0: **no clearing subset**

### Quantile bins by |pred|

| bin | |pred| lo–hi | n | mean gross | net@16 |
|---:|---|---:|---:|---:|
| 0 | 0.500–0.533 | 798 | -0.554 | -16.554 |
| 1 | 0.533–0.569 | 797 | -0.878 | -16.878 |
| 2 | 0.569–0.607 | 797 | -0.493 | -16.493 |
| 3 | 0.607–0.651 | 797 | -0.722 | -16.722 |
| 4 | 0.651–0.698 | 798 | -0.870 | -16.870 |
| 5 | 0.698–0.756 | 797 | -0.771 | -16.771 |
| 6 | 0.756–0.829 | 797 | -0.529 | -16.529 |
| 7 | 0.829–0.939 | 797 | -0.105 | -16.105 |
| 8 | 0.939–1.130 | 797 | -1.157 | -17.157 |
| 9 | 1.130–2.976 | 798 | -1.009 | -17.009 |

## lgbm_8h_20260730

- n trades: **4901**
- mean gross bps: **-1.254**
- mean pair RT fee bps: **95.08**
- mean net (pair fees): **-96.339**
- any |pred| slice clears 16 bps: **False**

### Cumulative from top |pred|

| top frac | n | abs_pred min | mean gross | net@16 | clears@16 |
|---:|---:|---:|---:|---:|:---:|
| 100% | 4901 | 0.500 | -1.254 | -17.254 | no |
| 50% | 2451 | 0.620 | -1.485 | -17.485 | no |
| 25% | 1226 | 0.746 | -2.451 | -18.451 | no |
| 10% | 491 | 0.921 | -4.593 | -20.593 | no |
| 5% | 246 | 1.041 | -6.549 | -22.549 | no |
| 2% | 99 | 1.254 | -8.673 | -24.673 | no |
| 1% | 50 | 1.487 | -11.397 | -27.397 | no |

### Fee-clearing gates (in-sample mean net > 0)

- fee=0.0: clears with loosest `|pred|≥2.799` (n=5, mean net=1.414)
- fee=5.0: **no clearing subset**
- fee=10.0: **no clearing subset**
- fee=15.0: **no clearing subset**
- fee=16.0: **no clearing subset**

### Quantile bins by |pred|

| bin | |pred| lo–hi | n | mean gross | net@16 |
|---:|---|---:|---:|---:|
| 0 | 0.500–0.519 | 491 | -1.294 | -17.294 |
| 1 | 0.519–0.540 | 490 | -0.987 | -16.987 |
| 2 | 0.540–0.561 | 490 | -1.153 | -17.153 |
| 3 | 0.561–0.588 | 490 | -0.547 | -16.547 |
| 4 | 0.588–0.620 | 490 | -1.130 | -17.130 |
| 5 | 0.620–0.660 | 490 | -0.467 | -16.467 |
| 6 | 0.660–0.711 | 490 | -0.467 | -16.467 |
| 7 | 0.711–0.786 | 490 | -0.888 | -16.888 |
| 8 | 0.786–0.921 | 490 | -1.005 | -17.005 |
| 9 | 0.921–2.835 | 490 | -4.601 | -20.601 |

## Verdict

Under the published LightGBM campaign book, **no `|pred|` confidence slice** has positive in-sample mean gross above a 16 bps round-trip fee. A fee-aware gate therefore declines essentially all trades; profitability requires a different target (bps-net-of-cost) or lower-cost execution, not a tighter z-filter alone.

Additional findings:

- Calibrated `E[gross_bps | |pred|]` is **negative everywhere** (max ≈ −0.69 bps on Jul31). Applying the live gate at `min_expected_gross_bps=16` keeps **0 / 7973** Jul31 trades.
- Higher `|pred|` bins are *worse*, not better (top 1% mean gross ≈ −2.0 bps vs full-book −0.71).
- Only a tiny top slice clears **fee=0** (gross > 0): Jul31 `|pred|≥2.72` keeps n=5. Nothing clears fee ≥ 5 bps.
- Pair-specific 4-leg taker fees from `scripts/fees.py` average ~**105 bps** on Jul31 (Coinbase 60 bps/leg dominates) — far above the flat 16 bps assumption.

### Live usage

Default paper trader behavior is unchanged. To enforce the gate:

```bash
python -m experiments.paper_trade_lgbm \
  --model statarb/outputs/statarb_lgbm.txt \
  --enable-fee-gate \
  --pred-bps-calib data/paper_trading/July31st_8_hr/pred_bps_calib.json \
  --round-trip-fee-bps 16
```

With the current calib this will skip nearly all entries (`n_skip_fee_gate` in `summary.json`). Re-run calibration after changing the model/target:

```bash
python scripts/fee_aware_trade_gate.py
```
