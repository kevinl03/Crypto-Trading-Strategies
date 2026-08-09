# Bps-net-of-cost retrain results

Retrained LightGBM on the Jul25–28 protocol with spread-change targets.
Flat fee assumption: **16.0 bps** round-trip. Pair fees from `scripts/fees.py`.

| model | R² | DirAcc | n trades | mean gross | mean net@16 | mean net@pair | win net@16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| zscore_fwd | 0.1328 | 0.629 | 275775 | -0.864675987051931 | -16.864675987051932 | -104.80879710027828 | 0.008800652706010334 |
| bps_gross | 0.0030 | 0.393 | 0 | — | — | — | — |
| bps_gross_entry0 | 0.0030 | 0.393 | 1302628 | 0.17687058768418173 | -15.823129412315817 | -83.66900866564063 | 0.008096709114190697 |
| bps_net_flat16 | 0.0030 | 0.990 | 0 | — | — | — | — |
| bps_net_pair_fee | 0.9870 | 1.000 | 0 | — | — | — | — |

## Interpretation

No trained variant achieves **positive mean net@16** on the Jul25–28 test set. Changing the target alone is not enough at this horizon / fee level; need longer holds, maker pairs, or lower execution cost.