# Feature-engineering vs raw market features

Protocol: H=1, Jul 25–28 offline test, filter τ=0.9, fixed rounds = **49**.

Identity (`coin`, `pair`) is an entity control, not the FE question:
- `coin`: which asset
- `pair`: which two venues the cross-exchange mean-reversion lives on

| Variant | # | R² | R²_f | DirAcc_f |
|---|---:|---:|---:|---:|
| raw + identity | 53 | 0.014 | 0.012 | 77.8% |
| engineered + identity | 11 | 0.132 | 0.535 | 85.4% |
| engineered only | 9 | 0.129 | 0.525 | 84.7% |
| raw only | 51 | -0.003 | -0.343 | 60.0% |
| full | 62 | 0.133 | 0.528 | 85.3% |
