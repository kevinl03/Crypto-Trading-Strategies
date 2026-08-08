# V4C — Abstract + short/long live OOS framing

- **Date:** 2026-08-08
- **Branch:** `paper/incorporate-review-feedback`
- **Artifact:** `versions/v4c-gradient-boosting-cross-market-spread-prediction.pdf`
- **What changed vs v4b:** Abstract rewritten for next-z formulation, Campaign C (~73h) primary live metrics, explicit short (~8h) + long live OOS clause; intro contribution bullet synced; softened absolute claims.

## Build (Tectonic)

From the repo root:

```powershell
cd paper
tectonic gradient-boosting-cross-market-spread-prediction.tex
```

Windows (if `tectonic` is not on PATH):

```powershell
cd paper
& "$env:LOCALAPPDATA\tectonic\tectonic.exe" gradient-boosting-cross-market-spread-prediction.tex
```

Then snapshot:

```powershell
Copy-Item -Force gradient-boosting-cross-market-spread-prediction.pdf versions\v4c-gradient-boosting-cross-market-spread-prediction.pdf
```
