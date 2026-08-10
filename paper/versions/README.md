# Paper version snapshots

**Current checkpoint:** [`../Baseline-gradient-boosting-cross-market-spread-prediction.pdf`](../Baseline-gradient-boosting-cross-market-spread-prediction.pdf)  
(Tree/forest baseline framing vs DNN/LSTM peers; Random Forest bagging control; setting-first contribution.)

Prior checkpoint: [`../Validation-set-gradient-boosting-cross-market-spread-prediction.pdf`](../Validation-set-gradient-boosting-cross-market-spread-prediction.pdf)  
(Train/test/validation rename; τ table on validation-only; W figure at τ=0.9.)

Also: `W-tau-gradient-boosting-cross-market-spread-prediction.pdf`, `v12-gradient-boosting-cross-market-spread-prediction.pdf` (Sections 1–2).

Older PDF checkpoints and iteration notes live under [`archive/`](archive/).

## Build with Tectonic

```powershell
cd paper
tectonic gradient-boosting-cross-market-spread-prediction.tex
Copy-Item -Force gradient-boosting-cross-market-spread-prediction.pdf Baseline-gradient-boosting-cross-market-spread-prediction.pdf
```
