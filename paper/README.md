# Paper Compilation Instructions

## Main Document

**File**: `gradient-boosting-cross-market-spread-prediction.tex`  
**Conference**: ICAIF '26 (7th ACM International Conference on AI in Finance)  
**Format**: ACM sigconf, double-blind review  
**Current Status**: 11 pages (3 pages over 8-page limit — needs cuts)

## Prerequisites

### Tectonic (recommended)

Tectonic is a self-contained LaTeX engine that automatically downloads packages.

**Windows**:
```powershell
# Download latest release
$url = "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-x86_64-pc-windows-msvc.zip"
$dest = "$env:LOCALAPPDATA\tectonic"
New-Item -ItemType Directory -Force -Path $dest
Invoke-WebRequest -Uri $url -OutFile "$env:TEMP\tectonic.zip"
Expand-Archive -Path "$env:TEMP\tectonic.zip" -DestinationPath $dest -Force
$env:PATH = "$dest;$env:PATH"
```

**macOS** (Homebrew):
```bash
brew install tectonic
```

**Linux**:
```bash
# AppImage
wget https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-x86_64.AppImage
chmod +x tectonic-0.17.0-x86_64.AppImage
./tectonic-0.17.0-x86_64.AppImage --help
```

### Alternative: Full TeX Distribution

If you have TeXLive or MiKTeX installed, you can use `pdflatex` + `bibtex`:

```bash
cd paper
pdflatex gradient-boosting-cross-market-spread-prediction.tex
bibtex gradient-boosting-cross-market-spread-prediction
pdflatex gradient-boosting-cross-market-spread-prediction.tex
pdflatex gradient-boosting-cross-market-spread-prediction.tex
```

## Compilation

### Using Tectonic (one command)

```bash
cd paper
tectonic gradient-boosting-cross-market-spread-prediction.tex
```

Output: `gradient-boosting-cross-market-spread-prediction.pdf`

### With verbose output

```bash
tectonic --print gradient-boosting-cross-market-spread-prediction.tex
```

### Keep intermediate files

```bash
tectonic -k gradient-boosting-cross-market-spread-prediction.tex
```

This keeps `.log`, `.aux`, `.bbl`, `.out` files for debugging.

## File Structure

```
paper/
├── gradient-boosting-cross-market-spread-prediction.tex  # Main document
├── gradient-boosting-cross-market-spread-prediction.pdf  # Compiled output
├── references.bib                                        # Bibliography
├── sections/
│   ├── abstract.tex
│   ├── introduction.tex
│   ├── related_work.tex
│   ├── methodology.tex
│   ├── experimental_setup.tex
│   ├── results.tex
│   ├── ablation.tex
│   ├── discussion.tex
│   ├── conclusion.tex
│   └── ethics.tex
├── figures/
│   ├── fig1_ablation_r2_diracc.png
│   ├── fig2_pred_vs_realized_jul31.png
│   ├── fig3_filter_r2_lift.png
│   ├── fig4_feature_importance_top20.png
│   ├── fig5_cum_pnl_proxy_jul31.png
│   └── fig6_model_minus_naive_diracc.png
├── SUBMISSION_GUIDE.md                                   # ICAIF '26 requirements
├── COMPILATION_ISSUES.md                                 # Known issues
└── README.md                                             # This file
```

## Double-Blind vs Camera-Ready

### Current (double-blind submission)

```latex
\documentclass[sigconf,anonymous,review]{acmart}
% ...
\anonsubmissiontrue
```

### After acceptance (camera-ready)

```latex
\documentclass[sigconf]{acmart}
% ...
\anonsubmissionfalse
```

Then add the ACM rights-form metadata from the acceptance email:
```latex
\setcopyright{acmlicensed}
\acmDOI{...}
\acmISBN{...}
```

## Known Issues

See `COMPILATION_ISSUES.md` for:
- Page count (11 pages, need to cut 3)
- Z-unit vs basis-point sign reversal
- Feature importance discrepancies
- Sharpe ratio context
- Bibliography corrections applied

## Warnings

Compilation produces ~37 overfull/underfull hbox warnings. These are cosmetic (line-breaking in tight columns) and don't break layout. The most significant are in the ablation section's feature discussion.

## Page Count Check

After compilation, count pages:
```bash
# With pypdf (Python)
python -c "import pypdf; print(f'{len(pypdf.PdfReader(\"gradient-boosting-cross-market-spread-prediction.pdf\").pages)} pages')"

# Manual: open PDF and check page indicator
```

Target: ≤8 pages including figures and references.

## Support

For tectonic issues: https://github.com/tectonic-typesetting/tectonic/issues  
For ACM template issues: https://www.acm.org/publications/proceedings-template
