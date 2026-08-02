# ICAIF '26 Submission Guide

Target venue: **7th ACM International Conference on AI in Finance (ICAIF '26)**, November 14–17, 2026, Milan, Italy.

## Hard Constraints

| Constraint | Value |
|-----------|-------|
| **Page limit** | **8 pages total** (including figures and references) |
| **Format** | ACM sigconf two-column |
| **Over-length** | Rejected without review |
| **Supplementary materials** | Not accepted (no appendices) |
| **Submission deadline** | August 9, 2026, 23:59 AOE |
| **Author limit** | Max 6 submissions per individual author |
| **Author list** | Final at submission — no additions/removals after |
| **Notification** | September 27, 2026 |

## Template

- **ACM acmart v2.19** (2026-06-27), `sigconf` format.
- `acmart.cls` and `ACM-Reference-Format.bst` are included in this directory for reproducibility.
- LaTeX source: https://www.acm.org/publications/proceedings-template

## Double-Blind Review Rules

ICAIF '26 uses **double-blind** review. Both of the following must hold:

### In the LaTeX source

The documentclass uses `anonymous,review`:
```latex
\documentclass[sigconf,anonymous,review]{acmart}
```
This automatically suppresses author names, affiliations, emails, and the acknowledgments section in the compiled PDF. Line numbers are added for reviewer convenience.

### In the paper content

- **No self-identifying language.** Do not write "In our prior work [cite], we showed...". Instead use third-person: "Litvin et al. [cite] showed...".
- **Self-citation is permitted** only in third person.
- **Do not include links** to personal websites, GitHub repos, or institutions that reveal identity.
- **Acknowledgments** will be suppressed automatically, but ensure no identifying info leaks into other sections.
- **Supplementary material / code** — if submitted, must also be anonymized (no author names in filenames, READMEs, or commit history).

## ACM Formatting Rules (Enforced — Paper Will Be Returned If Violated)

1. **No margin adjustments.**
2. **No font substitutions.** The template uses Libertine — do not override.
3. **No `\vspace`, `\hspace`, or manual spacing adjustments** between elements.
4. **No redefinition of sectioning commands** (`\section`, `\subsection`, etc.).
5. **Sections through `\subsubsection` must be numbered.** Do not remove numbering.
6. **Do not simulate sections** with bold/italic first words.

## Accessibility Requirements

### Figure Descriptions (Mandatory)

Every `\includegraphics` inside a `figure` environment **must** have a `\Description{...}` command:

```latex
\begin{figure}[h]
\centering
\includegraphics[width=\columnwidth]{figures/example.pdf}
\caption{Short caption.}
\Description{Plain-text alternative description for visually impaired readers. Must be < 2000 characters. Must NOT repeat the caption. Should describe the key visual information: axis labels, trends, notable data points.}
\label{fig:example}
\end{figure}
```

### Color Usage

- Figures must be readable in greyscale.
- Do not encode information using only color differences.
- Use ColorBrewer (http://colorbrewer2.org/) or ACE (http://daprlab.com/ace/) for palette selection.

## CCS Concepts

- Generated from: https://dl.acm.org/ccs/ccs.cfm
- The `\begin{CCSXML}...\end{CCSXML}` block and the `\ccsdesc[significance]{Category~Term}` commands **must match exactly**. Both are generated together by the tool.
- Required for papers > 2 pages.

## Required Sections

| Section | Notes |
|---------|-------|
| Abstract | Required |
| CCS Concepts + Keywords | Required for papers > 2 pages |
| Introduction | Standard |
| Related Work | Standard |
| Methodology | Standard |
| Experimental Setup | Standard |
| Results | Standard |
| Discussion | Standard |
| Conclusion | Standard |
| Acknowledgments | Use `\begin{acks}...\end{acks}` environment (auto-suppressed in anonymous mode) |
| Ethics and Privacy Statement | Recommended by ACM (use `\section*{}`, unnumbered) |
| References | Use `ACM-Reference-Format` bst style |

## Bibliography

- Use BibTeX with `ACM-Reference-Format.bst`.
- Full author names (not initials): "Donald E. Knuth" not "D. E. Knuth".
- Include: title, year, volume, number, pages, DOI where available.
- Reference format: https://www.acm.org/publications/authors/bibtex-formatting

## Switching to Camera-Ready (After Acceptance)

When accepted, change the documentclass to:
```latex
\documentclass[sigconf]{acmart}
```

Then fill in the rights-form commands provided by ACM:
```latex
\setcopyright{...}       % from rights form
\copyrightyear{2026}
\acmYear{2026}
\acmDOI{...}             % from rights form
\acmISBN{...}            % from rights form
\acmBooktitle{...}       % if different from default
```

Remove `anonymous` and `review` options. Author info will render normally.

## File Structure

```
paper/
├── acmart.cls                                            # ACM class file v2.19
├── ACM-Reference-Format.bst                              # Bibliography style
├── stochastic-cross-venue-ohlcv-trading.tex              # Old OU/OHLCV draft (reference only; matches content)
├── references.bib                                        # Paper bibliography
├── generate_figures.py                                   # Script to produce figures
├── figures/                                              # Paper figures
├── acm-sample/                                           # Official ACM sample (isolated; not the paper)
│   ├── sigconf-sample.tex
│   ├── sigconf-sample.pdf
│   ├── sample-base.bib
│   ├── sampleteaser.pdf
│   └── sample-franklin.png
└── SUBMISSION_GUIDE.md
```

The **new ICAIF submission** (gradient boosting / LightGBM) will be a **new** `.tex` file created during regeneration (e.g. `gradient-boosting-cross-market-spread-prediction.tex`), started from `acm-sample/sigconf-sample.tex`. Do not overwrite the stochastic reference draft in place.

### Compiling

**Old reference draft** (from `paper/`): uses only `acmart.cls`, `references.bib`, and `figures/`. The `acm-sample/` folder is unused and does not interfere.

```powershell
cd paper
tectonic stochastic-cross-venue-ohlcv-trading.tex
```

**ACM sample** (needs parent dir on the search path so `acmart.cls` / `.bst` resolve):

```powershell
cd paper\acm-sample
tectonic -Z search-path=.. sigconf-sample.tex
```
