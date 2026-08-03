# V3 Page Reduction Summary: 11 Pages → 8 Pages

## Quick Overview

**Must cut**: 3 pages  
**Strategy**: Remove redundant sections, compress verbose explanations, merge related content

---

## Section-by-Section Reduction Plan

### 🔴 Related Work: Cut 1.0 page (2.0p → 1.0p)

| Subsection | Action | Rationale | Pages Saved |
|------------|--------|-----------|-------------|
| 2.4 Cross-Exchange Dislocations | **DELETE** | Observational only, minimal technical comparison | 0.2p |
| 2.5 Reinforcement Learning | **DELETE** | Single RL paper, orthogonal methodology | 0.25p |
| 2.1 Rule-Based Crypto | **COMPRESS** | Replace paragraph descriptions with summary table | 0.3p |
| 2.2 ML Spread Models | **COMPRESS** | Remove "what they don't report" lists | 0.2p |
| 2.3 Equities ML | **KEEP** | Essential comparison | — |
| 2.6 Positioning Table | **KEEP** | Critical differentiation | — |

**Subtotal**: 0.95 pages

---

### 🔴 Discussion: Cut 1.0 page (2.0p → 1.0p)

| Subsection | Action | Rationale | Pages Saved |
|------------|--------|-----------|-------------|
| 7.1 What Forecast Adds | **MERGE** into Results 5.3 | Interprets baseline comparison table | 0.3p |
| 7.2 Why Persistence Beats MR | **DELETE** | Already explained in Results, redundant | 0.3p |
| 7.3 Z-Score Units | **COMPRESS** | Remove inversion mechanics paragraph | 0.2p |
| 7.4 Sharpe Characterization | **COMPRESS** | Merge to single paragraph | 0.15p |
| 7.5 Baseline Sensitivity | **DELETE** | Greedy allocation not reported result | 0.25p |
| 7.6 Threats to Validity | **KEEP** | Required limitations section | — |

**Subtotal**: 1.2 pages

---

### 🟡 Ablation Studies: Cut 0.5 pages (1.5p → 1.0p)

| Subsection | Action | Rationale | Pages Saved |
|------------|--------|-----------|-------------|
| 6.1 Confidence Filter | **KEEP** | Critical model design explanation | — |
| 6.2 Direction vs Rules | **DELETE** | Duplicate of Results 5.3 | 0.3p |
| 6.3 Feature Importance | **COMPRESS** | Remove enumerated list (info in figure) | 0.25p |
| 6.4 Generalization | **MERGE** into Results 5.1 | Pertains to offline eval | 0.1p |

**Subtotal**: 0.65 pages

---

### 🟢 Minor Compressions: Cut 0.3 pages

| Section | Action | Pages Saved |
|---------|--------|-------------|
| Introduction 1.1 Dataset | Compress 2 paragraphs → 1 paragraph | 0.1p |
| Results 5.2 (Figure 2) | **DELETE** scatter plot + paragraph | 0.15p |
| Conclusion 8.1 Future Work | Compress 3 sentences → 1 sentence | 0.05p |

**Subtotal**: 0.3 pages

---

## Total Impact: 2.8 Pages Cut

| Priority | Section | Cut |
|----------|---------|-----|
| **Priority 1** | Related Work | 1.0p |
| **Priority 2** | Discussion | 1.0p |
| **Priority 3** | Ablation | 0.5p |
| **Priority 4** | Minor | 0.3p |
| **TOTAL** | | **2.8p** |

*0.2 page buffer remains for adjustments*

---

## Literature Justification

### Comparison to Similar Papers:

| Paper | Related Work | Results/Experiments | Discussion | Ablation |
|-------|-------------|---------------------|------------|----------|
| **Sarmento 2024** | 0.5p | 2.5p (33%) | 1.0p | integrated |
| **Shen 2022** | 1.0p | 2.0p | 1.0p | integrated |
| **Perrone 2026** | 1.0p | 2.0p | — | integrated |
| **Ko 2023** | 1.5p | 2.5p | — | minimal |
| **Average** | **1.0p** | **2.25p** | **0.5-1.0p** | **integrated** |
| | | | | |
| **Our V2 (current)** | 2.0p ❌ | 2.0p ✓ | 2.0p ❌ | 1.5p ❌ |
| **Our V3 (target)** | 1.0p ✓ | 2.25p ✓ | 1.0p ✓ | 1.0p ✓ |

**Key Finding**: We are 1-2 pages over on Related Work and Discussion compared to literature norms. Results section size is appropriate.

---

## What Gets Deleted (No Recovery)

### Entire Sections Removed:
1. Section 2.4: Cross-Exchange Price Dislocations (Makarov & Schoar)
2. Section 2.5: Reinforcement Learning for Spread Trading (Ning & Lee)
3. Section 7.2: Why Persistence Beats Mean-Reversion at One Snapshot
4. Section 7.5: Sensitivity of the Baseline Margin
5. Section 6.2: Learned Direction vs Mechanical Rules

### Figures Removed:
- Figure 2: Predicted vs Realized Z-Scores scatter plot

### Content Compressed (50%+ reduction):
- Section 2.1: Rule-Based Z-Score Pairs Trading (paragraph → table + 1 paragraph)
- Section 2.2: ML Models for Spread Prediction (remove negative statements)
- Section 7.3: Z-Score Units (3 paragraphs → 2 paragraphs)
- Section 7.4: Sharpe Ratio (2 paragraphs → 1 paragraph)
- Section 6.3: Feature Importance (remove enumerated list)

---

## What Gets Merged/Relocated

| Original Location | Merged Into | Reason |
|-------------------|-------------|--------|
| Discussion 7.1 | Results 5.3 | Interprets Table 5 baseline comparison |
| Ablation 6.4 | Results 5.1 | Pertains to offline evaluation metrics |
| (Optional) 2.4 snippet | Introduction | Motivation for cross-exchange study |

---

## Backup Option: Remove Campaign A

**If still >8 pages after above cuts:**

Delete all Campaign A (Jul 30) content:
- Experimental Setup 4.4: Campaign A bullet list
- Results 5.2: Campaign A subsection + Table 4
- Results 5.5: Protocol Differences section
- Figure 1: Remove Jul 30 bars (or remake figure)

**Pages saved**: 0.3–0.5 additional pages

**Rationale**: Campaign B (Jul 31) matches training parameters and is more robust; Campaign A uses different model/hyperparams

---

## Preservation Priorities (Do NOT Cut)

### Tables (All Required):
- ✅ Table 1: Positioning summary (unique differentiation)
- ✅ Table 2: Hyperparameters
- ✅ Table 3: Offline evaluation
- ✅ Table 4: Jul 30 results (unless Campaign A removed)
- ✅ **Table 5: Baseline comparison** ← **MOST CRITICAL**
- ✅ Table 6: Z-unit vs basis-point comparison

### Figures (Keep):
- ✅ Figure 1: Model vs naive R² and dir-acc across campaigns
- ✅ Figure 3: Filter R² lift
- ✅ Figure 4: Feature importance (top 20)
- ✅ Figure 5: Cumulative PnL proxy

### Sections (Preserve):
- ✅ Methodology (all 6 subsections) — technical foundation
- ✅ Results Section 5.3 — baseline comparison (expand with 7.1 merge)
- ✅ Threats to Validity 7.6 — required limitations

---

## Risk Assessment

### Low Risk Cuts (High Confidence):
- ✅ Section 2.5 (RL): Orthogonal methodology
- ✅ Section 7.2 (Persistence explanation): Redundant
- ✅ Section 6.2 (Direction ablation): Duplicate of Results 5.3
- ✅ Figure 2 (scatter): Reviewer said "depletes our credibility"

### Medium Risk Cuts (Verify Impact):
- ⚠️ Section 2.4 (Makarov & Schoar): Foundational citation, but observational
- ⚠️ Section 7.5 (Baseline sensitivity): Honest disclosure of greedy allocation result

### High Risk Cuts (Avoid Unless Desperate):
- ❌ Table 5 or Section 5.3: Core contribution
- ❌ Feature importance content: Dataset justification
- ❌ Threats to Validity: Honesty/transparency requirement

---

## Success Metrics for V3

- ✅ Page count: Exactly 8 pages
- ✅ All core claims intact (forecast R², dir-acc, baseline comparison)
- ✅ All reviewer V2 feedback preserved
- ✅ Narrative flow: no abrupt jumps from cuts
- ✅ References valid (no orphaned `\cite{}`)
- ✅ Compiles without errors

---

## Next Steps

1. **Review this summary** and approve cut priorities
2. **Execute handoff plan** in `PAGE_REDUCTION_HANDOFF.md`
3. **Compile V3 PDF** and verify 8-page target
4. **If overbudget**: Remove Campaign A (backup strategy)
5. **Commit and push** V3 to branch

---

## Files to Modify

**Heavy edits**:
- `paper/sections/related_work.tex`
- `paper/sections/discussion.tex`
- `paper/sections/ablation.tex`
- `paper/sections/results.tex` (receive merged content)

**Minor edits**:
- `paper/sections/introduction.tex`
- `paper/sections/conclusion.tex`

**No changes**:
- `paper/sections/abstract.tex`
- `paper/sections/methodology.tex`
- `paper/sections/experimental_setup.tex` (unless Campaign A removed)
- `paper/sections/ethics.tex`
- `paper/references.bib` (verify after cuts)
- `paper/gradient-boosting-cross-market-spread-prediction.tex` (main file)

---

**End of Summary**
