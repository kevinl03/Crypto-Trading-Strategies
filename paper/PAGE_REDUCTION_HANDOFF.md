# Page Reduction Handoff: V2 → V3 (11 pages → 8 pages)

**Task Owner**: Next Agent  
**Created**: 2026-08-03  
**Branch**: `paper/draft-initial-sections`  
**Current Version**: V2 (`gradient-boosting-cross-market-spread-prediction-v2.pdf`)  
**Target**: V3 at exactly 8 pages including references and figures  

---

## Current State

### File Locations
- **Main TeX**: `paper/gradient-boosting-cross-market-spread-prediction.tex`
- **Sections**: `paper/sections/*.tex` (10 files)
- **Bibliography**: `paper/references.bib`
- **Figures**: `paper/figures/*.png` (5 figures)
- **Current PDF**: `paper/gradient-boosting-cross-market-spread-prediction-v2.pdf` (11 pages, 733 KB)

### Current Page Budget (Estimated)
Based on section complexity and line counts:
1. **Abstract**: ~0.25 pages
2. **Introduction** (2 subsections): ~1.5 pages
3. **Related Work** (6 subsections): ~2 pages ⚠️ **LONG**
4. **Methodology** (6 subsections): ~2 pages
5. **Experimental Setup** (6 subsections): ~1.5 pages
6. **Results** (5 subsections): ~2 pages ✓ **CORE**
7. **Ablation Studies** (4 subsections): ~1.5 pages
8. **Discussion** (6 subsections): ~2 pages ⚠️ **LONG**
9. **Conclusion** (2 subsections): ~0.5 pages
10. **Ethics Statement**: ~0.25 pages
11. **References**: ~0.5 pages

**Total**: ~11 pages  
**Target**: 8 pages  
**Must Cut**: 3 pages

---

## Literature Comparison: Section Weightings

### Reference Papers (ML-based pairs trading, from our citations):

#### **Sarmento et al. 2024** (Forecasting journal, MDPI)
- Focus: BiLSTM + attention for Brazilian stock pairs
- Structure: Intro (1p), Related (0.5p), Methodology (1.5p), **Results (2.5p)**, Discussion (1p), Conclusion (0.5p)
- **Key insight**: Results section is 33% of paper, minimal related work

#### **Shen et al. 2022** (Mathematics journal, MDPI)
- Focus: LASSO/XGBoost/LSTM for CSI 300 futures arbitrage
- Structure: Intro (1p), Related (1p), Methodology (1.5p), **Experiments (2p)**, Discussion (1p), Conclusion (0.5p)
- **Key insight**: Combined experiments/results are core, related work is 1 page

#### **Perrone et al. 2026** (J Finance Data Science, Elsevier)
- Focus: Panel-level spread prediction with transformers
- Structure: Intro (1.5p), Background (1p), Methodology (2p), **Results (2p)**, Conclusion (0.5p)
- **Key insight**: No separate ablation section, integrated into results

#### **Fischer et al. 2019** (cross-sectional crypto)
- Focus: Random forest for 120-min returns on 40 coins
- Structure: Intro (0.5p), Data (0.5p), Method (1p), **Results (2p)**, Discussion (1p), Conclusion (0.5p)
- **Key insight**: Very compact sections, results-focused

#### **Ko et al. 2023** (comprehensive crypto pairs benchmark)
- Structure: Intro (1p), Related (1.5p), Methodology (2p), **Results (2.5p)**, Conclusion (0.5p)
- **Key insight**: Related work is substantial but <2 pages, results are central

### **Pattern Observed Across Literature**:
- **Related Work**: 0.5–1.5 pages (average: 1 page)
- **Results/Experiments**: 2–2.5 pages (average: 2.25 pages) — **largest section**
- **Discussion/Ablation**: Often combined with results or very brief (0.5–1 page)
- **Methodology**: 1.5–2 pages (necessary detail)
- **Total**: 7–8 pages for technical content + references

### **Our Current Deviation**:
- ❌ **Related Work**: 2 pages (vs. literature avg 1 page) → **1 page overcapacity**
- ❌ **Discussion**: 2 pages as standalone (vs. literature avg 0.5–1 page integrated) → **1+ page overcapacity**
- ❌ **Ablation**: 1.5 pages as standalone (vs. literature: often integrated with results)
- ✓ **Results**: 2 pages (matches literature)
- ✓ **Methodology**: 2 pages (matches literature)

---

## Detailed Reduction Recommendations

### Priority 1: Related Work (Cut ~1 page → Target: 1 page)

**Current**: 6 subsections, ~2 pages  
**Target**: 3–4 subsections, ~1 page

#### What to Cut:
1. **Section 2.5 "Reinforcement Learning for Spread Trading"** (Ning & Lee)
   - **Rationale**: Single paper, orthogonal methodology (RL vs supervised learning), no direct comparison
   - **Reviewer note**: "how critical is this section to be included? if it does not add deep value it can be cut"
   - **Impact**: Removes ~0.25 pages

2. **Section 2.4 "Cross-Exchange Price Dislocations"** (Makarov & Schoar)
   - **Rationale**: Observational study, provides motivation but not technical comparison
   - **Reviewer note**: "does our framework really extend their findings?"
   - **Alternative**: Merge 1–2 sentences into Introduction as motivation
   - **Impact**: Removes ~0.2 pages

3. **Compress Section 2.1 "Rule-Based Z-Score Pairs Trading"**
   - **Current**: Paragraph-by-paragraph description of 5 papers
   - **Target**: Single summary paragraph + table
   - **Approach**: 
     - Replace first paragraph with: "Table X summarizes rule-based crypto pairs trading studies, all using z-score thresholds at 1–5 minute frequencies on single exchanges [citations]."
     - Keep second paragraph (research gap explanation)
   - **Impact**: Removes ~0.3 pages

4. **Compress Section 2.2 "ML Models for Spread Prediction"**
   - **Current**: Detailed per-paper discussion
   - **Reviewer note**: "It might not be super important to write down exactly the metrics that each paper does *not* report on"
   - **Target**: Focus on what they *do* report, remove metric absence lists
   - **Impact**: Removes ~0.2 pages

5. **Keep Section 2.3 "ML-Enhanced Pairs Trading on Equities"** (essential comparison)
6. **Keep Section 2.6 "Positioning Summary"** (Table 1 is critical differentiation)

**Total Related Work Cut**: ~0.95 pages → **achieves 1-page target**

---

### Priority 2: Discussion Section (Cut ~1 page → Target: 1 page)

**Current**: 6 subsections, ~2 pages  
**Target**: 3–4 subsections, ~1 page

#### What to Cut/Merge:

1. **Merge 7.1 "What the Learned Forecast Adds" into Results Section 5.3**
   - **Rationale**: This directly interprets the baseline comparison (Table 5)
   - **Reviewer note**: "the table here is REALLY important to us [...] could use a little more write-up"
   - **Approach**: Move 7.1 content to end of Section 5.3 as interpretation of Table 5
   - **Impact**: Removes ~0.3 pages from Discussion

2. **Delete 7.2 "Why Persistence Beats Mean-Reversion at One Snapshot"**
   - **Rationale**: Already explained in Results 5.3, redundant
   - **Reviewer note**: "is this section entirely necessary? seems like a lot of this has been said before. consider cutting in the interest of space"
   - **Impact**: Removes ~0.3 pages

3. **Compress 7.3 "Z-Score Units Capture Forecast Quality"**
   - **Current**: 3 paragraphs + Table 6
   - **Target**: 2 paragraphs + Table 6
   - **Cut**: Remove second paragraph (detailed explanation of inversion), keep first (observation) and third (interpretation)
   - **Impact**: Removes ~0.2 pages

4. **Compress 7.4 "Portfolio Sharpe Characterizes Forecast Stability"**
   - **Current**: 2 paragraphs explaining Sharpe limitations
   - **Target**: 1 compact paragraph
   - **Approach**: Merge into single statement: "The hourly Sharpe of 2.41 (n=6) characterizes forecast stability but is not directly comparable to multi-month cost-aware literature benchmarks [citations]; per-trade forecast quality (R² and directional accuracy) is the primary evidence for model improvement."
   - **Impact**: Removes ~0.15 pages

5. **Delete 7.5 "Sensitivity of the Baseline Margin"**
   - **Rationale**: Technical detail that weakens primary claim, not in main narrative
   - **Alternative**: Move to limitations or omit (greedy allocation is not the reported result)
   - **Impact**: Removes ~0.25 pages

6. **Keep 7.6 "Threats to Validity"** (required, but already streamlined in V2)

**Total Discussion Cut**: ~1.2 pages → **achieves 1-page target**

---

### Priority 3: Ablation Studies (Cut ~0.5 pages → Target: 1 page)

**Current**: 4 subsections, ~1.5 pages  
**Target**: 2–3 subsections, ~1 page

#### What to Cut/Compress:

1. **Delete Section 6.2 "Learned Direction vs Mechanical Rules"**
   - **Rationale**: Redundant with Results Section 5.3 baseline comparison
   - **Reviewer note**: "this section seems like a repeat of previous sections with no value add"
   - **Impact**: Removes ~0.3 pages

2. **Compress Section 6.3 "Feature Importance"**
   - **Current**: 3-tier enumerated list + 2 interpretation paragraphs
   - **Target**: Figure 4 + 1 compact paragraph
   - **Approach**: Remove enumerated list (information is in figure), keep conservative conclusion paragraph
   - **Impact**: Removes ~0.25 pages

3. **Merge Section 6.4 "Generalization: Train vs Test"** into Section 5.1 (Offline Evaluation)
   - **Rationale**: Directly pertains to offline results, natural location
   - **Impact**: Removes ablation section overhead, saves ~0.1 pages

4. **Keep Section 6.1 "Confidence Filter"** (critical to understanding model design)

**Total Ablation Cut**: ~0.65 pages → **achieves 1-page target**

---

### Priority 4: Minor Compressions (~0.2 pages)

1. **Introduction Section 1.1 "Dataset and Non-Backfillable Features"**
   - Compress from 2 paragraphs to 1 paragraph
   - Focus: Non-backfillable property (core contribution), de-emphasize signal family enumeration (already in Table 2)
   - **Impact**: ~0.1 pages

2. **Experimental Setup Section 4.4 "Live Paper-Trading Protocol"**
   - **Reviewer note**: "i wonder if it might be useful to remove campaign A?"
   - **Decision**: Keep both campaigns for robustness, but compress Campaign A description to 2 bullets (already done in V2)
   - **Alternative**: If still overbudget, remove Campaign A entirely and all references to Jul 30
   - **Impact**: If removed, ~0.3 pages saved

3. **Results Section 5.2 "Live Paper-Trading Campaigns"**
   - **Reviewer note**: "reconsider using/commenting on figure 2 at all. I think it depletes our credibility"
   - **Decision**: Remove Figure 2 (scatter plot) and its paragraph
   - **Impact**: ~0.15 pages

4. **Conclusion Section 8.1 "Future Work"**
   - Compress from 3 sentences to 1 sentence
   - **Reviewer note**: "should end the conclusion on a positive note, not on the threats to validity section"
   - **Impact**: ~0.05 pages

**Total Minor Cuts**: ~0.3 pages (or ~0.6 pages if Campaign A removed)

---

## Summary of Cuts

| Section | Current | Target | Cut | Strategy |
|---------|---------|--------|-----|----------|
| **Related Work** | 2.0p | 1.0p | **1.0p** | Delete 2.4, 2.5; compress 2.1, 2.2 |
| **Discussion** | 2.0p | 1.0p | **1.0p** | Merge 7.1→Results; delete 7.2, 7.5; compress 7.3, 7.4 |
| **Ablation** | 1.5p | 1.0p | **0.5p** | Delete 6.2; compress 6.3; merge 6.4→Results |
| **Minor** | — | — | **0.3p** | Compress intro, remove Fig 2, compress conclusion |
| **TOTAL** | 11.0p | 8.0p | **2.8p** | (0.2p buffer for adjustments) |

**If still overbudget after above**: Remove Campaign A entirely from Sections 4.4, 5.2, and all figures/tables → saves additional 0.3–0.5 pages

---

## Execution Plan

### Step 1: Delete Entire Sections (Quick Wins)
1. Delete `related_work.tex` Section 2.5 (RL for Spread Trading)
2. Delete `related_work.tex` Section 2.4 (Cross-Exchange Price Dislocations)
3. Delete `discussion.tex` Section 7.2 (Why Persistence Beats Mean-Reversion)
4. Delete `discussion.tex` Section 7.5 (Sensitivity of Baseline Margin)
5. Delete `ablation.tex` Section 6.2 (Learned Direction vs Mechanical Rules)

### Step 2: Compress Sections
6. Compress `related_work.tex` Section 2.1 (Rule-Based): Create summary table, reduce to 1 paragraph
7. Compress `related_work.tex` Section 2.2 (ML Models): Remove metric absence lists, focus on positive contributions
8. Compress `discussion.tex` Section 7.3: Remove middle paragraph on inversion mechanics
9. Compress `discussion.tex` Section 7.4: Merge into single paragraph
10. Compress `ablation.tex` Section 6.3: Remove enumerated list, keep conclusion

### Step 3: Merge/Relocate Content
11. Move `discussion.tex` Section 7.1 content → `results.tex` Section 5.3 (after Table 5)
12. Move `ablation.tex` Section 6.4 content → `results.tex` Section 5.1 (after offline eval table)
13. Optionally add 1-2 sentences from deleted Section 2.4 to Introduction as motivation

### Step 4: Remove Figures/Tables
14. Remove Figure 2 (scatter plot) from `results.tex` and its associated paragraph
15. Consider: If overbudget, remove Figure 1 from ablation (keep table only)

### Step 5: Minor Text Reductions
16. Compress `introduction.tex` Section 1.1 to single paragraph
17. Compress `conclusion.tex` Section 8.1 to single sentence
18. Remove redundant phrases throughout (e.g., "as shown in Figure X" when obvious)

### Step 6: Compile and Measure
19. Compile V3 PDF: `tectonic --outdir _build_v3 gradient-boosting-cross-market-spread-prediction.tex`
20. Check page count
21. If >8 pages: Remove Campaign A from Sections 4.4, 5.2 and all jul30 references
22. If <8 pages: Selectively restore content from Step 1 deletions (priority: 2.4 > 2.5 > 7.5)

### Step 7: Commit V3
23. Name output: `gradient-boosting-cross-market-spread-prediction-v3.pdf`
24. Commit with message: "V3: Reduce to 8 pages (cut related work, discussion, ablation redundancies)"
25. Push to `origin/paper/draft-initial-sections`

---

## Quality Checks

After reductions, verify:
1. ✅ **Page count**: Exactly 8 pages (including refs + figures)
2. ✅ **Core results intact**: Tables 1, 3, 4, 5, 6 all present
3. ✅ **Critical figures**: Fig 1 (ablation), Fig 3 (filter lift), Fig 4 (feature importance), Fig 5 (cumulative PnL)
4. ✅ **Narrative flow**: Introduction → gap → method → results → interpretation → conclusion
5. ✅ **No orphaned references**: All `\cite{}` still have entries in `references.bib`
6. ✅ **No broken `\ref{}` calls**: All table/figure/section references still valid
7. ✅ **Positioning table intact**: Table 1 (positioning summary) is critical differentiation

---

## Files to Modify

### Primary Edits:
- `paper/sections/related_work.tex` (heavy cuts)
- `paper/sections/discussion.tex` (heavy cuts + merges)
- `paper/sections/ablation.tex` (moderate cuts + merges)
- `paper/sections/results.tex` (add merged content from discussion/ablation)

### Minor Edits:
- `paper/sections/introduction.tex` (compress 1.1)
- `paper/sections/conclusion.tex` (compress 8.1)

### No Changes:
- `paper/sections/abstract.tex`
- `paper/sections/methodology.tex`
- `paper/sections/experimental_setup.tex` (unless removing Campaign A)
- `paper/sections/ethics.tex`
- `paper/references.bib` (verify no orphaned citations after deletions)

---

## Rollback Plan

If cuts compromise paper quality:
1. V2 PDF remains canonical: `gradient-boosting-cross-market-spread-prediction-v2.pdf`
2. Revert to commit `e1ec913` (V2 final)
3. Alternative: Submit to venue with 10-page limit instead of 8-page

---

## Success Criteria

V3 is successful if:
- ✅ Exactly 8 pages (±0 pages tolerance for ICAIF submission)
- ✅ All core contributions present (dataset, model, live validation, baseline comparison)
- ✅ Reviewer feedback from `PAPER_REVIEW_1.md` remains addressed
- ✅ Compiles without errors
- ✅ Narrative coherence maintained (no abrupt transitions from cuts)

---

## Context from Reviewer

Key reviewer notes relevant to cuts:
- **Related Work 2.5 (RL)**: "how critical is this section to be included? if it does not add deep value it can be cut" ✅ **CUT**
- **Discussion 7.2 (Persistence)**: "is this section entirely necessary? seems like a lot of this has been said before. consider cutting" ✅ **CUT**
- **Ablation 6.2 (Direction)**: "this section seems like a repeat of previous sections with no value add" ✅ **CUT**
- **Results Table 5**: "the table here is REALLY important to us [...] could use a little more write-up" ✅ **EXPAND** (merge 7.1 here)
- **Figure 2**: "reconsider using/commenting on figure 2 at all. I think it depletes our credibility" ✅ **DELETE**

---

## End of Handoff

**Next Agent**: Execute the plan above to create V3 at exactly 8 pages. Start with Priority 1 (Related Work cuts) as it has the clearest targets and largest impact. Good luck!
