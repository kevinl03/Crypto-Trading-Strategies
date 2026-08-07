# Manual review of the paper 


## Lit Review 

In this section I will check all in-text citations against the pdf of the cited paper to ensure that a) the citation matches a claim/result from the cited paper, b) the authorship, title, etc are properly attributed, and c) if there are any missing citations that should be added. 

There are a few papers missing from the citations/in-text. They have been verified and can be added in the citation list. The links are below: 

1. Sarmento et al. - https://www.mdpi.com/2571-9394/6/2/24

2. Shen et al. - https://pmc.ncbi.nlm.nih.gov/articles/PMC9601484/

3. Perrone et al. - https://www.sciencedirect.com/science/article/pii/S2405918826000024


____

## Paper Content 

In this section I check the content in terms of problem framing/explanation, content accuracy, any irrelevant or redundant or missing information.


### Abstract

1. First sentence - this is a true statement but I would like to have a stronger emphasis on WHY we would want/prefer using a predictive model over a rule-based trading structure. 

2. i don't know if claiming "strictly stronger" is entirely honest, but stronger for sure 


### 1. Introduction 

1. check the accuracy of "co-moving" - is this a scientific term to be backed up by some calculation, or is it just a descriptor. if the latter then ok. if the former, then come up with some generalized synonym. 

2. would be careful about saying "every existing pairs-trading study" because theoretically our lit review could have missed some stuff. possibly replace with "existing literature mainly uses [...]" or something similar. 

3. our lightGBM model can predict the next-snapshot z-score, replacing the mechanical threshold... but why? again it is not clear why someone would take the time and computing resources to do it our way over just checking z-score at every time step. NEED TO BE CLEAR WHY THE EXTRA EFFORT IN BUILDING THE MODEL PAYS OFF IN TERMS OF RESULTS. 

4. the sentence "the model ingests 68 features spanning [...]" is an exact duplication of the one in the abstract - this needs to be reworded. 

5. "acting as a learned abstention mechanism" is deeply pretentious wording. leave out entirely to say "[...], allowing the model to select high-quality entries from the forecast distribution to trade on." 

#### 1.1 Dataset and Non-Backfillable Features

1. check something - are the spread matrices precomputed or do we build them from the data? 

2. "at approximate one-minute granularity" - this needs to be added to the last sentence bc claiming strict one-minute granularity is untrue. 

3. is withholding the url standard practice here? i know we can't publish it because it says SFU as well as our first and last names in there but just as a double check am flagging this

#### 1.2 Contributions 

1. I would possibly reorder this to be in chronological order, so that it's dataset, model, paper trading, results (ie. dataset first) but am open to keeping it this way as well

2. in point (3) replace "percentage points" with "%"


### 2. Related Work 

1. it says we organize along 2 axes but there are more than 2 subsections. why say only 2? or why not group everything differently - but the current grouping works nicely. 

#### 2.1 Rule-Based Z-Score Pairs Trading in Crypto 

1. why does everyone use 26 Binance coins? is this true or a hallucination - need to check literature

2. what did Ko et al get in their results? 

3. generally speaking the first paragraph is a lot of regurgitation of results. the second paragraph is good at showing the research gap between us and them, but it would be nice to have a bit of a conclusion sentence in the first paragraph tying together the common findings/conclusions. 

#### 2.2 ML Models for Spread Prediction 

1. At first glance, this looks pretty long, much longer than sectio 2.1. is there a good reason for this or can it be tightened. 

2. as a flag, we are messy with the notation. sometimes i've seen $|\hat{z}|$ and sometimes it's just normal $|z|$. 

3. It might not be super important to write down exactly the metrics that each paper does *not* report on. 

4. Additionally, it would be better if we also had a small overarching paraghaph summarizing results here as it was above. 

#### 2.3 ML-Enhanced Pairs Trading on Equities

1. FIX THE CITATION FOR SARMENTO ET AL!! (the link is: https://www.mdpi.com/2571-9394/6/2/24), claims from the paper are good 

2. FIX CITATION FOR SHEN ET AL!!! (the link is: https://pmc.ncbi.nlm.nih.gov/articles/PMC9601484/), claims/citations made in the paper are correct

3. FIX CITATION FOR PERRONE ET AL!!!!! (the link is: https://www.sciencedirect.com/science/article/pii/S2405918826000024), claims in the paper are all ok

4. in the last paragraph here can we also make note of a similarity between us and their work, rather than only citing differences. 

#### 2.4 Cross-Exchange Price Dislocations

1. define price dislocations 

2. does our framework really extend their findings? 

#### 2.5 Reinforcement Learning for Spread Trading 

1. how critical is this section to be included? if it does not add deep value it can be cut

#### 2.6 Positioning Summary

1. would it be a good idea to order the papers in descending order of matching features between us and them? 


### 3. Methodology 

#### 3.1 Cross-Exchange Spread and Z-Score Target 

1. again we are mixing hats, bars, and no embellishments. it is standard practice to use hat for estimators, bar for estimates, and no embellishment for theoretical population-level quantities. please ensure that our notation matches this. 

#### 3.2 Supervised Prediction Target 

1. TYPO: here it says "production model" -> change to "prediction model" 

2. is MIN_PERIODS defined anywhere before telling that MIN_PERIODS = 90? 

#### 3.3 Feature Engineering 

1. do we need to explain why we turned off ohlcv? 

#### 3.4 Model: LightGBM Gradient Boosting 

1. note that we tested turning off early stopping, and the model never improved test scores, indicating that we aren't stuck in a local min (this is to guard against hypothetical criticism)

#### 3.5 Trading Policy 

1. did we really not account for fees? will we have problems with this during review? how can we somehow control/hedge for this? 

#### 3.6 Evaluation Metrics

1. need to have a note to justify why we approximate risk-free rate as 0. this is generally very bad to do in practice but it is okay because in theory due to the law of one price we should assume that trading from one coin to another would give us 0% return, and because we are trading minute-by-minute it should also be 0% 


### 4. Experimental Setup 

#### 4.1 Data Collection 

ok here. 

#### 4.2 Asset Universe 

ok 

#### 4.3 Train/Test Split 

1. good job flagging that row count is not IID. 

#### 4.4 Live Paper-Trading Protocol 

1. i have a question - what are the two different models used between campaigns A and B? should we write a note telling readers the difference? 

2. Don't tell them our trading collector crashed after 8 hours. it was an 8 hour campaign and leave it there. 

#### 4.5 Baseline Definitions 

ok

#### 4.6 Reproducibility 

ok 


### 5. Results 

intro paragraph ok

#### 5.1 Offline Evaluation 

ok

#### 5.2 Live Paper-Trading Campaigns

1. again, do not remove anything but I wonder if it might be useful to remove campaign A? since the parameters are different between A and B, and B matches the params used in the training set. 

2. reconsider using/commenting on figure 2 at all. I think it depletes our credibility publishing this image. use the one for the test set of campaign B if we really want a scatterplot or nothing at all . 

#### 5.3 Baseline Comparison: Learned Forecast vs. Mechanical Z-Score Rules 

1. the table here is REALLY important to us. this basically shows that we are better than the mechanical baseline which is used in a lot of previous literature. this is big, and could use a little more write-up because this is very much a key finding. 

#### 5.4 Portfolio Risk-Adjusted Performance

1. might be a good idea to explain why we did hour-to-hour sharpe ratio, not just what it is. 

#### 5.5 Protocol Differences Between Campaigns 

1. if we decide to get rid of Campaign A entirely, this will be unnecessary.  


### Ablation Studies 

#### 6.1 Confidence Filter 

1. AGAIN SARMENTO ET AL WHAT IS THE CITATION (https://www.mdpi.com/2571-9394/6/2/24 - as above)

2. change the framing of rhe last paragraph. yes it's true that the dir_acc comes a lot from the persistence of high z-scores, but we need to give more emphasis to the fact that the model is predicting magnitude accurately BEFORE we talk about why dir_acc is comparable to naive persistence 

#### 6.2 Learned Direction vs Mechanical Rules 

1. this section seems like a repeat of previous sections with no value add. 

#### 6.3 Feature Importance: Role of Microstructure Inputs 

1. the list of grouped features is nice. 

2. is the last paragraph weakening our paper? should we run this study and add it? this might be possible to be put instead of our exising ablation study but do not change the paper or any code, this is just a note for ourselves for later consideration. 

#### 6.4 Generalization: Train vs. Test

ok for now 


### 7. Discussion 

#### 7.1 What the Learned Forecast Adds 

1. in the first paragraph, good that we identify that this tells us magnitude and not only direction. however we need to right off the bat explain what implications this has for us in practice. 

2. the second paragraph is pretty weak. should mention what the dir_acc is before filtering, and also say that filtering on high predictions raises the dir_acc, and not say the sentence about "would be measuring mostly the filter" 

#### 7.2 Why Persistence Beats Mean-Reversion at One Snapshot 

1. is this section entirely necessary? seems like a lot of this has been said before. consider cutting in the interest of space if not entirely critical? 

#### 7.3 Z-Score Units Are Not Economic Profit 

1. need to review and reframe if possible. this pokes a ton of holes but need to reframe in alignment with what our thesis was

#### 7.4 Sharpe Ratio is Not the Discriminating Metric 

1. again this seems overly critical right off the bat. should lead with the fact that we only report for context, and that it measures some different things than the other literature because, as we said, the others are multi-month, capital normalized, etc. so we should caution against direct comparison to the 0.95-7.94 range. 

#### 7.5 Sensitivity of the Baseline Margin 

fine for now 

#### 7.6 Threats to Validity 

1. do not include the operational artifacts subitem. cut it entirely. 

2. dependence structure point is good as is 

3. execution latency point is good. 

4. proxy point is good 

5. evaluation horizon point is okay but should not mention the two different models and the issues about bad ablation. 


### 8. Conclusion 

1. should end the conclusion on a positive note, not on the threats to validity section 

#### 8.1 

good 


### Ethics and Privacy Statement

good


____
 
## Paper Format

This is stylistic in terms of font sizes and spacing, table width and alignment, and images. 

1. CCS Concepts - there is inconsistency in the boldface and italics here 

2. Please be sure to use hat for estimated values (eg. $\hat{z}$) and to follow proper statistical conventions for this as stated above. 

3. Table 1 is overly wide and exits its column

4. multiple issues with lines escaping their columns 

5. Table 4 and 5 and 6 are all too wide and breaks the col
