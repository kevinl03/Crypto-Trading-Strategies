# Manual review of the paper (pt. 2)

After receiving feedback on the paper, we are updating it incrementally. The following document outlines changes to be made in terms of the intro and related works sections. 

## Introduction 

There are a few missing key pieces of information in the current intro. 

We talk a lot about the fragmented markets, law of one price, etc. but we don't say why exactly the current work (which uses rule-based z-score thresholds) is insufficient, as directly related to the crypto market. 

A paragraph to add somewhere would be something along the lines of: 

"Much of the existing work in this space focuses on rule-based thresholds for market entry and exit, but the crypto cross-exchange environment may be better suited for different methods due to some structural constraints. First, spreads can revert quickly or slowly due to different market conditions, especially at minute-by-minute granularity; a strictly rule-based trading algorithm has no way of distinguishing which is which, and bets on mean reversion regardless. Additionally, different venues can have structural differences in terms of fees, liquidity, or latency. Using a machine learning model that takes into consideration both the coin and the venue as categorical variables can learn these differences, and trade accordingly." 

The above is a draft and can be tightened or reworded as need be, but the gist is what i would like to be included. 

## Related Work 

This section is much too long and not informative enough overall. I would like to get rid of a lot of unnecessary content, tighten up the results, and really drive homethe point about 1) why our paper is uniquely positioned both by problem context and 2) its methodology. 

I think the table of similarities and differences can be removed entirely, and I would rather have one or two cohesive paragraphs instead of a new subsection for every little difference. Additionally, I find that the section includes a lot of facts but gives very little information about how exactly our work is preceded by, and differentiated from the other papers on a deeper level. 

One main issue I have is below the fact that the Fil and Kristoufek paper (Miroslav Fil and Ladislav Kristoufek. 2020. Pairs Trading in Cryptocurrency Markets. IEEE Access 8 (2020), 172644–172651) is mentioned only a tiny bit in passing. I think we should lead with this as a related works paper. The repercussions from their paper for our work are: that classical methods for statarb/pairs trading do not work in the crypto setting, so with this in mind we decided to build an ML model instead. 

The next issue I have is the framing of the Fisher et al (Thomas Günter Fischer, Christopher Krauss, and Alexander Deinert. 2019. Statistical Arbitrage in Cryptocurrency Markets. Journal of Risk and Financial Management 12, 1 (2019), 31. https://doi.org/10.3390/jrfm12010031) in section 2.2 we have some information that a) makes their paper look larger and more significant than ours, and b) immediately mentions a threat to the validity of our results. I would like to reframe this by stating that, in their results, they found that statistically significant returns on doing statistical arbitrage of a portfolio of crypto coins would be possible, and that this helped motivate our decision to work on same-coin cross-exchange arbitrage. We do not need to mention the decay in profitability after a time delay here (as the time delay is baked into our collection and trading process). 