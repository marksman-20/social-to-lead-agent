# Trader Performance vs Market Sentiment Analysis 

## Methodology
The assignment involved processing and merging two datasets: `sentiment.csv` (Bitcoin Fear/Greed Index values by date) and `trades.csv` (Historical Hyperliquid trader execution data).
1. **Cleaning & Standardization**: Missing values and duplicates were handled. Time stamps were localized to Indian Standard Time (IST), explicitly matching the daily format `DD-MM-YYYY` to extract the corresponding date (`YYYY-MM-DD`). 
2. **Merging**: The trades were inner-joined to the sentiment dataset using the standardized date.
3. **Metric Calculation**: Daily metrics per trader were computed, including daily PnL (aggregating absolute PnL), Trade Frequency (counts of executions), Win Rate (percentage of positions with PnL > 0), Average Trade Size (USD proxy for leverage appetite), and Long/Short transaction ratios.
4. **Segmentation**: Traders were categorized based on their lifetime behavior into 'Frequent' (above median daily trades) and 'Infrequent' (below median). 
5. **Predictive Modeling**: To predict next-day profitability, a Random Forest Classifier model was trained on the lag features (current-day PnL, Win Rate, Size, Frequency, Sentiment value).

---

## Key Insights
1. **Higher Volatility in PnL during Fear Regimes**:
   It was empirically observed that average daily PnL is highest during **Fear** conditions (~$4,445), closely followed by **Greed** (~$4,398). However, neutral days yield substantially worse average performance (~$2,709). This indicates that volatility inherent in strong sentiment days allows for greater profit generation overall if managed correctly.
2. **Behavioral Shifts During Greed**:
   Traders significantly amplify their trading intensity during **Greed** days. The average daily trade execution frequency almost doubles (105 trades per trader vs 57 on Fear days) alongside a heavy shift towards Long bias (Long/Short ratio rises from ~1.05 to ~1.54). Despite this intense scaling, their mean win-rate drops from 69.7% (Fear) to 66.9% (Greed).
3. **Frequent Traders Underperform in Greed relative to Fear**:
   The segmentation analysis shows distinct patterns: "Frequent" traders average higher gains on Fear Days (~$5,968) heavily outperforming on these "panicked" days, whereas "Infrequent" traders surprisingly dominate during generic Greed days (~$4,987). Frequent traders suffer heavily on average in Greed, dropping to ~$3,846 meaning their heightened activity translates poorly, whereas Infrequent traders drop massively if they trade during Fear (~$3,090).

---

## Strategy Recommendations (Actionable Output)
Based on the observations quantified above, we propose two empirical rules-of-thumb:

**Strategy 1: Mean Reversion Moderation for Frequent Traders**
> *"During Greed days, drastically scale down leverage and limit frequency constraints if you are a statistically Frequent Trader."*
Our data indicates that Frequent Traders lose massive efficiency on Greed days despite trying to execute heavily. Setting a dynamic cap to reduce automated executions on Greed regimes forces mean-reverted safety while retaining extreme edge during Fear regimes.

**Strategy 2: Fear Filtering for Low-Frequency (Infrequent) Traders**
> *"During Fear metrics <= 40, Infrequent traders should avoid discretionary trend-holding."*
As Infrequent Traders generally underperform on high-volatility Fear days, systems trading these accounts should force smaller position sizes and tighten stop-losses, acting only on verified "Greed" tailwinds where they exhibit highest relative returns.
