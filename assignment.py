import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings

warnings.filterwarnings('ignore')

# Provide a clean styling to charts
sns.set_theme(style="whitegrid", palette="muted")

def main():
    print("--- PART A: DATA PREPARATION ---")
    # 1. Load Data
    sentiment = pd.read_csv('data/sentiment.csv')
    trades = pd.read_csv('data/trades.csv')
    
    print(f"Sentiment data: {sentiment.shape[0]} rows, {sentiment.shape[1]} columns")
    print(f"Trades data: {trades.shape[0]} rows, {trades.shape[1]} columns")
    
    print(f"Missing in Sentiment:\n{sentiment.isnull().sum()}")
    print(f"Missing in Trades:\n{trades.isnull().sum()}")
    
    print(f"Duplicates in Sentiment: {sentiment.duplicated().sum()}")
    print(f"Duplicates in Trades: {trades.duplicated().sum()}")
    
    # 2. Convert timestamps and align
    sentiment['date'] = pd.to_datetime(sentiment['date'])
    # Trades Timestamp IST format observed as 'DD-MM-YYYY HH:MM'
    trades['Timestamp IST'] = pd.to_datetime(trades['Timestamp IST'], format='%d-%m-%Y %H:%M')
    trades['date'] = trades['Timestamp IST'].dt.date
    trades['date'] = pd.to_datetime(trades['date'])

    # Merge trades with sentiment based on date
    df = pd.merge(trades, sentiment[['date', 'classification', 'value']], on='date', how='left')
    
    # Drop rows without sentiment data (if any)
    df.dropna(subset=['classification'], inplace=True)
    
    print("Data alignment complete. Merged data shape:", df.shape)

    # 3. Create Key Metrics (Daily level)
    # First, let's normalize "Closed PnL" and sizes.
    # Convert 'Closed PnL' to numeric if it's not
    df['Closed PnL'] = pd.to_numeric(df['Closed PnL'], errors='coerce').fillna(0)
    df['Size USD'] = pd.to_numeric(df['Size USD'], errors='coerce').fillna(0)
    df['Start Position'] = pd.to_numeric(df['Start Position'], errors='coerce').fillna(0)
    df['is_win'] = (df['Closed PnL'] > 0).astype(int)
    
    # Calculate Leverage proxy: Size USD / (Closed PnL + some base margin).
    # Since we lack explicit margin, typical leverage proxy could be Position Size USD vs absolute PnL or assuming start position.
    # We will just use Size USD as a relative proxy for leverage appetite, or derive if 'leverage' field existed. The prompt mentions "leverage distribution", but the field is missing from our list. We'll proxy it by `Size USD` scale or Trade Frequency.
    
    daily_metrics = df.groupby(['date', 'Account', 'classification', 'value']).agg(
        daily_pnl=('Closed PnL', 'sum'),
        trades_count=('Trade ID', 'count'),
        wins=('is_win', 'sum'),
        total_size_usd=('Size USD', 'sum'),
        avg_trade_size=('Size USD', 'mean'),
        long_trades=('Side', lambda x: (x.str.upper() == 'BUY').sum()),
        short_trades=('Side', lambda x: (x.str.upper() == 'SELL').sum())
    ).reset_index()
    
    daily_metrics['win_rate'] = daily_metrics['wins'] / daily_metrics['trades_count']
    daily_metrics['long_short_ratio'] = daily_metrics['long_trades'] / (daily_metrics['short_trades'] + 1e-5)
    
    print("\n--- PART B: ANALYSIS ---")
    
    # 1. Performance vs Sentiment (Fear vs Greed)
    # Simplify classifications to broader categories if needed, or keep 'Fear', 'Greed', 'Extreme Fear', 'Extreme Greed', 'Neutral'
    def map_sentiment(c):
        if 'Fear' in c: return 'Fear'
        if 'Greed' in c: return 'Greed'
        return 'Neutral'
        
    daily_metrics['sentiment_group'] = daily_metrics['classification'].apply(map_sentiment)
    
    perf_by_sentiment = daily_metrics.groupby('sentiment_group').agg(
        mean_pnl=('daily_pnl', 'mean'),
        mean_win_rate=('win_rate', 'mean'),
        avg_trades=('trades_count', 'mean')
    )
    print("\nPerformance by Sentiment:")
    print(perf_by_sentiment)
    
    # Plot PnL by Sentiment
    plt.figure(figsize=(10, 6))
    sns.barplot(data=daily_metrics, x='sentiment_group', y='daily_pnl')
    plt.title('Average Daily PnL by Market Sentiment')
    plt.savefig('pnl_by_sentiment.png')
    plt.close()
    
    # 2. Behavioral Shifts (Trade frequency, size, long/short bias)
    behav_by_sentiment = daily_metrics.groupby('sentiment_group').agg(
        mean_trade_freq=('trades_count', 'mean'),
        mean_trade_size=('avg_trade_size', 'mean'),
        mean_ls_ratio=('long_short_ratio', 'median')
    )
    print("\nBehavioral Shifts by Sentiment:")
    print(behav_by_sentiment)
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=daily_metrics, x='sentiment_group', y='trades_count', showfliers=False)
    plt.title('Trade Frequency by Market Sentiment')
    plt.savefig('freq_by_sentiment.png')
    plt.close()

    # 3. Segments (Frequent vs Infrequent Traders)
    trader_profile = df.groupby('Account').agg(
        total_trades=('Trade ID', 'count'),
        total_pnl=('Closed PnL', 'sum'),
        avg_trade_size=('Size USD', 'mean')
    ).reset_index()
    
    q_trades = trader_profile['total_trades'].quantile(0.5)
    trader_profile['frequency_segment'] = np.where(trader_profile['total_trades'] > q_trades, 'Frequent', 'Infrequent')
    
    q_size = trader_profile['avg_trade_size'].quantile(0.5)
    trader_profile['size_segment'] = np.where(trader_profile['avg_trade_size'] > q_size, 'High Size', 'Low Size')
    
    print("\nTrader Segments Breakdown:")
    print(trader_profile['frequency_segment'].value_counts())
    
    # Merge segments back to daily metrics
    daily_metrics = pd.merge(daily_metrics, trader_profile[['Account', 'frequency_segment', 'size_segment']], on='Account', how='left')
    
    # Analyze segment behavior across sentiment
    segment_analysis = daily_metrics.groupby(['frequency_segment', 'sentiment_group'])['daily_pnl'].mean().unstack()
    print("\nAverage Daily PnL by Segment and Sentiment:")
    print(segment_analysis)
    
    segment_analysis.plot(kind='bar', figsize=(10, 6))
    plt.title('Daily PnL by Trader Segment across Sentiment')
    plt.ylabel('Average Daily PnL (USD)')
    plt.tight_layout()
    plt.savefig('segment_pnl.png')
    plt.close()

    print("\n--- PART C: ACTIONABLE OUTPUT & BONUS ---")
    
    # Bonus: Simple Predictive Model
    # Predict next day profitability bucket (Profit (>0) vs Loss (<=0))
    print("Training Predictive Model...")
    
    # We need to shift the profitability to the next trading day per account
    daily_metrics = daily_metrics.sort_values(by=['Account', 'date'])
    daily_metrics['next_day_pnl'] = daily_metrics.groupby('Account')['daily_pnl'].shift(-1)
    
    model_df = daily_metrics.dropna(subset=['next_day_pnl'])
    model_df['target_profitable'] = (model_df['next_day_pnl'] > 0).astype(int)
    
    # Features: sentiment value, current day pnl, current win rate, trades count, ls ratio, avg trade size
    features = ['value', 'daily_pnl', 'win_rate', 'trades_count', 'long_short_ratio', 'avg_trade_size']
    
    X = model_df[features].fillna(0)
    y = model_df['target_profitable']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    # Feature importances
    importances = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
    print("\nFeature Importances:\n", importances)
    
    importances.plot(kind='bar', figsize=(8, 5))
    plt.title('Feature Importances for Next-Day Profitability')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

if __name__ == "__main__":
    main()
