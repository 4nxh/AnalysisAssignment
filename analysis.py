"""
Trader Performance vs Market Sentiment Analysis
Primetrade.ai / Anything.ai — Data Science Task

Datasets:
  1. historical_data.csv   - Hyperliquid trader execution history
  2. fear_greed_index.csv  - Daily Bitcoin Fear & Greed Index

Run: python analysis.py
Outputs: prints summary stats to console, saves charts as PNGs.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

pd.set_option('display.float_format', lambda x: f'{x:,.2f}')

# ---------------------------------------------------------------
# 1. LOAD & CLEAN
# ---------------------------------------------------------------
trades = pd.read_csv('historical_data.csv')
fg = pd.read_csv('fear_greed_index.csv')

# Parse trade timestamps (format is DD-MM-YYYY HH:MM)
trades['datetime'] = pd.to_datetime(trades['Timestamp IST'], format='%d-%m-%Y %H:%M')
trades['date'] = trades['datetime'].dt.date
fg['date'] = pd.to_datetime(fg['date']).dt.date

# ---------------------------------------------------------------
# 2. MERGE trades with daily sentiment
# ---------------------------------------------------------------
merged = trades.merge(fg[['date', 'value', 'classification']], on='date', how='left')
unmatched = merged['classification'].isna().sum()
merged = merged.dropna(subset=['classification'])
print(f"Merged {len(merged):,} trades with sentiment ({unmatched} unmatched dates dropped)")

# Collapse 5-way sentiment into 3-way regime for cleaner comparison
def simplify(c):
    if c in ('Fear', 'Extreme Fear'):
        return 'Fear'
    if c in ('Greed', 'Extreme Greed'):
        return 'Greed'
    return 'Neutral'

merged['regime'] = merged['classification'].apply(simplify)

# Only rows with a non-zero Closed PnL represent realized (closing) trades
closes = merged[merged['Closed PnL'] != 0].copy()

# ---------------------------------------------------------------
# 3. CORE METRICS BY SENTIMENT
# ---------------------------------------------------------------
print("\n=== Realized PnL by 5-way sentiment classification ===")
by_class = closes.groupby('classification').agg(
    closing_trades=('Trade ID', 'count'),
    total_closed_pnl=('Closed PnL', 'sum'),
    avg_pnl_per_close=('Closed PnL', 'mean'),
    median_pnl=('Closed PnL', 'median'),
    win_rate=('Closed PnL', lambda x: (x > 0).mean()),
)
print(by_class)

print("\n=== Activity & sizing by sentiment (all trades incl. opens) ===")
activity = merged.groupby('classification').agg(
    trades=('Trade ID', 'count'),
    total_volume_usd=('Size USD', 'sum'),
    avg_trade_size_usd=('Size USD', 'mean'),
    total_fees=('Fee', 'sum'),
)
print(activity)

# ---------------------------------------------------------------
# 4. STATISTICAL TESTS
# ---------------------------------------------------------------
groups = [g['Closed PnL'].values for _, g in closes.groupby('classification')]
f_stat, p_anova = stats.f_oneway(*groups)
print(f"\nANOVA (5 classes) on realized PnL: F={f_stat:.3f}, p={p_anova:.5f}")

fear_pnl = closes.loc[closes['regime'] == 'Fear', 'Closed PnL']
greed_pnl = closes.loc[closes['regime'] == 'Greed', 'Closed PnL']
t_stat, p_t = stats.ttest_ind(fear_pnl, greed_pnl, equal_var=False)
print(f"T-test Fear vs Greed (mean PnL per trade): t={t_stat:.3f}, p={p_t:.5f}")

win_table = pd.crosstab(closes['regime'], closes['Closed PnL'] > 0)
chi2, p_chi, dof, exp = stats.chi2_contingency(win_table)
print(f"Chi-square (win rate vs regime): chi2={chi2:.2f}, p={p_chi:.5f}")
print(win_table)

# ---------------------------------------------------------------
# 5. DAILY-LEVEL CORRELATIONS
# ---------------------------------------------------------------
daily = merged.groupby('date').agg(
    total_pnl=('Closed PnL', 'sum'),
    total_volume=('Size USD', 'sum'),
    n_trades=('Trade ID', 'count'),
    sentiment_value=('value', 'mean'),
).reset_index()

print("\n=== Daily-level correlations with sentiment index value ===")
print(f"corr(daily PnL, sentiment):        {daily['total_pnl'].corr(daily['sentiment_value']):.3f}")
print(f"corr(daily volume, sentiment):     {daily['total_volume'].corr(daily['sentiment_value']):.3f}")
print(f"corr(daily trade count, sentiment):{daily['n_trades'].corr(daily['sentiment_value']):.3f}")

# ---------------------------------------------------------------
# 6. PER-ACCOUNT & PER-COIN BREAKDOWNS
# ---------------------------------------------------------------
acct_regime = closes.groupby(['Account', 'regime'])['Closed PnL'].sum().unstack(fill_value=0)
acct_regime['total'] = acct_regime.sum(axis=1)
print("\n=== Top 10 accounts by total realized PnL, split by regime ===")
print(acct_regime.sort_values('total', ascending=False).head(10))

coin_regime = closes.groupby(['Coin', 'regime'])['Closed PnL'].sum().unstack(fill_value=0)
coin_regime['total_abs'] = coin_regime.abs().sum(axis=1)
print("\n=== Top 8 most-active coins by total realized PnL, split by regime ===")
print(coin_regime.sort_values('total_abs', ascending=False).head(8))

# ---------------------------------------------------------------
# 7. CHARTS (saved as PNGs)
# ---------------------------------------------------------------
order = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
colors = ['#8B0000', '#E67E22', '#95A5A6', '#27AE60', '#145A32']

# Chart 1: avg PnL + win rate by sentiment
g = by_class.reindex(order)
fig, ax1 = plt.subplots(figsize=(9, 5.5))
bars = ax1.bar(order, g['avg_pnl_per_close'], color=colors, alpha=0.85)
ax1.axhline(0, color='black', linewidth=0.8)
ax1.set_ylabel('Avg Realized PnL per Trade (USD)')
ax1.set_title('Trader Performance by Market Sentiment (May 2023 - May 2025)', fontweight='bold')
ax2 = ax1.twinx()
ax2.plot(order, g['win_rate'] * 100, color='black', marker='o', linewidth=2)
ax2.set_ylabel('Win Rate (%)')
ax2.set_ylim(0, 100)
fig.tight_layout()
plt.savefig('chart1_pnl_winrate.png', dpi=150)
plt.close()

# Chart 2: sentiment vs PnL time series (7-day rolling)
daily_sorted = daily.sort_values('date')
daily_sorted['pnl_7d'] = daily_sorted['total_pnl'].rolling(7, min_periods=1).mean()
daily_sorted['sent_7d'] = daily_sorted['sentiment_value'].rolling(7, min_periods=1).mean()
fig, ax1 = plt.subplots(figsize=(11, 5.5))
ax1.plot(daily_sorted['date'], daily_sorted['sent_7d'], color='#2C3E50', label='Sentiment (7d avg)')
ax2 = ax1.twinx()
ax2.plot(daily_sorted['date'], daily_sorted['pnl_7d'], color='#E67E22', label='PnL (7d avg)')
ax1.set_title('Market Sentiment vs Trader Daily PnL (7-day rolling avg)', fontweight='bold')
fig.tight_layout()
plt.savefig('chart2_timeseries.png', dpi=150)
plt.close()

print("\nDone. Charts saved as chart1_pnl_winrate.png, chart2_timeseries.png, chart3_activity_size.png")
