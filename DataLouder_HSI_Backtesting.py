# HSI Social Media Sentiment-led Strategy Backtesting Report

#### IMPORTING LIBRARIES AND DATASET ####

import pandas as pd # type: ignore

# First use Pandas to load in HSI.xlsx
hsi_full_dataset = pd.read_excel('./data/HSI.xlsx')

# Find no. of missing values
print('')
print('*** Missing Values Overview ***')
print('')

print(hsi_full_dataset.isnull().sum()) # There are 195 missing sentiment/votes values

print('')
print('*** HSI.xlsx Dataset Overview ***')
print('')

print(hsi_full_dataset)


#### MULTIPLE IMPUTATION FOR MISSING SENTIMENT DATA ####

# Multiple Imputation (MI) variables
data_for_mi = hsi_full_dataset[['Open', 'High', 'Low', 'Close', 'Up votes', 'Down votes']].copy()

from sklearn.experimental import enable_iterative_imputer # type: ignore
from sklearn.impute import IterativeImputer

# Initialize the MICE-based imputer
imp = IterativeImputer(max_iter=10, random_state=0)

# Fit and transform
imputed_array = imp.fit_transform(data_for_mi)

# Convert back to DataFrame
imputed_dataset = pd.DataFrame(imputed_array, columns=data_for_mi.columns, index=data_for_mi.index)

# Update original dataset
hsi_full_dataset['Up votes'] = imputed_dataset['Up votes'].round(2)
hsi_full_dataset['Down votes'] = imputed_dataset['Down votes'].round(2)


hsi_full_dataset.to_csv("./data/HSI_Imputed_new.csv", index = True)



#### INITIAL DATA AND TREND VISUALIZATION ####

# Initial data and trend exploration

# Import the imputed dataset
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
import pandas as pd
import datetime

hsi_imputed_dataset = pd.read_csv('./data/DataLouder Backtesting/HSI_Imputed_new.csv')

print('')
print('*** HSI Imputed Dataset Overview ***')
print('')

print(hsi_imputed_dataset)

# Convert 'Date' into date format
hsi_imputed_dataset['Date'] = pd.to_datetime(hsi_imputed_dataset['Date'])
hsi_imputed_dataset.set_index('Date', inplace = True)


hsi_imputed_dataset['Total votes'] = hsi_imputed_dataset['Up votes'] + hsi_imputed_dataset['Down votes']
hsi_imputed_dataset['Up_percent'] = hsi_imputed_dataset['Up votes'] / hsi_imputed_dataset['Total votes'] * 100
hsi_imputed_dataset['Down_percent'] = 100 - hsi_imputed_dataset['Up_percent']


# Moving Average (MA) 3, 5, 10, 30 for Up Vote (%)
hsi_imputed_dataset['Up_percent_MA2'] = hsi_imputed_dataset['Up_percent'].rolling(2).mean()
hsi_imputed_dataset['Up_percent_MA5'] = hsi_imputed_dataset['Up_percent'].rolling(5).mean()
hsi_imputed_dataset['Up_percent_MA10'] = hsi_imputed_dataset['Up_percent'].rolling(10).mean()
hsi_imputed_dataset['Up_percent_MA30'] = hsi_imputed_dataset['Up_percent'].rolling(30).mean()

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(20, 8), sharex=True,
    gridspec_kw={'height_ratios': [1, 1]}
)
ax1.plot(hsi_imputed_dataset.index, hsi_imputed_dataset['Up_percent'], alpha=0.3, label='Up vote %', color='gray')
ax1.plot(hsi_imputed_dataset.index, hsi_imputed_dataset['Up_percent_MA2'], label='2-Day MA', color='blue')
ax1.plot(hsi_imputed_dataset.index, hsi_imputed_dataset['Up_percent_MA5'], label='5-Day MA', color='green')
ax1.plot(hsi_imputed_dataset.index, hsi_imputed_dataset['Up_percent_MA10'], label='10-Day MA', color='orange')

ax1.set_ylabel('Up Vote %')
ax1.set_title('Daily Up Vote % with 2, 5 & 10-Day Moving Averages')
ax1.legend()
ax1.grid(True)

# Plot 2: Plotting HSI data
dates = mdates.date2num(hsi_imputed_dataset.index.to_pydatetime())

for i in range(len(hsi_imputed_dataset)):
    # Coding red and green candles
    open, high, low, close = hsi_imputed_dataset.iloc[i][['Open', 'High', 'Low', 'Close']]
    color = 'green' if close >= open else 'red'
    ax2.add_patch(Rectangle((dates[i] - 0.05, min (open, close)), 0.3, abs(open - close), color = color))
    ax2.plot([dates[i], dates[i]], [low, high], color = color)


import datetime

# Set X-axis limits closer to dataset's actual start/end to reduce white space
ax2.set_xlim(datetime.datetime(2022, 2, 14), datetime.datetime(2025, 3, 22))

# X-axis as date
ax2.xaxis_date()

ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.title('HSI Daily Candlestick Chart: 24 February 2022 to 12 March 2025')

# X & Y axis label names
plt.xlabel("Date")
plt.ylabel("Price")

# Add grids to graph
plt.grid(True)
plt.tight_layout()

plt.show()


#### STRATEGY DESIGN/DEVELOPMENT ####


### STRATEGY 1: Basic model - Majority Up Vote = Buy, Less than Majority = Sell ###

# Calculate daily returns

hsi_imputed_dataset['Return'] = hsi_imputed_dataset['Close'].pct_change()

# Strategy 1: Basic
hsi_imputed_dataset['Strat1_Position'] = 0
holding = False

for i in range(1, len(hsi_imputed_dataset)):
    upvote = hsi_imputed_dataset['Up votes'].iloc[i]

    if upvote > 0.55 and not holding:
        hsi_imputed_dataset.at[hsi_imputed_dataset.index[i], 'Strat1_Position'] = 1
        holding = True
    elif upvote < 0.50 and holding:
        hsi_imputed_dataset.at[hsi_imputed_dataset.index[i], 'Strat1_Position'] = 0
        holding = False
    else:
        hsi_imputed_dataset.at[hsi_imputed_dataset.index[i], 'Strat1_Position'] = hsi_imputed_dataset['Strat1_Position'].iloc[i - 1]


hsi_imputed_dataset['Strategy1_Return'] = hsi_imputed_dataset['Strat1_Position'].shift(1) * hsi_imputed_dataset['Return']

# Strategy 1 cumulative returns
hsi_imputed_dataset['Cumulative_Strategy1'] = (1 + hsi_imputed_dataset['Strategy1_Return']).cumprod()
hsi_imputed_dataset['Cumulative_BuynHold'] = (1 + hsi_imputed_dataset['Return']).cumprod()


# ### STRATEGY 2 ENHANCED: Extreme Fear and Greed, higher selling thereshold in Bull Market ###

print('')
print('Strategy 2: Contrarian with higher bull market threshold')
print('')

hsi_imputed_dataset['Return'] = hsi_imputed_dataset['Close'].pct_change()

# Strategy 2 Enhanced: Extreme Fear / Greed
hsi_imputed_dataset['Strat2e_Position'] = 0
holding = False

# Add bull market threshold (30-day moving average)
hsi_imputed_dataset['MA30'] = hsi_imputed_dataset['Close'].rolling(window=30).mean()
ma30 = hsi_imputed_dataset['MA30']

# Strategy 2: Extreme Fear Or Greed with Bull Market Threshold

for i in range(30, len(hsi_imputed_dataset)):
    closing_price = hsi_imputed_dataset['Close'].iloc[i]  # current close price
    upvote = hsi_imputed_dataset['Up votes'].iloc[i]

    if upvote <= 0.35 and not holding:
        hsi_imputed_dataset.at[hsi_imputed_dataset.index[i], 'Strat2e_Position'] = 1
        holding = True
    elif upvote >= 0.65 and closing_price < ma30.iloc[i] and holding:
        hsi_imputed_dataset.at[hsi_imputed_dataset.index[i], 'Strat2e_Position'] = 0
        holding = False
    else:
        hsi_imputed_dataset.at[hsi_imputed_dataset.index[i], 'Strat2e_Position'] = hsi_imputed_dataset['Strat2e_Position'].iloc[i - 1]


# Shift strategy position by 1 to avoid lookahead bias
hsi_imputed_dataset['Strategy2e_Return'] = hsi_imputed_dataset['Strat2e_Position'].shift(1) * hsi_imputed_dataset['Return']

# Strategy 2 Enhanced: cumulative returns vs Buy and Hold
hsi_imputed_dataset['Cumulative_Strategy2e'] = (1 + hsi_imputed_dataset['Strategy2e_Return']).cumprod()
hsi_imputed_dataset['Cumulative_BuynHold'] = (1 + hsi_imputed_dataset['Return']).cumprod()


# STRATEGY 3: Gap Up/Down and MA10 Sentiment
# Ensure date index is sorted
hsi_imputed_dataset = hsi_imputed_dataset.sort_index()

# Compute previous high and low
hsi_imputed_dataset['Prev_High'] = hsi_imputed_dataset['High'].shift(1)
hsi_imputed_dataset['Prev_Low'] = hsi_imputed_dataset['Low'].shift(1)

# Compute MA10 of Upvote %
hsi_imputed_dataset['Upvote_MA10'] = hsi_imputed_dataset['Up_percent'].rolling(window=10).mean()

# Initialize columns
hsi_imputed_dataset['Buy_Signal_3'] = False
hsi_imputed_dataset['Sell_Signal_3'] = False
hsi_imputed_dataset['Strategy3_Position'] = 0

# Strategy logic
holding = False
buy_streak = 0
sell_streak = 0

for i in range(1, len(hsi_imputed_dataset)):
    today = hsi_imputed_dataset.index[i]
    yesterday = hsi_imputed_dataset.index[i - 1]

    open_today = hsi_imputed_dataset.at[today, 'Open']
    high_yesterday = hsi_imputed_dataset.at[yesterday, 'High']
    low_yesterday = hsi_imputed_dataset.at[yesterday, 'Low']
    ma10 = hsi_imputed_dataset.at[today, 'Upvote_MA10']

    # Buy streak
    if open_today > high_yesterday:
        buy_streak += 1
    else:
        buy_streak = 0

    # Sell streak
    if open_today < low_yesterday:
        sell_streak += 1
    else:
        sell_streak = 0

    # Conditions for Buy in
    if not holding and buy_streak >= 2:
        hsi_imputed_dataset.at[today, 'Buy_Signal_3'] = True
        hsi_imputed_dataset.at[today, 'Strategy3_Position'] = 1
        holding = True
        sell_streak = 0
        continue

    # Conditions for Selling
    if holding and (sell_streak >= 2 or ma10 > 60):
        hsi_imputed_dataset.at[today, 'Sell_Signal_3'] = True
        hsi_imputed_dataset.at[today, 'Strategy3_Position'] = 0
        holding = False
        buy_streak = 0
        sell_streak = 0
    else:
        hsi_imputed_dataset.at[today, 'Strategy3_Position'] = hsi_imputed_dataset.at[yesterday, 'Strategy3_Position']

hsi_imputed_dataset['Strategy3_Return'] = hsi_imputed_dataset['Strategy3_Position'].shift(1) * hsi_imputed_dataset['Return']
hsi_imputed_dataset['Cumulative_Strategy3'] = (1 + hsi_imputed_dataset['Strategy3_Return']).cumprod()



# Calculate strategy returns
hsi_imputed_dataset['Strategy3_Return'] = hsi_imputed_dataset['Strategy3_Position'].shift(1) * hsi_imputed_dataset['Return']
hsi_imputed_dataset['Cumulative_Strategy3'] = (1 + hsi_imputed_dataset['Strategy3_Return']).cumprod()


#### COMBINED GRAPH ####
plt.figure(figsize=(12, 6))

# Plot each strategy's cumulative return
plt.plot(hsi_imputed_dataset['Cumulative_Strategy1'], label='Strategy 1: Basic', color='#006400')
plt.plot(hsi_imputed_dataset['Cumulative_Strategy2e'], label='Strategy 2: Contrarian (Extreme Fear/Greed)', color='#90EE90')
plt.plot(hsi_imputed_dataset['Cumulative_Strategy3'], label='Strategy 3: Gap Up/Down & Up Vote MA10',color='#228B22')
plt.plot(hsi_imputed_dataset['Cumulative_BuynHold'], label='Buy and Hold', color = 'red')

plt.legend()
plt.title('HSI Cumulative Returns: Active Sentiment-based Management Strategies vs. Buy and Hold')
plt.xlabel('Date')
plt.ylabel('Cumulative Return (%)')
plt.grid(True)
plt.tight_layout()
plt.show()




####### BACKTESTING METRICS #######
import quantstats as qs  # type: ignore
qs.extend_pandas()


# Strategy 1: Basic
cumulative_return_1 = hsi_imputed_dataset['Cumulative_Strategy1'].iloc[-1] - 1
sharpe_ratio_1 = hsi_imputed_dataset['Strategy1_Return'].mean() / hsi_imputed_dataset['Strategy1_Return'].std() * (252 ** 0.5)
max_drawdown_1 = (hsi_imputed_dataset['Cumulative_Strategy1'] / hsi_imputed_dataset['Cumulative_Strategy1'].cummax() - 1).min()
sortino_ratio_1 = hsi_imputed_dataset['Strategy1_Return'].dropna().sortino()


print('')
print('*** Backtesting measures for Strategy 1: Basic ***')
print(f"Cumulative Return: {cumulative_return_1:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio_1:.2f}")
print(f"Max Drawdown: {max_drawdown_1:.2%}")
print(f"Sortino Ratio: {sortino_ratio_1:.2f}")
print(f"Win Rate: {hsi_imputed_dataset['Strategy2e_Return'].dropna().win_rate():.2%}")




# Strategy 2: Extreme Fear/Greed
cumulative_return_2 = hsi_imputed_dataset['Cumulative_Strategy2e'].iloc[-1] - 1
sharpe_ratio_2 = hsi_imputed_dataset['Strategy2e_Return'].mean() / hsi_imputed_dataset['Strategy2e_Return'].std() * (252 ** 0.5)
max_drawdown_2 = (hsi_imputed_dataset['Cumulative_Strategy2e'] / hsi_imputed_dataset['Cumulative_Strategy2e'].cummax() - 1).min()
sortino_ratio_2 = hsi_imputed_dataset['Strategy2e_Return'].dropna().sortino()

print('')
print('*** Backtesting measures for Strategy 2: Ext reme Fear/Greed ***')
print('')

print(f"Cumulative Return: {cumulative_return_2:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio_2:.2f}")
print(f"Max Drawdown: {max_drawdown_2:.2%}")
print(f"Sortino Ratio: {sortino_ratio_2:.2f}")
print(f"Win Rate: {hsi_imputed_dataset['Strategy2e_Return'].dropna().win_rate():.2%}")


# Strategy 3: Gap Up/Down & Up Vote MA10
cumulative_return_3 = hsi_imputed_dataset['Cumulative_Strategy3'].iloc[-1] - 1
sharpe_ratio_3 = hsi_imputed_dataset['Strategy3_Return'].mean() / hsi_imputed_dataset['Strategy3_Return'].std() * (252 ** 0.5)
max_drawdown_3 = (hsi_imputed_dataset['Cumulative_Strategy3'] / hsi_imputed_dataset['Cumulative_Strategy3'].cummax() - 1).min()
sortino_ratio_3 = hsi_imputed_dataset['Strategy3_Return'].dropna().sortino()

print('')
print('*** Strategy 3: Gap Up/Down & Up Vote MA10 ***')
print('')

print(f"Cumulative Return: {cumulative_return_3:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio_3:.2f}")
print(f"Max Drawdown: {max_drawdown_3:.2%}")
print(f"Sortino Ratio: {sortino_ratio_3:.2f}")
print(f"Win Rate: {hsi_imputed_dataset['Strategy3_Return'].dropna().win_rate():.2%}")



# Passive: Buy and Hold
cumulative_return_bh = hsi_imputed_dataset['Cumulative_BuynHold'].iloc[-1] - 1
sharpe_ratio_bh = hsi_imputed_dataset['Return'].mean() / hsi_imputed_dataset['Return'].std() * (252 ** 0.5)
max_drawdown_bh = (hsi_imputed_dataset['Cumulative_BuynHold'] / hsi_imputed_dataset['Cumulative_BuynHold'].cummax() - 1).min()
sortino_bh = hsi_imputed_dataset['Return'].dropna().sortino()

print('')
print('*** Backtesting measures for Buy and Hold ***')
print('')

print(f"Cumulative Return: {cumulative_return_bh:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio_bh:.2f}")
print(f"Max Drawdown: {max_drawdown_bh:.2%}")
print(f"Win Rate: {hsi_imputed_dataset['Cumulative_BuynHold'].dropna().win_rate():.2%}")
print(f"Sortino Ratio: {sortino_bh:.2f}")




