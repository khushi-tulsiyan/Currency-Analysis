import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

start_date = "2023-01-01"
end_date = "2024-09-30"

# Download EUR/INR data from Yahoo Finance
eur_inr_data = yf.download('EURINR=X', start=start_date, end=end_date)

eur_inr_data['1D_MA'] = eur_inr_data['Close'].rolling(window=1).mean()  # 1-day moving average
eur_inr_data['7D_MA'] = eur_inr_data['Close'].rolling(window=7).mean()  # 7-day moving average

# Bollinger Bands Calculation
eur_inr_data['20D_MA'] = eur_inr_data['Close'].rolling(window=20).mean()
eur_inr_data['20D_STD'] = eur_inr_data['Close'].rolling(window=20).std()
eur_inr_data['Upper_Band'] = eur_inr_data['20D_MA'] + (eur_inr_data['20D_STD'] * 2)
eur_inr_data['Lower_Band'] = eur_inr_data['20D_MA'] - (eur_inr_data['20D_STD'] * 2)

# Commodity Channel Index (CCI) Calculation
def CCI(df, ndays):
    TP = (df['High'] + df['Low'] + df['Close']) / 3
    CCI = (TP - TP.rolling(window=ndays).mean()) / (0.015 * TP.rolling(window=ndays).std())
    return CCI

eur_inr_data['CCI'] = CCI(eur_inr_data, 20)

# Generate Buy/Sell/Neutral Signals
def signal_decision(row):
    if row['Close'] > row['1D_MA']:
        return 'BUY'
    elif row['Close'] < row['1D_MA']:
        return 'SELL'
    else:
        return 'NEUTRAL'

eur_inr_data['Decision_MA_1D'] = eur_inr_data.apply(signal_decision, axis=1)

# Check for Buy/Sell signal from Bollinger Bands
def bollinger_decision(row):
    if row['Close'] > row['Upper_Band']:
        return 'SELL'
    elif row['Close'] < row['Lower_Band']:
        return 'BUY'
    else:
        return 'NEUTRAL'

eur_inr_data['Decision_Bollinger'] = eur_inr_data.apply(bollinger_decision, axis=1)

# CCI Signal
def cci_decision(row):
    if row['CCI'] > 100:
        return 'SELL'
    elif row['CCI'] < -100:
        return 'BUY'
    else:
        return 'NEUTRAL'

eur_inr_data['Decision_CCI'] = eur_inr_data.apply(cci_decision, axis=1)

# Handle date lookup for the final date's decisions
end_date_ts = pd.Timestamp(end_date)

# Find the closest available date on or before the specified end_date
final_decision_date = eur_inr_data.index.asof(end_date_ts)

if final_decision_date is not None:
    final_decisions = eur_inr_data.loc[final_decision_date]
    print(f"Final decisions on {final_decision_date}:\n", final_decisions)
else:
    print(f"No data available on or before {end_date}")

# Save to CSV for further analysis
eur_inr_data.to_csv('eur_inr_analysis.csv')

# Plot the graphs
plt.figure(figsize=(14, 7))
plt.title('EUR/INR Closing Price and Moving Averages')
plt.plot(eur_inr_data['Close'], label='EUR/INR Close')
plt.plot(eur_inr_data['1D_MA'], label='1 Day MA', linestyle='--')
plt.plot(eur_inr_data['7D_MA'], label='7 Day MA', linestyle='--')
plt.legend()
plt.show()

# Additional plotting for Bollinger Bands and CCI
def plot_bollinger_bands(data):
    plt.figure(figsize=(14, 7))
    plt.title('Bollinger Bands for EUR/INR')
    plt.plot(data['Close'], label='Close Price', color='blue')
    plt.plot(data['Upper_Band'], label='Upper Bollinger Band', linestyle='--', color='red')
    plt.plot(data['Lower_Band'], label='Lower Bollinger Band', linestyle='--', color='green')
    plt.fill_between(data.index, data['Lower_Band'], data['Upper_Band'], color='lightgray')
    plt.legend()
    plt.show()

def plot_cci(data):
    plt.figure(figsize=(14, 7))
    plt.title('CCI (Commodity Channel Index) for EUR/INR')
    plt.plot(data['CCI'], label='CCI', color='purple')
    plt.axhline(100, linestyle='--', color='red', label='Overbought')
    plt.axhline(-100, linestyle='--', color='green', label='Oversold')
    plt.legend()
    plt.show()

# --- Call the plotting functions ---
plot_bollinger_bands(eur_inr_data)
plot_cci(eur_inr_data)
