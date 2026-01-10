import yfinance as yf
import pandas as pd
import seaborn as sns  # type: ignore
import matplotlib.pyplot as plt
import sys

# --- CONFIGURATION ---
TICKERS = ['GOOGL', 'AMZN', 'BN', 'INTC', 'PYPL', 'LLY', 'VST', 'FLR', 'LULU', 'PINS', 'META', 'STX', 'UPS', 'DECK', 'TSSI', 'FLNC', 'POET']
YEARS_HISTORY = 10 # Check the last 10 years of history

print(f"--- ANALYZING SEASONALITY FOR {len(TICKERS)} STOCKS ({YEARS_HISTORY} YEARS) ---")

def calculate_seasonality(ticker_list, years):
    results = {}
    
    for ticker in ticker_list:
        try:
            # Download data from the beginning of the required period
            start_date = (pd.Timestamp.now() - pd.DateOffset(years=years)).strftime('%Y-%m-%d')
            data = yf.download(ticker, start=start_date, progress=False, auto_adjust=False)
            
            # Handle case where download returns DataFrame or Series
            if isinstance(data, pd.DataFrame):
                data = data['Adj Close']
            elif not isinstance(data, pd.Series):
                continue
            
            if data.empty or len(data) < 12:  # Need at least 12 months of data
                continue
            
            # Calculate monthly returns
            monthly_prices = data.resample('M').last()
            monthly_returns = monthly_prices.pct_change().dropna()
            
            # Group by year and month
            monthly_returns_df = pd.DataFrame({
                'Year': monthly_returns.index.year,
                'Month': monthly_returns.index.month,
                'Return': monthly_returns.values
            })
            
            # Calculate the AVERAGE return for each month across all years
            avg_monthly_returns = monthly_returns_df.groupby('Month')['Return'].mean() * 100
            
            results[ticker] = avg_monthly_returns
            
        except Exception as e:
            # print(f"Skipping {ticker} due to data error: {e}")
            pass
            
    # Convert results into a single DataFrame
    seasonality_df = pd.DataFrame(results).T
    
    # Rename columns to month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    seasonality_df.columns = month_names
    
    return seasonality_df

# Run the calculation
seasonality_data = calculate_seasonality(TICKERS, YEARS_HISTORY)

# Check if we have any data
if seasonality_data.empty:
    print("No data available for any tickers. Please check your ticker symbols.")
    sys.exit(1)

# --- PLOTTING ---
plt.figure(figsize=(14, 10))

# Custom color map: Green = Positive Return, Red = Negative Return
sns.heatmap(seasonality_data, 
            annot=True, 
            fmt=".1f", # Display as percentage (e.g., 2.5)
            cmap='RdYlGn', # Red (bad) to Green (good)
            vmin=-5, vmax=5, # Cap color scale at -5% and +5% for readability
            center=0,
            linewidths=.5, 
            linecolor='black')

plt.title(f'Historical Average Monthly Returns (Seasonality) - Last {YEARS_HISTORY} Years', 
          fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Ticker', fontsize=12)
plt.xlabel('Month', fontsize=12)
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()

print("\n--- SEASONALITY REPORT (Avg Monthly Return %) ---")
print("Green = Best Months (Avg +5% or more)")
print("Red = Worst Months (Avg -5% or more)")
print("\n" + seasonality_data.to_string(float_format="%.1f"))