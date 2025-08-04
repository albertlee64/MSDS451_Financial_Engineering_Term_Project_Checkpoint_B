# %%
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import skewnorm

# %%
tickers = [
    'AAPL', 'JPM', 'HON', 'NOC', 'MSFT', 'JNJ', 'PG',
    'SPY', 'IWM', 'QQQ', 'ACWI', 'TLT', 'GLD', 'VNQ', 'UUP'
]
benchmark = 'SPY'
start_date = "1999-01-01"
end_date = "2024-12-31"
initial_investment = 1_000_000
risk_free_rate = 0.01

# %%
np.random.seed(42)

# %%
data = {}
for ticker in tickers:
    hist = yf.Ticker(ticker).history(start=start_date, end=end_date)
    if {'Close', 'Dividends'}.issubset(hist.columns):
        data[ticker] = {
            "Adj Close": hist["Close"],  # Substituting Close for Adj Close
            "Dividends": hist["Dividends"]
        }
    else:
        print(f"{ticker} missing expected columns: {hist.columns.tolist()}")

# Combine into DataFrames
adj_close_df = pd.DataFrame({ticker: df["Adj Close"] for ticker, df in data.items()}).ffill()
dividends_df = pd.DataFrame({ticker: df["Dividends"] for ticker, df in data.items()}).fillna(0)

if adj_close_df.empty:
    raise ValueError("No valid data loaded into adj_close_df")

# %%
prices = yf.download(tickers + [benchmark], start=start_date, end=end_date)['Close']
returns = prices.pct_change().dropna()
asset_returns = returns[tickers]
benchmark_returns = returns[benchmark]

adj_close_df.to_csv("adj_close_prices.csv")
dividends_df.to_csv("dividends.csv")

# %%
valid_start = adj_close_df.apply(lambda x: x.first_valid_index())
valid_end = adj_close_df.apply(lambda x: x.last_valid_index())

price_return = pd.Series(dtype=float)
total_return = pd.Series(dtype=float)
total_dividends = pd.Series(dtype=float)

for ticker in adj_close_df.columns:
    start = valid_start[ticker]
    end = valid_end[ticker]
    if pd.isna(start) or pd.isna(end):
        continue
    p_start = adj_close_df.loc[start, ticker]
    p_end = adj_close_df.loc[end, ticker]
    div_total = dividends_df.loc[start:end, ticker].sum()
    price_return[ticker] = (p_end / p_start) - 1
    total_return[ticker] = ((p_end + div_total) / p_start) - 1
    total_dividends[ticker] = div_total

summary_table = pd.DataFrame({
    "Start Date": valid_start,
    "End Date": valid_end,
    "Capital Appreciation (%)": price_return * 100,
    "Total Dividends ($)": total_dividends,
    "Total Return (%)": total_return * 100
}).round(2).sort_values("Total Return (%)", ascending=False)

print("Total Returns Summary with Start Dates (Based on First Available Date of Asset)")
print(summary_table)

# %%
#Normalize to base = 100 and compute equal-weighted portfolio
normalized_prices = adj_close_df / adj_close_df.iloc[0] * 100
equal_weighted_portfolio = normalized_prices.mean(axis=1)
equal_weighted_portfolio.index = equal_weighted_portfolio.index.tz_localize(None)

#Plot the portfolio and highlight market crises
plt.figure(figsize=(14, 8))
plt.plot(equal_weighted_portfolio, label="Equal-Weighted Portfolio", linewidth=2, color='navy')

highlight_periods = [
    ("2000-03", "2002-10", "Dot-Com Bubble"),
    ("2007-07", "2009-03", "Subprime Crisis"),
    ("2007-07", "2007-08", "Quant Meltdown"),
    ("2009-01", "2014-12", "European Debt Crisis"),
    ("2020-02", "2021-12", "COVID-19")
]

for start, end, label in highlight_periods:
    plt.axvspan(pd.to_datetime(start), pd.to_datetime(end), color='gray', alpha=0.2, label=label)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=8)
plt.title("Growth of $100 Investment (Equal-Weighted Portfolio)")
plt.ylabel("Indexed Value (Base = 100)")
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
highlight_periods = [
    ("2000-03-01", "2002-10-31", "Dot-Com Bubble"),
    ("2007-07-01", "2009-03-31", "Subprime Crisis"),
    ("2007-07-01", "2007-08-31", "Quant Meltdown"),
    ("2009-01-01", "2014-12-31", "European Debt Crisis"),
    ("2020-02-01", "2021-12-31", "COVID-19")
]

returns_data = []

for start, end, label in highlight_periods:
    start_date = equal_weighted_portfolio.index.asof(pd.to_datetime(start))
    end_date = equal_weighted_portfolio.index.asof(pd.to_datetime(end))
    start_value = equal_weighted_portfolio.loc[start_date]
    end_value = equal_weighted_portfolio.loc[end_date]
    period_return = (end_value / start_value - 1) * 100
    returns_data.append((label, f"{start_date.date()} to {end_date.date()}", round(period_return, 2)))

start_val = equal_weighted_portfolio.iloc[0]
end_val = equal_weighted_portfolio.iloc[-1]
total_return = (end_val / start_val - 1) * 100
returns_data.append(("Total Cumulative Return", f"{equal_weighted_portfolio.index[0].date()} to {equal_weighted_portfolio.index[-1].date()}", round(total_return, 2)))

returns_df = pd.DataFrame(returns_data, columns=["Period", "Date Range", "Portfolio Return (%)"])
print("Portfolio Returns Summary")
print(returns_df)

# %% [markdown]
# # Method 1: Monte Carlo Dirichlet Process (DP) #

# %%
daily_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()

# %%
# Annualized expected returns and covariance matrix
mean_returns = daily_returns.mean() * 252
cov_matrix = daily_returns.cov() * 252

# %%
n_simulations = 1000
results = np.zeros((n_simulations, len(mean_returns)+3))

for i in range(n_simulations):
    weights = np.random.dirichlet(np.ones(len(mean_returns)), size=1)[0]
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - 0.01) / portfolio_volatility
    results[i, :len(mean_returns)] = weights
    results[i, -3] = portfolio_return
    results[i, -2] = portfolio_volatility
    results[i, -1] = sharpe_ratio

# %%
columns = list(mean_returns.index) + ['Return', 'Volatility', 'Sharpe']
results_df = pd.DataFrame(results, columns=columns)

# %%
optimal_idx = results_df['Sharpe'].idxmax()
worst_idx = results_df['Sharpe'].idxmin()
optimal_portfolio = results_df.loc[optimal_idx]
worst_portfolio = results_df.loc[worst_idx]
optimal_weights = optimal_portfolio[:len(mean_returns)]

# %%
print("Optimal Portfolio Allocation via Monte Carlo")
print(optimal_weights.sort_values(ascending=False))

print("Portfolio Performance:")
print(f"Expected Annual Return: {optimal_portfolio['Return']:.2%}")
print(f"Annual Volatility: {optimal_portfolio['Volatility']:.2%}")
print(f"Sharpe Ratio: {optimal_portfolio['Sharpe']:.2f}")

# %%
plt.figure(figsize=(10,6))
plt.scatter(results_df['Volatility'], results_df['Return'], c=results_df['Sharpe'], cmap='viridis', alpha=0.7)
plt.colorbar(label='Sharpe Ratio')
plt.scatter(optimal_portfolio['Volatility'], optimal_portfolio['Return'], c='blue', marker='o', s=100, label='Max Sharpe')
plt.scatter(worst_portfolio['Volatility'], worst_portfolio['Return'], c='red', marker='o', s=100, label='Min Sharpe')
plt.title("Monte Carlo Simulated Efficient Frontier")
plt.xlabel("Annualized Volatility")
plt.ylabel("Annualized Return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# # Method 2: Monte Carlo Deterministic Optimization Technique #

# %%
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.dot(weights, mean_returns)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, volatility

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.01):
    returns, volatility = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(returns - risk_free_rate) / volatility

num_assets = len(mean_returns)
args = (mean_returns, cov_matrix)
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for _ in range(num_assets))
initial_weights = num_assets * [1. / num_assets]

optimal = minimize(negative_sharpe_ratio, initial_weights, args=args,
                   method='SLSQP', bounds=bounds, constraints=constraints)
classic_weights = pd.Series(optimal.x, index=mean_returns.index)

# %%
n_simulations = 1000
years = 25
days = years * 252
mc_results = []

# %%
for _ in range(n_simulations):
    simulated_returns = np.random.multivariate_normal(mean_returns / 252, cov_matrix / 252, days)
    
    portfolio_returns = np.dot(simulated_returns, classic_weights)
    
    cumulative_return = np.prod(1 + portfolio_returns)
    annualized_return = cumulative_return**(1 / years) - 1
    
    mc_results.append(annualized_return)

mc_results = np.array(mc_results)

# %%
summary = {
    "Mean Annual Return": np.mean(mc_results),
    "Median Annual Return": np.median(mc_results),
    "5th Percentile": np.percentile(mc_results, 5),
    "95th Percentile": np.percentile(mc_results, 95)
}
summary_df = pd.DataFrame(summary, index=["Monte Carlo Results"])
print("Monte Carlo Return Distribution Summary Annualized Returns")
print(summary_df)

# %%
plt.figure(figsize=(10,6))
plt.hist(mc_results, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(np.mean(mc_results), color='red', linestyle='dashed', linewidth=2, label='Mean')
plt.axvline(np.percentile(mc_results, 5), color='orange', linestyle='dashed', linewidth=2, label='5th Percentile')
plt.axvline(np.percentile(mc_results, 95), color='green', linestyle='dashed', linewidth=2, label='95th Percentile')
plt.title("Monte Carlo Simulated 25-Year Cumulative Return Distribution")
plt.xlabel("Cumulative Return")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
print("Optimized Portfolio Weights % (Rounded)")
rounded_weights = classic_weights.sort_values(ascending=False).apply(lambda x: round(x * 100, 4))
print(rounded_weights)

# %% [markdown]
# # Method 1 for Backtesting since 1999

# %%
tickers = [
    'AAPL', 'JPM', 'HON', 'NOC', 'MSFT', 'JNJ', 'PG',
    'SPY', 'IWM', 'QQQ', 'ACWI', 'TLT', 'GLD', 'VNQ', 'UUP'
]
benchmark = 'SPY'
risk_free_rate = 0.01

# %%
price_data = yf.download(tickers, start='1999-01-01', end='2024-12-31')['Close']

# %%
returns = np.log(price_data / price_data.shift(1))

# %%
start_year = 1999
end_year = 2024
allocations_by_year = pd.DataFrame(index=range(start_year, end_year + 1), columns=tickers)

# %%
for year in range(start_year, end_year + 1):
    train_end = f"{year}-12-31"
    training_data = returns.loc[:train_end]

    # Ensure at least 252 days of data for inclusion
    available_tickers = [t for t in tickers if training_data[t].dropna().shape[0] >= 252]
    
    if len(available_tickers) == 0:
        allocations_by_year.loc[year] = 0.0
        continue

    data = training_data[available_tickers].copy()
    data = data.fillna(method='ffill').fillna(method='bfill')

    mean_returns = data.mean()
    cov_matrix = data.cov()

    n_simulations = 1000
    results = np.zeros((n_simulations, len(available_tickers) + 3))

    for i in range(n_simulations):
        weights = np.random.dirichlet(np.ones(len(available_tickers)))
        port_return = np.dot(weights, mean_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (port_return - 0.01) / port_vol
        results[i, :len(available_tickers)] = weights
        results[i, -3:] = [port_return, port_vol, sharpe]

    best_weights = results[results[:, -1].argmax(), :len(available_tickers)]
    
    # Create full allocation vector
    full_weights = pd.Series(0.0, index=tickers)
    full_weights[available_tickers] = best_weights
    allocations_by_year.loc[year] = full_weights

# %%
allocations_by_year.to_csv("portfolio_allocations_1999_2024.csv")
print(allocations_by_year.round(4))

# %%
allocations = pd.read_csv('portfolio_allocations_1999_2024.csv', index_col=0)

# Ensure the index is integer years
allocations.index = allocations.index.astype(int)

# Create the stacked bar plot
fig, ax = plt.subplots(figsize=(14, 7))
allocations.plot(kind='bar', stacked=True, ax=ax, width=0.85)

# Formatting
ax.set_title('Portfolio Asset Allocation by Year (1999–2024)', fontsize=16)
ax.set_ylabel('Allocation Weight', fontsize=12)
ax.set_xlabel('Year', fontsize=12)
ax.legend(title='Ticker', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()

# Show the chart
plt.show()

# %%
weights_df = pd.read_csv("portfolio_allocations_1999_2024.csv")
weights_df.rename(columns={"Unnamed: 0": "Year"}, inplace=True)
weights_df["Year"] = weights_df["Year"].astype(int)
weights_df.set_index("Year", inplace=True)
weights_df.loc[1999] = weights_df.loc[2000]

# %%
adj_close_df = pd.read_csv("adj_close_prices.csv", index_col=0, parse_dates=True)
adj_close_df.index = pd.DatetimeIndex([ts.replace(tzinfo=None) for ts in adj_close_df.index])

# %%
returns_full = adj_close_df.pct_change()

# %%
portfolio_returns = pd.Series(index=returns_full.index, dtype=float)

# %%
for year in weights_df.index:
    start_date = returns_full[returns_full.index.year == year].index.min()
    end_date = returns_full[returns_full.index.year == year].index.max()
    if pd.isna(start_date) or pd.isna(end_date):
        continue

    current_weights = weights_df.loc[year].fillna(0)
    current_weights = current_weights[current_weights > 0]  # Only non-zero weights

    if current_weights.empty:
        continue

    tickers = current_weights.index.tolist()
    yearly_returns = returns_full.loc[start_date:end_date, tickers].dropna(how='any')

    if yearly_returns.empty:
        continue

    normalized_weights = current_weights / current_weights.sum()
    weighted_returns = yearly_returns.dot(normalized_weights)
    portfolio_returns.loc[yearly_returns.index] = weighted_returns

# %%
portfolio_returns = portfolio_returns.dropna()
portfolio_cumulative = (1 + portfolio_returns).cumprod()
spy_cumulative = (1 + returns_full["SPY"].loc[portfolio_cumulative.index].dropna()).cumprod()

# === Combine for Plotting ===
cumulative_df = pd.DataFrame({
    "Optimized Portfolio": portfolio_cumulative,
    "S&P 500 (SPY)": spy_cumulative
})

# %%
plt.figure(figsize=(12, 6))
plt.plot(cumulative_df["Optimized Portfolio"], label="Optimized Portfolio", linewidth=2)
plt.plot(cumulative_df["S&P 500 (SPY)"], label="S&P 500 (SPY)", linewidth=2)
plt.title("Cumulative Returns: Optimized Portfolio vs S&P 500 (1999–2024)")
plt.ylabel("Cumulative Return")
plt.xlabel("Date")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
risk_free_rate = 0.01
daily_rf_rate = (1 + risk_free_rate) ** (1 / 252) - 1

# %%
cumulative_df = pd.DataFrame({
    'Optimized Portfolio': portfolio_cumulative,
    'S&P 500 (SPY)': spy_cumulative
})
annual_returns = cumulative_df.resample('Y').last().pct_change().dropna()
annual_returns.index = annual_returns.index.year
annual_returns_pct = annual_returns * 100


# %%
sharpe_by_year_combined = {"Optimized Portfolio": {}, "S&P 500 (SPY)": {}}

for year in annual_returns.index:
    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year}-12-31")

    # Portfolio Sharpe
    port_group = portfolio_returns.loc[start:end]
    if len(port_group) >= 2:
        port_excess = port_group - daily_rf_rate
        std = port_excess.std()
        sharpe = (port_excess.mean() / std) * np.sqrt(252) if std > 0 else np.nan
        sharpe_by_year_combined["Optimized Portfolio"][year] = sharpe
    else:
        sharpe_by_year_combined["Optimized Portfolio"][year] = np.nan

    # SP500 Sharpe
    spy_group = returns_full["SPY"].loc[start:end]
    if len(spy_group.dropna()) >= 2:
        spy_excess = spy_group - daily_rf_rate
        std = spy_excess.std()
        sharpe = (spy_excess.mean() / std) * np.sqrt(252) if std > 0 else np.nan
        sharpe_by_year_combined["S&P 500 (SPY)"][year] = sharpe
    else:
        sharpe_by_year_combined["S&P 500 (SPY)"][year] = np.nan

# %%
sharpe_df = pd.DataFrame(sharpe_by_year_combined)
combined_df = annual_returns_pct.merge(sharpe_df, left_index=True, right_index=True)

# %%
print(combined_df)

# %%
initial_investment = 1_000_000 
risk_free_rate = 0.01 
daily_rf_rate = (1 + risk_free_rate) ** (1 / 252) - 1

# %%
def monte_carlo_projection(daily_returns, years, n_simulations=10000, seed=42):
    np.random.seed(seed)
    trading_days = 252
    n_days = years * trading_days
    mean_return = daily_returns.mean()
    std_return = daily_returns.std()

    simulations = np.zeros(n_simulations)
    for i in range(n_simulations):
        simulated_returns = np.random.normal(loc=mean_return, scale=std_return, size=n_days)
        cumulative_return = np.prod(1 + simulated_returns)
        simulations[i] = initial_investment * cumulative_return

    return simulations

# %%
mc_results = {
    "5-Year": monte_carlo_projection(portfolio_returns, 5),
    "10-Year": monte_carlo_projection(portfolio_returns, 10),
    "15-Year": monte_carlo_projection(portfolio_returns, 15),
    "25-Year": monte_carlo_projection(portfolio_returns, 25)
}

# %%
print(summary_df)

# %%
data = [mc_results[horizon] for horizon in ["5-Year", "10-Year", "15-Year", "25-Year"]]
labels = ["5-Year", "10-Year", "15-Year", "25-Year"]

fig, ax = plt.subplots(figsize=(10, 6))
box = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)

means = [np.mean(d) for d in data]
ax.plot(range(1, 5), means, 'o', color='black', label='Mean')

ax.set_title("Monte Carlo Projected Portfolio Values ($1M Investment)", fontsize=14)
ax.set_ylabel("Portfolio Value ($)", fontsize=12)
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.show()

# %%
projection_summary = pd.DataFrame({
    "Median ($)": [np.median(v) for v in mc_results.values()],
    "Mean ($)": [np.mean(v) for v in mc_results.values()],
    "5th Percentile ($)": [np.percentile(v, 5) for v in mc_results.values()],
    "95th Percentile ($)": [np.percentile(v, 95) for v in mc_results.values()],
}, index=["5-Year", "10-Year", "15-Year", "25-Year"])

projection_summary_formatted = projection_summary.applymap(lambda x: f"${x:,.0f}")

print(projection_summary_formatted)

# %%
n_simulations = 1000
n_months = 300  # 25 years * 12 months
initial_value = 1_000_000
monthly_return_mean = 0.005  # ~6% annualized
monthly_return_std = 0.02    # ~24% annualized volatility

# %%
all_simulations = []

for _ in range(n_simulations):
    values = [initial_value]
    for _ in range(n_months):
        r = np.random.normal(loc=monthly_return_mean, scale=monthly_return_std)
        new_value = values[-1] * (1 + r)
        values.append(new_value)
    all_simulations.append(values[1:])  # skip initial value to keep length = 300


# %%
mc_results = {
    "full_projection": np.array(all_simulations)  # shape: (n_simulations, 300)
}

# %%
portfolio_array = mc_results["full_projection"]

monthly_projection = pd.DataFrame({
    "Median ($)": np.median(portfolio_array, axis=0),
    "Mean ($)": np.mean(portfolio_array, axis=0),
    "5th Percentile ($)": np.percentile(portfolio_array, 5, axis=0),
    "95th Percentile ($)": np.percentile(portfolio_array, 95, axis=0),
})

monthly_projection.index = [f"Year {i // 12 + 1} Month {i % 12 + 1}" for i in range(n_months)]
monthly_projection_formatted = monthly_projection.applymap(lambda x: f"${x:,.0f}")

# %%
csv_path = "25_Year_Monthly_Projection.csv"
monthly_projection.to_csv(csv_path)

# %%
print(monthly_projection_formatted.head(30))

# %% [markdown]
# # Fee Section #

# %%
annual_mgmt_fee = 0.02           # 2% annual management fee
monthly_turnover = 0.05          # 5% of portfolio rebalanced monthly
trading_fee_rate = 0.001         # 0.1% transaction cost
monthly_mgmt_fee = annual_mgmt_fee / 12

# %%
mean_values = monthly_projection["Mean ($)"]

monthly_projection["Management Fee ($)"] = mean_values * monthly_mgmt_fee
monthly_projection["Trading Fee ($)"] = mean_values * monthly_turnover * trading_fee_rate
monthly_projection["Total Fees ($)"] = (
    monthly_projection["Management Fee ($)"] + monthly_projection["Trading Fee ($)"]
)
monthly_projection["Net Mean Value ($)"] = (
    monthly_projection["Mean ($)"] - monthly_projection["Total Fees ($)"]
)

# %%
monthly_projection[[
    "Mean ($)", 
    "Management Fee ($)", 
    "Trading Fee ($)", 
    "Total Fees ($)", 
    "Net Mean Value ($)"
]].head()

# %%
monthly_projection.to_csv("monthly_projection_with_fees_only.csv", 
                          columns=["Mean ($)", "Management Fee ($)", "Trading Fee ($)", "Total Fees ($)", "Net Mean Value ($)"])

# %%
monthly_projection["Cumulative Total Fees ($)"] = monthly_projection["Total Fees ($)"].cumsum()

# %%
all_simulations = []

for _ in range(n_simulations):
    values = [initial_value]
    for _ in range(n_months):
        r = np.random.normal(loc=monthly_return_mean, scale=monthly_return_std)
        new_value = values[-1] * (1 + r)
        values.append(new_value)
    all_simulations.append(values[1:])

portfolio_array = np.array(all_simulations)
base_mean_curve = np.mean(portfolio_array, axis=0)

# %%
fee_levels = [0.00, 0.01, 0.02, 0.03, 0.04]  # 0% to 4%
monthly_fees = [f / 12 for f in fee_levels]

years = np.arange(1, n_months + 1) / 12
fee_curves = {}
cumulative_fee_curves = {}

for annual_fee, monthly_fee in zip(fee_levels, monthly_fees):
    value = base_mean_curve[0]
    net_values = [value]
    cumulative_fees = [0]
    for r in base_mean_curve[1:] / base_mean_curve[:-1] - 1:
        fee_amt = net_values[-1] * monthly_fee
        new_value = net_values[-1] * (1 + r) - fee_amt
        net_values.append(new_value)
        cumulative_fees.append(cumulative_fees[-1] + fee_amt)
    label = f"{int(annual_fee * 100)}% Fee"
    fee_curves[label] = net_values
    cumulative_fee_curves[label] = cumulative_fees

# %%
plt.figure(figsize=(12, 6))

for label, values in fee_curves.items():
    plt.plot(years, values, label=label)

    # Add label at end of Year 25
    final_value = values[-1]
    cumulative_fee = cumulative_fee_curves[label][-1]
    plt.text(years[-1], final_value,
             f"{label}\nFinal Value: ${final_value:,.0f}\nFees: ${cumulative_fee:,.0f}",
             fontsize=9, verticalalignment='center',
             bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))

plt.title("Hypothetical Portfolio (assumne 6% annual compounding returns) Value After 25 Years at Different Fee Levels")
plt.xlabel("Years")
plt.ylabel("Portfolio Value ($)")
plt.grid(True)
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()


