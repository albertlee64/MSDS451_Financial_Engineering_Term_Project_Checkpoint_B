# MSDS451 Financial Engineering Term Project Checkpoint_B

# Monte Carlo Portfolio Optimization and Backtesting

# Assets Chosen for Portfolio

Ticker	Name
AAPL	Apple Inc.
JPM	JPMorgan Chase & Co.
HON	Honeywell International Inc.
NOC	Northrop Grumman Corporation
MSFT	Microsoft Corporation
JNJ	Johnson & Johnson
PG	The Procter & Gamble Company
SPY	SPDR S&P 500 ETF Trust (tracks the S&P 500 index)
IWM	iShares Russell 2000 ETF (tracks the Russell 2000 index)
QQQ	Invesco QQQ Trust (tracks the Nasdaq-100 Index)
ACWI	iShares MSCI ACWI ETF (All Country World Index)
TLT	iShares 20+ Year Treasury Bond ETF (tracks the US Treasury with remaining maturities greater than 20 years)
GLD	SPDR Gold Shares (tracks the price performance of gold bullion)
VNQ	Vanguard Real Estate ETF (invests in stocks issued by REIT’s)
UUP	Invesco DB U.S. Dollar Index Bullish Fund (tracks US dollar)
<img width="468" height="272" alt="image" src="https://github.com/user-attachments/assets/778c53d9-536e-40cf-b429-3a4ab0e3586d" />


Assume Equal-Weighted Portfolio Return Performance since 1999

<img width="1389" height="790" alt="image" src="https://github.com/user-attachments/assets/71b8d42b-e251-40db-8d80-4485df636cc0" />

<img width="468" height="98" alt="image" src="https://github.com/user-attachments/assets/700f8b0a-f6b9-4462-a041-2d989c7f646d" />


# Methods Implemented

Method 1: Monte Carlo Simulation using Dirichlet Process
Randomly generates portfolio weights using the Dirichlet distribution.

Simulates expected returns, volatility, and Sharpe ratios.

Selects the best-performing portfolio based on Sharpe ratio.

Method 2: Deterministic Optimization using SLSQP
Optimizes weights to maximize the Sharpe ratio using SciPy’s minimize function.

Applies constraints to ensure valid portfolio weights (sum to 1, no shorting).

Uses historical mean returns and covariance matrices.

# Backtesting
Applies Method 1’s optimal weights annually from 1999 onward.

Compares optimized portfolio performance to S&P 500 benchmark (SPY).

Includes visualizations of cumulative returns, crisis period shading, and asset rebalancing over time.


