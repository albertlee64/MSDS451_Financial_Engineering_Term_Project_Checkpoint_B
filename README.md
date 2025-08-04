# MSDS451 Financial Engineering Term Project Checkpoint_B

# Monte Carlo Portfolio Optimization and Backtesting

# Assets Chosen for Portfolio

<img width="468" height="272" alt="image" src="https://github.com/user-attachments/assets/1b7ebb29-624d-4b26-8910-4f9bc1feee40" />

Assume Equal-Weighted Portfolio Return Performance since 1999

<img width="1389" height="790" alt="image" src="https://github.com/user-attachments/assets/71b8d42b-e251-40db-8d80-4485df636cc0" />


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


