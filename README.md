# MSDS451 Financial Engineering Term Project Checkpoint_B

# Monte Carlo Portfolio Optimization and Backtesting

# Assets Chosen for Portfolio

<img width="468" height="272" alt="image" src="https://github.com/user-attachments/assets/1b7ebb29-624d-4b26-8910-4f9bc1feee40" />

Assume Equal-Weighted Portfolio Return Performance since 1999

<img width="1389" height="790" alt="image" src="https://github.com/user-attachments/assets/71b8d42b-e251-40db-8d80-4485df636cc0" />

<img width="468" height="98" alt="image" src="https://github.com/user-attachments/assets/431507e4-7de0-4094-bdfa-24dd73b502d9" />

# Methods Implemented

Method 1: Monte Carlo Simulation using Dirichlet Process
Randomly generates portfolio weights using the Dirichlet distribution.

Simulates expected returns, volatility, and Sharpe ratios.

Selects the best-performing portfolio based on Sharpe ratio.

<img width="930" height="590" alt="image" src="https://github.com/user-attachments/assets/192c11f7-2e05-491e-aa7b-7771294543f1" />


Method 2: Deterministic Optimization using SLSQP
Optimizes weights to maximize the Sharpe ratio using SciPy’s minimize function.

Applies constraints to ensure valid portfolio weights (sum to 1, no shorting).

Uses historical mean returns and covariance matrices.

<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/a92a5a6b-9748-4e43-9e45-7328ea0d608c" />


# Backtesting

Applies Method 1’s optimal weights annually from 1999 onward.

Compares optimized portfolio performance to S&P 500 benchmark (SPY).

Includes visualizations of cumulative returns, crisis period shading, and asset rebalancing over time.

<img width="1394" height="690" alt="image" src="https://github.com/user-attachments/assets/b3eec489-5a7c-490a-a771-54b915de425b" />

# Results

The optimized portfolio outperformed the S&P 500 over the 25-year backtest period (1999–2024), especially during periods of heightened volatility. The Dirichlet-based Monte Carlo method (Method 1) demonstrated robust performance by adapting annual weight allocations to shifting market conditions. While the deterministic optimizer (Method 2) provided consistent Sharpe-maximizing allocations, the randomized simulations often captured a more diversified and resilient risk-return profile. Both strategies highlighted the value of disciplined rebalancing and multi-asset diversification compared to a passive benchmark.

<img width="989" height="589" alt="image" src="https://github.com/user-attachments/assets/5218678b-acb2-4f72-83d6-6a6b9249943c" />


# Notes

Designed for long-only portfolios.

Annual rebalancing strategy is applied.

Crisis periods are shaded for clarity in return plots.


# How to Run

1. Clone the repository.
   
2. Open the notebook in Jupyter or VSCode.

3. Run cells sequentially to:

4. Import data

5. Simulate portfolio weights

6. Run optimization

7. Conduct historical backtest

8. Visualize results

# AI Usage

AI was used to help with the Monte Carlo models along with the backend testing as some of the assets chosen did not cover the full span of the 25 years shown in the model. Some of the ETF's were not started until the early to late 2000's creating complexity to optimize for the portfolio allocation.

# License

MIT License.


