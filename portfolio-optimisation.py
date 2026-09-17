import yfinance as yf
import pandas as pd
import datetime
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

# Data and returns

tickers = [
    # Technology / Communication
    "AAPL", "MSFT", "META", "NVDA",

    # Consumer
    "AMZN", "DIS", "LOW", "SBUX",

    # Financials
    "JPM", "USB", "AFL",

    # Healthcare
    "JNJ", "PFE", "MDT",

    # Energy
    "XOM", "COP",

    # Industrials
    "CAT", "EMR", "CSX",

    # Utilities
    "DUK", "SO",

    # Materials / Real Estate
    "NUE", "O"
]

start = datetime.date(2020, 1, 1)
end = datetime.date(2025, 1, 1)

data = yf.download(tickers, start, end)

close_prices = data["Close"]

returns = close_prices.pct_change().dropna()
returns = returns[tickers]

mean_returns = returns.mean().to_numpy()
cov_matrix = returns.cov().to_numpy()

# Efficient frontier

def portfolio_variance_cov_matrix(w):
    w = np.array(w)
    cov = np.array(cov_matrix)
    return w.T @ cov @ w

def portfolio_return(w):
    w = np.array(w) 
    mr = np.array(mean_returns)
    return w.T @ mr

def minimum_variance_for_target(x):
    w0 = np.ones(len(tickers))/(len(tickers))
    constraint1 = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    constraint2 = {"type": "eq", "fun": lambda w: portfolio_return(w) - x}
    bounds = [(0, 1)] * len(tickers)
    result = minimize(fun=portfolio_variance_cov_matrix, x0=w0, method="SLSQP", bounds=bounds, constraints=[constraint1,constraint2], options={"ftol": 1e-12, "maxiter": 1000, "disp": False})
    if not result.success:
        raise RuntimeError(result.message)
    return (result.fun)

minimum_return = min(mean_returns)
maximum_return = max(mean_returns)
target_returns = np.linspace(minimum_return, maximum_return, 50)

variances = []
for target in target_returns:
    variances.append(minimum_variance_for_target(target))

variances = np.array(variances)
gmv_index = np.argmin(variances)
efficient_returns = target_returns[gmv_index:]
efficient_variances = variances[gmv_index:]
efficient_volatilities = np.sqrt(efficient_variances)

plt.plot(efficient_volatilities, efficient_returns)
plt.xlabel("Daily Volatility")
plt.ylabel("Expected daily return")
plt.title("Efficient Frontier")
plt.savefig("efficient_frontier.png", dpi=300, bbox_inches="tight")
plt.show()

def portfolio_variance(w, cov_matrix):
    w = np.array(w)
    return w.T @ cov_matrix @ w

# Covariance estimators

window_size = 252
holding_period = 21


def training_cov(training_returns):
    training_mean_returns = training_returns.mean().to_numpy()
    training_cov_matrix = training_returns.cov().to_numpy()
    return training_mean_returns, training_cov_matrix

def training_exponential(training_returns, lmbd=0.94):
    data = training_returns.to_numpy()
    T = len(data)
    training_mean_returns = training_returns.mean().to_numpy()
    weights = np.array([((1 - lmbd) * (lmbd ** (T - 1 - t))) / (1 - lmbd ** T) for t in range(T)])
    exp_mean = np.sum(weights[:, None] * data, axis=0)
    X = data - exp_mean
    exp_cov_matrix = X.T @ (weights[:, None] * X)
    return training_mean_returns, exp_cov_matrix

def training_ledoit(training_returns):
    training_mean_returns = training_returns.mean().to_numpy()
    lw = LedoitWolf()
    lw.fit(training_returns.to_numpy())
    training_cov_matrix = lw.covariance_
    return training_mean_returns, training_cov_matrix
    
# Global minimum variance optimisation

def optimal_portfolio(training_returns, estimator, max_weight=1.0):

    training_mean_returns, training_cov_matrix = estimator(training_returns)
    w0 = np.ones(len(tickers)) / len(tickers)
    bounds = [(0, max_weight)] * len(tickers)
    constraint = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    result = minimize(fun=portfolio_variance, x0=w0, args=(training_cov_matrix,), method="SLSQP", bounds=bounds, constraints=constraint, options={"ftol": 1e-12, "maxiter": 1000, "disp": False})
    if not result.success:
        raise RuntimeError(result.message)
    return result.x

# Hierarchical Risk Parity

def inverse_variance_weights(cov_matrix):
    variances = np.diag(cov_matrix)
    inverse_variances = 1.0 / variances
    weights = inverse_variances / inverse_variances.sum()
    return weights

def cluster_variance(cov_matrix):
    weights = inverse_variance_weights(cov_matrix)
    return weights @ cov_matrix @ weights

def recursive_bisection(cov_matrix, ordered_assets):
    weights = pd.Series(1.0, index=ordered_assets)
    clusters = [ordered_assets]
    while clusters:
        new_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            split_point = len(cluster) // 2
            left_cluster = cluster[:split_point]
            right_cluster = cluster[split_point:]
            left_cov = cov_matrix.loc[left_cluster, left_cluster].to_numpy()
            right_cov = cov_matrix.loc[right_cluster, right_cluster].to_numpy()
            left_variance = cluster_variance(left_cov)
            right_variance = cluster_variance(right_cov)
            total_variance = left_variance + right_variance
            alpha_left = right_variance / total_variance
            alpha_right = left_variance / total_variance
            weights.loc[left_cluster] *= alpha_left
            weights.loc[right_cluster] *= alpha_right
            new_clusters.extend([left_cluster, right_cluster])
        clusters = new_clusters
    weights /= weights.sum()
    return weights

def hrp_weights(training_returns):
    cov_matrix = training_returns.cov()
    corr_matrix = training_returns.corr()
    distance_matrix = np.sqrt(np.maximum((1.0 - corr_matrix) / 2.0, 0.0))
    np.fill_diagonal(distance_matrix.values, 0.0)
    condensed_distance = squareform(distance_matrix.to_numpy())
    linkage_matrix = linkage(condensed_distance, method="single")
    leaf_order = leaves_list(linkage_matrix)
    ordered_assets = list(training_returns.columns[leaf_order])
    weights = recursive_bisection(cov_matrix, ordered_assets)
    weights = weights.reindex(training_returns.columns)
    return weights.to_numpy()

# Rolling out-of-sample backtests

def run_gmv_backtest(window_size, estimator, max_weight=1.0):
    weight_history = []
    portfolio_returns = []
    for i in range(0, len(returns) - window_size, holding_period):
        training_returns = returns.iloc[i:i + window_size]
        test_returns = returns.iloc[i + window_size: i + window_size + holding_period]
        weights = optimal_portfolio(training_returns, estimator, max_weight=max_weight)
        weight_history.append(weights)
        portfolio_returns.extend(test_returns.to_numpy() @ weights)
    return (np.array(portfolio_returns), np.array(weight_history))

def run_hrp_backtest(window_size):
    weight_history = []
    portfolio_returns = []
    for i in range(0, len(returns) - window_size, holding_period):
        training_returns = returns.iloc[i:i + window_size]
        test_returns = returns.iloc[i + window_size:i + window_size + holding_period]
        weights = hrp_weights(training_returns)
        weight_history.append(weights)
        portfolio_returns.extend(test_returns.to_numpy() @ weights)
    return np.array(portfolio_returns), np.array(weight_history)


w_even = np.ones(len(tickers)) / len(tickers)
portfolio_returns_even = []

for i in range(0,len(returns) - window_size, holding_period):
    test_returns = returns.iloc[i + window_size: i + window_size + holding_period]
    portfolio_returns_even.extend(test_returns.to_numpy() @ w_even)

# Main results

portfolio_returns_sample, weight_history_sample = run_gmv_backtest(252, training_cov)

portfolio_returns_exp, weight_history_exp = run_gmv_backtest(252, training_exponential)

portfolio_returns_shrinkage, weight_history_shrinkage = run_gmv_backtest(252, training_ledoit)

portfolio_returns_hrp, weight_history_hrp = run_hrp_backtest(252)


sample_returns = np.array(portfolio_returns_sample)
exp_returns = np.array(portfolio_returns_exp)
shrinkage_returns = np.array(portfolio_returns_shrinkage)
hrp_returns = np.array(portfolio_returns_hrp)
even_returns = np.array(portfolio_returns_even)


sample_wealth = np.cumprod(1 + sample_returns)
exp_wealth = np.cumprod(1 + exp_returns)
shrinkage_wealth = np.cumprod(1 + shrinkage_returns)
even_wealth = np.cumprod(1 + even_returns)
hrp_wealth = np.cumprod(1 + hrp_returns)

plt.figure(figsize=(10, 6))

plt.plot(sample_wealth, label="Sample Covariance")
plt.plot(exp_wealth, label="Exponential")
plt.plot(shrinkage_wealth, label="Ledoit-Wolf")
plt.plot(even_wealth, label="Equal Weight")
plt.plot(hrp_wealth, label="HRP")

plt.xlabel("Out-of-Sample Trading Days")
plt.ylabel("Cumulative Wealth")
plt.title("Out-of-Sample Cumulative Wealth")
plt.legend()

plt.savefig("cumulative_wealth.png", dpi=300, bbox_inches="tight")
plt.show()

# Performance metrics

def performance_metrics(portfolio_returns):
    portfolio_returns = np.array(portfolio_returns)
    annual_return = portfolio_returns.mean() * 252
    annual_volatility = (portfolio_returns.std() * np.sqrt(252))

    sharpe = (annual_return - 0.03) / annual_volatility

    wealth = (1 + portfolio_returns).cumprod()
    running_max = np.maximum.accumulate(wealth)
    drawdown = (wealth - running_max) / running_max
    max_drawdown = np.min(drawdown)

    return (annual_return, annual_volatility, sharpe, max_drawdown)

sample_stats = performance_metrics(sample_returns)
exp_stats = performance_metrics(exp_returns)
shrinkage_stats = performance_metrics(shrinkage_returns)
even_stats = performance_metrics(even_returns)
hrp_stats = performance_metrics(hrp_returns)

results = pd.DataFrame({"Sample": sample_stats, "Exponential": exp_stats, "Shrinkage": shrinkage_stats, "Equal": even_stats, "HRP": hrp_stats}, index=["Annual Return","Annual Volatility","Sharpe Ratio","Maximum Drawdown"])

print(results)

# Portfolio concentration and turnover

sample_weights = np.array(weight_history_sample)
exp_weights = np.array(weight_history_exp)
shrinkage_weights = np.array(weight_history_shrinkage)
hrp_weights_array = np.array(weight_history_hrp)


def portfolio_instability(weights):
    weights = np.asarray(weights, dtype=float)

    turnover = np.sum(np.abs(np.diff(weights, axis=0)), axis=1)
    average_turnover = np.mean(turnover)

    max_weights = np.max(weights, axis=1)
    average_max_weight = np.mean(max_weights)

    hhi = np.sum(weights ** 2, axis=1)
    average_hhi = np.mean(hhi)

    return average_turnover, average_max_weight, average_hhi

sample_instability = portfolio_instability(sample_weights)
exp_instability = portfolio_instability(exp_weights)
shrinkage_instability = portfolio_instability(shrinkage_weights)
hrp_instability = portfolio_instability(hrp_weights_array)

instability_results = pd.DataFrame({"Sample": sample_instability, "Exponential": exp_instability, "Shrinkage": shrinkage_instability,"HRP": hrp_instability},index=["Average Turnover","Average Maximum Weight","Average HHI"])

print(instability_results)

# Transaction costs

def apply_transaction_costs(portfolio_returns, weight_history, cost_rate=0.001):
    net_returns = portfolio_returns.copy()
    for i in range(1, len(weight_history)):
        turnover = np.sum(np.abs(weight_history[i] - weight_history[i - 1]))
        cost = cost_rate * turnover
        net_returns[i * holding_period] -= cost
    return net_returns

sample_net_returns = apply_transaction_costs(sample_returns, sample_weights)
exp_net_returns = apply_transaction_costs(exp_returns, exp_weights)
shrinkage_net_returns = apply_transaction_costs(shrinkage_returns, shrinkage_weights)
hrp_net_returns = apply_transaction_costs(hrp_returns, hrp_weights_array)

sample_net_stats = performance_metrics(sample_net_returns)
exp_net_stats = performance_metrics(exp_net_returns)
shrinkage_net_stats = performance_metrics(shrinkage_net_returns)
hrp_net_stats = performance_metrics(hrp_net_returns)

net_results = pd.DataFrame({"Sample": sample_net_stats, "Exponential": exp_net_stats, "Shrinkage": shrinkage_net_stats, "HRP": hrp_net_stats}, index=["Annual Return","Annual Volatility","Sharpe Ratio","Maximum Drawdown"])

print(net_results)

# Estimation window robustness

robustness_windows = [126, 252, 504]
window_results = []

estimators = {"Sample": training_cov, "Exponential": training_exponential, "Shrinkage": training_ledoit}

for window in robustness_windows:
    for estimator_name, estimator_function in estimators.items():
        port_returns, weights = run_gmv_backtest(window,estimator_function)
        stats = performance_metrics(port_returns)
        instability = portfolio_instability(weights)
        window_results.append({"Window": window,"Method": estimator_name,"Annual Return": stats[0],"Annual Volatility": stats[1],"Sharpe Ratio": stats[2],"Maximum Drawdown": stats[3],"Average Turnover": instability[0],"Average Maximum Weight": instability[1],"Average HHI": instability[2]})
    port_returns, weights = run_hrp_backtest(window)
    stats = performance_metrics(port_returns)
    instability = portfolio_instability(weights)
    window_results.append({"Window": window, "Method": "HRP", "Annual Return": stats[0], "Annual Volatility": stats[1], "Sharpe Ratio": stats[2], "Maximum Drawdown": stats[3], "Average Turnover": instability[0], "Average Maximum Weight": instability[1], "Average HHI": instability[2]})
window_robustness_results = pd.DataFrame(window_results)

print("\nESTIMATION WINDOW ROBUSTNESS")
print(window_robustness_results.round(4))



volatility_by_window = window_robustness_results.pivot(index="Window", columns="Method", values="Annual Volatility")

print("\nANNUAL VOLATILITY BY ESTIMATION WINDOW")
print(volatility_by_window.round(4))


hhi_by_window = window_robustness_results.pivot(index="Window", columns="Method", values="Average HHI")

print("\nAVERAGE HHI BY ESTIMATION WINDOW")
print(hhi_by_window.round(4))



window_robustness_results.to_csv("window_robustness_results.csv", index=False)

# Weight Cap Robustness

weight_caps = [1.00, 0.20, 0.10]
constraint_results = []
for cap in weight_caps:
    for estimator_name, estimator_function in estimators.items():
        port_returns, weights = run_gmv_backtest(window_size=252, estimator=estimator_function, max_weight=cap)
        stats = performance_metrics(port_returns)
        instability = portfolio_instability(weights)
        constraint_results.append({"Maximum Weight Constraint": cap, "Method": estimator_name, "Annual Return": stats[0], "Annual Volatility": stats[1], "Sharpe Ratio": stats[2], "Maximum Drawdown": stats[3], "Average Turnover": instability[0], "Average Maximum Weight": instability[1], "Average HHI": instability[2]})

constraint_robustness_results = pd.DataFrame(constraint_results)

print("\nMAXIMUM-WEIGHT CONSTRAINT ROBUSTNESS")
print(constraint_robustness_results.round(4))


constraint_volatility = constraint_robustness_results.pivot(index="Maximum Weight Constraint", columns="Method", values="Annual Volatility")

print("\nANNUAL VOLATILITY BY WEIGHT CONSTRAINT")
print(constraint_volatility.round(4))



constraint_hhi = constraint_robustness_results.pivot(index="Maximum Weight Constraint", columns="Method", values="Average HHI")

print("\nAVERAGE HHI BY WEIGHT CONSTRAINT")
print(constraint_hhi.round(4))



constraint_robustness_results.to_csv("constraint_robustness_results.csv", index=False)



plt.figure(figsize=(10, 6))

for method in volatility_by_window.columns:
    plt.plot(volatility_by_window.index, volatility_by_window[method], marker="o", label=method)

plt.xlabel("Estimation Window (Trading Days)")
plt.ylabel("Annualised OOS Volatility")
plt.title("Sensitivity of Out-of-Sample Volatility to Estimation Window")
plt.legend()
plt.grid(alpha=0.3)

plt.savefig("window_robustness_volatility.png", dpi=300, bbox_inches="tight")
plt.show()


plt.figure(figsize=(10, 6))

for method in constraint_volatility.columns:
    plt.plot(constraint_volatility.index, constraint_volatility[method], marker="o", label=method)

plt.xlabel("Maximum Individual Asset Weight")
plt.ylabel("Annualised OOS Volatility")
plt.title("Effect of Portfolio Constraints on GMV Performance")
plt.legend()
plt.grid(alpha=0.3)

plt.savefig("constraint_robustness_volatility.png", dpi=300, bbox_inches="tight")
plt.show()




