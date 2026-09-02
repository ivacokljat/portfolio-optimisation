#!/usr/bin/env python
# coding: utf-8

# In[166]:


import yfinance as yf
import pandas as pd
import datetime
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf


tickers = ['AAPL' , 'MSFT', 'META', 'AMZN', 'NVDA', 'JPM', 'PFE', 'XOM', 'DIS', 'JNJ']
start = datetime.date(2020, 1, 1)
end = datetime.date(2025, 1, 1)

data = yf.download(tickers, start, end)

close_prices = data["Close"]

returns = close_prices.pct_change().dropna()
returns = returns[tickers]

mean_returns = returns.mean().to_numpy()
cov_matrix = returns.cov().to_numpy()

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
    bounds = [(0, 1)] * 10
    result = minimize(fun=portfolio_variance_cov_matrix, x0=w0, method="SLSQP", bounds=bounds, constraints=[constraint1,constraint2], options={"ftol": 1e-12, "maxiter": 1000, "disp": False})
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
plt.show()

annual_risk_free_rate = 0.03
daily_risk_free_rate = (1 + annual_risk_free_rate)**(1/252) - 1

def negative_sharpe(w, mean_returns, cov_matrix, daily_risk_free_rate):
    portfolio_return = w @ mean_returns
    portfolio_volatility = np.sqrt(w @ cov_matrix @ w)
    return -(portfolio_return - daily_risk_free_rate) / portfolio_volatility

def training_cov(training_returns):
    training_mean_returns = training_returns.mean().to_numpy()
    training_cov_matrix = training_returns.cov().to_numpy()

    return training_mean_returns, training_cov_matrix

def optimal_portfolio(training_returns,training_example):

    training_mean_returns, training_cov_matrix = training_example(training_returns)

    w0 = np.ones(len(tickers)) / len(tickers)
    bounds = [(0, 1)] * len(tickers)

    constraint = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    result = minimize(fun=negative_sharpe, x0=w0, args=(training_mean_returns, training_cov_matrix, daily_risk_free_rate), method="SLSQP", bounds=bounds, constraints=constraint, options={"ftol": 1e-12, "maxiter": 1000, "disp": False})

    if not result.success:
        raise RuntimeError(result.message)
    
    return result.x
    
window_size = 252
holding_period = 21

weight_history_cov = []
portfolio_returns_cov = []

for i in range(0,len(returns) - window_size, holding_period):
    training_returns = returns.iloc[i: i + window_size]
    test_returns = returns.iloc[i + window_size: i + window_size + holding_period]
    weights = optimal_portfolio(training_returns, training_cov)
    weight_history_cov.append(weights)
    portfolio_returns_cov.extend(test_returns.to_numpy() @ weights)


# In[153]:


def training_exponential(training_returns, lmbd=0.94):
    data = training_returns.to_numpy()
    T = len(data)

    training_mean_returns = training_returns.mean().to_numpy()

    weights = np.array([((1 - lmbd) * (lmbd ** (T - 1 - t))) / (1 - lmbd ** T) for t in range(T)])

    exp_mean = np.sum(weights[:, None] * data, axis=0)
    X = data - exp_mean

    exp_cov_matrix = X.T @ (weights[:, None] * X)

    return training_mean_returns, exp_cov_matrix

weight_history_exp = []
portfolio_returns_exp = []

for i in range(0,len(returns) - window_size, holding_period):
    training_returns = returns.iloc[i: i + window_size]
    test_returns = returns.iloc[i + window_size: i + window_size + holding_period]
    weights = optimal_portfolio(training_returns, training_exponential)
    weight_history_exp.append(weights)
    portfolio_returns_exp.extend(test_returns.to_numpy() @ weights)


# In[157]:


def training_ledoit(training_returns):
    training_mean_returns = training_returns.mean().to_numpy()

    lw = LedoitWolf()
    lw.fit(training_returns.to_numpy())

    training_cov_matrix = lw.covariance_

    return training_mean_returns, training_cov_matrix

weight_history_ledoit = []
portfolio_returns_ledoit = []

for i in range(0,len(returns) - window_size, holding_period):
    training_returns = returns.iloc[i: i + window_size]
    test_returns = returns.iloc[i + window_size: i + window_size + holding_period]
    weights = optimal_portfolio(training_returns, training_ledoit)
    weight_history_ledoit.append(weights)
    portfolio_returns_ledoit.extend(test_returns.to_numpy() @ weights)


# In[160]:


w_even = np.ones(10) / 10

portfolio_returns_even = []

for i in range(0,len(returns) - window_size, holding_period):
    test_returns = returns.iloc[i + window_size: i + window_size + holding_period]
    portfolio_returns_even.extend(test_returns.to_numpy() @ w_even)


# In[162]:


sample_returns = np.array(portfolio_returns_cov)
exp_returns = np.array(portfolio_returns_exp)
shrinkage_returns = np.array(portfolio_returns_ledoit)
even_returns = np.array(portfolio_returns_even)


# In[168]:


sample_wealth = np.cumprod(1 + sample_returns)
exp_wealth = np.cumprod(1 + exp_returns)
shrinkage_wealth = np.cumprod(1 + shrinkage_returns)
even_wealth = np.cumprod(1 + even_returns)

plt.figure(figsize=(10, 6))

plt.plot(sample_wealth, label="Sample Covariance")
plt.plot(exp_wealth, label="Exponential")
plt.plot(shrinkage_wealth, label="Ledoit-Wolf")
plt.plot(even_wealth, label="Equal Weight")

plt.xlabel("Out-of-Sample Trading Days")
plt.ylabel("Cumulative Wealth")
plt.title("Out-of-Sample Cumulative Wealth")
plt.legend()

plt.show()


# In[164]:


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

results = pd.DataFrame({"Sample": sample_stats, "Exponential": exp_stats, "Shrinkage": shrinkage_stats, "Equal": even_stats}, index=["Annual Return","Annual Volatility","Sharpe Ratio","Maximum Drawdown"])

print(results)


# In[62]:


#"How sensitive is mean–variance portfolio optimisation to covariance estimation error, and do alternative covariance estimators improve out-of-sample portfolio performance?"


# In[ ]:


#Do shrinkage and exponentially weighted covariance estimators improve the out-of-sample performance of mean–variance portfolios relative to the sample covariance matrix?


# In[170]:


sample_weights = np.array(weight_history_cov)
exp_weights = np.array(weight_history_exp)
shrinkage_weights = np.array(weight_history_ledoit)


# In[172]:


def portfolio_instability(weights):
    turnover = np.sum(np.abs(np.diff(weights, axis=0)), axis=1)
    average_turnover = np.mean(turnover)

    max_weights = np.max(weights, axis=1)
    average_max_weight = np.mean(max_weights)

    return average_turnover, average_max_weight

sample_instability = portfolio_instability(sample_weights)
exp_instability = portfolio_instability(exp_weights)
shrinkage_instability = portfolio_instability(shrinkage_weights)

instability_results = pd.DataFrame({"Sample": sample_instability, "Exponential": exp_instability, "Shrinkage": shrinkage_instability}, index=["Average Turnover", "Average Maximum Weight"])

print(instability_results)


# In[174]:


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

sample_net_stats = performance_metrics(sample_net_returns)
exp_net_stats = performance_metrics(exp_net_returns)
shrinkage_net_stats = performance_metrics(shrinkage_net_returns)

net_results = pd.DataFrame({"Sample": sample_net_stats, "Exponential": exp_net_stats, "Shrinkage": shrinkage_net_stats}, index=["Annual Return","Annual Volatility","Sharpe Ratio","Maximum Drawdown"])

print(net_results)

