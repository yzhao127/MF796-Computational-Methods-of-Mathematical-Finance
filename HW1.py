# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 14:25:00 2019

@author: yyang
"""

## 5
import pandas as pd
import numpy as np
import math
from scipy.stats import norm

# b
S0 = 100
K = S0
r = 0.0
beta = 1.0
sigma = 0.25
N = 10000
T = 252
t = 1

def stock_price_simu(S0, r, sigma, T, N, beta):
    deltaT = 1/T
    deltaS = pd.DataFrame(index = range(0, T), columns = range(0, N))
    S = pd.DataFrame(index = range(0, T), columns = range(0, N))
    deltaS.iloc[0] = 0
    S.iloc[0] = S0
    for i in range(1, T):
        deltaS.iloc[i] = r * S.iloc[i - 1].map(lambda x: x * deltaT) + sigma * S.iloc[i - 1].map(lambda x: x ** beta) * np.random.randn(N) * (deltaT ** 0.5)
        S.iloc[i] = S.iloc[i - 1] + deltaS.iloc[i]
    return S

def price_eu_call(r, T, K, S):
    payoff = S.iloc[T-1] - K
    for i in range(0, len(payoff)):
        if payoff[i] < 0:
            payoff[i] = 0
    payoff_mean = payoff.mean()
    call_price_eu = payoff_mean * math.exp(-r * T)
    return call_price_eu

S_simu = stock_price_simu(S0, r, sigma, T, N, beta)
eu_call_simu = price_eu_call(r, T, K, S_simu)

# c
def price_eu_call_BS(S0, K, r, sigma, t):
    d1 = (math.log(S0 / K) + (r + (sigma ** 2) / 2) * t) / (sigma * (t ** 0.5))
    d2 = d1 - sigma * (t ** 0.5)
    call_option_BS = S0 * norm.cdf(d1) - K * math.exp(-r * t) * norm.cdf(d2)
    return call_option_BS

eu_call_bs = price_eu_call_BS(S0, K, r, sigma, t)

# d
def delta_eu_call(S0, K, r, sigma, t):
    d1 = (math.log(S0 / K) + (r + (sigma ** 2) / 2) * t) / (sigma * (t ** 0.5))
    delta = norm.cdf(d1)
    return delta

eu_call_delta = delta_eu_call(S0, K, r, sigma, t)

# e
stock_share = eu_call_delta

# f
def payoff_deltahedge(S0, r, T, K, S, delta):
    payoff_option = S.iloc[T-1] - K
    for i in range(0, len(payoff_option)):
        if payoff_option[i] < 0:
            payoff_option[i] = 0
    portfolio_price = payoff_option.mean() * math.exp(-r * T) - (S.iloc[T-1].mean() * math.exp(-r * T) - S0) * delta
    return portfolio_price

portf_price = payoff_deltahedge(S0, r, T, K, S_simu, stock_share)

def payoff_deltahedge_change(S0, r, T, K, S, delta):
    payoff_option = S - K
    if payoff_option < 0:
        payoff_option = 0
    portfolio_price = payoff_option * math.exp(-r * T) - (S * math.exp(-r * T) - S0) * delta
    return portfolio_price

S = 90
portf_price_S = payoff_deltahedge_change(S0, r, T, K, S, stock_share)

#portf_price_S0 = []
#step = 100
#for S0 in range(100-step, 100+step):
#    S_simu_S0 = stock_price_simu(S0, r, sigma, T, 500, beta)
#    portf_price_S0.append(payoff_deltahedge(S0, r, T, K, S_simu_S0, stock_share))
#
#portf_price_S0_df = pd.DataFrame(data = portf_price_S0)
#portf_price_S0_df.plot()
#
#portf_price_K = []
#step = 100
#S0 = 100
#for K in range(100-step, 100+step):
#    S_simu_K = stock_price_simu(S0, r, sigma, T, 500, beta)
#    portf_price_K.append(payoff_deltahedge(S0, r, T, K, S_simu_K, stock_share))
#
#portf_price_r = []
#K = 100
#for r in np.linspace(0, 0.5, 50):
#    S_simu_r = stock_price_simu(S0, r, sigma, T, 500, beta)
#    portf_price_r.append(payoff_deltahedge(S0, r, T, K, S_simu_r, stock_share))

# g
S_simu_2 = stock_price_simu(S0, r, sigma, T, N, 0.5)
portf_price_2 = payoff_deltahedge(S0, r, T, K, S_simu_2, stock_share)

# h
S_simu_3 = stock_price_simu(S0, r, 0.4, T, N, beta)
portf_price_3 = payoff_deltahedge(S0, r, T, K, S_simu_3, stock_share)









