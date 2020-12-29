# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 15:35:28 2019

@author: yyang
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm

# Prob 1
S0 = 10
K = 12
r = 0.04
sigma = 0.2
T = 3/12

# 1
def price_eu_call_BS(S0, K, r, sigma, t):
    d1 = (math.log(S0 / K) + (r + (sigma ** 2) / 2) * t) / (sigma * (t ** 0.5))
    d2 = d1 - sigma * (t ** 0.5)
    call_option_BS = S0 * norm.cdf(d1) - K * math.exp(-r * t) * norm.cdf(d2)
    return call_option_BS

eu_call_bs = price_eu_call_BS(S0, K, r, sigma, T)

# 2
# (a) & (b)

def reimann_rule(a, b, N, interval_type, integrand):
    w = (b - a) / N
    sum = 0
    
    for i in range(1, N + 1):
        if interval_type == 'left':
            x = a + w * (i - 1)
        elif interval_type == 'right':
            x = a + w * i
        elif interval_type == 'mid':
            x = a + w * (i - 0.5)
        sum += integrand(x)
    return sum * w

def reimann_error(N, interval_type):
    if interval_type == 'mid':
        return 1 / (N**2)
    else:
        return 1 / N
    
def gauss_nodes(a, b, N, integrand):
    x, w = np.polynomial.legendre.leggauss(N)
    t = 0.5 * (x + 1) * (b - a) + a
    return sum(w * integrand(t)) * 0.5*(b - a)

def gauss_error(N):
    return 1 / N ** (2 * N)

def integrand(x):
    f = (np.exp(-r * T) * (np.exp(x) - K) / (np.sqrt(2 * math.pi * T) * sigma) * np.exp(-(x - (np.log(S0) + (r - 0.5 * sigma ** 2) * T)) ** 2 / (2* T * sigma**2)))
    return f

a = np.log(K)
b = np.log(S0)+(r - 0.5 * sigma**2) * T + 3 * sigma * np.sqrt(T)

N_nodes = [5, 10, 50, 100]

reimann_left_price = []
reimann_mid_price = []
gauss_price = []

reimann_left_error = []
reimann_mid_error = []
gauss_error_list = []

for n in N_nodes:
    reimann_left_price.append(reimann_rule(a, b, n, 'left', integrand))
    reimann_mid_price.append(reimann_rule(a, b, n, 'mid', integrand))
    gauss_price.append(gauss_nodes(a, b, n, integrand))
    
    reimann_left_error.append(reimann_error(n, 'left'))
    reimann_mid_error.append(reimann_error(n, 'mid'))
    gauss_error_list.append(gauss_error(n))
#
#reimann_left_cal_err = reimann_left_price - eu_call_bs
#reimann_mid_cal_err = reimann_mid_price - eu_call_bs
#gauss_price_cal_err = gauss_price - eu_call_bs

for n in range(1, 101):
    reimann_left_price.append(reimann_rule(a, b, n, 'left', integrand))
    reimann_mid_price.append(reimann_rule(a, b, n, 'mid', integrand))
    gauss_price.append(gauss_nodes(a, b, n, integrand))
    
    reimann_left_error.append(reimann_error(n, 'left'))
    reimann_mid_error.append(reimann_error(n, 'mid'))
    gauss_error_list.append(gauss_error(n))

reimann_left_cal_err = reimann_left_price - eu_call_bs
reimann_mid_cal_err = reimann_mid_price - eu_call_bs
gauss_price_cal_err = gauss_price - eu_call_bs

# 3 & 4      
plt.figure(1)
plt.title('The calculation error')
plt.plot(range(1, 101), reimann_left_cal_err)
plt.plot(range(1, 101), reimann_mid_cal_err)
plt.plot(range(1, 101), gauss_price_cal_err)
plt.legend(['Left Riemann','Mid Point','Gaussian'])
plt.show() 
 
plt.figure(2)
plt.title('The experimental and theoretical error of Left Riemann rule')
plt.plot(range(1, 101), reimann_left_cal_err, reimann_left_error)
plt.legend(['experimental','theoretical'])
plt.show()

plt.figure(3)
plt.title('The experimental and theoretical error of Midpoint rule')
plt.plot(range(1, 101), reimann_mid_cal_err, reimann_mid_error)
plt.legend(['experimental','theoretical'])
plt.show()

plt.figure(4)
plt.title('The experimental and theoretical error of Gauss nodes')
plt.plot(range(1, 101), gauss_price_cal_err, gauss_error_list)
plt.legend(['experimental','theoretical'])
plt.show()

# Prob 2
sigma1 = 20
sigma2 = 15
p = 0.95
r1 = 0
r2 = 0
K = 260
Kc = 250
S = 273.16
T1 = 1
T2 = 1/2

lb1 = K
ub1 = S + 3*sigma1
ub2 =Kc
lb2 = S - 3*sigma2
N = 100


def integrand2(x):
    f2 = (np.exp(-r1 * T1) * (x - K) / (np.sqrt(2 * math.pi) * sigma1) * np.exp(-(x - (S + r1 * T1)) ** 2 / (2 * sigma1**2)))
    return f2
    
def integrand3(x1,x2,p):
    f3 = (np.exp(-r1 * T1) * (x1 - K) / (2 * math.pi * sigma1 *sigma2 * np.sqrt(1 - p**2)) *np.exp((-(x1 - S)**2 / (sigma1**2) - (x2 - S)**2 / (sigma2**2) + 2 * p * (x2 - S)*(x1 - S) / (sigma1 * sigma2)) / (2*(1 - p**2))))
    return f3

# 1
def reimann_double(a1, b1, a2, b2, p, N, integrand):
    w1 = (b1 - a1)/ N
    w2 = (b2 - a2)/ N
    sum = 0
    for i in range(N):
        for j in range(N):
            sum += integrand(a1 + (i+0.5) * w1, a2 + (j+0.5) * w2, p)
    return sum * w1 * w2

vani_option_price = reimann_rule(lb1, ub1, N, 'mid',integrand2)

# 2
contin_option_price = reimann_double(lb1, ub1, lb2, ub2, p, N, integrand3)

# 3
contin_option_p_list = []
p_list = [0.8, 0.5, 0.2]
for p in p_list:
    contin_option_p_list.append(reimann_double(lb1, ub1, lb2, ub2, p, N, integrand3))

# 5
K_list = [240, 230, 220]
contin_option_K_list = []
lb2 = S - 5*sigma2
p = 0.95
for ub2 in K_list:
    contin_option_K_list.append(reimann_double(lb1, ub1, lb2, ub2, p, N, integrand3))




            