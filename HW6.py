# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 14:44:17 2019

@author: yyang
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy.linalg import eig


def des_pde(s0,smin,smax,t,Nt,Ns,sigma,r,k1,k2,contract,option='european'):
    s0 = s0
    smin = smin
    smax = smax
    t = t
    Nt = Nt
    Ns = Ns
    ht = t / Nt
    hs = (smax -  smin) / Ns
    sigma = sigma
    r = r
    k1 = k1
    k2 = k2
    
    # compute the matrix A
    s = np.arange(smin, smax + hs, hs)
    a = 1 - (sigma * s) ** 2 * ht / (hs ** 2) - r * ht
    l = ((sigma * s) ** 2) / 2 * (ht / (hs ** 2)) - (r * s * ht) / (2 * hs)
    u = ((sigma * s) ** 2) / 2 * (ht / (hs ** 2)) + (r * s * ht) / (2 * hs)
    A = np.diag(a[1: Ns])
    upper_list = u[1: Ns - 1]
    lower_list = l[2: Ns]
    
    for i in range(len(upper_list)):
        A[i][i + 1] = upper_list[i]
        A[i + 1][i] = lower_list[i]    
    
    # explicit Euler discretization to price options and spread
    if contract == 'call':
        k = k1 # using k1 as the call option's strike
        c_early = (s - k)[1: Ns]
        c_early[c_early < 0] = 0
        c_vec = c_early
        
        for i in range(Nt):
            c_vec = A.dot(c_vec)
            c_vec[-1] = c_vec[-1] + u[Ns - 1] * (smax - k * np.exp(-r * i * ht))
            if option == 'American':
                c_vec = [x if x > y else y for x, y in zip(c_vec, c_early)]
                
    elif contract == 'callspread':
        long = (s - k1)[1: Ns]
        short = (s - k2)[1: Ns]
        long[long < 0] = 0
        short[short < 0] = 0
        c_vec = long - short
        c_early = c_vec
        
        for i in range(Nt):
            c_vec = A.dot(c_vec)
            c_vec[-1] = c_vec[-1] + u[Ns - 1] * (smax - (k1 - k2) * np.exp(-r * i * ht))
            if option == 'American':
                c_vec = [x if x > y else y for x, y in zip(c_vec, c_early)]
                
    # get option price with linear interpolation
    c_price = np.interp(s0, s[1: Ns], c_vec)
    
    return (A, c_price)

# use BS formula to price call options               
def price_eu_call_BS(S0, K, r, sigma, t):
    d1 = (math.log(S0 / K) + (r + (sigma ** 2) / 2) * t) / (sigma * (t ** 0.5))
    d2 = d1 - sigma * (t ** 0.5)
    call_option_BS = S0 * norm.cdf(d1) - K * math.exp(-r * t) * norm.cdf(d2)
    return call_option_BS




if __name__ == '__main__':
    s0 = 276.11
    k1 = 285
    k2 = 290
    t = 144/252
    r = 0.0247
    smin = 0
    smax = 500
    Nt = 1000
    Ns = 250
    sigma = 0.1331

    #4. make decisions about the choice of smax, hs, ht
    # smax
    smax_list = [500, 1000, 1500]
    Nt_list = [1000, 5000, 10000]
    call_price_BS = price_eu_call_BS(s0, k1, r, sigma, t)
    call_price_pde = []
    relative_error = []
    for i in range(len(smax_list)):
        call_price_pde += [des_pde(s0,smin,smax_list[i],t,Nt_list[i],Ns,sigma,r,k1,k2,'call', 'European')[1]]
    
    relative_error = abs((call_price_pde - call_price_BS)/call_price_BS)

    #5. find out its eigenvalues and check their absolute values
    A = des_pde(s0,smin,smax,t,Nt,Ns,sigma,r,k1,k2,'call', 'European')[0]
    eig_val = eig(A)[0]
    abs_eigval = sorted(abs(eig_val), reverse = True)
    top_eigval = abs_eigval[0]
    
    #6. find today's price of the call spread without the right of early exercise
    spread_eu = des_pde(s0,smin,smax,t,Nt,Ns,sigma,r,k1,k2,'callspread', 'European')[1]
    
    #7. calculate the price of the call spread with the right of early exercise
    spread_am = des_pde(s0,smin,smax,t,Nt,Ns,sigma,r,k1,k2,'callspread', 'American')[1]
    
    #8. calculate the premium
    spread_premium = spread_am - spread_eu
    
    
    
    
    