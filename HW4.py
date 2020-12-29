# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 09:27:21 2019

@author: yyang
"""
import os as os
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
from sklearn.decomposition import PCA
from numpy import linalg as LA
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# define the objective function for optimization
def fun(args):
    R, a, C = args
    obj_fun = lambda w: - R.dot(w) + a * np.transpose(w).dot(C).dot(w)
    return obj_fun

# define the constraints
def con(args):
    G, c = args
    cons = ({'type': 'eq', 'fun': lambda w: G[0].dot(w) - c[0]},\
             {'type': 'eq', 'fun': lambda w: G[1].dot(w) - c[1]})
    return cons


if __name__ == '__main__':
    ## Prob 1
    # 1
    # Get 100 stock tickers
    os.chdir('C:/Users/yyang/OneDrive/Documents/Documents/BU/Courses/19Spring/MF796/HW/HW4')
    stk_ticker = pd.read_csv('yahoo_stock_tickers.csv', header = 0, index_col = 0, usecols = ['Tickers']).index.tolist()
    tickers = stk_ticker[0: 150]
    
    # Get stock price from yahoo finance
    yf.pdr_override()
    stock_info = pdr.get_data_yahoo(tickers, start="2014-01-01", end="2019-01-01")
    stock_close = stock_info['Adj Close']
    stock_close = stock_close.drop(labels = 'BRK.AX', axis = 1)
    stock_close = stock_close.dropna(axis = 0, how = 'all')
    stock_close = stock_close.dropna(axis = 1, how = 'any')
    stock_close = stock_close.ix[:, 0: 100]
    
    # 2
    # Generate stock returns
    stock_ret = stock_close.pct_change().iloc[1: , : ]
    
    # 3
    # Covariance matrix of returns
    ret_cov = stock_ret.cov()
    eig_val, eig_vec = LA.eig(ret_cov)

    # perform PCA on the matrix
    # pca = PCA(n_components = 'mle', svd_solver = 'full')
    # pca.fit(ret_cov)
    # eig_val = pca.singular_values_
    
    plt.title('Eigenvalue')
    plt.xlabel('Number')
    plt.ylabel('Value')
    plt.plot(eig_val)
    plt.show()
    
    # count eigenvalues
    pos = sum(n > 0 for n in eig_val)
    neg = sum(n < 0 for n in eig_val)
    
    # 4
    # Get variance ratio
    # eig_rat = pca.explained_variance_ratio_
    eig_val_sum = eig_val.sum()
    eig_val_cum = eig_val.cumsum()
    cum_rat = eig_val_cum / eig_val_sum
    cum_50 = np.argwhere(cum_rat < 0.5)[-1] + 2
    cum_90 = np.argwhere(cum_rat < 0.9)[-1] + 2
    
    
    ## Prob 2
    # 3
    # Generate G and c
    G = np.array([[1] * 100, [1] * 17 + [0] * (100 - 17)])
    c = np.array([1, 0.1])
    
    inv_GCG = np.linalg.inv(G.dot(np.linalg.inv(ret_cov)).dot(np.transpose(G)))
    
    # 4
    # Calculate lambda and weights using the equation
    # a = [1] * len(R.transpose())
    a = 1
    R = np.transpose(stock_ret.mean())
    
    lamb = inv_GCG.dot(np.transpose(np.transpose(G.dot(np.linalg.inv(ret_cov)).dot(R)) - 2 * a * c))
    w_inv = - (1 / 2) * np.linalg.inv(ret_cov).dot(np.transpose(G).dot(lamb) - R)
    
    # Calculate lambda and weights using optimization
    # define the constants in the objective function
    args_fun = (R, a, ret_cov)
    # define the constants in the constraints
    args_con = (G, c)
    cons = con(args_con)
    # set the initial values
    w0 = np.asarray(([0] * 100))
    # apply the optimization method
    res = minimize(fun(args_fun), w0, method = 'SLSQP', constraints = cons)
    w_opt = res.x
    print(w_opt)
    
    # draw plot for weigts
    plt.title('Weigts derived from equations')
    plt.xlabel('Number of weights')
    plt.ylabel('Value')
    plt.plot(np.transpose(w_inv))
    plt.show()
    
    plt.title('Weigts derived from optimization')
    plt.xlabel('Number of weights')
    plt.ylabel('Value')
    plt.plot(np.transpose(w_opt))
    plt.show()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
