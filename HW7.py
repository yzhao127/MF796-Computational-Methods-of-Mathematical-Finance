# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:29:58 2019

@author: yyang
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.stats import norm


def simulate_Heston(S0,r,q,para,T,N):
    t = T/N
    k =  para[0]
    theta =  para[1]
    sig=  para[2]
    p =  para[3]
    v0 =  para[4]

    cov = t * np.matrix([[1, p], [p, 1]])
    z = np.random.multivariate_normal((0, 0), cov, N)
    W1 = z[:,0]
    W2 = z[:,1]

    v = []
    S = []

    for i in range(N):
        if i == 0:
            S += [S0 + (r - q) * S0 * t + np.sqrt(v0) * S0 * W1[i]]
            v += [v0+k*(theta-v0)*t+sig*np.sqrt(v0)*W2[i]]

        else:
            S += [S[-1] + (r - q) * S[-1] * t + np.sqrt(max([0,v[-1]])) * S[-1] * W1[i]]
            v += [v[-1] + k * (theta - v[-1])*t + sig * np.sqrt(max([0,v[-1]])) * W2[i]]

    return S

def price_European(S0,K,r,q,para,T,N,M,type):
    price = []
    if type == 'put':
        for i in range(M):
            S = simulate_Heston(S0, r, q, para, T, N)
            price += [max([0.0,K-S[-1]])]
    else:
        for i in range(M):
            S = simulate_Heston(S0, r, q, para, T, N)
            price += [max([0.0,S[-1]-K])]
    return np.exp(-r*T)*np.average(price)

def c_Heston(u,sig,v0,k,p,theta,S0,t,r,q):
    ii = complex(0, 1)
    lamda = np.sqrt(sig**2*(u**2+ii*u)+(k-p*u*sig*ii)**2)
    w = np.exp((np.log(S0)+(r-q)*t)*u*ii+k*theta*t*(k-p*sig*u*ii)/(sig**2))/((np.cosh(lamda*t/2)+np.sinh(lamda*t/2)*(k-p*sig*u*ii)/lamda)**(2*k*theta/(sig**2)))
    c = w*np.exp(-1*(u**2+u*ii)*v0/(lamda/np.tanh(lamda*t/2)+k-ii*p*sig*u))

    return c

def c_derivative(v,a,sig,v0,k,p,theta,S0,t,r,q):
    ii = complex(0, 1)
    u =v-(a+1)*ii
    c_m = c_Heston(u,sig,v0,k,p,theta,S0,t,r,q)
    c = 1/((a+v*ii)*(a+1+v*ii))*c_m
    return c

def x(j,v_List,beta,a,sig,v0,k,p,theta,S0,t,r,q,f):
    ii = complex(0, 1)
    x = np.exp(-1*v_List[j]*beta*ii)*f(v_List[j],a,sig,v0,k,p,theta,S0,t,r,q)*w(j)
    return x

def w(j):
    if j == 0:
        return 1
    else:
        return 2

def fft(S0 ,K,N,B,a,f, para,t):
    k =  para[0]
    theta =  para[1]
    sig=  para[2]
    p =  para[3]
    v0 =  para[4]

    dv = B / N
    dk = 2 * math.pi / (dv * N)
    v_List = list((np.cumsum(np.ones((N, 1))) - 1) * dv)
    beta = np.log(S0) - dk * N / 2
    k_List = list(beta + (np.cumsum(np.ones((N, 1))) - 1) * dk)
    xi = []
    for j in range(N):
        xi  += [x(j,v_List,beta,a,sig,v0,k,p,theta,S0,t,r,q,f)*dv/2]

    zz = np.fft.fft(xi)
    zz =  np.array(zz) * (np.exp(-1*a * np.array(k_List))/ math.pi)
    Kt = np.exp(np.array(k_List))
    Kz = []
    Z = []
    zz = np.real(zz)
    for i in range(len(Kt)):
        if( Kt[i]>1e-16 )&(Kt[i] < 1e16)& ( Kt[i] != float("inf"))&( Kt[i] != float("-inf")) &( zz[i] != float("inf"))&(zz[i] != float("-inf")) & (zz[i] is not  float("nan")):
            Kz += [Kt[i]]
            Z += [zz[i]]
    tck = interpolate.splrep(Kz , np.real(Z))
    y =  np.exp(-r*t)*interpolate.splev(K, tck)
    return y

def  price_up_and_out(S0,K1,K2,r,q,para,T,N,M):
    price = []

    for i in range(M):
        S = simulate_Heston(S0, r, q, para, T, N)
        if max(S)<K2:
            price += [max([0.0, S[-1] - K1])]
        else:
            price += [0]

    return np.exp(-r * T) * np.mean(price)

def  price_up_and_out_control(S0,K1,K2,r,q,para,T,N,M):
    S_fft = fft(S0, K1, N1, B, a, c_derivative, para, T)
    price_uo = []
    price_eu = []
    for i in range(M):
        S = simulate_Heston(S0, r, q, para, T, N)
        if max(S)<K2:
            price_uo += [max([0.0, S[-1] - K1])]
        else:
            price_uo += [0]
        price_eu += [max([0.0,  S[-1] - K1])]
    co = np.vstack((np.exp(-r * T) *np.array(price_uo), np.exp(-r * T) *np.array(price_eu)))
    c =-1* np.cov(co)[1,0]/np.var(np.exp(-r * T) *np.array(price_eu))
    print(c)
    return np.mean (np.exp(-r * T) *np.array(price_uo) + c * (np.exp(-r * T) *np.array(price_eu) -  S_fft ))

#Prob 1
para =np.array([3.51,0.052,1.17,-0.77,0.034])
r = 0.015
q = 0.0177
S0 = 282
T = 1
K = 285

#Prob 2
B = 1000
N1 = 2 ** 12
a = 5
S_fft =fft(S0,K,N1,B,a,c_derivative,para,T)

#Prob 3
error_list = []
price_list = []
for N in [1000,5000,10000,20000]:
    error = []
    price = []
    for M in [500,1000,2000]:
        error_mean = []
        price_mean = []
        for i in range(1):
            price_mean += [price_European(S0,K,r,q,para,T,M,N,'call')]
            error_mean += [(price_mean[-1]-S_fft)**2]
        error += [np.average(error_mean)]
        price += [np.average(price_mean)]
    error_list += [error]
    price_list += [price]

M=1000
N=10000
S_s = price_European(S0,K,r,q,para,T,M,N,'call')

#Prob 4
error_list_1 = []
price_uo = []
K1 =285
K2 = 315
M = 1000
for N in [500,1000,2500,5000,10000,20000,40000]:
    price = []
    error = []
    for i in range(1):
        price += [price_up_and_out(S0,K1,K2,r,q,para,T,M,N)]
        if len(price_uo) != 0:
            error += [(price [-1]-price_uo[-1])**2]
    if len(price_uo) != 0:
        error_list_1 += [np.average(error)]
    price_uo += [np.average(price)]

#Prob 5
error_list_2 = []
price_cv = []
for N in [500,1000,2500,5000,10000,20000,40000]:
    price = []
    error = []
    for i in range(1):
        price += [price_up_and_out_control(S0,K1,K2,r,q,para,T,M,N)]
        if len(price_cv) != 0:
            error += [(price[-1] - price_cv[-1]) ** 2]
    if len(price_cv) != 0:
        error_list_2 += [np.average(error)]
    price_cv += [np.average(price)]




