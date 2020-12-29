# -*- coding: utf-8 -*-
"""
@author: yyangzh
"""

import math
import numpy as np
import pandas as pd
from scipy.stats import norm, kurtosis, skew, mode
from scipy.optimize import root, minimize
from scipy import interpolate
import matplotlib.pyplot as plt


class Euro_BSformula:
    def __init__(self, S0, K, r, sigma, q, T, kind='put'):
        if kind not in ['call', 'put']:
            raise ValueError('Option type must be \'call\' or \'put\'')

        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.q = q
        self.kind = kind

        self.d1 = ((np.log(self.S0 / self.K)
                + (self.r -self.q + 0.5 * self.sigma ** 2) * self.T)
                / (self.sigma * np.sqrt(self.T)))
        self.d2 = ((np.log(self.S0 / self.K)
                + (self.r - 0.5 * self.sigma ** 2) * self.T)
                / (self.sigma * np.sqrt(self.T)))


    def price(self):
        if self.kind == "put":
            return (self.K * math.exp(-self.r * self.T) * norm.cdf(-self.d2)
                    - self.S0 * np.exp(-self.q * self.T) * norm.cdf(-self.d1))
        else:
            return (self.S0 * np.exp(-self.q * self.T) * norm.cdf(self.d1)
                    - self.K * math.exp(-self.r * self.T) * norm.cdf(self.d2))
    
    def calc_delta(self):
        if self.kind == "put": return np.exp(-self.q * self.T) * norm.cdf(self.d1) - 1
        else: return np.exp(-self.q * self.T) * norm.cdf(self.d1)
   
    def calc_vega(self):
        return np.exp(-self.q * self.T) * self.S0 * norm.pdf(self.d1) * np.sqrt(self.T)
    
class FFT:
    def __init__(self,sigma,eta0,kappa,rho,theta,S0,r,q,T):
        self.sigma = sigma
        self.eta0 = eta0
        self.kappa = kappa
        self.rho = rho
        self.theta = theta
        self.S0 = S0
        self.r = r
        self.q = q
        self.T = T
        
    def Heston_fft(self,alpha,n,B,K):
        r = self.r
        T = self.T
        S0 = self.S0
        N = 2**n
        Eta = B / N
        Lambda_Eta = 2 * math.pi / N
        Lambda = Lambda_Eta / Eta
        
        J = np.arange(1,N+1,dtype = complex)
        vj = (J-1) * Eta
        m = np.arange(1,N+1,dtype = complex)
        Beta = np.log(S0) - Lambda * N / 2
        km = Beta + (m-1) * Lambda
        
        ii = complex(0,1)
        
        Psi_vj = np.zeros(len(J),dtype = complex)
        
        for zz in range(0,N):
            u = vj[zz] - (alpha + 1) * ii
            numer = self.Heston_cf(u)
            denom = (alpha + vj[zz] * ii) * (alpha + 1 + vj[zz] * ii)
            
            Psi_vj [zz] = numer / denom
            
        xx = (Eta/2) * Psi_vj * np.exp(-ii * Beta * vj) * (2 - self.dirac(J-1))
        zz = np.fft.fft(xx)
        
        Mul = np.exp(-alpha * np.array(km)) / np.pi
        zz2 = Mul * np.array(zz).real
        k_List = list(Beta + (np.cumsum(np.ones((N, 1))) - 1) * Lambda)
        Kt = np.exp(np.array(k_List))
       
        Kz = []
        Z = []
        for i in range(len(Kt)):
            if( Kt[i]>1e-16 )&(Kt[i] < 1e16)& ( Kt[i] != float("inf"))&( Kt[i] != float("-inf")) &( zz2[i] != float("inf"))&(zz2[i] != float("-inf")) & (zz2[i] is not  float("nan")):
                Kz += [Kt[i]]
                Z += [zz2[i]]
        tck = interpolate.splrep(Kz , np.real(Z))
        price =  np.exp(-r*T)*interpolate.splev(K, tck).real
        
        return price
    
    def dirac(self,n):
        y = np.zeros(len(n),dtype = complex)
        y[n==0] = 1
        return y
        
    def Heston_cf(self,u):
        sigma = self.sigma
        eta0 = self.eta0
        kappa = self.kappa
        rho = self.rho
        theta = self.theta
        S0 = self.S0
        r = self.r
        q = self.q
        T = self.T
        
        ii = complex(0,1)
        
        l = np.sqrt(sigma**2*(u**2+ii*u)+(kappa-ii*rho*sigma*u)**2)
        w = np.exp(ii*u*np.log(S0)+ii*u*(r-q)*T+kappa*theta*T*(kappa-ii*rho*sigma*u)/sigma**2)/(np.cosh(l*T/2)+(kappa-ii*rho*sigma*u)/l*np.sinh(l*T/2))**(2*kappa*theta/sigma**2)
        y = w*np.exp(-(u**2+ii*u)*eta0/(l/np.tanh(l*T/2)+kappa-ii*rho*sigma*u))
        
        return y
    
    
def extract_strike(params):
    sigma = params[0]
    T = params[1]
    delta = params[2]

    if delta < 0:
        return root(lambda K: norm.ppf(delta + 1) - (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T)), S0).x
    else:
        return root(lambda K: norm.ppf(delta) - (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T)), S0).x

    
def density_changs(K, coef, T):
    h = 0.01
    dens = []
    for i in range(len(K)):
        c = Euro_BSformula( S0, K[i], r, coef[0]*K[i] + coef[1], q, T, kind='put').price()
        cpos = Euro_BSformula( S0, K[i]+h, r, coef[0]*(K[i]+h) + coef[1], q,T,  kind='put').price()
        cneg = Euro_BSformula( S0, K[i]-h, r, coef[0]*(K[i]-h) + coef[1], q,T,  kind='put').price()
        dens += [np.exp(r * T) * (cneg - 2*c + cpos)/ h**2]
    return dens
    
def density_consts(K, sigma, T):
    h = 0.1
    dens = []
    for i in range(len(K)):
        c = Euro_BSformula( S0, K[i], r, sigma, q,T, kind='call').price()
        cpos = Euro_BSformula( S0, K[i]+h, r, sigma, q,T,  kind='call').price()
        cneg = Euro_BSformula( S0, K[i]-h, r, sigma, q,T,  kind='call').price()
        dens += [np.exp(r * T) * (cneg - 2*c + cpos)/ h**2]
    return dens

def digit_px(K, coef, T, kind):
    h = 0.1 
    p = ((Euro_BSformula( S0, K+h, r, coef[0]*(K+h)+ coef[1], q,T,  kind).price() -
         Euro_BSformula( S0, K-h, r, coef[0]*(K-h)+ coef[1], q,T,  kind).price())/(2*h))
    if kind == "put":
        return p
    else:
        return -p

def remove_arb(K, c, p):
    ind=[]
    for i in range(len(K) - 1):        
        rmv = False
        if (c[i] <=  c[i+1]) or (p[i] >=  p[i+1]): rmv = True
        elif (((c[i] - c[i+1])/(K[i+1]-K[i]) <= 0) or ((c[i] - c[i+1])/(K[i+1]-K[i]) >= 1)or 
              ((p[i] - p[i+1])/(K[i+1]-K[i]) >= 0) or ((p[i] - p[i+1])/(K[i+1]-K[i]) <= -1)):
            rmv = True
        if i < (len(K) - 2):
            if ( (c[i] + c[i+2] - 2*c[i+1]) <= 0) or ((p[i] + p[i+2] - 2*p[i+1]) <= 0):
                rmv = True  
        if rmv: ind += [i+1]
    return ind

def ew_minf(par):
    sse = 0
    for i in expd:
        alpha = 1.0
        n = 12
        B = 1000
        sub = opt[opt['expDays'] == i]
        a = FFT(par[0],par[1],par[2],par[3],par[4],S0,r,q,sub['expT'].values[0])    
        for j in range(len(sub['K'])):
            sse += (a.Heston_fft(alpha,n,B,sub['K'].values[j]) -
                    cmid[opt['expDays'] == i].values[j]) ** 2
        alpha = -1.5
        for j in range(len(sub['K'])):
            sse += (a.Heston_fft(alpha,n,B,sub['K'].values[j]) - 
                    pmid[opt['expDays'] == i].values[j]) ** 2
    return sse

def w_minf(par):
    sse = 0
    for i in expd:
        alpha = 1.0
        n = 12
        B = 1000
        sub = opt[opt['expDays'] == i]
        a = FFT(par[0],par[1],par[2],par[3],par[4],S0,r,q,sub['expT'].values[0])    
        for j in range(len(sub['K'])):
            sse +=  1 / csprd[opt['expDays'] == i].values[j] * (a.Heston_fft(alpha,n,B,sub['K'].values[j]) -
                    cmid[opt['expDays'] == i].values[j]) ** 2
        alpha = -1.5
        for j in range(len(sub['K'])):
            sse += 1 / psprd[opt['expDays'] == i].values[j] * (a.Heston_fft(alpha,n,B,sub['K'].values[j]) - 
                    pmid[opt['expDays'] == i].values[j]) ** 2
    return sse

def imp_vol(sig):
    return root(lambda sig: px - Euro_BSformula( S0, K, r, sig, q,expT, kind).price(), 0.1).x

                
if __name__=='__main__':
    #
    # Prob 1
    #
     S0 = 100.0
     r = 0.0
     q = 0.0
     vol = pd.read_csv('C:/Users/yyang/OneDrive/Documents/Documents/BU/Courses/19Spring/MF796/HW/HW5/vol.csv')
#     # (a) 
     strikes1 = []
     strikes3 = []
     for i in range(len(vol)):
         params = [float(vol.iloc[i]['1M']), 1/12, float(vol.iloc[i]['Delta'])]
         strikes1 += [extract_strike(params)]
         params = [float(vol.iloc[i]['3M']), 3/12, float(vol.iloc[i]['Delta'])]
         strikes3 += [extract_strike(params)]
     vol.insert(2,'K1',np.asarray(strikes1))
     vol.insert(4,'K3',np.asarray(strikes3))
#     # (b) 
     vol1coef = np.polyfit(vol['K1'], vol['1M'],1)
     vol3coef = np.polyfit(vol['K3'], vol['3M'],1)
     K = np.arange(85, 110, 1)
     volf1 = vol1coef[0]*K + vol1coef[1]    
     volf3 = vol3coef[0]*K + vol3coef[1]
     
     plt.plot(K, volf1)
     plt.plot(K, volf3)
     plt.title('Volatility Function')
     plt.xlabel('Strikes')
     plt.ylabel('Volatility')
     plt.show()  
#     # (c) 
     K = np.arange(50,130,1)
     dens1 = density_changs(K, vol1coef, 1/12)
     dens3 = density_changs(K, vol3coef, 3/12)    
     plt.plot(K,dens1)
     plt.plot(K,dens3)
     plt.title('Risk Neutral Density')
     plt.xlabel('ST')
     plt.ylabel('Probability')
     plt.legend(['1 month','3 month'])
     plt.show()
     
     print('          1 month option             3 month option')
     print('Mean      '+ str(np.mean(dens1))+'       '+str(np.mean(dens3)))
     print('Variance  '+ str(np.var(dens1))+'      '+str(np.var(dens3)))
     print('Skewness  '+ str(skew(dens1))+'          '+str(skew(dens3)))
     print('Kurtosis  '+ str(kurtosis(dens1))+'          '+str(kurtosis(dens3)))
#     
#     # (d) 
     K = np.arange(50,130,1)
     sig1_cons = vol.iloc[3]['1M']
     dens1_d = density_consts(K, sig1_cons, 1/12)
     plt.plot(K,dens1)
     plt.plot(K,dens1_d)
     plt.title('Risk Neutral Density')
     plt.xlabel('ST')
     plt.ylabel('Probability')
     plt.legend(['changeable volatility','constant volatility'])
     plt.show()
     
     print('1 month option       changeable volatility    constant volatility')
     print('Mean                 '+ str(np.mean(dens1))+'      '+str(np.mean(dens1_d)))
     print('Variance             '+ str(np.var(dens1))+'     '+str(np.var(dens1_d)))
     print('Skewness             '+ str(skew(dens1))+'         '+str(skew(dens1_d)))
     print('Kurtosis             '+ str(kurtosis(dens1))+'         '+str(kurtosis(dens1_d)))
     
     K = np.arange(50,130,1)
     sig3_cons = vol.iloc[3]['3M']
     dens3_d = density_consts(K, sig3_cons, 3/12) 
     plt.plot(K,dens3)
     plt.plot(K,dens3_d)
     plt.title('Risk Neutral Density')
     plt.xlabel('ST')
     plt.ylabel('Probability')
     plt.legend(['changeable volatility','constant volatility'])
     plt.show()
     
     print('3 month option       changeable volatility       constant volatility')
     print('Mean                 '+ str(np.mean(dens3))+'        '+str(np.mean(dens3_d)))
     print('Variance             '+ str(np.var(dens3))+'       '+str(np.var(dens3_d)))
     print('Skewness             '+ str(skew(dens3))+'           '+str(skew(dens3_d)))
     print('Kurtosis             '+ str(kurtosis(dens3))+'          '+str(kurtosis(dens3_d)))
#     
#     # (e) 
#     # i. 
     T1 = 1/12
     K1 = 110
     p1 = digit_px(K1, vol1coef, T1, 'put')
     print('The price of 1M European Digital Put Option with Strike 110 is '+str(p1))
     # ii. 
     T2 = 3/12
     K2 = 105
     p2 = digit_px(K2, vol3coef, T2, 'call')
     print('The price of 3M European Digital Call Option with Strike 105 is '+str(p2))
     # iii. 
     vol2coef = (vol3coef + vol1coef)/2
     K3 = 100
     T3 = 2/12
     Kn = 0.1
     K = np.arange(K3,110,Kn)
     dens2 = density_changs(K, vol2coef, T3)
     p3 = Kn * np.dot(dens2, K - K3)
     print('The price of 2M European Call Option with Strike 100 is '+str(p3))
# 
    #
    # Prob 2
    #
    r = 0.015
    q = 0.0177
    S0 = 267.15

    opt = pd.read_excel('C:/Users/yyang/OneDrive/Documents/Documents/BU/Courses/19Spring/MF796/HW/HW5/mf796-hw5-opt-data.xlsx')
    cmid = (opt['call_bid'] + opt['call_ask'])/2
    pmid = (opt['put_bid'] + opt['put_ask'])/2
    opt.insert(5,'call_mid',np.asarray(cmid))
    opt.insert(8,'put_mid',np.asarray(pmid))
    
    # (a) 
    expd = np.unique(opt['expDays'])
    rmv_ind = []
    for i in expd:
        sub = opt[opt['expDays'] == i]
        rmv_ind += remove_arb(sub['K'].values, sub['call_mid'].values, sub['put_mid'].values)
    if rmv_ind != []:
        opt.drop(opt.index[rmv_ind])
        
    # (b) 
    par0 = [0.25, 0.1, 1.5, -0.5, 0.1]
    bnds = ((0,2.0), (0, 1.0), (0.001, 5.0), (-1,1), (0,2.0))
    res = minimize(ew_minf, par0, method='L-BFGS-B', bounds=bnds)

    # (c) 
    par01 = [0.2,0.03,0.05,-0.6,0.2]
    res1 = minimize(ew_minf, par01, method='L-BFGS-B', bounds=bnds)
    par02 = [0.8,0.09,0.25,-0.25,0.06]
    res2 = minimize(ew_minf, par02, method='L-BFGS-B', bounds=bnds)
    
    bnds1 = ((0,2.0), (0, 0.5), (0.001, 2.5), (-1,0.2), (0,1.0))
    res3 = minimize(ew_minf, par0, method='L-BFGS-B', bounds=bnds1)
    res4 = minimize(ew_minf, par01, method='L-BFGS-B', bounds=bnds1)
    res5 = minimize(ew_minf, par02, method='L-BFGS-B', bounds=bnds1)
    
    #  d) 
    csprd = opt['call_ask'] - opt['call_bid']
    psprd = opt['put_ask'] - opt['put_bid']
    res_w = minimize(w_minf, par0, method='SLSQP', bounds=bnds)
    
    res_w1 = minimize(w_minf, par01, method='L-BFGS-B', bounds=bnds)
    res_w2 = minimize(w_minf, par02, method='L-BFGS-B', bounds=bnds)
    res_w3 = minimize(w_minf, par0, method='L-BFGS-B', bounds=bnds1)
    res_w4 = minimize(w_minf, par01, method='L-BFGS-B', bounds=bnds1)
    res_w5 = minimize(w_minf, par02, method='L-BFGS-B', bounds=bnds1)
    
    #
    # Prob 3
    #
     expT = 3/12
     K = 275
     r = 0.015
     q = 0.0177
     S0 = 267.15
     kind = 'call'
#     
#     # (a) 
     sigma = 1.18
     eta0 = 0.034
     kappa = 3.52
     rho = -0.77
     theta = 0.052       
     
     alpha = 1
     n = 15
     B = 1000
     h = 0.01
     H_delta = (FFT(sigma,eta0,kappa,rho,theta,S0+h,r,q,expT).Heston_fft(alpha,n,B,K) - 
                FFT(sigma,eta0,kappa,rho,theta,S0-h,r,q,expT).Heston_fft(alpha,n,B,K))/(2*h)
# 
#     # i) 
     px = FFT(sigma,eta0,kappa,rho,theta,S0,r,q,expT).Heston_fft(alpha,n,B,K)
     BS_impvol = imp_vol(sigma)
     BS_delta = Euro_BSformula(S0, K, r, BS_impvol, q, expT, kind).calc_delta()
#     
#     # (b) 
#     # i. 
     s = 0.01 * eta0
     H_vega = (FFT(sigma,eta0+s,kappa,rho,theta+s,S0,r,q,expT).Heston_fft(alpha,n,B,K) - 
                FFT(sigma,eta0-s,kappa,rho,theta-s,S0,r,q,expT).Heston_fft(alpha,n,B,K))/(2*s)
#     
#     # ii. 
     BS_vega = Euro_BSformula(S0, K, r, BS_impvol, q, expT, kind).calc_vega()
