#!/usr/bin/env python
# coding: utf-8

# # Risk Premia Estimation using GMM

# Start by importing the modules and functions needed

# In[ ]:


from numpy import hstack, ones, array, mat, tile, reshape, squeeze, eye, asmatrix
from numpy.linalg import inv
from pandas import read_csv, Series 
from scipy.linalg import kron
from scipy.optimize import minimize
import numpy as np
import statsmodels.api as sm


# Next a callable function is used to produce iteration-by-iteration output when using the non-linear optimizer.

# In[ ]:


iteration = 0
last_value = 0
function_count = 0

def iter_print(params):
    global iteration, last_value, function_count
    iteration += 1
    print(f'Func value: {last_value:6.6g}, Iteration: {iteration}, Function Count: {function_count}')


# The GMM objective, which is minimized, is defined next.

# In[ ]:


def gmm_objective(params, p_rets, f_rets, Winv, out=False):
    global last_value, function_count
    t,n = p_rets.shape
    t,k = f_rets.shape
    beta = squeeze(array(params[:(n*k)]))
    lam = squeeze(array(params[(n*k):]))
    beta = reshape(beta,(n,k))
    lam = reshape(lam,(k,1))
    betalam = beta @ lam
    expected_ret = f_rets @ beta.T
    e = p_rets - expected_ret
    instr = tile(f_rets,n)
    moments1  = kron(e,ones((1,k)))
    moments1 = moments1 * instr
    moments2 = p_rets - betalam.T
    moments = hstack((moments1,moments2))

    avg_moment = moments.mean(axis=0)
    
    J = t * mat(avg_moment) * mat(Winv) * mat(avg_moment).T
    J = J[0,0]
    last_value = J
    function_count += 1
    if not out:
        return J
    else:
        return J, moments


# The `G` matrix, which is the derivative of the GMM moments with respect to the parameters, is defined.

# In[ ]:


def gmm_G(params, p_rets, f_rets):
    t,n = p_rets.shape
    t,k = f_rets.shape
    beta = squeeze(array(params[:(n*k)]))
    lam = squeeze(array(params[(n*k):]))
    beta = reshape(beta,(n,k))
    lam = reshape(lam,(k,1))
    G = np.zeros((n*k+k,n*k+n))
    ffp = (f_rets.T @ f_rets) / t
    G[:(n*k),:(n*k)]=kron(eye(n),ffp)
    G[:(n*k),(n*k):] = kron(eye(n),-lam)
    G[(n*k):,(n*k):] = -beta.T
    
    return G


# Next, the data is imported and a subset of the test portfolios is selected to make the estimation faster.

# In[ ]:


data = read_csv('FamaFrench.csv')

# Split using both named colums and ix for larger blocks
dates = data['date'].values
factors = data[['VWMe','SMB','HML']].values
riskfree = data['RF'].values
portfolios = data.iloc[:,5:].values

t,n = portfolios.shape
portfolios = portfolios[:,np.arange(0,n,2)]
t,n = portfolios.shape
excess_ret = portfolios - np.reshape(riskfree,(t,1))
k = np.size(factors,1)


# Starting values for the factor loadings and rick premia are estimated using OLS and simple means.

# In[ ]:


betas = []
for i in range(n):
    res = sm.OLS(excess_ret[:,i],sm.add_constant(factors)).fit()
    betas.append(res.params[1:])

avg_return = excess_ret.mean(axis=0)
avg_return.shape = n,1
betas = array(betas)
res = sm.OLS(avg_return, betas).fit()
risk_premia = res.params


# The starting values are computed the first step estimates are found using the non-linear optimizer.  The initial weighting matrix is just the identify matrix.

# In[ ]:


risk_premia.shape = 3
starting_vals = np.concatenate((betas.flatten(),risk_premia))

Winv = np.eye(n*(k+1))
args = (excess_ret, factors, Winv)
iteration = 0
function_count = 0
opt = minimize(gmm_objective, starting_vals, args=args, callback=iter_print)
step1opt = opt.x


# Here we look at the risk premia estimates from the first step (inefficient) estimates.

# In[ ]:


premia = step1opt[-3:]
premia = Series(premia,index=['VWMe', 'SMB', 'HML'])
print('Annualized Risk Premia (First step)')
print(12 * premia)


# Next the first step estimates are used to estimate the moment conditions which are in-turn used to estimate the optimal weighting matrix for the moment conditions.  This is then used as an input for the 2nd-step estimates.

# In[ ]:


out = gmm_objective(step1opt, excess_ret, factors, Winv, out=True)
S = np.cov(out[1].T)
Winv2 = inv(S)
args = (excess_ret, factors, Winv2)

iteration = 0
function_count = 0
opt = minimize(gmm_objective, step1opt, args=args, callback=iter_print)
step2opt = opt.x


# Finally the VCV of the parameter estimates is computed.

# In[ ]:


out = gmm_objective(step2opt, excess_ret, factors, Winv2, out=True)
G = gmm_G(step2opt, excess_ret, factors)
S = np.cov(out[1].T)
vcv = inv(G @ inv(S) @ G.T)/t


# The annualized risk premia and their associated t-stats.

# In[ ]:


premia = step2opt[-3:]
premia = Series(premia,index=['VWMe', 'SMB', 'HML'])
premia_vcv = vcv[-3:,-3:]
print('Annualized Risk Premia')
print(12 * premia)

premia_stderr = np.diag(premia_vcv)
premia_stderr = Series(premia_stderr,index=['VWMe', 'SMB', 'HML'])
print('T-stats')
print(premia / premia_stderr)

