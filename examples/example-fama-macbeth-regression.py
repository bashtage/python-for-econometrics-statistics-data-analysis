#!/usr/bin/env python
# coding: utf-8

# # Estimating the Risk Premia using Fama-MacBeth Regressions

# This example highlights how to implement a Fama-MacBeth 2-stage regression to estimate factor risk premia, make inference on the risk premia, and test whether a linear factor model can explain a cross-section of portfolio returns. This example closely follows [Cochrane::2001] (See also [JagannathanSkoulakisWang::2010]). As in the previous example, the first segment contains the imports. 

# In[ ]:


from numpy import mat, cov, mean, hstack, multiply,sqrt,diag,     squeeze, ones, array, vstack, kron, zeros, eye, savez_compressed
from numpy.linalg import inv
from scipy.stats import chi2
from pandas import read_csv
import statsmodels.api as sm


# Next, the data are imported. I formatted the data downloaded from Ken French's website into an easy-to-import CSV which can be read by `pandas.read_csv`. The data is split using named columns for the small sets of variables and `ix` for the portfolios. The code uses pure NumPy arrays, and so `values` is used to retrieve the array from the DataFrame. The dimensions are determined using `shape`. Finally the risk free rate is forced to have 2 dimensions so that it will be broadcastable with the portfolio returns in the construction of the excess returns to the Size and Value-weighted portfolios. `asmatrix` is used to return matrix views of all of the arrays. This code is linear algebra-heavy and so matrices are easier to use than arrays.

# In[ ]:


data = read_csv('FamaFrench.csv')

# Split using both named colums and ix for larger blocks
dates = data['date'].values
factors = data[['VWMe', 'SMB', 'HML']].values
riskfree = data['RF'].values
portfolios = data.iloc[:, 5:].values

# Use mat for easier linear algebra
factors = mat(factors)
riskfree = mat(riskfree)
portfolios = mat(portfolios)

# Shape information
t,k = factors.shape
t,n = portfolios.shape
# Reshape rf and compute excess returns
riskfree.shape = t,1
excess_returns = portfolios - riskfree


# 
# The next block does 2 things:
# 
# 1. Compute the time-series $\beta$s. This is done be regressing the full array of excess returns on the factors (augmented with a constant) using lstsq.
# 2. Compute the risk premia using a cross-sectional regression of average excess returns on the estimates $\beta$s. This is a standard regression where the step 1 $\beta$ estimates are used as regressors, and the dependent variable is the average excess return.

# In[ ]:


# Time series regressions
x = sm.add_constant(factors)
ts_res = sm.OLS(excess_returns, x).fit()
alpha = ts_res.params[0]
beta = ts_res.params[1:]
avgexcess_returns = mean(excess_returns, 0)
# Cross-section regression
cs_res = sm.OLS(avgexcess_returns.T, beta.T).fit()
risk_premia = cs_res.params


# 
# The asymptotic variance requires computing the covariance of the demeaned returns and the weighted pricing errors. The problem is formulated using 2-step GMM where the moment conditions are 
# \begin{equation}
# g_{t}\left(\theta\right)=\left[\begin{array}{c}
# \epsilon_{1t}\\
# \epsilon_{1t}f_{t}\\
# \epsilon_{2t}\\
# \epsilon_{2t}f_{t}\\
# \vdots\\
# \epsilon_{Nt}\\
# \epsilon_{Nt}f_{t}\\
# \beta u_{t}
# \end{array}\right]
# \end{equation}
# 
# where $\epsilon_{it}=r_{it}^{e}-\alpha_{i}-\beta_{i}^{\prime}f_{t}$, $\beta_{i}$ is a $K$ by 1 vector of factor loadings, $f_{t}$ is a $K$ by 1 set of factors, $\beta=\left[\beta_{1}\,\beta_{2}\ldots\beta_{N}\right]$ is a $K$ by $N$ matrix of all factor loadings, $u_{t}=r_{t}^{e}-\beta'\lambda$ are the $N$ by 1 vector of pricing errors and $\lambda$ is a $K$  by 1 vector of risk premia. 
# The vector of parameters is then $\theta= \left[\alpha_{1}\:\beta_{1}^{\prime}\:\alpha_{2}\:\beta_{2}^{\prime}\:\ldots\:\alpha_{N}\,\beta_{N}^{\prime}\:\lambda'\right]'$
#  To make inference on this problem, the derivative of the moments with respect to the parameters, $\partial g_{t}\left(\theta\right)/\partial\theta^{\prime}$ is needed. With some work, the estimator of this matrix can be seen to be 
#  
# \begin{equation}
#  G=E\left[\frac{\partial g_{t}\left(\theta\right)}{\partial\theta^{\prime}}\right]=\left[\begin{array}{cc}
# -I_{n}\otimes\Sigma_{X} & 0\\
# G_{21} & -\beta\beta^{\prime}
# \end{array}\right].
# \end{equation}
# 
# where $X_{t}=\left[1\: f_{t}^{\prime}\right]'$  and $\Sigma_{X}=E\left[X_{t}X_{t}^{\prime}\right]$. $G_{21}$ is a matrix with the structure 
# 
# \begin{equation}
# G_{21}=\left[G_{21,1}\, G_{21,2}\,\ldots G_{21,N}\right]
# \end{equation}
# 
# where 
# 
# \begin{equation}
# G_{21,i}=\left[\begin{array}{cc} 
# 0_{K,1} & \textrm{diag}\left(E\left[u_{i}\right]-\beta_{i}\odot\lambda\right)\end{array}\right]\end{equation}
# 
# and where $E\left[u_{i}\right]$ is the expected pricing error. In estimation, all expectations are replaced with their sample analogues.

# In[ ]:


# Moment conditions
X = sm.add_constant(factors)
p = vstack((alpha, beta))
epsilon = excess_returns - x @ p
moments1 = kron(epsilon, ones((1, k + 1)))
moments1 = multiply(moments1, kron(ones((1, n)), x))
u = excess_returns - risk_premia[None,:] @ beta
moments2 = u * beta.T
# Score covariance
S = mat(cov(hstack((moments1, moments2)).T))
# Jacobian
G = mat(zeros((n * k + n + k, n * k + n + k)))
sigma_x = (x.T @ x) / t
G[:n * k + n, :n * k + n] = kron(eye(n), sigma_x)
G[n * k + n:, n * k + n:] = -beta @ beta.T
for i in range(n):
    temp = zeros((k, k + 1))
    values = mean(u[:, i]) - multiply(beta[:, i], risk_premia)
    temp[:, 1:] = diag(values)
    G[n * k + n:, i * (k + 1):(i + 1) * (k + 1)] = temp

vcv = inv(G.T) * S * inv(G) / t


# The $J$-test examines whether the average pricing errors, $\hat{\alpha}$, are zero. The $J$ statistic has an asymptotic $\chi_{N}^{2}$  distribution, and the model is badly rejected.

# In[ ]:


vcv_alpha = vcv[0:n * k + n:4, 0:n * k + n:4]
J = alpha @ inv(vcv_alpha) @ alpha.T
J = J[0, 0]
Jpval = 1 - chi2(25).cdf(J)


# The final block using formatted output to present all of the results in a readable manner.

# In[ ]:


vcvrisk_premia = vcv[n * k + n:, n * k + n:]
annualized_rp = 12 * risk_premia
arp = list(squeeze(annualized_rp))
arp_se = list(sqrt(12 * diag(vcvrisk_premia)))
print('        Annualized Risk Premia')
print('           Market       SMB        HML')
print('--------------------------------------')
print(f'Premia     {arp[0]:0.4f}    {arp[1]:0.4f}     {arp[2]:0.4f}')
print(f'Std. Err.  {arp_se[0]:0.4f}    {arp_se[1]:0.4f}     {arp_se[2]:0.4f}')
print('\n\n')

print(f'J-test:   {J:0.4f}')
print(f'P-value:   {Jpval:0.4f}')

i = 0
beta_se = []
for j in range(5):
    for m in range(5):
        a = alpha[i]
        b = beta[:, i]
        variances = diag(vcv[(k + 1) * i:(k + 1) * (i + 1), (k + 1) * i:(k + 1) * (i + 1)])
        beta_se.append(sqrt(variances))
        s = sqrt(variances)
        c = hstack((a, b))
        t = c / s
        print(f'Size: {j+1}, Value:{m+1}   Alpha   Beta(VWM)   Beta(SMB)   Beta(HML)')
        print(f'Coefficients: {a:>10,.4f}  {b[0]:>10,.4f}  {b[1]:>10,.4f}  {b[2]:>10,.4f}')
        print(f'Std Err.      {s[0]:>10,.4f}  {s[1]:>10,.4f}  {s[2]:>10,.4f}  {s[3]:>10,.4f}')
        print(f'T-stat        {t[0]:>10,.4f}  {t[1]:>10,.4f}  {t[2]:>10,.4f}  {t[3]:>10,.4f}')
        print('')
        i += 1


# The final block converts the standard errors of $\beta$ to be an array and saves the results.

# In[ ]:


beta_se = array(beta_se)
savez_compressed('fama-macbeth-results', alpha=alpha, beta=beta,
                 beta_se=beta_se, arp_se=arp_se, arp=arp, J=J, Jpval=Jpval)


# ## Save Results
# 
# Save the estimated values for use in the $\LaTeX$ notebook. 

# In[ ]:


from numpy import savez
savez('fama-macBeth-results.npz', arp=arp, beta=beta, arp_se=arp_se,
      beta_se=beta_se, J=J, Jpval=Jpval)

