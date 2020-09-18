"""
To get line profiling data, first install line-profiler

pip install line-profiler --upgrade

Then run the following two commands

kernprof -l examples_gjrgarch_lineprofiler.py
python -m line_profiler examples_gjrgarch_lineprofiler.py.lprof
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import size, log, pi, sum, array, zeros, diag, asarray, sqrt, copy
from numpy.linalg import inv
import pandas as pd
from scipy.optimize import minimize
import builtins

try:
    builtins.profile
    print("Running with kernprof")
except AttributeError:
    # No line profiler, provide a pass-through version
    def profile(func):
        return func


    builtins.profile = profile
    print("Running without kernprof")


@profile
def gjr_garch_likelihood(parameters, data, sigma2, out=None):
    """Negative log-likelihood for GJR-GARCH(1,1,1) model"""
    mu = parameters[0]
    omega = parameters[1]
    alpha = parameters[2]
    gamma = parameters[3]
    beta = parameters[4]

    T = size(data, 0)
    eps = data - mu
    # Data and sigma2 are T by 1 vectors
    for t in range(1, T):
        sigma2[t] = (omega + alpha * eps[t - 1] ** 2
                     + gamma * eps[t - 1] ** 2 * (eps[t - 1] < 0) + beta * sigma2[t - 1])

    logliks = 0.5 * (log(2 * pi) + log(sigma2) + eps ** 2 / sigma2)
    loglik = sum(logliks)

    if out is None:
        return loglik
    else:
        return loglik, logliks, copy(sigma2)


def gjr_constraint(parameters):
    """ Constraint that alpha+gamma/2+beta<=1"""

    alpha = parameters[2]
    gamma = parameters[3]
    beta = parameters[4]

    return array([1 - alpha - gamma / 2 - beta])


constraint = {"type": "ineq", "fun": gjr_constraint}


def hessian_2sided(fun, theta, args):
    f = fun(theta, *args)
    h = 1e-5 * np.abs(theta)
    thetah = theta + h
    h = thetah - theta
    K = size(theta, 0)
    h = np.diag(h)

    fp = zeros(K)
    fm = zeros(K)
    for i in range(K):
        fp[i] = fun(theta + h[i], *args)
        fm[i] = fun(theta - h[i], *args)

    fpp = zeros((K, K))
    fmm = zeros((K, K))
    for i in range(K):
        for j in range(i, K):
            fpp[i, j] = fun(theta + h[i] + h[j], *args)
            fpp[j, i] = fpp[i, j]
            fmm[i, j] = fun(theta - h[i] - h[j], *args)
            fmm[j, i] = fmm[i, j]

    hh = (diag(h))
    hh = hh.reshape((K, 1))
    hh = hh @ hh.T

    H = zeros((K, K))
    for i in range(K):
        for j in range(i, K):
            H[i, j] = (fpp[i, j] - fp[i] - fp[j] + f
                       + f - fm[i] - fm[j] + fmm[i, j]) / hh[i, j] / 2
            H[j, i] = H[i, j]

    return H


# Import data
ftse = pd.read_csv('FTSE_1984_2012.csv', parse_dates=[0])
# Set index
ftse.index = ftse.pop('Date')
# Flip upside down
ftse = ftse.iloc[::-1]
# Compute returns
ftse_price = ftse['Adj Close']
ftse_return = 100 * ftse_price.pct_change().dropna()

# Starting values
starting_vals = array([ftse_return.mean(),
                       ftse_return.var() * .01,
                       .03, .09, .90])

# Estimate parameters
finfo = np.finfo(np.float64)
bounds = [(-10 * ftse_return.mean(), 10 * ftse_return.mean()),
          (finfo.eps, 2 * ftse_return.var()),
          (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

T = ftse_return.shape[0]
sigma2 = np.ones(T) * ftse_return.var()
# Pass a NumPy array, not a pandas Series
args = (np.asarray(ftse_return), sigma2)
opt = minimize(gjr_garch_likelihood,
               starting_vals,
               constraints=constraint,
               bounds=bounds,
               args=args)
estimates = opt.x

loglik, logliks, sigma2final = gjr_garch_likelihood(estimates, ftse_return,
                                                    sigma2, out=True)

step = 1e-5 * estimates
scores = zeros((T, 5))
for i in range(5):
    h = step[i]
    delta = np.zeros(5)
    delta[i] = h

    loglik, logliksplus, sigma2 = gjr_garch_likelihood(estimates + delta, np.asarray(ftse_return),
                                                       sigma2, out=True)
    loglik, logliksminus, sigma2 = gjr_garch_likelihood(estimates - delta, np.asarray(ftse_return),
                                                        sigma2, out=True)

    scores[:, i] = (logliksplus - logliksminus) / (2 * h)

I = (scores.T @ scores) / T

J = hessian_2sided(gjr_garch_likelihood, estimates, args)
J = J / T
Jinv = inv(J)
vcv = Jinv @ I @ Jinv / T
vcv = asarray(vcv)


output = np.vstack((estimates, sqrt(diag(vcv)), estimates / sqrt(diag(vcv)))).T
print('Parameter   Estimate       Std. Err.      T-stat')
param = ['mu', 'omega', 'alpha', 'gamma', 'beta']
for i in range(len(param)):
    print(
        f'{param[i]:<11} {output[i, 0]:>0.6f}        {output[i, 1]:0.6f}    {output[i, 2]: 0.5f}')

# Register date converters
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

# Produce a plot
dates = ftse.index[1:]
fig = plt.figure()
ax = fig.add_subplot(111)
volatility = pd.DataFrame(np.sqrt(252 * sigma2), index=dates)
ax.plot(volatility)
ax.autoscale(tight='x')
fig.autofmt_xdate()
fig.tight_layout(pad=1.5)
ax.set_ylabel('Volatility')
ax.set_title('FTSE Annualized Volatility (GJR GARCH(1,1,1))')
plt.show()
