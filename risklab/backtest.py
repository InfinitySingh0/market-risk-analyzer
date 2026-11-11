import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2
from statsmodels.stats.proportion import proportion_confint
from .risk import estimate


def rolling_forecasts(returns, window=250, confidence=.99, method='Normal parametric', simulations=20000, seed=42, ewma_lambda=.94, mean_mode='sample'):
    if window < 2 or len(returns) <= window:
        raise ValueError('Need more return observations than the estimation window.')
    if not returns.index.is_monotonic_increasing or returns.index.has_duplicates:
        raise ValueError('Return dates must be sorted and unique.')
    rows = []
    for i in range(window, len(returns)):
        # Fixed seed provides common random numbers, making MC reproducible across windows.
        risk = estimate(returns.iloc[i-window:i], confidence, method, simulations, seed, ewma_lambda, mean_mode)
        loss = -float(returns.iloc[i])
        rows.append((returns.index[i], returns.index[i-1], loss, risk.var, risk.es, loss > risk.var))
    return pd.DataFrame(rows, columns=['date','training_end','loss','var','es','breach']).set_index('date')


def loglik(successes, failures, probability):
    return float(xlogy(successes, probability) + xlogy(failures, 1-probability))


def coverage_tests(breaches, confidence):
    b = np.asarray(breaches, dtype=bool)
    n, k = len(b), int(b.sum())
    if n == 0:
        raise ValueError('No out-of-sample forecasts.')
    alpha = 1-confidence
    lr_uc = max(0., 2*(loglik(k,n-k,k/n)-loglik(k,n-k,alpha)))
    result = {'observations':n, 'breaches':k, 'expected_breaches':n*alpha,
              'breach_rate':k/n, 'kupiec_lr':lr_uc, 'kupiec_p':float(chi2.sf(lr_uc,1))}
    lo, hi = proportion_confint(k,n,method='wilson')
    result.update(breach_ci_low=float(lo), breach_ci_high=float(hi))
    counts = np.zeros((2,2), dtype=int)
    for a,c in zip(b[:-1], b[1:]):
        counts[int(a),int(c)] += 1
    n00,n01,n10,n11 = counts.ravel()
    result.update(n00=int(n00),n01=int(n01),n10=int(n10),n11=int(n11))
    if n < 3 or n00+n01 == 0 or n10+n11 == 0:
        result.update(independence_lr=np.nan, independence_p=np.nan,
                      independence_note='Not identifiable: both prior breach states need observations.')
    else:
        p = (n01+n11)/(n-1)
        unrestricted = loglik(n01,n00,n01/(n00+n01))+loglik(n11,n10,n11/(n10+n11))
        restricted = loglik(n01+n11,n00+n10,p)
        lr = max(0.,2*(unrestricted-restricted))
        result.update(independence_lr=lr, independence_p=float(chi2.sf(lr,1)), independence_note='Asymptotic test; sparse transitions weaken interpretation.')
    result['warning'] = ('Small sample: fewer than 10 expected breaches; p-values are fragile.' if n*alpha < 10 else 'Tests assess VaR coverage and independence, not ES accuracy.')
    return result
