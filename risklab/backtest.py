import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2, binom
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


def coverage_tests(breaches, confidence, losses=None, es_forecasts=None):
    if not 0 < confidence < 1:
        raise ValueError('Confidence must be between zero and one.')
    b = np.asarray(breaches, dtype=bool)
    if b.ndim != 1:
        raise ValueError('Breaches must be a one-dimensional sequence.')
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
    if np.isfinite(result['independence_lr']):
        lr_cc = lr_uc + result['independence_lr']
        result.update(conditional_coverage_lr=lr_cc,
                      conditional_coverage_p=float(chi2.sf(lr_cc,2)),
                      conditional_coverage_note='Joint coverage/independence test; asymptotic chi-square(2).')
    else:
        result.update(conditional_coverage_lr=np.nan, conditional_coverage_p=np.nan,
                      conditional_coverage_note=result['independence_note'])
    result.update(traffic_light(b, confidence))
    result.update(es_diagnostic(b, losses, es_forecasts))
    return result


def traffic_light(breaches, confidence=.99):
    """Basel-style educational zones: binomial CDF(k;250,.01)."""
    b = np.asarray(breaches, dtype=bool)
    if b.ndim != 1 or not 0 < confidence < 1:
        raise ValueError('Require one-dimensional breaches and valid confidence.')
    if not np.isclose(confidence, .99, rtol=0, atol=1e-12) or len(b)<250:
        return {'traffic_light':'N/A','traffic_light_breaches':np.nan,'traffic_light_cdf':np.nan,
                'traffic_light_note':'Requires 99% one-day VaR and at least 250 forecasts.'}
    k=int(b[-250:].sum())
    probability=float(binom.cdf(k,250,.01))
    zone='green' if probability<.95 else ('yellow' if probability<.9999 else 'red')
    return {'traffic_light':zone,'traffic_light_breaches':k,'traffic_light_cdf':probability,
            'traffic_light_note':'Last 250 forecasts only; educational Basel-style zones, not a regulatory assessment.'}


def es_diagnostic(breaches, losses=None, es_forecasts=None):
    """Descriptive fallback only. No Acerbi–Szekely statistic or p-value claimed."""
    b=np.asarray(breaches,dtype=bool)
    result={'es_test':'Descriptive fallback (not a formal ES backtest)', 'es_p_value':np.nan,
            'es_tail_mean_loss':np.nan, 'es_tail_mean_forecast':np.nan, 'es_tail_gap':np.nan}
    if losses is None or es_forecasts is None:
        result['es_note']='Realized losses and ES forecasts not supplied.'
        return result
    loss=np.asarray(losses,dtype=float); es=np.asarray(es_forecasts,dtype=float)
    if b.ndim!=1 or loss.shape!=b.shape or es.shape!=b.shape or not np.isfinite(loss).all() or not np.isfinite(es).all():
        raise ValueError('Losses, ES forecasts and breaches must be aligned finite vectors.')
    if not b.any():
        result['es_note']='No VaR breaches: tail diagnostic unavailable.'
        return result
    result.update(es_tail_mean_loss=float(loss[b].mean()), es_tail_mean_forecast=float(es[b].mean()),
                  es_tail_gap=float((loss[b]-es[b]).mean()),
                  es_note='Mean loss minus forecast ES on VaR-breach dates; selection bias and small samples prevent formal inference. No simulated p-value.')
    return result
