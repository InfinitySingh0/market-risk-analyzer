from dataclasses import dataclass
import numpy as np
from functools import lru_cache
from scipy.optimize import minimize
from scipy.stats import norm, t

METHODS = ['Normal parametric', 'Historical simulation', 'Monte Carlo (normal)',
           'EWMA normal', 'Filtered historical simulation', 'Student-t parametric']

@dataclass(frozen=True)
class Risk:
    var: float
    es: float


def validate(returns, confidence):
    x = np.asarray(returns, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('At least two finite returns are required.')
    if not 0 < confidence < 1:
        raise ValueError('Confidence must be between zero and one.')
    return x


def empirical(losses, confidence):
    """Inverse empirical CDF VaR; integrated empirical quantile ES."""
    x = validate(losses, confidence)
    var = float(np.quantile(x, confidence, method='inverted_cdf'))
    descending = np.sort(x)[::-1]
    mass = len(x) * (1 - confidence)
    whole = int(np.floor(mass))
    fraction = mass - whole
    tail_sum = descending[:whole].sum()
    if fraction > 0 and whole < len(x):
        tail_sum += fraction * descending[whole]
    return Risk(var, float(tail_sum / mass))


def normal_risk(returns, confidence=.99):
    x = validate(returns, confidence)
    mu, sigma = x.mean(), x.std(ddof=1)
    z = norm.ppf(confidence)
    return Risk(float(-mu + sigma*z), float(-mu + sigma*norm.pdf(z)/(1-confidence)))


def estimate(returns, confidence=.99, method=METHODS[0], simulations=20000, seed=42, ewma_lambda=.94, mean_mode="sample"):
    x = validate(returns, confidence)
    if method == METHODS[0]:
        return normal_risk(x, confidence)
    if method == METHODS[1]:
        return empirical(-x, confidence)
    if method == METHODS[2]:
        if simulations < 1000:
            raise ValueError('Use at least 1,000 simulations.')
        losses = np.random.default_rng(seed).normal(-x.mean(), x.std(ddof=1), simulations)
        return empirical(losses, confidence)
    if method == 'EWMA normal':
        mu, _, next_vol = ewma_volatility(x, ewma_lambda, mean_mode)
        z = norm.ppf(confidence)
        return Risk(float(-mu+next_vol*z), float(-mu+next_vol*norm.pdf(z)/(1-confidence)))
    if method == 'Filtered historical simulation':
        if simulations < 1000:
            raise ValueError('Use at least 1,000 simulations.')
        if x.std(ddof=1) < 1e-14:
            return empirical(-x, confidence)  # Deterministic limit; no stable residual scaling.
        mu, vol, next_vol = ewma_volatility(x, ewma_lambda, mean_mode)
        residuals = np.divide(x-mu, vol, out=np.zeros_like(x), where=vol>0)
        sampled = np.random.default_rng(seed).choice(residuals, size=simulations, replace=True)
        return empirical(-(mu + next_vol*sampled), confidence)
    if method == 'Student-t parametric':
        df, loc, scale = fit_student_t(x)
        return student_t_risk(df, loc, scale, confidence)
    raise ValueError('Unknown risk method.')


def ewma_volatility(returns, ewma_lambda=.94, mean_mode='sample'):
    """Pre-return conditional vols and next-day forecast, seeded with window variance.

    The seed and optional mean use the entire training window. This is an in-window
    filter initialization, never a claim of out-of-sample historical residuals.
    Return i is divided by sigma_i BEFORE its own squared innovation updates it.
    """
    x = validate(returns, .99)
    if not 0 < ewma_lambda < 1:
        raise ValueError('EWMA lambda must be strictly between zero and one.')
    if mean_mode not in ('zero', 'sample'):
        raise ValueError('Mean mode must be zero or sample.')
    mu = float(x.mean()) if mean_mode == 'sample' else 0.
    variance = float(x.var(ddof=1))
    vols = np.empty(len(x))
    for i, value in enumerate(x):
        vols[i] = np.sqrt(max(variance, 0.))
        variance = ewma_lambda*variance + (1-ewma_lambda)*(value-mu)**2
    return mu, vols, float(np.sqrt(max(variance, 0.)))


def _finite_mean_optimizer(func, initial, args=(), disp=0):
    # Finite-mean constrained MLE: 1.01 <= df <= 1000, positive scale; no new dependency.
    result = minimize(func, initial, args=args, method='Nelder-Mead',
                      bounds=[(1.01, 1000.), (None, None), (1e-8, None)],
                      options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-8})
    if not result.success or not np.isfinite(result.fun):
        raise ValueError('Student-t MLE did not converge: '+str(result.message))
    return result.x


@lru_cache(maxsize=4096)
def _fit_standardized(values):
    return t.fit(np.asarray(values), 8., loc=0., scale=.8, optimizer=_finite_mean_optimizer)


def fit_student_t(returns):
    """Location/scale/df MLE, constrained to 1.01 <= df <= 1000 for finite ES.

    Affine normalization improves optimization; 9-decimal rounding stabilizes
    cache keys. Constant windows use a deterministic point-mass limit.
    """
    x = validate(returns, .99)
    mu, sigma = float(x.mean()), float(x.std(ddof=1))
    if sigma < 1e-14:
        return float('inf'), mu, 0.
    df, loc, scale = _fit_standardized(tuple(np.round((x-mu)/sigma, 9)))
    return float(df), float(mu+sigma*loc), float(sigma*scale)


def student_t_risk(df, loc, scale, confidence=.99):
    """Student-t return model: scale is NOT the distribution standard deviation."""
    if not 0 < confidence < 1 or not np.isfinite(loc) or not np.isfinite(scale) or scale < 0 or np.isnan(df) or df <= 1:
        raise ValueError('Require df > 1, finite location, nonnegative scale and valid confidence.')
    if scale == 0:
        return Risk(-float(loc), -float(loc))
    if np.isinf(df):
        z = norm.ppf(confidence)
        return Risk(-loc+scale*z, -loc+scale*norm.pdf(z)/(1-confidence))
    q = t.ppf(confidence, df)
    return Risk(float(-loc+scale*q),
                float(-loc+scale*(df+q*q)/(df-1)*t.pdf(q,df)/(1-confidence)))
