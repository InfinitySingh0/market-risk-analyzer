from dataclasses import dataclass
import numpy as np
from scipy.stats import norm

METHODS = ['Normal parametric', 'Historical simulation', 'Monte Carlo (normal)']

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


def estimate(returns, confidence=.99, method=METHODS[0], simulations=20000, seed=42):
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
    raise ValueError('Unknown risk method.')
