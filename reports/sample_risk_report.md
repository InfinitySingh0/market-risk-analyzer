# Sample risk report

**Synthetic demonstration only — not actual SPY or Bitcoin market history.**

Fixed configuration: $100,000 long exposure; daily simple returns; 2022–2025 synthetic SPY weekday observations (including holidays); 250-return training window; 95% and 99% confidence; 20,000 draws; seed 42; EWMA lambda 0.94 and sample mean. No parameters were tuned against this evaluation.

## Results

| Confidence | Model | Next VaR | Next ES | Breaches / N | Kupiec p | CC p | Traffic light |
|---|---|---:|---:|---:|---:|---:|---|
| 95% | Normal parametric | 1.41% | 1.79% | 23/792 | 0.0034 | 0.0069 | N/A |
| 95% | Historical simulation | 1.29% | 1.61% | 38/792 | 0.7929 | 0.7578 | N/A |
| 95% | Monte Carlo (normal) | 1.43% | 1.80% | 23/792 | 0.0034 | 0.0069 | N/A |
| 95% | EWMA normal | 1.39% | 1.77% | 32/792 | 0.2005 | 0.1143 | N/A |
| 95% | Filtered historical simulation | 1.37% | 1.83% | 40/792 | 0.9481 | 0.7058 | N/A |
| 95% | Student-t parametric | 1.40% | 1.96% | 30/792 | 0.1026 | 0.0809 | N/A |
| 99% | Normal parametric | 2.03% | 2.34% | 6/792 | 0.4738 | 0.7390 | green |
| 99% | Historical simulation | 1.78% | 2.01% | 6/792 | 0.4738 | 0.7390 | green |
| 99% | Monte Carlo (normal) | 2.04% | 2.34% | 6/792 | 0.4738 | 0.7390 | green |
| 99% | EWMA normal | 2.01% | 2.32% | 11/792 | 0.2989 | 0.4992 | green |
| 99% | Filtered historical simulation | 2.19% | 2.32% | 10/792 | 0.4754 | 0.6820 | yellow |
| 99% | Student-t parametric | 2.28% | 2.92% | 4/792 | 0.1217 | 0.2959 | green |

Next VaR/ES use the final window; tests evaluate earlier forecasts. Multiply fractional risks by $100,000 for monetary risk.

## Model comparison findings

- At 95%, Normal parametric: 23/792 breaches (2.90%) against 39.6 expected; Kupiec p=0.0034 rejects coverage at the 5% test level.
- At 95%, Historical simulation: 38/792 breaches (4.80%) against 39.6 expected; Kupiec p=0.7929 does not reject coverage at the 5% test level.
- The normal rejection at 95% comes from fewer breaches than expected; a rejection does not necessarily mean risk was understated.
- At 95%, models rejected by Kupiec: Normal parametric, Monte Carlo (normal).
- At 95%, next-day ES ranges from 1.61% to 1.96% across the six specifications.
- At 99%, Normal parametric: 6/792 breaches (0.76%) against 7.9 expected; Kupiec p=0.4738 does not reject coverage at the 5% test level.
- At 99%, Historical simulation: 6/792 breaches (0.76%) against 7.9 expected; Kupiec p=0.4738 does not reject coverage at the 5% test level.
- At 99%, models rejected by Kupiec: none.
- At 99%, next-day ES ranges from 2.01% to 2.92% across the six specifications.

The normal and normal-Monte-Carlo specifications share the same assumed distribution; Monte Carlo is a numerical check. EWMA and FHS respond to recent volatility through the filter. Student-t estimates tail shape under finite-mean constrained MLE. Historical estimates depend on observed tail order statistics. The normal-versus-historical coverage contrast at 95% is specific to this deterministic synthetic sample, not a ranking that generalizes to markets.

At 99%, fewer exceptions reduce power: non-rejection is not proof of adequate tails. The last-250 traffic light and full-history coverage tests use different evaluation lengths. Conditional coverage includes both breach frequency and first-order clustering. Sparse transitions weaken asymptotic inference.

## ES and stress limitations

ES diagnostics are descriptive only: average loss minus forecast ES on breach dates, recorded in the CSV. No Acerbi–Székely test or simulated p-value is implemented. At 95% the training tail mass is 12.5 observations, versus 2.5 at 99%; empirical ES uses fractional boundary weight.

A hypothetical −20% shock causes a $20,000 loss. Historical GFC and COVID results remain unavailable for synthetic data. Real historical scenarios require genuine data with exact endpoints; multi-day cumulative losses are not one-day VaR.

## Reproduce

`python generate_sample.py` then `python generate_report.py`. Neither script uses current time or live data. CSV rows and columns have fixed ordering. Optimizer/library differences across environments may cause small floating-point differences; repeatability is checked within a fixed environment.
