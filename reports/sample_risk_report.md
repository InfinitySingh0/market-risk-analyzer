# Sample risk report

**Synthetic demonstration only — not evidence about actual SPY performance.**

Scope: $100,000 constant long exposure; daily simple returns; 99% confidence; trailing 250-return window; 20,000 normal Monte Carlo draws; seed 42. Offline weekday data, 2022–2025; exchange holidays are not removed.

## Results

| Model | Next-day VaR | Next-day ES | Breaches / forecasts | Expected breaches | Kupiec p | Independence p |
|---|---:|---:|---:|---:|---:|---:|
| Normal parametric | $2,029 | $2,338 | 6/792 | 7.92 | 0.4738 | 0.7620 |
| Historical simulation | $1,781 | $2,012 | 6/792 | 7.92 | 0.4738 | 0.7620 |
| Monte Carlo (normal) | $2,040 | $2,339 | 6/792 | 7.92 | 0.4738 | 0.7620 |

## Interpretation

Normal analytical and normal Monte Carlo estimates use the same distribution and parameter estimates. Their differences arise from Monte Carlo sampling. Historical estimates reflect the observed tail rather than a normal assumption. The demo generator uses Student-t innovations and volatility regimes, so normal model misspecification is deliberate; this is not evidence that historical simulation will always outperform.

Each forecast uses only preceding observations. Actual breach rates and p-values in the table describe this fixed configuration; no settings were optimized against the evaluation sample. Small expected breach counts weaken asymptotic tests. A p-value above 0.05 does not establish model adequacy. Kupiec tests unconditional coverage; Christoffersen tests first-order breach independence. Neither validates ES.

## Tail estimation and stress limitations

At 99% confidence, a 250-observation window contains only 2.5 observations of tail mass. Historical ES gives fractional weight to the boundary loss. Estimates are consequently sensitive to a few observations.

A hypothetical −20% shock produces a $20,000 loss. Historical GFC and COVID results are intentionally unavailable for synthetic data. Use genuine prices with exact scenario endpoints to run these scenarios. Multi-day cumulative scenarios are distinct from one-day VaR.

## Reproduction

`python generate_sample.py` then `python generate_report.py`. Machine-readable results and forecasts accompany this report.