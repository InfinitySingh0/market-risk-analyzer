from pathlib import Path
import pandas as pd
from risklab.data import load_sample,returns_from_prices
from risklab.risk import METHODS,estimate
from risklab.backtest import rolling_forecasts,coverage_tests
root=Path(__file__).parent/'reports';root.mkdir(exist_ok=True)
p,source=load_sample('SPY');r=returns_from_prices(p.loc['2022-01-01':'2025-12-31'])
rows=[];all_forecasts=[]
for m in METHODS:
    risk=estimate(r.iloc[-250:],.99,m,20000,42)
    f=rolling_forecasts(r,250,.99,m,20000,42);t=coverage_tests(f.breach,.99)
    rows.append({'method':m,'VaR':risk.var,'ES':risk.es,**t})
    all_forecasts.append(f.assign(method=m))
summary=pd.DataFrame(rows);summary.to_csv(root/'sample_model_results.csv',index=False)
pd.concat(all_forecasts).to_csv(root/'sample_forecasts.csv')
lines=['# Sample risk report','', '**Synthetic demonstration only — not evidence about actual SPY performance.**','',
'Scope: $100,000 constant long exposure; daily simple returns; 99% confidence; trailing 250-return window; 20,000 normal Monte Carlo draws; seed 42. Offline weekday data, 2022–2025; exchange holidays are not removed.','',
'## Results','', '| Model | Next-day VaR | Next-day ES | Breaches / forecasts | Expected breaches | Kupiec p | Independence p |', '|---|---:|---:|---:|---:|---:|---:|']
for x in rows:
    lines.append(f'| {x["method"]} | ${x["VaR"]*100000:,.0f} | ${x["ES"]*100000:,.0f} | {x["breaches"]}/{x["observations"]} | {x["expected_breaches"]:.2f} | {x["kupiec_p"]:.4f} | {x["independence_p"]:.4f} |')
lines+=['','## Interpretation','',
'Normal analytical and normal Monte Carlo estimates use the same distribution and parameter estimates. Their differences arise from Monte Carlo sampling. Historical estimates reflect the observed tail rather than a normal assumption. The demo generator uses Student-t innovations and volatility regimes, so normal model misspecification is deliberate; this is not evidence that historical simulation will always outperform.', '',
'Each forecast uses only preceding observations. Actual breach rates and p-values in the table describe this fixed configuration; no settings were optimized against the evaluation sample. Small expected breach counts weaken asymptotic tests. A p-value above 0.05 does not establish model adequacy. Kupiec tests unconditional coverage; Christoffersen tests first-order breach independence. Neither validates ES.','',
'## Tail estimation and stress limitations','',
'At 99% confidence, a 250-observation window contains only 2.5 observations of tail mass. Historical ES gives fractional weight to the boundary loss. Estimates are consequently sensitive to a few observations.', '',
'A hypothetical −20% shock produces a $20,000 loss. Historical GFC and COVID results are intentionally unavailable for synthetic data. Use genuine prices with exact scenario endpoints to run these scenarios. Multi-day cumulative scenarios are distinct from one-day VaR.','',
'## Reproduction','', '`python generate_sample.py` then `python generate_report.py`. Machine-readable results and forecasts accompany this report.']
(root/'sample_risk_report.md').write_text('\n'.join(lines))
print(summary[['method','VaR','ES','breaches','observations','kupiec_p','independence_p']].to_string(index=False))
