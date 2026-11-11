"""Deterministic synthetic case study; no parameters selected against this evaluation."""
from pathlib import Path
import pandas as pd
from risklab.data import load_sample, returns_from_prices
from risklab.risk import METHODS, estimate
from risklab.backtest import rolling_forecasts, coverage_tests

ROOT = Path(__file__).parent / 'reports'


def comparison_markdown(summary):
    lines = ['| Confidence | Model | Next VaR | Next ES | Breaches / N | Kupiec p | CC p | Traffic light |',
             '|---|---|---:|---:|---:|---:|---:|---|']
    for _, x in summary.iterrows():
        cc = f'{x.conditional_coverage_p:.4f}' if pd.notna(x.conditional_coverage_p) else 'N/A'
        lines.append(f'| {x.confidence:.0%} | {x.method} | {x.VaR:.2%} | {x.ES:.2%} | {x.breaches}/{x.observations} | {x.kupiec_p:.4f} | {cc} | {x.traffic_light} |')
    return '\n'.join(lines)


def generate_report():
    ROOT.mkdir(exist_ok=True)
    prices, _ = load_sample('SPY')
    returns = returns_from_prices(prices.loc['2022-01-01':'2025-12-31'])
    rows, forecasts = [], []
    for confidence in [.95, .99]:
        for method in METHODS:
            risk = estimate(returns.iloc[-250:], confidence, method, 20000, 42)
            f = rolling_forecasts(returns, 250, confidence, method, 20000, 42)
            stats = coverage_tests(f.breach, confidence, f.loss, f.es)
            rows.append({'confidence': confidence, 'method': method, 'VaR': risk.var, 'ES': risk.es, **stats})
            forecasts.append(f.assign(method=method, confidence=confidence))
    summary = pd.DataFrame(rows)
    summary.to_csv(ROOT/'sample_model_results.csv', index=False, float_format='%.12g')
    pd.concat(forecasts).to_csv(ROOT/'sample_forecasts.csv', float_format='%.12g', date_format='%Y-%m-%d')
    table = comparison_markdown(summary)
    lines = ['# Sample risk report', '', '**Synthetic demonstration only — not actual SPY or Bitcoin market history.**', '',
             'Fixed configuration: $100,000 long exposure; daily simple returns; 2022–2025 synthetic SPY weekday observations (including holidays); 250-return training window; 95% and 99% confidence; 20,000 draws; seed 42; EWMA lambda 0.94 and sample mean. No parameters were tuned against this evaluation.', '',
             '## Results', '', table, '', 'Next VaR/ES use the final window; tests evaluate earlier forecasts. Multiply fractional risks by $100,000 for monetary risk.', '', '## Model comparison findings', '']
    for confidence in [.95, .99]:
        group = summary[summary.confidence==confidence].set_index('method')
        for method in ['Normal parametric','Historical simulation']:
            x=group.loc[method]
            verdict='rejects coverage' if x.kupiec_p < .05 else 'does not reject coverage'
            lines.append(f'- At {confidence:.0%}, {method}: {x.breaches}/{x.observations} breaches ({x.breach_rate:.2%}) against {x.expected_breaches:.1f} expected; Kupiec p={x.kupiec_p:.4f} {verdict} at the 5% test level.')
        normal = group.loc['Normal parametric']
        if normal.kupiec_p < .05:
            direction = 'fewer' if normal.breaches < normal.expected_breaches else 'more'
            lines.append(f'- The normal rejection at {confidence:.0%} comes from {direction} breaches than expected; a rejection does not necessarily mean risk was understated.')
        rejected=group[group.kupiec_p<.05].index.tolist()
        lines.append(f'- At {confidence:.0%}, models rejected by Kupiec: '+(', '.join(rejected) if rejected else 'none')+'.')
        lines.append(f'- At {confidence:.0%}, next-day ES ranges from {group.ES.min():.2%} to {group.ES.max():.2%} across the six specifications.')
    lines += ['', 'The normal and normal-Monte-Carlo specifications share the same assumed distribution; Monte Carlo is a numerical check. EWMA and FHS respond to recent volatility through the filter. Student-t estimates tail shape under finite-mean constrained MLE. Historical estimates depend on observed tail order statistics. The normal-versus-historical coverage contrast at 95% is specific to this deterministic synthetic sample, not a ranking that generalizes to markets.', '',
              'At 99%, fewer exceptions reduce power: non-rejection is not proof of adequate tails. The last-250 traffic light and full-history coverage tests use different evaluation lengths. Conditional coverage includes both breach frequency and first-order clustering. Sparse transitions weaken asymptotic inference.', '',
              '## ES and stress limitations', '',
              'ES diagnostics are descriptive only: average loss minus forecast ES on breach dates, recorded in the CSV. No Acerbi–Székely test or simulated p-value is implemented. At 95% the training tail mass is 12.5 observations, versus 2.5 at 99%; empirical ES uses fractional boundary weight.', '',
              'A hypothetical −20% shock causes a $20,000 loss. Historical GFC and COVID results remain unavailable for synthetic data. Real historical scenarios require genuine data with exact endpoints; multi-day cumulative losses are not one-day VaR.', '',
              '## Reproduce', '', '`python generate_sample.py` then `python generate_report.py`. Neither script uses current time or live data. CSV rows and columns have fixed ordering. Optimizer/library differences across environments may cause small floating-point differences; repeatability is checked within a fixed environment.']
    (ROOT/'sample_risk_report.md').write_text('\n'.join(lines)+'\n')
    (ROOT/'model_comparison.md').write_text(table+'\n')
    print(summary[['confidence','method','breaches','observations','kupiec_p','conditional_coverage_p','traffic_light']].to_string(index=False))
    return summary


if __name__ == '__main__':
    generate_report()
