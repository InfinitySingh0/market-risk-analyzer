# Market Risk Analyzer

A Streamlit portfolio project for market risk analysts: one-day VaR, Expected Shortfall, rolling backtests, and stress testing for a single long SPY or BTC-USD position. No paid API is required.

## Run locally

Requires Python 3.11 or newer (tested on Python 3.12).

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. The app starts in offline demonstration mode. Choose Yahoo Finance download for market prices, or upload your own CSV. Yahoo access can be rate-limited or unavailable; the UI explicitly announces a synthetic fallback. Public access does not imply unrestricted redistribution; review provider terms for production use.

```bash
python -m pytest -q
python generate_sample.py
python generate_report.py
```

The generated report is `reports/sample_risk_report.md`. The accompanying CSVs contain model results and dated forecasts. Sample generation is deterministic and independent of network availability.

## Project structure

- `app.py`: Streamlit controls, charts, caching, error messages, and CSV exports.
- `risklab/data.py`: download, CSV validation, sample data, return and gap handling.
- `risklab/risk.py`: analytical normal, historical, and Monte Carlo risk engines.
- `risklab/backtest.py`: rolling forecasts, Kupiec, Christoffersen, and Wilson intervals.
- `risklab/stress.py`: historical endpoint and hypothetical shocks.
- `tests/`: numerical, leakage, statistical edge case, data and UI tests.
- `data/`: explicitly synthetic offline prices.
- `reports/`: reproducible example report and numerical exports.

## Data conventions

SPY is an ETF proxy for the S&P 500, not the index itself. Yahoo data uses `auto_adjust=True`; its adjusted Close incorporates provider corporate-action adjustments. BTC-USD uses its daily price series. Prices are nominal USD, and output is USD. No foreign-exchange conversion is modeled.

CSV schema:

```csv
Date,Adjusted Close
2024-01-02,100
2024-01-03,99
2024-01-04,100.5
```

Alternatively select the `Close` column explicitly; adjustment quality remains the user's responsibility. Invalid dates, duplicate dates, non-finite or non-positive prices cause validation errors. Dates are normalized to UTC dates and sorted. Returns are not forward filled and calendars are not merged. BTC gaps longer than one day and SPY gaps longer than four calendar days have their associated returns excluded. This heuristic does not identify every missing SPY session: users must supply complete exchange-session history. Unusual genuine exchange closures can also be excluded. Reported risk is per usable observed interval; where gaps are excluded, the estimation window contains the last N usable returns and may span longer elapsed time.

SPY's real observations follow its market sessions; BTC trades every calendar day. Friday-to-Monday is one SPY session but three BTC days. Cross-asset one-day estimates do not represent identical elapsed time. Daily volatility is not annualized.

Synthetic data spans 2018–2025. It uses seeded Student-t innovations and deliberately changing volatility, with weekday dates for SPY (including exchange holidays) and daily dates for BTC. It is not an approximation to actual historical prices. Historical crisis results are disabled for synthetic data. Student-t is used only in the demo generator, not as an additional fitted risk model.

## Risk formulas

Simple return r_t = P_t/P_(t-1) - 1; fractional loss L_t = -r_t. Monetary loss = position value × L_t. Negative values mean gains and are not clamped. Exposure is held constant for comparisons; this is not a compounded investment strategy.

For confidence c, tail probability α = 1-c, z = Φ⁻¹(c), trailing sample mean μ, and standard deviation σ estimated with ddof=1:

- Normal VaR(c) = -μ + σz.
- Normal ES(c) = -μ + σφ(z)/α.

Historical VaR uses NumPy's `inverted_cdf`: the smallest order statistic whose empirical cumulative probability reaches c. No linear quantile interpolation is applied. Historical ES is the integrated empirical quantile over [c,1]. Sort losses descending, let m=Nα and k=floor(m); sum the worst k losses plus (m-k) times the next loss, then divide by m. This gives exactly the intended tail mass and handles ties without counting every threshold tie as a full tail observation. With fewer than one observation of tail mass, ES equals the worst observation, emphasizing the lack of tail information.

Monte Carlo simulates normally distributed simple-return losses with mean -μ and standard deviation σ and applies the same empirical VaR/ES definitions. This is parametric Monte Carlo, not a separate distribution-free model. It should converge to the normal analytical values as simulation count increases. A fixed seed is reused across rolling windows (common random numbers), keeping differences across dates from being driven by changing random draws. Simulation discretization and sampling error remain. A normal simple-return distribution allows losses exceeding 100%, an economic limitation particularly relevant to volatile assets; draws are not clipped silently.

## Out-of-sample backtesting

For realized return at index i, the training slice is [i-window, i), never including return i or later. The forecast is compared with L_i; a breach is strictly L_i > VaR_i. Ties can affect empirical coverage for discrete distributions. Forecast dates, training end dates, losses, thresholds, and breaches are exported. The headline VaR/ES cards use the final training window to forecast the next observation; backtest charts contain previously evaluated forecasts.

Kupiec: compare Bernoulli likelihood under α against the MLE breach rate k/N. LR_uc = 2(log L_MLE - log L_null), with an asymptotic χ²(1) reference. Zero-count terms use 0 log 0 = 0, so all/zero breaches produce finite coverage statistics.

Christoffersen independence: compare a two-state first-order Markov likelihood against an iid transition likelihood using n00,n01,n10,n11. LR_ind has an asymptotic χ²(1) reference. If either prior-state row has no observations, return N/A with an explanation rather than implying the model passed. Transition counts are available from the API. Sparse transitions and low expected counts weaken inference; the dashboard highlights fewer than 10 expected breaches. Wilson 95% intervals summarize breach-rate uncertainty.

These are VaR tests, not ES tests. Mean realized loss on breach dates versus mean ES on those dates is only descriptive: selecting on breaches, changing forecasts, and sample size preclude treating it as a formal ES validation.

No automatic model tuning is performed. Interactive inspection can itself introduce selection bias. For research, decide settings on a training/validation period, then evaluate once on a separately reserved final holdout. Repeatedly changing settings against the same backtest is not an untouched test.

## Stress scenarios

- GFC / Lehman week: 2008-09-12 → 2008-09-19.
- COVID selloff: 2020-02-19 → 2020-03-23.
- COVID Friday–Monday: 2020-03-13 → 2020-03-16 (one SPY session, three BTC intervals with complete data).

Historical return = P_end/P_start - 1; monetary stress loss = -return × position. Exact endpoints must exist in the selected dataset; unavailable scenarios remain unavailable. No Bitcoin history is invented for 2008. Users can also specify instantaneous price shocks from -100% to +100%. Multi-day cumulative losses are labeled separately from one-day VaR. No scenario probabilities are inferred. These endpoint scenarios need not equal maximum drawdown within the interval.

## Dashboard and exports

Risk overview includes current-window loss histogram, VaR/ES markers, daily rolling volatility, and rolling VaR for all models. Backtesting includes realized losses, risk forecasts, breach markers, coverage statistics, and descriptive ES diagnostics. Stress testing shows historical availability and hypothetical shocks. The exports tab provides individual CSV downloads and a ZIP containing forecasts, breaches, stress results, configuration, and current model comparisons.

## Validation and limitations

Tests cover closed-form normal results, empirical fractional tails, ties, risk signs, translation invariance, confidence-level monotonicity, ES ≥ VaR, Monte Carlo convergence and repeatability, look-ahead prevention across all methods, zero/all breach cases, clustered breaches, CSV errors, calendar gaps, stress losses, and the default/BTC Streamlit flows.

Risk estimates depend on the selected window. Regime shifts, tails, liquidity, transaction costs, slippage, taxes, model uncertainty, and market closures may make realized results differ. This project does not model credit, operational or liquidity risk, nor derivatives or multi-asset dependence. Historical simulation is not guaranteed to outperform parametric methods.

## Source references

- Yahoo retrieval adapter: https://github.com/ranaroussi/yfinance
- NumPy empirical quantiles: https://numpy.org/doc/stable/reference/generated/numpy.quantile.html
- SciPy normal distribution: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
- Wilson intervals: https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportion_confint.html
- Kupiec (1995), “Techniques for Verifying the Accuracy of Risk Measurement Models.”
- Christoffersen (1998), “Evaluating Interval Forecasts.”

## Additional risk models

EWMA normal uses RiskMetrics variance recursion v_(t+1) = λv_t + (1−λ)(r_t−μ)², λ=0.94 by default. The seed is the training window's sample variance. The UI/API support zero mean or sample mean (default sample, preserving translation invariance). The final variance forecasts the next observation. Normal VaR/ES formulas use its square root.

Filtered historical simulation uses pre-update sigma_t to standardize each training innovation, bootstraps these residuals with a fixed seed, and constructs next returns μ+sigma_next×residual. VaR/ES use the existing empirical quantile definitions. Seed and sample mean use the entire training window: these are in-sample filtered residuals, not historical out-of-sample predictions. A zero denominator maps a zero innovation to zero. No observations outside [i-window,i) enter a forecast.

Student-t uses scipy.stats.t.fit with location and scale and a constrained MLE optimizer: 1.01 ≤ df ≤ 1000 ensures finite mean and ES; variance may still be infinite when df ≤ 2. Standardizing inputs and rounding standardized observations to 9 decimal places improves fitting/cache stability. Constant windows use a point-mass limit. Failed fits produce errors rather than silently switching models. For q=t_ppf(c,ν), VaR=−loc+scale×q and ES=−loc+scale×(ν+q²)/(ν−1)×t_pdf(q,ν)/(1−c). The fitted scale is not standard deviation. Unbounded t returns, like normal returns, can imply losses greater than 100%.

Monte Carlo (normal) remains a numerical check of the analytical normal model, not independent distributional evidence. Model-specific EWMA settings are exported with the configuration. All six models use the same simple-return/loss conventions.
The Student-t df upper bound of 1000 provides a numerical near-normal limit when the likelihood is flat toward infinite df. These are constrained fits, not unrestricted MLE; boundary estimates and small samples should be interpreted cautiously. The standardized scale lower bound is 1e-8.
