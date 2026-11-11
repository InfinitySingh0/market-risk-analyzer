# Upgrade validation

Local environment: Python 3.12, existing NumPy/SciPy/Streamlit dependencies. No new dependency was added.

## Required commands

The project virtual environment was activated before running:

```bash
python -m pytest -q
python generate_sample.py
python generate_report.py
```

- Tests: **49 passed** in 169.13 seconds. Existing numerical and UI assertions remain; parameterized invariance and no-look-ahead tests now include all six methods. Added tests cover EWMA hand recursion, pre-update FHS filtering, degenerate windows, seed repeatability, t formulas/fit, conditional coverage, traffic-light boundaries and descriptive ES diagnostics.
- One deprecation warning comes from the environment's preinstalled anyio pytest plugin, concerning a renamed pytest internal type. No project test failed.
- Browser checks: charts and comparison table rendered, the project ZIP was absent until explicitly prepared, the downloaded archive included LICENSE and the CI workflow and excluded Git history/virtual environments, and analysis exports included backtest_diagnostics.csv.
- Screenshot: docs/screenshot.png captures the synthetic demonstration dashboard.

## Scope decisions

- Student-t fits constrain 1.01 ≤ df ≤ 1000 to make ES finite and stabilize the near-normal limit; location and positive scale are fitted. This numerical constraint is documented in the UI and README.
- ES uses the user-authorized descriptive fallback. No formal Acerbi–Székely Z2 statistic or simulated p-value is claimed.
- CI is configured for Python 3.11 and 3.12. The local result above is Python 3.12; remote workflow results are separate.

## Regeneration results

Both generation commands exited successfully. SHA-256 comparisons confirmed that all two sample CSVs and four report artifacts were **byte-identical** to their preceding generated versions in this environment. Reports contain all six methods at both 95% and 99% confidence.

At 95%, normal VaR produced 23/792 exceptions versus 39.6 expected (Kupiec p=0.003417), rejecting coverage due to too few exceptions. Historical simulation produced 38/792 (p=0.792894), so coverage was not rejected. At 99%, both produced 6/792 (p=0.473793). The report explains that weak rejection power and sample selection prevent broad model rankings.
