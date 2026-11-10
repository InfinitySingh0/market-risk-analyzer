from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def clean_prices(frame, price_column='Adjusted Close'):
    if 'Date' not in frame or price_column not in frame:
        raise ValueError(f'CSV needs Date and {price_column} columns.')
    dates = pd.to_datetime(frame['Date'], errors='coerce', utc=True).dt.tz_convert(None).dt.normalize()
    prices = pd.to_numeric(frame[price_column], errors='coerce')
    if dates.isna().any():
        raise ValueError('Invalid dates: correct the CSV before analysis.')
    if dates.duplicated().any():
        raise ValueError('Duplicate dates: provide one observation per date.')
    if prices.isna().any() or not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError('Prices must be finite and positive; missing prices are not filled.')
    series = pd.Series(prices.to_numpy(), index=pd.DatetimeIndex(dates), name='price').sort_index()
    if len(series) < 3:
        raise ValueError('At least three prices are required.')
    return series


def returns_from_prices(prices):
    # No forward fill. Exclude suspicious gaps to avoid calling multi-session returns one-day.
    r = prices.pct_change(fill_method=None).dropna()
    return r.rename('return')


def calendar_check(prices, asset):
    gaps = prices.index.to_series().diff().dt.days
    limit = 1 if asset == 'BTC-USD' else 4
    bad = gaps > limit
    return bad, int(bad.sum())


def load_sample(asset):
    p = clean_prices(pd.read_csv(ROOT/'data'/f'{asset}_synthetic.csv'))
    return p, 'Synthetic demonstration • seeded regime-switching Student-t returns; not market history'


def download(asset, start, end):
    import yfinance as yf
    frame = yf.download(asset, start=str(start), end=str(pd.Timestamp(end)+pd.Timedelta(days=1))[:10],
                        auto_adjust=True, progress=False, threads=False, timeout=15)
    if frame.empty:
        raise ValueError('Provider returned no observations.')
    close = frame['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:,0]
    data = pd.DataFrame({'Date':close.index, 'Adjusted Close':close.to_numpy()})
    return clean_prices(data), 'Yahoo Finance via yfinance • auto-adjusted Close (provider adjustments)'
