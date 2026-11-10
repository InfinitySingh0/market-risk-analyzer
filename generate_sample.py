"""Generate explicitly synthetic offline datasets; never passed off as market history."""
from pathlib import Path
import numpy as np
import pandas as pd

root=Path(__file__).parent/'data'
root.mkdir(exist_ok=True)
for asset,frequency,scale,initial,seed,start in [('SPY','B',.009,200,101,'2018-01-01'),('BTC-USD','D',.025,8000,202,'2018-01-01')]:
    dates=pd.date_range(start,'2025-12-31',freq=frequency)
    rng=np.random.default_rng(seed)
    vol=np.full(len(dates),scale)
    vol[500:650]*=2.7
    vol[1100:1250]*=1.8
    r=np.clip(.00025+vol*rng.standard_t(5,len(dates))/np.sqrt(5/3),-.8,.8)
    prices=initial*np.cumprod(1+r)
    pd.DataFrame({'Date':dates,'Adjusted Close':prices}).to_csv(root/f'{asset}_synthetic.csv',index=False)
