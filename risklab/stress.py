import pandas as pd

SCENARIOS = [('GFC: Lehman week','2008-09-12','2008-09-19'),
             ('COVID selloff','2020-02-19','2020-03-23'),
             ('COVID single session','2020-03-13','2020-03-16')]


def historical_stress(prices, asset, position, synthetic=False):
    rows=[]
    for name,start,end in SCENARIOS:
        a,b=pd.Timestamp(start),pd.Timestamp(end)
        status='Available'
        if synthetic:
            status='Unavailable: synthetic data is not historical evidence'
        elif a not in prices.index or b not in prices.index:
            status='Unavailable: exact endpoints not in selected dataset'
        change=float(prices.loc[b]/prices.loc[a]-1) if status=='Available' else float('nan')
        sessions = int(((prices.index>a)&(prices.index<=b)).sum()) if status=='Available' else None
        rows.append({'scenario':name,'start':start,'end':end,'intervals':sessions,
                     'scope':'One-session' if sessions==1 else 'Cumulative multi-day',
                     'return':change,'loss_pct':-change,'loss_amount':-change*position,'status':status})
    return pd.DataFrame(rows)


def hypothetical_stress(shock_percent, position):
    if not -100 <= shock_percent <= 100:
        raise ValueError('Shock must be between -100% and +100%.')
    return pd.DataFrame([{'scenario':'User-defined instantaneous shock','return':shock_percent/100,
                          'loss_pct':-shock_percent/100,'loss_amount':-position*shock_percent/100,
                          'scope':'Instantaneous hypothetical','status':'Assumed; no assigned probability'}])
