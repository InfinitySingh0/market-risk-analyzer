import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from risklab.risk import estimate, empirical, normal_risk, METHODS
from risklab.backtest import rolling_forecasts, coverage_tests
from risklab.data import clean_prices, returns_from_prices, calendar_check
from risklab.stress import historical_stress, hypothetical_stress


def test_normal_closed_form():
    risk=normal_risk([-.02,0,.02],.99)
    assert risk.var==pytest.approx(.02*norm.ppf(.99))
    assert risk.es==pytest.approx(.02*norm.pdf(norm.ppf(.99))/.01)


def test_fractional_tail_and_ties():
    r=empirical([0,1,2,3,4],.7)
    assert r.var==3
    assert r.es==pytest.approx((4+.5*3)/1.5)
    assert empirical([1,2,2,2],.75).es==2

@pytest.mark.parametrize('method',METHODS)
def test_monotonicity_signs_and_translation(method):
    x=np.random.default_rng(1).normal(0,.02,1000)
    risks=[estimate(x,c,method,50000) for c in [.95,.975,.99]]
    assert all(r.es>=r.var for r in risks)
    assert risks[0].var<=risks[1].var<=risks[2].var
    assert risks[0].es<=risks[1].es<=risks[2].es
    assert estimate(x+.01,.99,method,50000).var==pytest.approx(risks[-1].var-.01)
    assert estimate(np.full(100,-.1),.99,method).var==pytest.approx(.1)
    assert estimate(np.full(100,.1),.99,method).var==pytest.approx(-.1)


def test_monte_carlo_agreement_and_seed():
    x=np.random.default_rng(10).normal(.001,.02,1000)
    a=normal_risk(x,.99); b=estimate(x,.99,METHODS[2],1000000,9)
    assert abs(a.var-b.var)<.0003
    assert abs(a.es-b.es)<.0003
    assert b==estimate(x,.99,METHODS[2],1000000,9)

@pytest.mark.parametrize('method',METHODS)
def test_no_lookahead(method):
    x=pd.Series(np.random.default_rng(3).normal(0,.01,90),index=pd.date_range('2024-01-01',periods=90))
    a=rolling_forecasts(x,30,.95,method,1000)
    changed=x.copy();changed.iloc[60:]=-.5
    b=rolling_forecasts(changed,30,.95,method,1000)
    np.testing.assert_allclose(a.loc[:x.index[60],['var','es']],b.loc[:x.index[60],['var','es']])
    assert (pd.to_datetime(a.training_end)<a.index).all()
    assert a.loc[x.index[60],'loss'] != b.loc[x.index[60],'loss']


def test_backtest_edge_cases_and_clustering():
    for values in [[False]*100,[True]*100]:
        t=coverage_tests(values,.99)
        assert np.isfinite(t['kupiec_p'])
        assert np.isnan(t['independence_p'])
    clustered=coverage_tests(([False]*90+[True]*10)*10,.9)
    assert clustered['kupiec_p']==pytest.approx(1)
    assert clustered['independence_p']<.001
    assert sum(clustered[k] for k in ['n00','n01','n10','n11'])==999


def test_data_rejection_and_returns():
    frame=pd.DataFrame({'Date':['2024-01-01','2024-01-02','2024-01-03'],'Adjusted Close':[100,90,99]})
    p=clean_prices(frame)
    np.testing.assert_allclose(returns_from_prices(p),[-.1,.1])
    frame.loc[1,'Adjusted Close']=np.nan
    with pytest.raises(ValueError):clean_prices(frame)
    frame.loc[1,'Adjusted Close']=90;frame.loc[1,'Date']='2024-01-01'
    with pytest.raises(ValueError):clean_prices(frame)


def test_calendar_and_stress_scope():
    p=pd.Series([100.,80.,90.],index=pd.to_datetime(['2020-03-13','2020-03-16','2020-03-17']))
    assert calendar_check(p,'BTC-USD')[1]==1
    assert calendar_check(p,'SPY')[1]==0
    h=historical_stress(p,'SPY',1000)
    assert h.iloc[2].loss_amount==pytest.approx(200)
    assert h.iloc[2].scope=='One-session'
    assert historical_stress(p,'SPY',1000,True).loss_amount.isna().all()
    assert hypothetical_stress(-20,1000).iloc[0].loss_amount==200

@pytest.mark.parametrize('x,c',[([1],.99),([1,np.nan],.99),([1,2],1),([1,2],0)])
def test_invalid_inputs(x,c):
    with pytest.raises(ValueError):estimate(x,c)
