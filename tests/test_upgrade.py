import numpy as np
import pandas as pd
import pytest
from scipy.stats import t, chi2
from risklab.risk import ewma_volatility, estimate, fit_student_t, student_t_risk, empirical
from risklab.backtest import coverage_tests, traffic_light, es_diagnostic, rolling_forecasts


def test_ewma_hand_recursion():
    mu, vols, forecast = ewma_volatility([1., 2., 3.], .5, 'sample')
    assert mu == 2
    np.testing.assert_allclose(vols, np.sqrt([1., 1., .5]))
    assert forecast == pytest.approx(np.sqrt(.75))
    mu, vols, forecast = ewma_volatility([1., 2., 3.], .5, 'zero')
    assert mu == 0
    np.testing.assert_allclose(vols, np.sqrt([1., 1., 2.5]))
    assert forecast == pytest.approx(np.sqrt(5.75))


@pytest.mark.parametrize('decay,mode',[(0,'sample'),(1,'zero'),(.94,'invalid')])
def test_ewma_invalid_settings(decay,mode):
    with pytest.raises(ValueError):ewma_volatility([1.,2.],decay,mode)


def test_fhs_pre_update_filter_and_bootstrap():
    residuals = np.array([-1., 0., 1/np.sqrt(.5)])
    draws = np.random.default_rng(17).choice(residuals,10000,replace=True)
    expected = empirical(-(2+np.sqrt(.75)*draws),.95)
    actual = estimate([1.,2.,3.],.95,'Filtered historical simulation',10000,17,.5,'sample')
    assert actual == expected


@pytest.mark.parametrize('method',['EWMA normal','Filtered historical simulation','Student-t parametric'])
def test_new_models_repeatable(method):
    x=np.random.default_rng(2).standard_t(6,100)*.01
    assert estimate(x,.99,method,5000,5)==estimate(x,.99,method,5000,5)


def test_fhs_seed_changes_sample():
    x=np.random.default_rng(2).normal(0,.01,500)
    a=estimate(x,.99,'Filtered historical simulation',1000,5)
    b=estimate(x,.99,'Filtered historical simulation',1000,6)
    assert a != b


@pytest.mark.parametrize('confidence',[.95,.975,.99])
def test_student_t_closed_form(confidence):
    df,loc,scale=5.,.001,.02
    q=t.ppf(confidence,df)
    actual=student_t_risk(df,loc,scale,confidence)
    assert actual.var==pytest.approx(t.ppf(confidence,df,loc=-loc,scale=scale))
    assert actual.es==pytest.approx(-loc+scale*(df+q*q)/(df-1)*t.pdf(q,df)/(1-confidence))


def test_t_fit_recovery_and_degenerate():
    x=t.rvs(5,loc=.003,scale=.02,size=10000,random_state=21)
    df,loc,scale=fit_student_t(x)
    assert 4<df<6.5
    assert abs(loc-.003)<.001
    assert abs(scale-.02)<.002
    assert estimate([.02]*30,.99,'Student-t parametric').es==pytest.approx(-.02)
    with pytest.raises(ValueError):student_t_risk(1,0,.01)


def test_conditional_coverage_identity_and_unidentified():
    result=coverage_tests(([False]*18+[True,False])*20,.95)
    assert result['conditional_coverage_lr']==pytest.approx(result['kupiec_lr']+result['independence_lr'])
    assert result['conditional_coverage_p']==pytest.approx(chi2.sf(result['conditional_coverage_lr'],2))
    unavailable=coverage_tests([False]*250,.99)
    assert np.isnan(unavailable['conditional_coverage_p'])
    assert 'Not identifiable' in unavailable['conditional_coverage_note']


@pytest.mark.parametrize('count,zone',[(0,'green'),(4,'green'),(5,'yellow'),(9,'yellow'),(10,'red'),(250,'red')])
def test_traffic_boundaries(count,zone):
    result=traffic_light([True]*count+[False]*(250-count),.99)
    assert result['traffic_light']==zone
    assert result['traffic_light_breaches']==count


def test_traffic_eligibility_and_recent_window():
    assert traffic_light([False]*249,.99)['traffic_light']=='N/A'
    assert traffic_light([False]*250,.95)['traffic_light']=='N/A'
    assert traffic_light([True]*100+[False]*250,.99)['traffic_light']=='green'


def test_es_descriptive_fallback():
    result=es_diagnostic([False,True,True],[0,.04,.08],[.03,.05,.06])
    assert result['es_tail_mean_loss']==pytest.approx(.06)
    assert result['es_tail_mean_forecast']==pytest.approx(.055)
    assert result['es_tail_gap']==pytest.approx(.005)
    assert np.isnan(result['es_p_value'])
    assert 'Descriptive' in result['es_test']
    assert 'No VaR breaches' in es_diagnostic([False],[0],[.03])['es_note']
    assert 'not supplied' in es_diagnostic([False])['es_note']
    with pytest.raises(ValueError):es_diagnostic([True],[.1,.2],[.1])


@pytest.mark.parametrize('method',['EWMA normal','Filtered historical simulation'])
def test_zero_mean_no_lookahead(method):
    x=pd.Series(np.random.default_rng(4).normal(0,.02,45),index=pd.date_range('2024-01-01',periods=45))
    y=x.copy();y.iloc[35:]=-.3
    a=rolling_forecasts(x,30,.99,method,1000,3,.8,'zero')
    b=rolling_forecasts(y,30,.99,method,1000,3,.8,'zero')
    np.testing.assert_allclose(a.loc[:x.index[35],['var','es']],b.loc[:x.index[35],['var','es']])


@pytest.mark.parametrize('mean_mode',['sample','zero'])
def test_fhs_constant_window(mean_mode):
    for value in [0.,.1,-.1]:
        risk=estimate([value]*30,.99,'Filtered historical simulation',1000,42,.94,mean_mode)
        assert risk.var==pytest.approx(-value)
        assert risk.es==pytest.approx(-value)
