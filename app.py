from datetime import date
import io
import json
import zipfile
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from risklab.data import load_sample, download, clean_prices, returns_from_prices, calendar_check
from risklab.risk import METHODS, estimate
from risklab.backtest import rolling_forecasts, coverage_tests
from risklab.stress import historical_stress, hypothetical_stress

st.set_page_config(page_title='Market Risk Analyzer', page_icon='◈', layout='wide')
st.markdown('''<style>
.stApp {background:#0b1220} [data-testid="stMetric"] {background:#152035;padding:18px;border-radius:10px;border:1px solid #29374c}
h1 {letter-spacing:-1.4px} .eyebrow {color:#50d7b0;letter-spacing:3px;font-size:12px;font-weight:700}
</style>''',unsafe_allow_html=True)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch(asset,start,end):
    return download(asset,start,end)

@st.cache_data(show_spinner=False)
def analyze(returns,window,confidence,simulations,seed,ewma_lambda=.94,mean_mode="sample"):
    latest=[]; forecasts={}; tests={}
    for method in METHODS:
        risk=estimate(returns.iloc[-window:],confidence,method,simulations,seed,ewma_lambda,mean_mode)
        forecasts[method]=rolling_forecasts(returns,window,confidence,method,simulations,seed,ewma_lambda,mean_mode)
        tests[method]=coverage_tests(forecasts[method].breach,confidence,forecasts[method].loss,forecasts[method].es)
        latest.append({'Method':method,'VaR':risk.var,'ES':risk.es})
    return pd.DataFrame(latest).set_index('Method'),forecasts,tests

with st.sidebar:
    st.markdown('### ◈ RISK / LAB')
    st.caption('Single asset · Long position · One-day risk')
    asset=st.selectbox('Asset',['SPY','BTC-USD'])
    source=st.radio('Data source',['Offline demonstration','Yahoo Finance download','Upload CSV'])
    start=st.date_input('Start date',date(2022,1,1))
    end=st.date_input('End date',date(2025,12,31))
    position=st.number_input('Position value (USD)',min_value=100.,value=100000.,step=1000.)
    confidence=st.selectbox('Confidence level',[.95,.975,.99],index=2,format_func=lambda x:f'{x:.1%}')
    window=st.number_input('Estimation window (returns)',min_value=30,max_value=2000,value=250,step=10)
    method=st.selectbox('Displayed model',METHODS,format_func=lambda m: m+' — numerical normal check' if m=='Monte Carlo (normal)' else m)
    with st.expander('EWMA / FHS settings'):
        ewma_lambda=st.slider('EWMA lambda',.50,.999,.94,.001)
        mean_mode=st.radio('EWMA / FHS mean',['sample','zero'],horizontal=True)
    with st.expander('Simulation settings'):
        simulations=st.select_slider('Simulation count',[1000,5000,20000,50000,100000],value=20000)
        seed=st.number_input('Random seed',min_value=0,max_value=2**32-1,value=42)
    uploaded=st.file_uploader('CSV: Date + Adjusted Close',type='csv') if source=='Upload CSV' else None
    price_column=st.selectbox('CSV price column',['Adjusted Close','Close']) if source=='Upload CSV' else 'Adjusted Close'
    st.caption('SPY: observed market sessions. BTC: calendar days. No forward filling. USD is the reporting currency.')

st.markdown('<div class="eyebrow">MARKET INTELLIGENCE / RISK ANALYTICS</div>',unsafe_allow_html=True)
st.title('Market Risk Analyzer')
st.write('Measure the downside. Examine the tail. Test the forecast.')
if start>=end:
    st.error('Start date must precede end date.'); st.stop()
synthetic=source=='Offline demonstration'
try:
    if source=='Upload CSV':
        if uploaded is None:
            st.info('Upload a CSV to begin. Choose the offline demonstration to explore immediately.'); st.stop()
        prices=clean_prices(pd.read_csv(uploaded),price_column)
        provenance=f'User upload: {uploaded.name} • {price_column}; adjustment quality is user-supplied'
        synthetic=st.checkbox('This upload contains synthetic/demo prices',value=False)
        st.info('CSV calendar: supply complete daily observations. Short gaps cannot always be distinguished from exchange holidays.')
    elif synthetic:
        prices,provenance=load_sample(asset)
    else:
        try:
            with st.spinner('Retrieving adjusted daily prices…'):
                prices,provenance=fetch(asset,start,end)
        except Exception as exc:
            st.warning(f'Download unavailable ({type(exc).__name__}). Showing explicitly synthetic offline data instead.')
            prices,provenance=load_sample(asset); synthetic=True
    prices=prices.loc[str(start):str(end)]
    if len(prices)<3:
        st.error('No usable prices in this range. Offline data spans 2018–2025.'); st.stop()
    gaps,gap_count=calendar_check(prices,asset)
    returns=returns_from_prices(prices)
    if gap_count:
        returns=returns.loc[~gaps.reindex(returns.index,fill_value=False)]
        st.warning(f'{gap_count} return intervals excluded for gaps exceeding '+('1 calendar day.' if asset=='BTC-USD' else '4 calendar days. Shorter missing-session gaps may remain; verify your calendar.'))
    if len(returns)<=window:
        st.error(f'{len(returns)} returns available. Select a longer date range or a window smaller than this count.'); st.stop()
except (ValueError,KeyError,pd.errors.ParserError) as exc:
    st.error(str(exc)); st.stop()

st.caption(f'{provenance} | Observed {prices.index.min():%d %b %Y} – {prices.index.max():%d %b %Y} | {len(returns):,} usable returns')
if synthetic:
    st.warning('DEMONSTRATION DATA — generated prices, not SPY or Bitcoin market history. SPY demo uses weekdays, including exchange holidays.')
if window*(1-confidence)<10:
    st.info(f'Only {window*(1-confidence):.1f} tail observations expected in each estimation window. Historical tail estimates are coarse; a longer window adds observations but may mix regimes.')
with st.spinner('Calculating reproducible forecasts and backtests…'):
    try:
        summary,forecasts,tests=analyze(returns,int(window),confidence,int(simulations),int(seed),ewma_lambda,mean_mode)
    except ValueError as exc:
        st.error(f'Risk estimation failed: {exc}'); st.stop()
selected=summary.loc[method]; bt=forecasts[method]; test=tests[method]
a,b,c,d=st.columns(4)
a.metric(f'{confidence:.1%} one-day VaR',f'${selected.VaR*position:,.0f}',f'{selected.VaR:.2%} of position',delta_color='off')
b.metric('Expected Shortfall',f'${selected.ES*position:,.0f}',f'{selected.ES:.2%} of position',delta_color='off')
c.metric('Worst observed daily loss',f'${(-returns.min())*position:,.0f}',f'{-returns.min():.2%}',delta_color='off')
d.metric('VaR breach rate',f'{test["breach_rate"]:.2%}',f'{test["breaches"]} / {test["observations"]} forecasts',delta_color='off')
st.caption('VaR and ES cards forecast the next observation after the dataset ends. Backtests below evaluate earlier forecasts. Amounts use a constant position value, not a compounded trading strategy.')

risk_tab,back_tab,stress_tab,method_tab=st.tabs(['Risk overview','Backtesting','Stress testing','Methodology & exports'])
with risk_tab:
    left,right=st.columns([1.5,1])
    with left:
        hist=px.histogram(x=-returns.iloc[-window:]*100,nbins=60,labels={'x':'Daily loss (%)','y':'Observations'},title='Loss distribution · current estimation window',color_discrete_sequence=['#50d7b0'])
        hist.add_vline(x=selected.VaR*100,line_color='#f7bf60',annotation_text='VaR')
        hist.add_vline(x=selected.ES*100,line_color='#ff7389',annotation_text='ES')
        st.plotly_chart(hist,width='stretch')
    with right:
        st.subheader('Model comparison')
        table=summary.copy()
        table['VaR (USD)']=table.VaR*position;table['ES (USD)']=table.ES*position
        st.dataframe(table.style.format({'VaR':'{:.2%}','ES':'{:.2%}','VaR (USD)':'${:,.0f}','ES (USD)':'${:,.0f}'}),width='stretch')
        st.caption('Monte Carlo uses the same fitted normal distribution as the analytical model. Differences reflect sampling and empirical tail estimation, not an independent economic model.')
    fig=px.line(pd.DataFrame({'Daily volatility':returns.rolling(window).std()*100}),title='Trailing daily volatility (%)')
    st.plotly_chart(fig,width='stretch')
    chart=pd.DataFrame({m:f['var']*100 for m,f in forecasts.items()})
    st.plotly_chart(px.line(chart,title='Out-of-sample one-day VaR forecasts (%)'),width='stretch')
with back_tab:
    st.subheader('Forecasts versus realized losses')
    fig=go.Figure()
    for name,color,values in [('Realized loss','#71839e',bt.loss),('VaR','#f7bf60',bt['var']),('Expected Shortfall','#50d7b0',bt.es)]:
        fig.add_trace(go.Scatter(x=bt.index,y=values*position,name=name,line={'color':color,'width':1}))
    breaches=bt[bt.breach]
    fig.add_trace(go.Scatter(x=breaches.index,y=breaches.loss*position,name='VaR breach',mode='markers',marker={'color':'#ff7389','size':7}))
    fig.update_layout(yaxis_title='Loss (USD)',legend={'orientation':'h'})
    st.plotly_chart(fig,width='stretch')
    rows=[]
    for m,t in tests.items():
        rows.append({'Model':m,'Forecasts':t['observations'],'Breaches':t['breaches'],'Expected':t['expected_breaches'],'Breach rate':t['breach_rate'],'Kupiec p':t['kupiec_p'],'Independence p':t['independence_p'],'Conditional coverage p':t['conditional_coverage_p'],'Traffic light':t['traffic_light'],'ES tail gap':t['es_tail_gap'],'ES assessment':t['es_test']})
    st.dataframe(pd.DataFrame(rows).set_index('Model'),width='stretch')
    st.caption(f'Wilson 95% interval for selected breach rate: {test["breach_ci_low"]:.2%}–{test["breach_ci_high"]:.2%}. Independence: {test["independence_note"]}')
    st.info(test['warning'])
    st.caption(test['conditional_coverage_note']+' '+test['traffic_light_note']+' '+test['es_note'])
    st.write('At a 5% test level, p < 0.05 is evidence against the respective null. A larger p-value does not prove adequacy. Tests evaluate VaR; they do not validate Expected Shortfall.')
    if len(breaches):
        st.write(f'Descriptive tail diagnostic: mean realized loss on breach dates **{breaches.loss.mean():.2%}**; mean ES forecast on those same dates **{breaches.es.mean():.2%}**. Selection on breaches and varying forecasts prevent treating this as a formal ES test.')
    else:
        st.write('No breaches: a realized tail-loss diagnostic is unavailable.')
    st.caption('Training windows advance through time and use only earlier returns. Choosing settings after inspecting this evaluation period can introduce model-selection bias; reserve a separate final holdout for research conclusions.')
with stress_tab:
    st.subheader('Scenario losses')
    shock=st.slider('Instantaneous price shock (%)',-100,100,-20)
    hypothetical=hypothetical_stress(shock,position)
    st.dataframe(hypothetical,width='stretch')
    historical=historical_stress(prices,asset,position,synthetic)
    st.write('Historical endpoint scenarios')
    st.dataframe(historical,width='stretch')
    st.caption('Historical return = ending adjusted price / starting adjusted price − 1. Exact endpoints must exist in the selected dataset. Expand the date range to include an event. Multi-day cumulative stress losses are not one-day VaR. For BTC, Friday to Monday spans three daily intervals. No scenario probabilities are assigned.')
with method_tab:
    st.subheader('Calculation conventions')
    st.markdown('''Simple return: rₜ = Pₜ/Pₜ₋₁ − 1. Loss: Lₜ = −rₜ. Negative losses represent gains; risk estimates are not clamped to zero.

For confidence c, z = Φ⁻¹(c), and sample standard deviation σ (ddof=1):
- Normal VaR = −μ + σz.
- Normal ES = −μ + σφ(z)/(1−c).
- Historical VaR uses the inverse empirical CDF (order statistic, no linear interpolation).
- Historical ES integrates the empirical quantile over the worst 1−c fraction. It includes fractional weight on the boundary observation, so tied observations and small samples do not enlarge the intended tail.
- EWMA normal uses recursive variance with configurable lambda (default 0.94) and sample or zero mean. The window sample variance seeds the recursion.
- FHS divides each training innovation by its pre-update EWMA volatility, bootstraps residuals with a fixed seed, and rescales by the next-day volatility. Initialization and sample mean use the training window only; filtered residuals are in-sample.
- Student-t fits location, scale and df by finite-mean constrained MLE (1.01 ≤ df ≤ 1000). Scale differs from standard deviation. Failed fits raise an explicit error. Constant windows use a point-mass limit.
- Monte Carlo draws normal simple-return losses using the same μ and σ. The normal model permits economically impossible losses above 100%; this is a model limitation, especially for volatile assets.

Returns are calculated only between supplied observations; missing prices are rejected and never filled. Suspicious long-gap returns are excluded. SPY short gaps cannot always be distinguished from holidays; CSV users must verify completeness. Volatility is daily, not annualized. Assets are evaluated on their own calendars and are not merged.

Risk reflects the chosen historical window. Changing regimes, tail dependence, liquidity, fees, slippage, taxes, and model uncertainty can make future losses differ. SPY represents ETF exposure and may differ from the S&P 500 index. These models cover market price risk, not a complete risk framework.''')
    config={'asset':asset,'source':provenance,'synthetic':synthetic,'requested_start':str(start),'requested_end':str(end),'observed_start':str(prices.index.min().date()),'observed_end':str(prices.index.max().date()),'position_usd':position,'confidence':confidence,'window':window,'simulations':simulations,'seed':seed,'ewma_lambda':ewma_lambda,'mean_mode':mean_mode,'return_convention':'simple','loss_convention':'negative return','excluded_gap_returns':gap_count,'displayed_model':method}
    combined=pd.concat(forecasts,names=['model','date']).reset_index()
    for col in ['loss','var','es']:
        combined[col+'_usd']=combined[col]*position
    stresses=pd.concat([historical,hypothetical],ignore_index=True)
    config_csv=pd.DataFrame([config]).to_csv(index=False)
    files={'forecasts.csv':combined.to_csv(index=False),'breaches.csv':combined[combined.breach].to_csv(index=False),'stress_results.csv':stresses.to_csv(index=False),'configuration.csv':config_csv,'model_comparison.csv':summary.to_csv(),'backtest_diagnostics.csv':pd.DataFrame(tests).T.rename_axis('model').to_csv()}
    bundle=io.BytesIO()
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,content in files.items():archive.writestr(name,content)
    st.download_button('Download analysis bundle (CSV ZIP)',bundle.getvalue(),'risk-analysis.zip','application/zip')
    for name,content in files.items():
        st.download_button(f'Download {name}',content,name,'text/csv',key=name)
    with st.expander('Current configuration'):
        st.json(config)
    st.divider()
    st.subheader('Project deliverables')
    from pathlib import Path
    project_root=Path(__file__).resolve().parent
    source_bundle=io.BytesIO()
    with zipfile.ZipFile(source_bundle,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(project_root.rglob('*')):
            relative=path.relative_to(project_root)
            if path.is_file() and not any(part in {'.venv','__pycache__','.pytest_cache','.git'} for part in relative.parts):
                archive.write(path,str(Path('market-risk-analyzer')/relative))
    st.download_button('Download complete Python project',source_bundle.getvalue(),'market-risk-analyzer.zip','application/zip')
    report_path=project_root/'reports'/'sample_risk_report.md'
    if report_path.exists():
        st.download_button('Download sample risk report',report_path.read_text(),'sample_risk_report.md','text/markdown')
