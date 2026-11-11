from pathlib import Path
from streamlit.testing.v1 import AppTest
from risklab.risk import METHODS
APP=Path(__file__).resolve().parents[1]/'app.py'

def test_default_app_and_bitcoin():
    app=AppTest.from_file(str(APP),default_timeout=180).run()
    assert not app.exception
    assert len(app.metric)==4
    assert any('DEMONSTRATION DATA' in w.value for w in app.warning)
    app.selectbox[0].select('BTC-USD').run()
    assert not app.exception
    app.selectbox[2].select('Historical simulation').run()
    assert not app.exception
    assert len(app.metric)==4

    for method in METHODS:
        app.selectbox[2].select(method).run()
        assert not app.exception
        assert len(app.metric)==4
    app.radio[1].set_value('zero').run()
    assert not app.exception
