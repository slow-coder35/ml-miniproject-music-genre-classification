"""Exercise Streamlit directly without opening or automating a browser."""
from pathlib import Path
import io
import numpy as np
import soundfile as sf
import streamlit as st
from streamlit.testing.v1 import AppTest
ROOT=Path(__file__).resolve().parents[1]

def test_empty_app_starts_without_errors():
    app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=30)
    assert not app.exception
    assert app.title[0].value=='Music Genre Classifier'
    assert len(app.info)==1

def test_upload_and_prediction_flow(monkeypatch):
    handle=io.BytesIO()
    sf.write(handle,.1*np.sin(2*np.pi*440*np.arange(66150)/22050),22050,format='WAV')
    class Upload:
        def getvalue(self):return handle.getvalue()
    class FakePredictor:
        def predict_audio(self,*args):
            return dict(genre='jazz',model='test_cnn',window_count=5,
                top_three=[{'genre':'jazz','score':.7},{'genre':'blues','score':.2},{'genre':'rock','score':.1}],
                scores={'jazz':.7,'blues':.2,'rock':.1})
    import predict
    monkeypatch.setattr(st,'file_uploader',lambda *a,**k:Upload())
    monkeypatch.setattr(predict,'Predictor',FakePredictor)
    app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=30)
    assert not app.exception
    assert app.button[0].label=='Predict genre'
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert app.subheader[0].value=='Jazz'
    st.cache_resource.clear()

def test_corrupt_upload_displays_error(monkeypatch):
    class Upload:
        def getvalue(self):return b'invalid audio'
    monkeypatch.setattr(st,'file_uploader',lambda *a,**k:Upload())
    app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=30)
    assert not app.exception
    assert 'valid WAV or FLAC' in app.error[0].value
