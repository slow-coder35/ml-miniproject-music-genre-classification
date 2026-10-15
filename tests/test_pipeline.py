import io
import json
import numpy as np
import pytest
import soundfile as sf
from config import SR,WINDOW_SAMPLES,FEATURE_SPEC,digest,write_json
from data import assert_disjoint
from features import standardize_raw,standardize_mel,log_mel_window,validate_cache_manifest,cache_signature
from predict import decode_upload,prediction_windows

def wav_bytes(seconds=3,sr=SR,stereo=False,silent=False):
    samples=np.zeros(int(seconds*sr)) if silent else .1*np.sin(2*np.pi*440*np.arange(int(seconds*sr))/sr)
    if stereo:samples=np.column_stack([samples,samples])
    handle=io.BytesIO();sf.write(handle,samples,sr,format='WAV');return handle.getvalue()

def test_rejects_parent_track_overlap_and_exact_duplicate_overlap():
    with pytest.raises(ValueError,match='overlap'):
        assert_disjoint({'train':['same'],'val':['same'],'test':['other']})
    splits={'train':['a'],'val':['b'],'test':['c']}
    with pytest.raises(ValueError,match='group overlap'):
        assert_disjoint(splits,{'a':'same','b':'same','c':'different'})
    assert_disjoint(splits,{'a':'one','b':'two','c':'three'})

def test_features_shape_finiteness_and_normalization():
    audio=np.sin(2*np.pi*440*np.arange(WINDOW_SAMPLES)/SR).astype(np.float32)
    mel=log_mel_window(audio)
    assert mel.shape==(64,173) and mel.dtype==np.float32 and np.isfinite(mel).all()
    raw=standardize_raw(np.stack([audio,np.zeros_like(audio)]))
    assert abs(raw[0].mean())<1e-5 and abs(raw[0].std()-1)<1e-5
    assert np.all(raw[1]==0)
    np.testing.assert_allclose(standardize_mel(mel,10,2),(mel-10)/2)
    with pytest.raises(ValueError): standardize_mel(mel,0,-1)

def test_decode_stereo_resampling_crop_and_error_cases():
    audio,details=decode_upload(wav_bytes(3,sr=44100,stereo=True))
    assert len(audio)==3*SR and details['sample_rate']==SR
    long,details=decode_upload(wav_bytes(31))
    assert len(long)==30*SR and details['source_duration_seconds']==31
    with pytest.raises(ValueError,match='two seconds'):decode_upload(wav_bytes(1))
    with pytest.raises(ValueError,match='silent'):decode_upload(wav_bytes(silent=True))
    with pytest.raises(ValueError,match='valid WAV'):decode_upload(b'broken audio')

def test_windows_are_deterministic_and_unique_for_short_clips():
    audio=np.arange(3*SR,dtype=np.float32)
    windows,starts=prediction_windows(audio)
    assert len(starts)==5 and starts[0]==0 and starts[-1]==SR
    np.testing.assert_array_equal(windows[2],audio[starts[2]:starts[2]+WINDOW_SAMPLES])
    a,b=prediction_windows(audio);np.testing.assert_array_equal(a,windows)
    assert len(prediction_windows(audio[:WINDOW_SAMPLES])[0])==1

def test_stale_cache_is_rejected(tmp_path):
    write_json(tmp_path/'splits.json',{'train':['a'],'val':['b'],'test':['c']})
    write_json(tmp_path/'split_meta.json',{'signature':'fixed'})
    write_json(tmp_path/'feature_meta.json',{'signature':cache_signature(tmp_path)})
    validate_cache_manifest(tmp_path,verify_source=False)
    write_json(tmp_path/'splits.json',{'train':['d'],'val':['b'],'test':['c']})
    with pytest.raises(ValueError,match='Stale'):validate_cache_manifest(tmp_path,verify_source=False)

def test_training_never_requests_test_partition(monkeypatch,tmp_path):
    import train
    observed=[]
    monkeypatch.setattr(train,'artifact_path',lambda e:tmp_path/'missing.keras')
    monkeypatch.setattr(train,'validate_cache_manifest',lambda:{'signature':'test'})
    class StopBeforeFit(Exception): pass
    def inspect_load(**kwargs):
        observed.append(kwargs['splits'])
        raise StopBeforeFit()
    monkeypatch.setattr(train,'load_feature_cache',inspect_load)
    with pytest.raises(StopBeforeFit):train.fit_experiment('cnn_mel')
    assert observed==[('train','val')]
