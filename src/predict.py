"""Shared saved-model inference used by both the CLI and the upload application."""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
from config import (ROOT, RESULTS_DIR, SR, WINDOW_SAMPLES, GENRES, FEATURE_SPEC,
                    read_json, initialize_gpu_environment)
from features import log_mel_window, standardize_mel

def decode_upload(source, max_seconds=30):
    if isinstance(source, bytes): source=io.BytesIO(source)
    try:
        with sf.SoundFile(source) as handle:
            source_duration=len(handle)/handle.samplerate
            source_sr=handle.samplerate
            audio=handle.read(frames=int(max_seconds*source_sr),dtype='float32',always_2d=True).mean(axis=1)
    except Exception as error:
        raise ValueError('Unable to read this audio. Upload a valid WAV or FLAC recording.') from error
    if not np.isfinite(audio).all(): raise ValueError('Audio contains invalid samples.')
    if source_sr!=SR: audio=librosa.resample(audio,orig_sr=source_sr,target_sr=SR)
    audio=np.asarray(audio,dtype=np.float32)
    if len(audio)<WINDOW_SAMPLES: raise ValueError('Please provide at least two seconds of audio.')
    if np.sqrt(np.mean(audio.astype(np.float64)**2))<1e-5:
        raise ValueError('This recording is silent or too quiet to classify.')
    return audio,dict(source_duration_seconds=source_duration,analyzed_seconds=len(audio)/SR,sample_rate=SR)

def prediction_windows(audio, count=5):
    if len(audio)<WINDOW_SAMPLES: raise ValueError('Audio must contain at least two seconds.')
    starts=np.unique(np.linspace(0,len(audio)-WINDOW_SAMPLES,count).round().astype(int))
    return np.stack([audio[s:s+WINDOW_SAMPLES] for s in starts]),starts

class Predictor:
    def __init__(self,selection_path=RESULTS_DIR/'demo_model.json'):
        from tensorflow import keras
        from gpu_check import configure_gpu
        self.gpus=configure_gpu(required=True)
        self.selection=read_json(selection_path)
        if self.selection['feature_spec']!=FEATURE_SPEC or self.selection['genres']!=list(GENRES):
            raise ValueError('Saved preprocessing specification or labels do not match this code.')
        self.model=keras.models.load_model(ROOT/self.selection['model_path'],compile=False)
    def predict_audio(self,audio,details=None):
        raw,starts=prediction_windows(audio)
        mel=np.stack([log_mel_window(window) for window in raw])
        X=standardize_mel(mel,self.selection['mel_mean'],self.selection['mel_std'])[...,None]
        scores=self.model(X,training=False).numpy().mean(axis=0)
        order=np.argsort(-scores,kind='stable')
        return dict(genre=GENRES[int(order[0])],model_score=float(scores[order[0]]),
            top_three=[dict(genre=GENRES[int(i)],score=float(scores[i])) for i in order[:3]],
            scores={genre:float(score) for genre,score in zip(GENRES,scores)},
            window_count=len(raw),window_starts_seconds=(starts/SR).tolist(),
            model=self.selection['experiment'],**(details or {}))
    def predict_file(self,source):
        audio,details=decode_upload(source)
        return self.predict_audio(audio,details)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audio',type=Path)
    parser.add_argument('--selection',type=Path,default=RESULTS_DIR/'demo_model.json')
    args=parser.parse_args()
    initialize_gpu_environment()
    print(json.dumps(Predictor(args.selection).predict_file(args.audio),indent=2))
if __name__=='__main__': main()
