"""Create matching raw/log-mel windows with train-only learned statistics."""
from __future__ import annotations
import argparse
import logging
from pathlib import Path
import librosa
import numpy as np
from config import (CACHE_DIR, GENRES, SEED, SPLITS, SR, WINDOW_SAMPLES, MEL_SHAPE,
                    FEATURE_SPEC, digest, read_json, write_json)
from data import load_audio, load_splits, inventory_source_signature, DATA_DIR
DEFAULT_CACHE_DIR = CACHE_DIR
LOGGER = logging.getLogger(__name__)

def standardize_raw(windows):
    windows = np.asarray(windows, dtype=np.float32)
    if windows.ndim != 2 or windows.shape[1] != WINDOW_SAMPLES:
        raise ValueError(f'Expected (N, {WINDOW_SAMPLES}) raw windows')
    means = windows.mean(axis=1, keepdims=True)
    std = windows.std(axis=1, keepdims=True)
    return np.asarray((windows-means)/np.maximum(std, np.float32(1e-8)), dtype=np.float32)

def log_mel_window(window):
    window = np.asarray(window, dtype=np.float32)
    if window.shape != (WINDOW_SAMPLES,):
        raise ValueError('Expected exactly two seconds of mono audio')
    power = librosa.feature.melspectrogram(y=window, sr=SR, n_fft=512, win_length=512,
        hop_length=256, n_mels=64, power=2.0, center=True, pad_mode='constant')
    mel = librosa.power_to_db(power, ref=1.0, amin=1e-10, top_db=80.0)
    if mel.shape != MEL_SHAPE or not np.isfinite(mel).all():
        raise ValueError(f'Invalid log-mel shape or values: {mel.shape}')
    return np.asarray(mel, dtype=np.float32)

def standardize_mel(mel, mean, std):
    if not np.isfinite([mean, std]).all() or std < 0:
        raise ValueError('Invalid training mel statistics')
    return np.asarray((np.asarray(mel, dtype=np.float32)-np.float32(mean))/max(float(std),1e-8), dtype=np.float32)

def load_mel_stats(cache_dir=CACHE_DIR):
    with np.load(Path(cache_dir)/'mel_stats.npz', allow_pickle=False) as stats:
        return float(stats['mean']), float(stats['std'])

def cache_signature(cache_dir):
    cache_dir = Path(cache_dir)
    return digest(dict(features=FEATURE_SPEC, splits=read_json(cache_dir/'splits.json'),
                       split_meta=read_json(cache_dir/'split_meta.json')))

def validate_cache_manifest(cache_dir=CACHE_DIR, verify_source=True):
    cache_dir = Path(cache_dir)
    meta = read_json(cache_dir/'feature_meta.json')
    if meta['signature'] != cache_signature(cache_dir):
        raise ValueError('Stale feature cache. Rebuild features with --force.')
    if verify_source:
        inventory = read_json(cache_dir/'inventory.json')
        if inventory['source_signature'] != inventory_source_signature(DATA_DIR):
            raise ValueError('Dataset files changed since audit. Rebuild data and features.')
    return meta

def load_feature_cache(cache_dir=CACHE_DIR, feature_type='raw', splits=SPLITS):
    if feature_type not in ('raw','mel'):
        raise ValueError('Expected raw or mel feature type')
    validate_cache_manifest(cache_dir)
    result = {}
    for split in splits:
        with np.load(Path(cache_dir)/f'{feature_type}_{split}.npz', allow_pickle=False) as cached:
            result[f'X_{split}'] = np.asarray(cached['X'], dtype=np.float32)
            result[f'y_{split}'] = np.asarray(cached['y'], dtype=np.int64)
    return result

def window_split(paths, split):
    rng = np.random.default_rng(SEED)
    count = 4 if split == 'train' else 1
    raw = np.empty((len(paths)*count, WINDOW_SAMPLES), dtype=np.float32)
    mel = np.empty((len(paths)*count,)+MEL_SHAPE, dtype=np.float32)
    labels = np.empty(len(paths)*count, dtype=np.int64)
    manifest = []
    used = 0
    for path in paths:
        audio = load_audio(path)
        if audio is None:
            raise ValueError(f'Audited recording became unreadable: {path}')
        starts = rng.integers(0, len(audio)-WINDOW_SAMPLES+1, size=count) if split == 'train' else [(len(audio)-WINDOW_SAMPLES)//2]
        for start in starts:
            start = int(start)
            window = audio[start:start+WINDOW_SAMPLES]
            raw[used], mel[used] = window, log_mel_window(window)
            labels[used] = GENRES.index(Path(path).parent.name)
            manifest.append(dict(track=path, start_sample=start, label=int(labels[used])))
            used += 1
        if used % 200 == 0:
            LOGGER.info('%s: generated %d windows', split, used)
    return standardize_raw(raw), mel, labels, manifest

def build_feature_cache(cache_dir=CACHE_DIR, force=False):
    cache_dir = Path(cache_dir)
    signature = cache_signature(cache_dir)
    expected = [cache_dir/f'{kind}_{s}.npz' for kind in ('raw','mel') for s in SPLITS]
    if not force and (cache_dir/'feature_meta.json').exists():
        validate_cache_manifest(cache_dir)
        if all(p.exists() for p in expected): return
        raise ValueError('Incomplete cache. Rebuild with --force.')
    splits = load_splits(cache_dir)
    windows = {}
    counts = {}
    for split in SPLITS:
        raw, mel, y, records = window_split(splits[split], split)
        if split == 'train':
            mean, std = float(mel.mean(dtype=np.float64)), float(mel.std(dtype=np.float64))
            np.savez(cache_dir/'mel_stats.npz', mean=mean, std=std)
        mel = standardize_mel(mel, mean, std)
        for kind, X in (('raw',raw),('mel',mel)):
            np.savez_compressed(cache_dir/f'{kind}_{split}.npz', X=X, y=y)
        windows[split], counts[split] = records, len(y)
        LOGGER.info('Saved %s: %d windows', split, len(y))
        del raw, mel, y
    write_json(cache_dir/'windows.json', windows)
    write_json(cache_dir/'feature_meta.json', dict(signature=signature, spec=FEATURE_SPEC, counts=counts,
                                                  mel_mean=mean, mel_std=std))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, default=CACHE_DIR)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    build_feature_cache(args.cache_dir, args.force)

if __name__ == '__main__': main()
