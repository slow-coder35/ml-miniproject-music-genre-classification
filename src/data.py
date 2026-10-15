"""Audit GTZAN and split parent recordings before sampling windows."""

from __future__ import annotations
import argparse
import csv
import hashlib
import logging
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
from sklearn.model_selection import StratifiedGroupKFold
from config import (
    ROOT,
    DATA_DIR,
    CACHE_DIR,
    GENRES,
    SEED,
    SR,
    WINDOW_SAMPLES,
    SPLITS,
    digest,
    read_json,
    write_json,
    resolve_track,
)

DEFAULT_DATA_DIR, DEFAULT_CACHE_DIR = DATA_DIR, CACHE_DIR
LOGGER = logging.getLogger(__name__)


def load_audio(path):
    try:
        audio, source_sr = sf.read(str(resolve_track(path)), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if source_sr != SR:
            audio = librosa.resample(audio, orig_sr=source_sr, target_sr=SR)
        if len(audio) < WINDOW_SAMPLES or not np.isfinite(audio).all():
            raise ValueError("audio is shorter than two seconds or contains nonfinite values")
        return np.asarray(audio, dtype=np.float32)
    except Exception as error:
        LOGGER.warning("Cannot use %s: %s", path, error)
        return None


def source_files(data_dir):
    return sorted(
        p
        for genre in GENRES
        for p in (Path(data_dir) / genre).glob("*")
        if p.suffix.lower() in (".wav", ".au", ".flac")
    )


def inventory_source_signature(data_dir):
    return digest(
        [
            (str(p.relative_to(data_dir)), p.stat().st_size, p.stat().st_mtime_ns)
            for p in source_files(data_dir)
        ]
    )


def audit_dataset(data_dir=DATA_DIR, cache_dir=CACHE_DIR):
    data_dir, cache_dir = Path(data_dir), Path(cache_dir)
    files = source_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No GTZAN audio in {data_dir}. Run scripts/download_dataset.py.")
    signature = inventory_source_signature(data_dir)
    manifest_path = cache_dir / "inventory.json"
    if manifest_path.exists():
        old = read_json(manifest_path)
        if old.get("source_signature") == signature:
            return old
    rows = []
    for i, path in enumerate(files):
        relative = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        row = dict(
            track_id=path.stem,
            path=relative,
            genre=path.parent.name,
            usable=False,
            reason="",
            source_sr=0,
            channels=0,
            duration=0,
            audio_hash="",
        )
        try:
            info = sf.info(path)
            audio = load_audio(path)
            if audio is None:
                raise ValueError("unreadable, nonfinite, or shorter than two seconds")
            row.update(
                usable=True,
                source_sr=info.samplerate,
                channels=info.channels,
                duration=float(info.duration),
                audio_hash=hashlib.sha256(audio.tobytes()).hexdigest(),
            )
        except Exception as error:
            row["reason"] = str(error)
        rows.append(row)
        if (i + 1) % 100 == 0:
            LOGGER.info("Audited %d/%d tracks", i + 1, len(files))
    # Identical audio with conflicting labels cannot support an unambiguous target.
    hash_genres = {}
    for row in rows:
        if row["usable"]:
            hash_genres.setdefault(row["audio_hash"], set()).add(row["genre"])
    for row in rows:
        if row["usable"] and len(hash_genres[row["audio_hash"]]) > 1:
            row.update(
                usable=False,
                reason="identical decoded audio has conflicting genre labels",
            )
    result = dict(source_signature=signature, tracks=rows)
    write_json(manifest_path, result)
    with (cache_dir / "inventory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


def assert_disjoint(splits, groups=None):
    tracks = {s: {str(resolve_track(p).resolve()) for p in splits[s]} for s in SPLITS}
    for s in SPLITS:
        if len(tracks[s]) != len(splits[s]):
            raise ValueError(f"duplicate recording in {s}")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if tracks[a] & tracks[b]:
            raise ValueError(f"{a}/{b} recording overlap")
        if groups and {groups[p] for p in splits[a]} & {groups[p] for p in splits[b]}:
            raise ValueError(f"{a}/{b} identical-audio group overlap")


def load_splits(cache_dir=CACHE_DIR):
    result = read_json(Path(cache_dir) / "splits.json")
    assert_disjoint(result)
    return result


def create_splits(data_dir=DATA_DIR, cache_dir=CACHE_DIR, force=False):
    cache_dir = Path(cache_dir)
    inventory = audit_dataset(data_dir, cache_dir)
    rows = [r for r in inventory["tracks"] if r["usable"]]
    paths = [r["path"] for r in rows]
    labels = np.array([GENRES.index(r["genre"]) for r in rows])
    groups = np.array([r["audio_hash"] for r in rows])
    group_map = dict(zip(paths, groups.tolist()))
    signature = digest(
        dict(
            source=inventory["source_signature"],
            seed=SEED,
            algorithm="SGKF10-fold0-test-fold1-val",
        )
    )
    meta_path = cache_dir / "split_meta.json"
    if (cache_dir / "splits.json").exists() and not force:
        if not meta_path.exists() or read_json(meta_path)["signature"] != signature:
            raise ValueError("Dataset or split settings changed. Rebuild with --force.")
        result = load_splits(cache_dir)
        assert_disjoint(result, group_map)
        return result
    counts = {genre: sum(r["genre"] == genre for r in rows) for genre in GENRES}
    if min(counts.values()) < 10:
        raise ValueError(f"At least ten usable tracks per class are required: {counts}")
    folds = list(
        StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED).split(
            np.zeros(len(rows)), labels, groups
        )
    )
    test, val = set(folds[0][1]), set(folds[1][1])
    result = {s: [] for s in SPLITS}
    for i, path in enumerate(paths):
        result["test" if i in test else "val" if i in val else "train"].append(path)
    for s in SPLITS:
        result[s].sort()
        if {GENRES.index(Path(p).parent.name) for p in result[s]} != set(range(10)):
            raise ValueError(f"Class coverage missing in {s}")
    assert_disjoint(result, group_map)
    write_json(cache_dir / "splits.json", result)
    write_json(
        meta_path,
        dict(
            signature=signature,
            seed=SEED,
            counts=counts,
            groups=group_map,
            split_counts={s: len(result[s]) for s in SPLITS},
        ),
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    splits = create_splits(args.data_dir, args.cache_dir, args.force)
    print({s: len(splits[s]) for s in SPLITS})


if __name__ == "__main__":
    main()
