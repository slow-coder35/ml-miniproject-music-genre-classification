"""One shared specification for preparation, training, and inference."""

from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "genres_original"
CACHE_DIR = ROOT / "data" / "cache"
RESULTS_DIR = ROOT / "results"
GENRES = (
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
)
SEED = 42
SR = 22050
WINDOW_SAMPLES = 44100
SPLITS = ("train", "val", "test")
MEL_SHAPE = (64, 173)
FEATURE_SPEC = dict(
    sample_rate=SR,
    mono=True,
    window_samples=WINDOW_SAMPLES,
    train_windows=4,
    evaluation_window="center",
    n_fft=512,
    hop_length=256,
    n_mels=64,
    power=2.0,
    center=True,
    pad_mode="constant",
    db_ref=1.0,
    db_amin=1e-10,
    db_top=80.0,
    raw_normalization="per_window_zscore",
    mel_normalization="training_global_zscore",
    version=2,
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def resolve_track(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def local_gpu_library_dirs():
    """Reuse laptop CUDA libraries when present, alongside pip NVIDIA packages."""
    import sys

    site = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    dirs = sorted(str(p) for p in (site / "nvidia").glob("*/lib"))
    ollama = Path("/usr/local/lib/ollama/cuda_v12")
    if ollama.is_dir():
        dirs.append(str(ollama))
    return dirs


def initialize_gpu_environment():
    """Re-exec before importing TensorFlow so the dynamic linker sees CUDA."""
    import sys

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "work" / "mpl"))
    if os.environ.get("MUSIC_GPU_ENV_READY") != "1":
        dirs = local_gpu_library_dirs()
        old = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(dirs + ([old] if old else []))
        os.environ["MUSIC_GPU_ENV_READY"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)
