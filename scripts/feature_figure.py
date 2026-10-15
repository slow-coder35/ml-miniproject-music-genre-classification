"""Plot genuine training evidence for the report and presentation."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import CACHE_DIR, RESULTS_DIR, SR, WINDOW_SAMPLES, read_json
from data import load_audio
from features import log_mel_window
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa.display

record = read_json(CACHE_DIR / "windows.json")["train"][0]
audio = load_audio(record["track"])[
    record["start_sample"] : record["start_sample"] + WINDOW_SAMPLES
]
mel = log_mel_window(audio)
fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
axes[0].plot(np.arange(len(audio)) / SR, audio, color="#007F8B", linewidth=0.4)
axes[0].set(
    xlabel="Time (seconds)",
    ylabel="Amplitude",
    title="Raw waveform: " + Path(record["track"]).stem,
)
img = librosa.display.specshow(
    mel, sr=SR, hop_length=256, x_axis="time", y_axis="mel", cmap="magma", ax=axes[1]
)
axes[1].set_title("Log-mel power spectrogram")
fig.colorbar(img, ax=axes[1], label="dB")
fig.tight_layout()
(RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
fig.savefig(RESULTS_DIR / "figures/feature_example.png", dpi=160)
