"""Validate the saved experiment evidence and the portable prediction bundle."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import (
    RESULTS_DIR,
    CACHE_DIR,
    read_json,
    write_json,
    initialize_gpu_environment,
)

initialize_gpu_environment()
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from data import assert_disjoint, load_splits, load_audio
from features import validate_cache_manifest
from predict import Predictor


def main():
    evaluation = read_json(RESULTS_DIR / "evaluation.json")
    freeze = read_json(RESULTS_DIR / "evaluation_freeze.json")
    assert_disjoint(load_splits(), read_json(CACHE_DIR / "split_meta.json")["groups"])
    validate_cache_manifest()
    import hashlib

    for experiment, expected in freeze["models"].items():
        extension = ".joblib" if experiment.startswith(("svm", "knn")) else ".keras"
        model_path = RESULTS_DIR / "models" / (experiment + extension)
        assert hashlib.sha256(model_path.read_bytes()).hexdigest() == expected
    checks = []
    for row in evaluation["rows"]:
        for split in ("train", "val", "test"):
            frame = pd.read_csv(RESULTS_DIR / "predictions" / f"{row['experiment']}_{split}.csv")
            assert (
                abs(accuracy_score(frame.true_index, frame.predicted_index) - row[f"{split}_acc"])
                < 1e-12
            )
            assert (
                abs(
                    f1_score(
                        frame.true_index,
                        frame.predicted_index,
                        labels=range(10),
                        average="macro",
                        zero_division=0,
                    )
                    - row[f"{split}_macro_f1"]
                )
                < 1e-12
            )
        checks.append(row["experiment"] + ": saved predictions reproduce all metrics")
    model = Predictor()
    track = load_splits()["val"][0]
    audio = load_audio(track)[: 30 * 22050]
    first = model.predict_audio(audio)
    second = Predictor().predict_audio(audio)
    np.testing.assert_allclose(
        list(first["scores"].values()),
        list(second["scores"].values()),
        rtol=1e-6,
        atol=1e-7,
    )
    checks.append("Independent model loads produce matching probabilities")
    assert evaluation["selected_demo"] == freeze["selection"]
    write_json(RESULTS_DIR / "validation.json", dict(passed=True, checks=checks, example=first))
    print("\n".join(checks))


if __name__ == "__main__":
    main()
