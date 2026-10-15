"""Fit eight main experiments and two bounded CNN trials, without loading test data."""

from __future__ import annotations
import argparse
import gc
import platform
import sys
from pathlib import Path
from time import perf_counter
from config import (
    ROOT,
    RESULTS_DIR,
    GENRES,
    SEED,
    FEATURE_SPEC,
    initialize_gpu_environment,
    read_json,
    write_json,
)

initialize_gpu_environment()
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from features import load_feature_cache, validate_cache_manifest, load_mel_stats

EXPERIMENTS = (
    "svm_mel",
    "knn_mel",
    "cnn_mel",
    "ffnn_mel",
    "svm_raw",
    "knn_raw",
    "ffnn_raw",
    "cnn_raw",
)
TRIALS = ("cnn_mel_lr0003", "cnn_mel_dropout040")


def artifact_path(experiment):
    return (
        RESULTS_DIR
        / "models"
        / f"{experiment}{'.joblib' if experiment.startswith(('svm', 'knn')) else '.keras'}"
    )


def neural_input(X, name):
    return X[..., None] if name == "cnn" else X


def fit_experiment(experiment, epochs=50, batch_size=32, force=False):
    name, kind, *_ = experiment.split("_")
    path = artifact_path(experiment)
    meta_path = RESULTS_DIR / f"{experiment}_run.json"
    cache_meta = validate_cache_manifest()
    if path.exists() and meta_path.exists() and not force:
        old = read_json(meta_path)
        if old["feature_signature"] != cache_meta["signature"]:
            raise ValueError("Existing model uses a different dataset/cache. Use --force.")
        print(f"Reusing {experiment}", flush=True)
        return old
    cache = load_feature_cache(feature_type=kind, splits=("train", "val"))
    X, y, Xv, yv = cache["X_train"], cache["y_train"], cache["X_val"], cache["y_val"]
    started = perf_counter()
    settings = dict(
        seed=SEED,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=0.001,
        dropout=0.25,
    )
    devices = []
    history = None
    if name in ("svm", "knn"):
        from classical_models import build_classical_model

        model = build_classical_model(name)
        model.fit(X, y)
        predictions = {s: model.predict(cache[f"X_{s}"]) for s in ("train", "val")}
        joblib.dump(model, path, compress=3)
        settings.update(pca_components=15, neighbors=10, kernel="rbf", C=1, gamma="scale")
    else:
        import tensorflow as tf
        from gpu_check import configure_gpu
        from nn_models import build_cnn, build_ffnn

        devices = [d.name for d in configure_gpu(required=True)]
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED)
        tf.keras.mixed_precision.set_global_policy("float32")
        settings["learning_rate"] = 0.0003 if experiment.endswith("lr0003") else 0.001
        settings["dropout"] = 0.40 if experiment.endswith("dropout040") else 0.25
        Xn, Xvn = neural_input(X, name), neural_input(Xv, name)
        model = (
            build_cnn(
                Xn.shape[1:],
                kind,
                learning_rate=settings["learning_rate"],
                dropout=settings["dropout"],
            )
            if name == "cnn"
            else build_ffnn(Xn.shape[1:])
        )

        # A bounded tf.data pool avoids creating a thread per host CPU.
        def dataset(a, b, training=False):
            ds = tf.data.Dataset.from_tensor_slices((a, b))
            if training:
                ds = ds.shuffle(len(b), seed=SEED, reshuffle_each_iteration=True)
            options = tf.data.Options()
            options.threading.private_threadpool_size = 2
            return ds.batch(batch_size).with_options(options).prefetch(1)

        class HistoryWriter(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                write_json(
                    RESULTS_DIR / f"{experiment}_progress.json",
                    dict(
                        epoch=epoch + 1,
                        logs={k: float(v) for k, v in (logs or {}).items()},
                    ),
                )

        history = model.fit(
            dataset(Xn, y, True),
            validation_data=dataset(Xvn, yv),
            epochs=epochs,
            shuffle=False,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=8, restore_best_weights=True
                ),
                HistoryWriter(),
            ],
            verbose=2,
        )
        predictions = {}
        for s, arr in (("train", Xn), ("val", Xvn)):
            scores = np.concatenate(
                [
                    model(arr[i : i + batch_size], training=False).numpy()
                    for i in range(0, len(arr), batch_size)
                ]
            )
            predictions[s] = scores.argmax(axis=1)
        # Share layers in an uncompiled inference model, excluding optimizer slots.
        inference = tf.keras.Model(model.inputs, model.outputs, name=model.name)
        inference.save(path)
        write_json(
            RESULTS_DIR / f"{experiment}_history.json",
            {k: [float(x) for x in v] for k, v in history.history.items()},
        )
        settings["epochs_run"] = len(history.history["loss"])
        settings["best_epoch"] = int(np.argmin(history.history["val_loss"])) + 1
        settings["best_val_loss"] = float(min(history.history["val_loss"]))
    metrics = {}
    for split in ("train", "val"):
        target = cache[f"y_{split}"]
        metrics[split] = dict(
            accuracy=float(accuracy_score(target, predictions[split])),
            macro_f1=float(
                f1_score(
                    target,
                    predictions[split],
                    labels=range(10),
                    average="macro",
                    zero_division=0,
                )
            ),
            count=len(target),
        )
    result = dict(
        experiment=experiment,
        model=name,
        input=kind,
        settings=settings,
        metrics=metrics,
        feature_signature=cache_meta["signature"],
        feature_spec=FEATURE_SPEC,
        genres=list(GENRES),
        fit_seconds=perf_counter() - started,
        model_bytes=path.stat().st_size,
        gpu_devices=devices,
        python=platform.python_version(),
        model_path=str(path.relative_to(ROOT)),
    )
    write_json(meta_path, result)
    print(f"{experiment}: {metrics}, {result['fit_seconds']:.1f}s", flush=True)
    del cache, X, Xv, model
    gc.collect()
    return result


def select_demo():
    candidates = []
    for experiment in ("cnn_mel",) + TRIALS:
        path = RESULTS_DIR / f"{experiment}_run.json"
        if path.exists():
            candidates.append(read_json(path))
    if not candidates:
        raise ValueError("Train cnn_mel before selecting the demo")
    best = max(
        candidates,
        key=lambda r: (
            r["metrics"]["val"]["macro_f1"],
            -r["settings"]["best_val_loss"],
        ),
    )
    mean, std = load_mel_stats()
    selection = dict(
        experiment=best["experiment"],
        model_path=best["model_path"],
        criterion="validation macro-F1, tie-break minimum validation loss",
        validation_macro_f1=best["metrics"]["val"]["macro_f1"],
        genres=list(GENRES),
        feature_spec=FEATURE_SPEC,
        feature_signature=best["feature_signature"],
        mel_mean=mean,
        mel_std=std,
        inference=dict(window_count=5, maximum_seconds=30, aggregation="mean_softmax"),
    )
    write_json(RESULTS_DIR / "demo_model.json", selection)
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        choices=["all", "tuning"] + list(EXPERIMENTS + TRIALS),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch size must be positive")
    (RESULTS_DIR / "models").mkdir(parents=True, exist_ok=True)
    selected = [
        e
        for e in EXPERIMENTS + TRIALS
        if "all" in args.models or e in args.models or ("tuning" in args.models and e in TRIALS)
    ]
    for experiment in selected:
        # Subprocess isolation releases all CUDA and host memory between runs.
        if len(selected) > 1:
            import subprocess

            command = [
                sys.executable,
                str(Path(__file__)),
                "--models",
                experiment,
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
            ]
            if args.force:
                command.append("--force")
            subprocess.run(command, check=True, cwd=ROOT)
        else:
            batch = args.batch_size
            while True:
                try:
                    fit_experiment(experiment, args.epochs, batch, args.force)
                    break
                except Exception as error:
                    if "ResourceExhausted" not in type(error).__name__ or batch <= 8:
                        raise
                    batch = max(8, batch // 2)
                    print(f"Retrying with batch size {batch}", flush=True)
    if (RESULTS_DIR / "cnn_mel_run.json").exists():
        print(select_demo())


if __name__ == "__main__":
    main()
