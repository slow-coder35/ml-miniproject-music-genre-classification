"""Evaluate frozen models once and export auditable predictions and figures."""

from __future__ import annotations
import argparse
import gc
import hashlib
from time import perf_counter
from config import (
    CACHE_DIR,
    RESULTS_DIR,
    GENRES,
    SPLITS,
    read_json,
    write_json,
    initialize_gpu_environment,
)

initialize_gpu_environment()
import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from features import load_feature_cache
from train import EXPERIMENTS, TRIALS, artifact_path, neural_input, select_demo

FIGURES = RESULTS_DIR / "figures"


def plot_confusion(target, predicted, path, title):
    matrix = confusion_matrix(target, predicted, labels=range(10))
    norm = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=GENRES,
        yticklabels=GENRES,
        ax=ax,
    )
    ax.set(xlabel="Predicted genre", ylabel="True genre", title=title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return matrix.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="Explicitly repeat a frozen evaluation"
    )
    args = parser.parse_args()
    experiments = [e for e in EXPERIMENTS + TRIALS if artifact_path(e).exists()]
    if any(e not in experiments for e in EXPERIMENTS):
        raise ValueError("Complete all eight main experiments before final evaluation.")
    selection = select_demo()
    freeze = dict(
        models={e: hashlib.sha256(artifact_path(e).read_bytes()).hexdigest() for e in experiments},
        selection=selection,
        feature_signature=read_json(CACHE_DIR / "feature_meta.json")["signature"],
    )
    freeze_path = RESULTS_DIR / "evaluation_freeze.json"
    if freeze_path.exists() and (RESULTS_DIR / "evaluation.json").exists() and not args.force:
        if read_json(freeze_path) == freeze:
            print("Identical frozen evaluation already exists. Reusing it.")
            return
        raise ValueError(
            "Models changed after evaluation. Use --force and disclose the new evaluation."
        )
    write_json(freeze_path, freeze)
    FIGURES.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "predictions").mkdir(exist_ok=True)
    windows = read_json(CACHE_DIR / "windows.json")
    rows = []
    reports = {}
    import tensorflow as tf
    from gpu_check import configure_gpu

    configure_gpu()
    for experiment in experiments:
        run = read_json(RESULTS_DIR / f"{experiment}_run.json")
        name, kind = run["model"], run["input"]
        cache = load_feature_cache(feature_type=kind)
        if name in ("svm", "knn"):
            model = joblib.load(artifact_path(experiment))
        else:
            model = tf.keras.models.load_model(artifact_path(experiment), compile=False)
        row = dict(
            experiment=experiment,
            model=name,
            input=kind,
            tuned=experiment in TRIALS,
            training_run_seconds=run["fit_seconds"],
            model_mb=run["model_bytes"] / 1e6,
        )
        report = {}
        for split in SPLITS:
            X, y = cache[f"X_{split}"], cache[f"y_{split}"]
            started = perf_counter()
            if name in ("svm", "knn"):
                pred = model.predict(X)
            else:
                X = neural_input(X, name)
                pred = np.concatenate(
                    [model([X[i : i + 32]], training=False).numpy() for i in range(0, len(X), 32)]
                ).argmax(axis=1)
            elapsed = perf_counter() - started
            row[f"{split}_acc"] = float(accuracy_score(y, pred))
            row[f"{split}_macro_f1"] = float(
                f1_score(y, pred, labels=range(10), average="macro", zero_division=0)
            )
            row[f"n_{split}"] = len(y)
            if split == "test":
                row["test_ms_per_window"] = 1000 * elapsed / len(y)
            report[split] = classification_report(
                y,
                pred,
                labels=range(10),
                target_names=GENRES,
                zero_division=0,
                output_dict=True,
            )
            pd.DataFrame(
                dict(
                    track=[r["track"] for r in windows[split]],
                    start_sample=[r["start_sample"] for r in windows[split]],
                    true_index=y,
                    predicted_index=pred,
                    true_genre=[GENRES[i] for i in y],
                    predicted_genre=[GENRES[i] for i in pred],
                )
            ).to_csv(RESULTS_DIR / "predictions" / f"{experiment}_{split}.csv", index=False)
            if split == "test":
                report["test_confusion_counts"] = plot_confusion(
                    y,
                    pred,
                    FIGURES / f"{experiment}_confusion.png",
                    f"{experiment}: central two-second windows",
                )
        reports[experiment] = report
        rows.append(row)
        if name not in ("svm", "knn"):
            history = read_json(RESULTS_DIR / f"{experiment}_history.json")
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
            for ax, key in zip(axes, ("accuracy", "loss")):
                ax.plot(np.arange(1, len(history[key]) + 1), history[key], label="Training")
                ax.plot(
                    np.arange(1, len(history[key]) + 1),
                    history["val_" + key],
                    label="Validation",
                )
                ax.set(xlabel="Epoch", ylabel=key.capitalize())
                ax.legend()
            fig.suptitle(experiment)
            fig.tight_layout()
            fig.savefig(FIGURES / f"{experiment}_learning.png", dpi=150)
            plt.close(fig)
        print(experiment, row["test_acc"], flush=True)
        del model, cache, X
        tf.keras.backend.clear_session()
        gc.collect()
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS_DIR / "results.csv", index=False)
    base = frame[~frame.tuned]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, title in zip(
        axes, ("test_acc", "test_macro_f1"), ("Test accuracy", "Test macro-F1")
    ):
        pivot = base.pivot(index="model", columns="input", values=metric).reindex(
            ["svm", "knn", "ffnn", "cnn"]
        )
        pivot.plot.bar(ax=ax, color=["#007f8b", "#ed9459"])
        ax.set(title=title, ylim=(0, 1), xlabel="")
        ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(FIGURES / "comparison.png", dpi=170)
    plt.close(fig)
    # The demo uses five windows. Evaluate that protocol separately from the main table.
    from predict import Predictor
    from data import load_audio, load_splits

    predictor = Predictor()
    multi = {}
    for split in ("val", "test"):
        truth = []
        predicted = []
        records = []
        for track in load_splits()[split]:
            audio = load_audio(track)[: 30 * 22050]
            result = predictor.predict_audio(audio)
            truth.append(GENRES.index(__import__("pathlib").Path(track).parent.name))
            predicted.append(GENRES.index(result["genre"]))
            records.append(dict(track=track, true_genre=GENRES[truth[-1]], **result))
        multi[split] = dict(
            accuracy=float(accuracy_score(truth, predicted)),
            macro_f1=float(
                f1_score(truth, predicted, labels=range(10), average="macro", zero_division=0)
            ),
            count=len(truth),
        )
        write_json(RESULTS_DIR / "predictions" / f"demo_multiwindow_{split}.json", records)
        if split == "test":
            plot_confusion(
                truth,
                predicted,
                FIGURES / "demo_multiwindow_confusion.png",
                "Selected CNN: five-window track predictions",
            )
    inventory = read_json(CACHE_DIR / "inventory.json")
    counts = (
        pd.Series([r["genre"] for r in inventory["tracks"] if r["usable"]])
        .value_counts()
        .reindex(GENRES)
    )
    fig, ax = plt.subplots(figsize=(9, 3.5))
    counts.plot.bar(ax=ax, color="#007f8b")
    ax.set(ylabel="Usable recordings", xlabel="", title="Audited GTZAN class counts")
    plt.xticks(rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(FIGURES / "class_counts.png", dpi=160)
    plt.close(fig)
    write_json(
        RESULTS_DIR / "evaluation.json",
        dict(
            experiments=reports,
            selected_demo=selection,
            multiwindow=multi,
            protocol="Main comparison: one central 2-second window per validation/test track; training accuracy uses four random windows per training track.",
            rows=rows,
        ),
    )
    print("Final evaluation saved.", flush=True)


if __name__ == "__main__":
    main()
