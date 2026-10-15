"""Specified sklearn pipelines for raw and mel features."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

from data import SEED


def flatten_features(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return X.reshape(len(X), -1)


def build_classical_model(name: str) -> Pipeline:
    if name == "svm":
        classifier = SVC(kernel="rbf", random_state=SEED)
    elif name == "knn":
        classifier = KNeighborsClassifier(n_neighbors=10, weights="distance")
    else:
        raise ValueError(f"Unknown classical model: {name}")
    return Pipeline([
        ("flatten", FunctionTransformer(flatten_features)),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=15, svd_solver="randomized", random_state=SEED)),
        ("classifier", classifier),
    ])
