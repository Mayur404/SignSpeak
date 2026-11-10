"""Dataset storage and personal model training."""
from __future__ import annotations

import datetime as _dt
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .features import DUAL_DIM, RICH_DIM
from .labels import is_trainable_label
from .paths import USER_DATASET, USER_MODEL, active_model_path

MIN_SAMPLES_PER_CLASS = 5
MAX_SAMPLES_PER_TAKE = 15
RECOMMENDED_SAMPLES = 30
LABEL_DTYPE = "<U128"


class TrainingError(RuntimeError):
    pass


@dataclass
class TrainingResult:
    accuracy: float
    algorithm: str
    classes: int
    samples: int
    class_counts: dict[str, int] = field(default_factory=dict)
    cv_scores: list[float] = field(default_factory=list)
    model_path: str = ""
    validation_method: str = "sample split"
    balanced_accuracy: float = 0.0


class DatasetStore:
    """A tiny thread-safe store of rich-feature samples per gesture."""

    def __init__(self, path: Path = USER_DATASET):
        self.path = path
        self._lock = threading.Lock()
        self._X: np.ndarray = np.empty((0, DUAL_DIM), dtype=np.float32)
        self._y: np.ndarray = np.empty((0,), dtype=LABEL_DTYPE)
        self._groups: np.ndarray = np.empty((0,), dtype=np.int64)
        self.load()

    def load(self) -> None:
        with self._lock:
            if not self.path.exists():
                return
            try:
                with np.load(self.path, allow_pickle=False) as data:
                    X = np.asarray(data["X"], dtype=np.float32)
                    y = np.asarray(data["y"], dtype=LABEL_DTYPE)
                    if "groups" in data.files:
                        groups = np.asarray(data["groups"], dtype=np.int64)
                    elif len(y):
                        # Older datasets were appended in recording batches.
                        # Label transitions recover those batches well enough
                        # for grouped validation without discarding samples.
                        groups = np.cumsum(
                            np.r_[True, y[1:] != y[:-1]], dtype=np.int64
                        ) - 1
                    else:
                        groups = np.empty((0,), dtype=np.int64)
                if X.ndim == 2 and X.shape[1] == RICH_DIM:
                    # Preserve existing single-hand recordings in the first slot.
                    X = np.pad(X, ((0, 0), (0, RICH_DIM)))
                if X.ndim == 2 and X.shape[1] == DUAL_DIM and len(X) == len(y):
                    if len(groups) == len(y):
                        self._X, self._y, self._groups = X, y, groups
            except (OSError, ValueError, KeyError):
                self._X = np.empty((0, DUAL_DIM), dtype=np.float32)
                self._y = np.empty((0,), dtype=LABEL_DTYPE)
                self._groups = np.empty((0,), dtype=np.int64)

    def save(self) -> None:
        with self._lock:
            X, y, groups = self._X.copy(), self._y.copy(), self._groups.copy()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, X=X, y=y, groups=groups)
            tmp.replace(self.path)

    def add_samples(self, label: str, samples: np.ndarray) -> int:
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 2 or samples.shape[1] not in (RICH_DIM, DUAL_DIM) or len(samples) == 0:
            return 0
        if samples.shape[1] == RICH_DIM:
            samples = np.pad(samples, ((0, 0), (0, RICH_DIM)))
        with self._lock:
            self._X = np.vstack([self._X, samples])
            self._y = np.concatenate(
                [self._y, np.full(len(samples), label, dtype=LABEL_DTYPE)]
            )
            group = int(self._groups.max()) + 1 if len(self._groups) else 0
            self._groups = np.concatenate(
                [self._groups, np.full(len(samples), group, dtype=np.int64)]
            )
        self.save()
        return len(samples)

    def counts(self) -> dict[str, int]:
        with self._lock:
            values, counts = np.unique(self._y, return_counts=True)
            return {str(v): int(c) for v, c in zip(values, counts)}

    def total(self) -> int:
        with self._lock:
            return int(len(self._y))

    def data(self) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            return self._X.copy(), self._y.copy()

    def training_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with self._lock:
            return self._X.copy(), self._y.copy(), self._groups.copy()

    def take_counts(self) -> dict[str, int]:
        with self._lock:
            counts: dict[str, int] = {}
            for label in np.unique(self._y):
                counts[str(label)] = int(np.unique(self._groups[self._y == label]).size)
            return counts

    def clear_label(self, label: str) -> None:
        with self._lock:
            mask = self._y != label
            self._X, self._y, self._groups = (
                self._X[mask], self._y[mask], self._groups[mask]
            )
        self.save()

    def clear(self) -> None:
        with self._lock:
            self._X = np.empty((0, DUAL_DIM), dtype=np.float32)
            self._y = np.empty((0,), dtype=LABEL_DTYPE)
            self._groups = np.empty((0,), dtype=np.int64)
        self.save()

    def has_model(self) -> bool:
        return active_model_path().exists()


def _build_candidates():
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    return [
        (
            "SVM · RBF kernel",
            CalibratedClassifierCV(
                SVC(
                    C=10.0,
                    gamma="scale",
                    class_weight="balanced",
                    random_state=42,
                ),
                method="sigmoid",
                cv=3,
                ensemble=False,
            ),
        ),
        (
            "Extra Trees · 300",
            ExtraTreesClassifier(
                n_estimators=300,
                min_samples_leaf=1,
                class_weight="balanced",
                # Avoid competing with the live camera/tracker for every CPU
                # core while personal training runs in the background.
                n_jobs=1,
                random_state=42,
            ),
        ),
        (
            "K-Nearest Neighbours · k=5",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "clf",
                        KNeighborsClassifier(
                            n_neighbors=5, weights="distance", metric="euclidean"
                        ),
                    ),
                ]
            ),
        ),
        (
            "Random Forest · 300",
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced_subsample",
                n_jobs=1,
                random_state=42,
            ),
        ),
    ]


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    progress: Callable[[str, int], None] | None = None,
    model_path: Path = USER_MODEL,
    groups: np.ndarray | None = None,
) -> TrainingResult:
    """Train and persist a personal classifier. Returns a training report."""

    def report(message: str, percent: int) -> None:
        if progress is not None:
            progress(message, percent)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    if X.ndim != 2 or X.shape[1] not in (RICH_DIM, DUAL_DIM):
        raise TrainingError("Dataset has an unexpected feature shape.")
    if X.shape[1] == RICH_DIM:
        X = np.pad(X, ((0, 0), (0, RICH_DIM)))
    if len(X) != len(y) or len(X) == 0:
        raise TrainingError("No samples available to train on.")
    if groups is not None:
        groups = np.asarray(groups)
        if groups.ndim != 1 or len(groups) != len(y):
            raise TrainingError("Recording groups do not match the dataset.")

    labels, counts = np.unique(y, return_counts=True)
    unknown = [label for label in labels if not is_trainable_label(str(label))]
    if unknown:
        raise TrainingError(f"Unknown labels in dataset: {unknown}")
    if len(labels) < 2:
        raise TrainingError("Record at least two different signs before training.")
    thin = [
        (str(label), int(count))
        for label, count in zip(labels, counts)
        if count < MIN_SAMPLES_PER_CLASS
    ]
    if thin:
        names = ", ".join(f"{label} ({count})" for label, count in thin[:6])
        raise TrainingError(
            f"Some signs have fewer than {MIN_SAMPLES_PER_CLASS} samples: {names}. "
            "Record a few more for each sign."
        )

    report("Preparing dataset…", 5)
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold, cross_validate

    min_count = int(counts.min())
    n_splits = max(2, min(5, min_count))
    cv_splits = None
    validation_method = "sample-level split (record more separate takes for a stronger check)"
    if groups is not None:
        group_count_by_label = {
            str(label): len(np.unique(groups[y == label])) for label in labels
        }
        min_groups = min(group_count_by_label.values())
        if min_groups >= 2:
            from sklearn.model_selection import StratifiedGroupKFold

            grouped_splits = min(5, min_groups)
            for seed in range(42, 52):
                candidate_splits = list(
                    StratifiedGroupKFold(
                        n_splits=grouped_splits, shuffle=True, random_state=seed
                    ).split(X, y, groups)
                )
                expected = set(labels.tolist())
                if all(
                    set(y[train_index].tolist()) == expected
                    and set(y[test_index].tolist()) == expected
                    for train_index, test_index in candidate_splits
                ):
                    cv_splits = candidate_splits
                    n_splits = grouped_splits
                    validation_method = "separate recording takes"
                    break
    if cv_splits is None:
        n_splits = max(2, min(5, min_count))
        cv_splits = list(
            StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42).split(X, y)
        )

    best_name = ""
    best_balanced = -1.0
    best_accuracy = 0.0
    best_model = None
    best_scores: list[float] = []
    candidates = _build_candidates()
    for index, (name, candidate) in enumerate(candidates):
        report(f"Evaluating {name}…", 10 + int(60 * index / len(candidates)))
        try:
            scores = cross_validate(
                candidate,
                X,
                y,
                cv=cv_splits,
                scoring={"accuracy": "accuracy", "balanced": "balanced_accuracy"},
                n_jobs=1,
                error_score="raise",
            )
            mean_balanced = float(np.mean(scores["test_balanced"]))
            mean_accuracy = float(np.mean(scores["test_accuracy"]))
        except Exception:
            continue
        if mean_balanced > best_balanced:
            best_name = name
            best_balanced = mean_balanced
            best_accuracy = mean_accuracy
            best_model = candidate
            best_scores = [float(s) for s in scores["test_balanced"]]

    if best_model is None:
        raise TrainingError("Training failed — try recording more samples.")

    report(f"Fitting {best_name}…", 78)
    best_model.fit(X, y)

    # Refit on a held-out check is already covered by CV; this is a sanity check.
    train_accuracy = float(accuracy_score(y, best_model.predict(X)))
    train_balanced_accuracy = float(balanced_accuracy_score(y, best_model.predict(X)))
    report("Saving model…", 92)

    class_counts = {str(label): int(count) for label, count in zip(labels, counts)}
    bundle = {
        "model": best_model,
        "labels": [str(label) for label in labels],
        "feature_mode": "rich126",
        "algorithm": best_name,
        "accuracy": best_accuracy,
        "balanced_accuracy": best_balanced,
        "train_accuracy": train_accuracy,
        "train_balanced_accuracy": train_balanced_accuracy,
        "trained_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "samples": int(len(X)),
        "class_counts": class_counts,
        "cv_scores": best_scores,
        "n_splits": n_splits,
        "validation_method": validation_method,
    }
    import joblib

    model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = model_path.with_suffix(".tmp.joblib")
    joblib.dump(bundle, tmp)
    tmp.replace(model_path)
    report("Done", 100)

    return TrainingResult(
        accuracy=best_accuracy,
        algorithm=best_name,
        classes=int(len(labels)),
        samples=int(len(X)),
        class_counts=class_counts,
        cv_scores=best_scores,
        model_path=str(model_path),
        validation_method=validation_method,
        balanced_accuracy=best_balanced,
    )


def _lower_process_priority() -> None:
    """Keep the live camera/recognition process ahead of training."""
    try:
        import os
        import sys
        if sys.platform == "win32":
            import ctypes
            BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.kernel32.SetPriorityClass(handle, BELOW_NORMAL_PRIORITY_CLASS)
        else:
            os.nice(10)
    except Exception:
        pass


def train_in_subprocess(X, y, groups, queue, model_path=None) -> None:
    """Run train_model in a child process, reporting through ``queue``.

    Messages: ("progress", message, percent), ("done", result_dict) or
    ("error", message). Running here instead of in a thread keeps model
    fitting from holding the main process's GIL, so live recognition keeps
    running at full speed while a model trains.
    """
    _lower_process_priority()
    try:
        def progress(message: str, percent: int) -> None:
            queue.put(("progress", str(message), int(percent)))

        res = train_model(X, y, progress=progress, groups=groups,
                          model_path=Path(model_path) if model_path else USER_MODEL)
        queue.put(("done", {
            "accuracy": float(res.accuracy),
            "algorithm": str(res.algorithm),
            "samples": int(res.samples),
            "classes": int(res.classes),
        }))
    except Exception as exc:  # report every failure to the parent
        queue.put(("error", str(exc) or exc.__class__.__name__))
