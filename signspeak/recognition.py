"""Model loading, prediction and temporal stabilisation."""
from __future__ import annotations

import pickle
import warnings
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .features import FEATURE_DIMS, extract
from .labels import LABELS, LEGACY_INDEX_TO_LABEL
from .paths import LEGACY_MODEL, USER_MODEL


class ModelLoadError(RuntimeError):
    pass


@dataclass
class Prediction:
    label: str
    confidence: float
    top: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class ModelInfo:
    kind: str
    name: str
    feature_mode: str
    algorithm: str
    accuracy: float | None = None
    trained_at: str | None = None
    samples: int | None = None
    classes: int | None = None
    validation_method: str | None = None
    balanced_accuracy: float | None = None

    @property
    def is_personal(self) -> bool:
        return self.kind == "user"

    @property
    def has_take_validation(self) -> bool:
        return (
            self.validation_method == "separate recording takes"
            and self.balanced_accuracy is not None
        )

    @property
    def subtitle(self) -> str:
        if self.balanced_accuracy is not None:
            return f"{self.algorithm} · {self.balanced_accuracy * 100:.1f}% balanced validation"
        if self.accuracy is not None:
            return f"{self.algorithm} · {self.accuracy * 100:.1f}% validation score"
        return self.algorithm


@dataclass
class RecognitionState:
    hand_present: bool = False
    label: str | None = None
    confidence: float = 0.0
    stable_label: str | None = None
    stable_ratio: float = 0.0
    hold_progress: float = 0.0
    armed: bool = True
    committed: str | None = None
    top: list[tuple[str, float]] = field(default_factory=list)

    @property
    def display_label(self) -> str | None:
        return self.stable_label or self.label

    @property
    def is_stable(self) -> bool:
        return self.stable_label is not None


class SignClassifier:
    """Thin wrapper around either the bundled RF or a personally trained model."""

    def __init__(self, model: Any, info: ModelInfo, classes: list[str]):
        self.model = model
        self.info = info
        self.classes = list(classes)
        self.feature_mode = info.feature_mode
        self.feature_dim = FEATURE_DIMS[self.feature_mode]

    def predict(self, features: np.ndarray) -> Prediction | None:
        vector = np.asarray(features, dtype=np.float32).reshape(1, -1)
        # Existing personal models and training examples contain one 63-value
        # hand. Place that hand in slot one when used with a new dual-hand model.
        if self.feature_mode == "rich126" and vector.shape[1] == 63:
            vector = np.pad(vector, ((0, 0), (0, 63)))
        if vector.shape[1] != self.feature_dim:
            return None
        try:
            probabilities = self.model.predict_proba(vector)[0]
        except Exception:
            try:
                index = int(self.model.predict(vector)[0])
            except Exception:
                return None
            label = self.classes[index] if 0 <= index < len(self.classes) else None
            return Prediction(label, 1.0, [(label, 1.0)]) if label else None

        model_classes = getattr(self.model, "classes_", None)
        pairs: list[tuple[str, float]] = []
        if model_classes is not None and len(model_classes) == len(probabilities):
            for raw_class, probability in zip(model_classes, probabilities):
                label = self._normalise_class(raw_class)
                if label is not None:
                    pairs.append((label, float(probability)))
        else:
            for label, probability in zip(self.classes, probabilities):
                pairs.append((label, float(probability)))

        if not pairs:
            return None
        pairs.sort(key=lambda item: item[1], reverse=True)
        return Prediction(pairs[0][0], pairs[0][1], pairs[:5])

    def _normalise_class(self, raw: Any) -> str | None:
        if isinstance(raw, (int, np.integer)):
            return LEGACY_INDEX_TO_LABEL.get(int(raw))
        text = str(raw)
        if text in self.classes:
            return text
        if text in LABELS:
            return text
        try:
            return LEGACY_INDEX_TO_LABEL.get(int(float(text)))
        except (TypeError, ValueError):
            return None


def _load_pickle(path: Path) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            from sklearn.exceptions import InconsistentVersionWarning

            warnings.simplefilter("ignore", InconsistentVersionWarning)
        except Exception:
            pass
        with path.open("rb") as handle:
            return pickle.load(handle)


def load_legacy_classifier(path: Path = LEGACY_MODEL) -> SignClassifier:
    if not path.exists():
        raise ModelLoadError(f"Bundled model not found: {path}")
    try:
        bundle = _load_pickle(path)
    except Exception as exc:  # pragma: no cover - corrupted file
        raise ModelLoadError(f"Could not load the bundled model: {exc}") from exc
    model = bundle.get("model") if isinstance(bundle, dict) else bundle
    if model is None:
        raise ModelLoadError("Bundled model file has an unexpected format.")
    classes = [LEGACY_INDEX_TO_LABEL[i] for i in range(len(LEGACY_INDEX_TO_LABEL))]
    info = ModelInfo(
        kind="legacy",
        name="Bundled ASL model",
        feature_mode="legacy42",
        algorithm="Random Forest · 100 trees",
        classes=len(classes),
        samples=None,
    )
    return SignClassifier(model, info, classes)


def load_user_classifier(path: Path = USER_MODEL) -> SignClassifier:
    if not path.exists():
        raise ModelLoadError("You have not trained a personal model yet.")
    try:
        import joblib

        bundle = joblib.load(path)
    except Exception as exc:
        raise ModelLoadError(f"Could not load your model: {exc}") from exc
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ModelLoadError("Personal model file has an unexpected format.")
    labels = [str(label) for label in bundle.get("labels", [])]
    model = bundle["model"]
    classes = [str(c) for c in getattr(model, "classes_", labels)]
    info = ModelInfo(
        kind="user",
        name="Your personal model",
        feature_mode=str(bundle.get("feature_mode", "rich63")),
        algorithm=str(bundle.get("algorithm", "Custom classifier")),
        accuracy=float(bundle["accuracy"]) if bundle.get("accuracy") is not None else None,
        trained_at=bundle.get("trained_at"),
        samples=int(bundle["samples"]) if bundle.get("samples") is not None else None,
        classes=len(classes),
        validation_method=str(bundle.get("validation_method", "sample-level split")),
        balanced_accuracy=(
            float(bundle["balanced_accuracy"])
            if bundle.get("balanced_accuracy") is not None
            else None
        ),
    )
    return SignClassifier(model, info, classes)


def load_classifier(kind: str) -> SignClassifier:
    return load_user_classifier() if kind == "user" else load_legacy_classifier()


class Stabilizer:
    """Turns noisy per-frame predictions into deliberate, typed characters.

    A character is committed only when it wins a majority vote over a short
    sliding window *and* is held for ``hold_ms``. After committing, the same
    character cannot repeat until the hand is released or a different sign is
    shown, which prevents one long hold from spamming the transcript.
    """

    def __init__(
        self,
        window: int = 12,
        threshold: float = 0.55,
        hold_ms: int = 650,
        min_ratio: float = 0.60,
        cooldown_ms: int = 250,
        release_ms: int = 320,
    ):
        self.window_size = max(3, int(window))
        self.threshold = float(threshold)
        self.hold_ms = max(50, int(hold_ms))
        self.min_ratio = float(min_ratio)
        self.cooldown_ms = max(0, int(cooldown_ms))
        self.release_ms = max(50, int(release_ms))
        self._window: deque[tuple[str | None, float]] = deque(maxlen=self.window_size)
        self._candidate: str | None = None
        self._candidate_since: int | None = None
        self._stable: str | None = None
        self._ratio = 0.0
        self._armed = True
        self._committed_label: str | None = None
        self._last_commit = -10**9
        self._last_seen: int | None = None
        self._none_since: int | None = None

    def configure(
        self,
        window: int | None = None,
        threshold: float | None = None,
        hold_ms: int | None = None,
        min_ratio: float | None = None,
        cooldown_ms: int | None = None,
        release_ms: int | None = None,
    ) -> None:
        if window is not None and int(window) != self.window_size:
            self.window_size = max(3, int(window))
            self._window = deque(self._window, maxlen=self.window_size)
        if threshold is not None:
            self.threshold = float(threshold)
        if hold_ms is not None:
            self.hold_ms = max(50, int(hold_ms))
        if min_ratio is not None:
            self.min_ratio = float(min_ratio)
        if cooldown_ms is not None:
            self.cooldown_ms = max(0, int(cooldown_ms))
        if release_ms is not None:
            self.release_ms = max(50, int(release_ms))

    def reset(self) -> None:
        self._window.clear()
        self._candidate = None
        self._candidate_since = None
        self._stable = None
        self._ratio = 0.0
        self._armed = True
        self._committed_label = None
        self._last_commit = -10**9
        self._last_seen = None
        self._none_since = None

    def _release(self) -> None:
        self._armed = True
        self._committed_label = None
        self._candidate_since = None

    def update(
        self, prediction: Prediction | None, hand_present: bool, now_ms: int
    ) -> RecognitionState:
        if not hand_present:
            if self._last_seen is None or now_ms - self._last_seen >= self.release_ms:
                self._window.clear()
                self._candidate = None
                self._stable = None
                self._ratio = 0.0
                self._candidate_since = None
                if self._committed_label is not None:
                    self._release()
            return RecognitionState(hand_present=False, armed=self._armed)

        self._last_seen = now_ms
        self._none_since = None
        confidence = prediction.confidence if prediction else 0.0
        raw_label = (
            prediction.label
            if prediction is not None and confidence >= self.threshold
            else None
        )
        self._window.append((raw_label, confidence))

        counts = Counter(label for label, _ in self._window if label is not None)
        stable: str | None = None
        ratio = 0.0
        if counts and len(self._window) >= min(5, self.window_size):
            top_label, top_count = counts.most_common(1)[0]
            ratio = top_count / len(self._window)
            mean_confidence = float(
                np.mean([c for label, c in self._window if label == top_label])
            )
            if ratio >= self.min_ratio and mean_confidence >= self.threshold:
                stable = top_label
        self._stable = stable
        self._ratio = ratio

        progress = 0.0
        committed: str | None = None
        if stable is None:
            if self._none_since is None:
                self._none_since = now_ms
            if (
                self._committed_label is not None
                and now_ms - self._none_since >= self.release_ms
            ):
                self._release()
            self._candidate = None
            self._candidate_since = None
        else:
            if stable != self._candidate:
                self._candidate = stable
                self._candidate_since = now_ms
            if self._committed_label is not None and stable != self._committed_label:
                self._release()
            if self._armed and self._candidate_since is not None:
                elapsed = now_ms - self._candidate_since
                progress = min(1.0, elapsed / self.hold_ms)
                if (
                    progress >= 1.0
                    and now_ms - self._last_commit >= self.cooldown_ms
                ):
                    committed = stable
                    self._last_commit = now_ms
                    self._committed_label = stable
                    self._armed = False
                    self._candidate_since = now_ms
                    progress = 0.0

        return RecognitionState(
            hand_present=True,
            label=prediction.label if prediction else None,
            confidence=confidence,
            stable_label=stable,
            stable_ratio=ratio,
            hold_progress=progress,
            armed=self._armed,
            committed=committed,
            top=prediction.top if prediction else [],
        )

    def commit_now(self, now_ms: int) -> str | None:
        label = self._stable
        if not label:
            return None
        self._last_commit = now_ms
        self._committed_label = label
        self._armed = False
        self._candidate_since = now_ms
        return label


class RecognitionEngine:
    """Feature extraction + classification + stabilisation."""

    def __init__(self, classifier: SignClassifier, settings):
        self.classifier = classifier
        self.stabilizer = Stabilizer(
            window=settings.smoothing,
            threshold=self._confidence_threshold(classifier, settings),
            hold_ms=settings.hold_ms,
            min_ratio=settings.min_vote_ratio,
            cooldown_ms=settings.cooldown_ms,
            release_ms=settings.release_ms,
        )
        self.last_error: str | None = None

    def set_classifier(self, classifier: SignClassifier) -> None:
        self.classifier = classifier
        self.stabilizer.reset()

    @staticmethod
    def _confidence_threshold(classifier: SignClassifier, settings) -> float:
        # Class probabilities depend on the number and balance of trained
        # classes. A fixed cutoff silently rejects signs as users add labels.
        # Temporal vote agreement and hold time are the recognition gates;
        # model probability is shown as diagnostic feedback only.
        return 0.0

    def apply_settings(self, settings) -> None:
        self.stabilizer.configure(
            window=settings.smoothing,
            threshold=self._confidence_threshold(self.classifier, settings),
            hold_ms=settings.hold_ms,
            min_ratio=settings.min_vote_ratio,
            cooldown_ms=settings.cooldown_ms,
            release_ms=settings.release_ms,
        )

    def reset(self) -> None:
        self.stabilizer.reset()

    def update(self, points: np.ndarray | None, now_ms: int) -> RecognitionState:
        if points is None:
            return self.stabilizer.update(None, False, now_ms)
        try:
            features = extract(points, self.classifier.feature_mode)
            prediction = self.classifier.predict(features)
            self.last_error = None
        except Exception as exc:  # keep the UI alive if a model misbehaves
            self.last_error = str(exc)
            prediction = None
        return self.stabilizer.update(prediction, True, now_ms)

    def commit_now(self, now_ms: int) -> str | None:
        return self.stabilizer.commit_now(now_ms)
