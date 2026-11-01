from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from signspeak.composer import TextComposer
from signspeak.features import LEGACY_DIM, RICH_DIM, extract
from signspeak.labels import BACKSPACE, PERIOD, SPACE, custom_word_label
from signspeak.recognition import (
    ModelInfo,
    Prediction,
    RecognitionEngine,
    SignClassifier,
    Stabilizer,
    load_legacy_classifier,
    load_user_classifier,
)
from signspeak.trainer import DatasetStore, train_model


class FeatureTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.points = rng.normal(size=(21, 3)).astype(np.float32)

    def test_feature_shapes_and_translation_invariance(self):
        legacy = extract(self.points, "legacy42")
        rich = extract(self.points, "rich63")
        shifted = self.points + np.array([8, -3, 2], dtype=np.float32)
        self.assertEqual(legacy.shape, (LEGACY_DIM,))
        self.assertEqual(rich.shape, (RICH_DIM,))
        np.testing.assert_allclose(legacy, extract(shifted, "legacy42"), atol=1e-6)
        np.testing.assert_allclose(rich, extract(shifted, "rich63"), atol=1e-5)


class ComposerTests(unittest.TestCase):
    def test_words_sentences_and_backspace(self):
        composer = TextComposer()
        composer.commit("H")
        composer.commit("I")
        self.assertEqual(composer.space().speak_word, "HI")
        composer.commit("Y")
        composer.commit("O")
        composer.commit("U")
        self.assertEqual(composer.commit(PERIOD).speak_sentence, "HI YOU")
        self.assertEqual(composer.transcript, "HI YOU.")
        self.assertTrue(composer.backspace())
        self.assertEqual(composer.transcript, "HI YOU")
        self.assertEqual(composer.commit(SPACE).changed, False)

    def test_custom_word_is_a_finished_word_and_backspace_gesture_deletes(self):
        composer = TextComposer()
        result = composer.commit(custom_word_label("good morning"))
        self.assertEqual(result.speak_word, "GOOD MORNING")
        self.assertEqual(composer.transcript, "GOOD MORNING")
        composer.commit("A")
        self.assertTrue(composer.commit(BACKSPACE).changed)
        self.assertEqual(composer.word, "")
        self.assertTrue(composer.commit(BACKSPACE).changed)
        self.assertEqual(composer.transcript, "")


class RecognitionTests(unittest.TestCase):
    def test_reference_model_uses_majority_vote_without_confidence_gate(self):
        from signspeak.config import Settings

        class DummyModel:
            pass

        settings = Settings(confidence=0.9)
        bundled = SignClassifier(
            DummyModel(), ModelInfo("legacy", "Reference", "legacy42", "RF"), ["A"]
        )
        personal = SignClassifier(
            DummyModel(), ModelInfo("user", "Personal", "rich63", "RF"), ["A"]
        )
        self.assertEqual(RecognitionEngine(bundled, settings).stabilizer.threshold, 0.0)
        self.assertEqual(RecognitionEngine(personal, settings).stabilizer.threshold, 0.0)

    def test_custom_word_classifier_label_is_preserved(self):
        custom = custom_word_label("THANK YOU")

        class DummyModel:
            classes_ = np.array([custom, BACKSPACE])

            def predict_proba(self, _vector):
                return np.array([[0.8, 0.2]])

        classifier = SignClassifier(
            DummyModel(),
            ModelInfo("user", "Personal", "rich63", "RF"),
            [custom, BACKSPACE],
        )
        prediction = classifier.predict(np.zeros(RICH_DIM, dtype=np.float32))
        self.assertIsNotNone(prediction)
        self.assertEqual(prediction.label, custom)

    def test_stabilizer_commits_once_until_release(self):
        stabilizer = Stabilizer(window=5, threshold=0.5, hold_ms=100, min_ratio=0.6,
                                cooldown_ms=0, release_ms=100)
        prediction = Prediction("A", 0.95, [("A", 0.95)])
        states = [stabilizer.update(prediction, True, t) for t in range(0, 351, 50)]
        commits = [state.committed for state in states if state.committed]
        self.assertEqual(commits, ["A"])
        for t in (400, 450, 500):
            stabilizer.update(None, False, t)
        repeated = [stabilizer.update(prediction, True, t).committed
                    for t in range(550, 1001, 50)]
        self.assertEqual(repeated.count("A"), 1)

    def test_bundled_model_predicts_legacy_vector(self):
        classifier = load_legacy_classifier()
        points = np.zeros((21, 3), dtype=np.float32)
        prediction = classifier.predict(extract(points, "legacy42"))
        self.assertIsNotNone(prediction)
        self.assertIn(prediction.label, classifier.classes)


class DatasetTests(unittest.TestCase):
    def test_dataset_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            store = DatasetStore(Path(directory) / "samples.npz")
            samples = np.ones((3, RICH_DIM), dtype=np.float32)
            self.assertEqual(store.add_samples("A", samples), 3)
            self.assertEqual(store.add_samples("A", samples[:1]), 1)
            loaded = DatasetStore(Path(directory) / "samples.npz")
            self.assertEqual(loaded.counts(), {"A": 4})
            self.assertEqual(loaded.take_counts(), {"A": 2})
            np.testing.assert_array_equal(loaded.data()[0], np.vstack([samples, samples[:1]]))
            loaded.clear_label("A")
            self.assertEqual(loaded.total(), 0)

    def test_custom_word_label_survives_dataset_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.npz"
            label = custom_word_label("thank you")
            store = DatasetStore(path)
            store.add_samples(label, np.ones((5, RICH_DIM), dtype=np.float32))
            loaded = DatasetStore(path)
            self.assertEqual(loaded.counts(), {label: 5})

    def test_synthetic_training_saves_loadable_model(self):
        rng = np.random.default_rng(12)
        X = np.vstack(
            [
                rng.normal(-1.0, 0.05, size=(10, RICH_DIM)),
                rng.normal(1.0, 0.05, size=(10, RICH_DIM)),
            ]
        ).astype(np.float32)
        custom = custom_word_label("GOOD MORNING")
        y = np.array(["A"] * 10 + [custom] * 10)
        groups = np.array([0] * 5 + [1] * 5 + [2] * 5 + [3] * 5)
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "user_model.joblib"
            result = train_model(X, y, model_path=model_path, groups=groups)
            classifier = load_user_classifier(model_path)
            prediction = classifier.predict(X[0])
            self.assertEqual(result.classes, 2)
            self.assertGreaterEqual(result.accuracy, 0.9)
            self.assertGreaterEqual(result.balanced_accuracy, 0.9)
            self.assertEqual(result.validation_method, "separate recording takes")
            self.assertTrue(model_path.exists())
            self.assertIsNotNone(prediction)
            self.assertEqual(prediction.label, "A")


if __name__ == "__main__":
    unittest.main()
