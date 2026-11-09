from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from signspeak.composer import TextComposer
from signspeak.config import Settings, load_settings, save_settings
from signspeak.features import DUAL_DIM, LEGACY_DIM, RICH_DIM, dual_hand_features, extract
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
from signspeak.vision import (
    CameraProbe,
    _camera_candidates,
    _safe_camera_source,
    sparsh_stream_urls,
)


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
        other_hand = self.points.copy()
        other_hand[:, 0] *= -1
        dual = dual_hand_features([self.points, other_hand])
        self.assertEqual(dual.shape, (DUAL_DIM,))
        self.assertFalse(np.array_equal(dual[:RICH_DIM], dual[RICH_DIM:]))
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
        self.assertEqual(result.speak_word, "good morning")
        self.assertEqual(composer.transcript, "good morning")
        composer.commit("A")
        self.assertTrue(composer.commit(BACKSPACE).changed)
        self.assertEqual(composer.word, "")
        self.assertTrue(composer.commit(BACKSPACE).changed)
        self.assertEqual(composer.transcript, "")

    def test_delete_word_removes_current_or_last_completed_word(self):
        composer = TextComposer()
        composer.commit("H")
        composer.commit("I")
        composer.space()
        composer.commit("Y")
        composer.commit("O")
        composer.commit("U")
        composer.period()

        self.assertTrue(composer.delete_word())
        self.assertEqual(composer.transcript, "HI")
        self.assertTrue(composer.delete_word())
        self.assertEqual(composer.transcript, "")
        self.assertFalse(composer.delete_word())

        composer.commit("O")
        composer.commit("K")
        self.assertTrue(composer.delete_word())
        self.assertEqual(composer.word, "")


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

    def test_stabilizer_ignores_ambiguous_frame_predictions(self):
        stabilizer = Stabilizer(window=5, threshold=0.0, hold_ms=100, min_ratio=0.6)
        ambiguous = Prediction("A", 0.52, [("A", 0.52), ("B", 0.50)])
        states = [stabilizer.update(ambiguous, True, t) for t in range(0, 400, 50)]
        self.assertTrue(all(state.stable_label is None for state in states))
        self.assertFalse(any(state.committed for state in states))

    def test_stabilizer_switches_sign_without_hand_leaving_frame(self):
        stabilizer = Stabilizer(
            window=5, threshold=0.0, hold_ms=100, min_ratio=0.6,
            cooldown_ms=0, release_ms=300,
        )
        sign_a = Prediction("A", 0.95, [("A", 0.95)])
        sign_b = Prediction("B", 0.95, [("B", 0.95)])
        commits = []
        for now in range(0, 301, 50):
            state = stabilizer.update(sign_a, True, now)
            if state.committed:
                commits.append(state.committed)
        for now in range(350, 701, 50):
            state = stabilizer.update(sign_b, True, now)
            if state.committed:
                commits.append(state.committed)
        self.assertEqual(commits, ["A", "B"])

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
            expected = np.pad(np.vstack([samples, samples[:1]]), ((0, 0), (0, RICH_DIM)))
            np.testing.assert_array_equal(loaded.data()[0], expected)
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
            prediction = classifier.predict(np.pad(X[0], (0, RICH_DIM)))
            self.assertEqual(result.classes, 2)
            self.assertGreaterEqual(result.accuracy, 0.9)
            self.assertGreaterEqual(result.balanced_accuracy, 0.9)
            self.assertEqual(result.validation_method, "separate recording takes")
            self.assertTrue(model_path.exists())
            self.assertIsNotNone(prediction)
            self.assertEqual(prediction.label, "A")

    def test_old_single_hand_dataset_is_migrated_without_losing_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old_samples.npz"
            old_samples = np.ones((4, RICH_DIM), dtype=np.float32)
            np.savez_compressed(path, X=old_samples, y=np.array(["A"] * 4))
            store = DatasetStore(path)
            upgraded, labels, _groups = store.training_data()
            self.assertEqual(upgraded.shape, (4, DUAL_DIM))
            np.testing.assert_array_equal(upgraded[:, :RICH_DIM], old_samples)
            np.testing.assert_array_equal(upgraded[:, RICH_DIM:], 0)
            self.assertEqual(labels.tolist(), ["A"] * 4)


class SettingsTests(unittest.TestCase):
    def test_settings_camera_source_persistence(self):
        settings = Settings(camera_source="rtsp://admin:admin123@192.168.128.10:554/h264/ch1/sub/av_stream")
        self.assertIn("rtsp://", settings.camera_source)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            save_settings(settings, path)
            loaded = load_settings(path)
            self.assertEqual(loaded.camera_source, settings.camera_source)


class CameraUrlTests(unittest.TestCase):
    def test_probe_offers_sparsh_streams_when_rtsp_port_is_reachable(self):
        results = []
        probe = CameraProbe(skip_source=0)
        probe.finishedList.connect(results.append)
        with patch("socket.create_connection"), patch(
            "signspeak.vision.cv2.VideoCapture", return_value=Mock(isOpened=lambda: False)
        ):
            probe.run()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][:2], sparsh_stream_urls())

    def test_sparsh_picker_offers_both_streams(self):
        urls = sparsh_stream_urls()
        self.assertEqual(len(urls), 2)
        self.assertIn("/sub/av_stream", urls[0][0])
        self.assertIn("/main/av_stream", urls[1][0])

    def test_sparsh_fallback_keeps_selected_stream_first(self):
        substream = sparsh_stream_urls()[0][0]
        candidates = _camera_candidates(substream)
        self.assertEqual(candidates[0], substream)
        self.assertIn("/avstream/channel=1/stream=1.sdp", candidates[1])
        self.assertIn("/main/av_stream", candidates[2])

    def test_other_custom_rtsp_url_is_not_rewritten(self):
        source = "rtsp://camera.local:8554/laptop"
        self.assertEqual(_camera_candidates(source), [source])

    def test_camera_error_url_does_not_expose_credentials_or_query(self):
        source = "rtsp://admin:secret@192.168.1.10:554/live?token=private"
        self.assertEqual(
            _safe_camera_source(source),
            "rtsp://192.168.1.10:554/live",
        )


if __name__ == "__main__":
    unittest.main()
