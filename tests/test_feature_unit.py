import os
import tempfile
import unittest
from pathlib import Path

import torch

from curator.data import AseDataset, collate_atomsdata, properties
from curator.layer._feature import (
    DistanceMetrics,
    FeatureCalculator,
    FeatureExtractor,
    FeatureStatistics,
    H5Feature,
    normalize_kernel,
)
from curator.select.active_learning import GeneralActiveLearning
from curator.utils import load_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = Path(os.environ.get("CURATOR_TEST_DATASET", PROJECT_ROOT / "example/LiFePO4.traj"))
MODEL_PATH = Path(os.environ.get("CURATOR_TEST_CKPT", PROJECT_ROOT / "example/best_model.ckpt"))


def _load_dataset(max_items: int = 2):
    if not DATASET_PATH.exists():
        raise unittest.SkipTest(f"Dataset not found at {DATASET_PATH}")
    dataset = AseDataset(str(DATASET_PATH))
    count = min(max_items, len(dataset))
    return torch.utils.data.Subset(dataset, list(range(count)))


def _load_model():
    if not MODEL_PATH.exists():
        raise unittest.SkipTest(f"Model checkpoint not found at {MODEL_PATH}")
    model = load_model(MODEL_PATH, device=torch.device("cpu"), load_compiled=False)
    model.eval()
    return model


class TestFeatureUtilities(unittest.TestCase):
    def test_normalize_kernel(self):
        self.assertEqual(normalize_kernel("full-g"), "full-gradient")
        self.assertEqual(normalize_kernel("ll-g"), "ll-gradient")
        self.assertEqual(normalize_kernel("local-full-g"), "local_full-gradient")
        self.assertEqual(normalize_kernel("local-gnn"), "local_gnn")

    def test_distance_metrics(self):
        feats = torch.randn(4, 8)
        metrics = DistanceMetrics(regularization=1e-6)
        metrics.fit(feats)
        scores = metrics.score(feats)
        self.assertEqual(scores.shape, (4,))

    def test_h5feature_roundtrip(self):
        feats = torch.randn(3, 5)
        image_idx = torch.tensor([0, 0, 1], dtype=torch.long)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.h5"
            store = H5Feature(path, num_models=1, kernels=["full-gradient"], dataset_size=3)
            store.append("full-gradient", 0, feats, image_idx)
            loaded, counts, idx = store.load_with_counts("full-gradient")
        self.assertEqual(loaded.shape, (1, 3, 5))
        self.assertEqual(counts.tolist(), [3])
        self.assertTrue(torch.equal(idx[0], image_idx))


class TestFeaturePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = _load_dataset()
        cls.model = _load_model()

    def _batch(self):
        batch = [self.dataset[i] for i in range(len(self.dataset))]
        return collate_atomsdata(batch)

    def test_feature_calculator_compute(self):
        calc = FeatureCalculator(
            extractor=FeatureExtractor(repr_callback=self.model),
            kernels=[("full-g", 8)],
        )
        batch = self._batch()
        feats = calc.compute(batch, predict=True)
        self.assertIsInstance(feats, torch.Tensor)
        self.assertEqual(feats.dim(), 2)
        self.assertEqual(feats.shape[0], len(self.dataset))

    def test_feature_statistics(self):
        stats = FeatureStatistics(
            models=[self.model],
            dataset=self.dataset,
            kernels=[("full-g", 8)],
            batch_size=1,
        )
        features = stats.get_features(normalize=False)
        self.assertIn("full-gradient", features)
        self.assertEqual(features["full-gradient"].shape[0], 1)

    def test_active_learning_select(self):
        selector = GeneralActiveLearning(
            models=[self.model],
            kernel="random",
        )
        selected = selector.select(
            pool_set=self.dataset,
            train_set=None,
            select_batch_size=1,
        )
        self.assertEqual(len(selected), 1)


if __name__ == "__main__":
    unittest.main()
