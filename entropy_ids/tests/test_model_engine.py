import unittest
import numpy as np
from models.model_engine import IsolationForestEngine

class TestModelEngine(unittest.TestCase):
    def setUp(self):
        self.model = IsolationForestEngine()
        # Mock 100 samples with 14 features
        self.X_train = np.random.rand(100, 14)

    def test_model_training(self):
        """Tests if the model trains and marks is_fitted successfully."""
        self.assertFalse(self.model.is_fitted)
        self.model.train(self.X_train)
        self.assertTrue(self.model.is_fitted)

    def test_get_scores(self):
        """Tests if get_scores returns expected decision function length."""
        self.model.train(self.X_train)
        scores = self.model.get_scores(self.X_train)
        self.assertEqual(len(scores), 100)
        self.assertTrue(isinstance(scores, np.ndarray))

    def test_get_scores_before_fit(self):
        """Tests exception raise when scoring before fitting."""
        with self.assertRaises(ValueError):
            self.model.get_scores(self.X_train)

if __name__ == "__main__":
    unittest.main()
