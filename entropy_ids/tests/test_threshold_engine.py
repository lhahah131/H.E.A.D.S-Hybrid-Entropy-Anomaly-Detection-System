import unittest
import numpy as np
from thresholds.threshold_engine import AdaptiveThreshold, StrictThreshold

class TestAdaptiveThreshold(unittest.TestCase):
    def test_adaptive_percentile(self):
        """Tests calculation of Percentile on array."""
        thresh = AdaptiveThreshold(percentile=50)
        # Using 100 values from 1 to 100, median is 50.5
        scores = np.arange(1, 101)
        t, scaler = thresh.compute(scores)
        self.assertEqual(t, 50.5)
        self.assertIsNone(scaler)

class TestStrictThreshold(unittest.TestCase):
    def test_strict_roc(self):
        """Tests strict ROC extraction limits."""
        thresh = StrictThreshold(multiplier=1.0)
        
        # Fake scores between -1 and 1
        scores = np.array([-0.9, -0.5, 0.0, 0.5, 0.9])
        
        # Ground truths (normal, anomali)
        y_true = np.array([0, 0, 1, 1, 1])
        
        t, scaler = thresh.compute(scores, y_true)
        # Verify scalar is built and threshold is float
        self.assertIsNotNone(scaler)
        self.assertTrue(isinstance(t, (float, np.float64, np.float32)))

    def test_strict_one_class(self):
        """Tests fallback mechanism if only 1 class is present."""
        thresh = StrictThreshold(multiplier=1.2)
        scores = np.random.rand(10)
        y_true = np.array([1] * 10)  # All identical
        
        t, scaler = thresh.compute(scores, y_true)
        # Default ROC output returns 0.5 * multiplier
        self.assertEqual(t, 0.5 * 1.2)

if __name__ == "__main__":
    unittest.main()
