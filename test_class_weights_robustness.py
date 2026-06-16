import unittest
import numpy as np

def calculate_class_weights(counts, total_samples):
    # This is the exact logic from train.py
    counts = counts.astype(np.float64)
    avg_2_3 = (counts[2] + counts[3]) / 2.0
    counts[2] = avg_2_3
    counts[3] = avg_2_3
    
    avg_4_5 = (counts[4] + counts[5]) / 2.0
    counts[4] = avg_4_5
    counts[5] = avg_4_5
    
    counts = np.maximum(counts, 1)
    raw_weights = total_samples / (7.0 * counts)
    clipped_weights = np.clip(raw_weights, 0.5, 3.0)
    return raw_weights, clipped_weights

class TestClassWeightsRobustness(unittest.TestCase):
    def test_normal_case(self):
        counts = np.array([100, 200, 150, 250, 50, 150, 100])
        total = int(np.sum(counts))
        raw, clipped = calculate_class_weights(counts, total)
        
        # Symmetry
        self.assertEqual(clipped[2], clipped[3])
        self.assertEqual(clipped[4], clipped[5])
        
        # Clipping bounds
        for w in clipped:
            self.assertTrue(0.5 <= w <= 3.0)
            
    def test_zero_counts(self):
        counts = np.zeros(7, dtype=int)
        total = 0
        raw, clipped = calculate_class_weights(counts, total)
        
        # No division by zero, all weights should be clipped to 0.5
        for w in clipped:
            self.assertEqual(w, 0.5)
            
    def test_extreme_skew(self):
        # 1 sample in class 0, 1 million in class 1
        counts = np.array([1, 1000000, 0, 0, 0, 0, 0])
        total = int(np.sum(counts))
        raw, clipped = calculate_class_weights(counts, total)
        
        # Class 0 has 1 count -> raw weight is total / (7 * 1) = 142857.28 -> clipped to 3.0
        self.assertEqual(clipped[0], 3.0)
        # Class 1 has 1M counts -> raw weight is total / (7 * 1M) = 1/7 = 0.1428 -> clipped to 0.5
        self.assertEqual(clipped[1], 0.5)
        
    def test_one_count_symmetric_classes(self):
        # Only class 2 has a sample, class 3 has 0
        counts = np.array([0, 0, 1, 0, 0, 0, 0])
        total = 1
        raw, clipped = calculate_class_weights(counts, total)
        
        # avg should be 0.5 for class 2 and 3
        # then np.maximum makes it 1.0
        # raw weight = 1 / (7 * 1) = 0.1428 -> clipped to 0.5
        self.assertEqual(clipped[2], 0.5)
        self.assertEqual(clipped[3], 0.5)

if __name__ == '__main__':
    unittest.main()
