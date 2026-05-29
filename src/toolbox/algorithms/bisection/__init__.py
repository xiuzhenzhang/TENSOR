import torch

from src.toolbox.algorithms.bisection.bisection_numpy import bisection_numpy
from src.toolbox.algorithms.bisection.bisection_torch import bisection_torch


def bisection(max_step, bisect_early_stop_threshold, bisect_func, threshold, *args, **kwargs):
    if torch.is_tensor(threshold):
        return bisection_torch(max_step, bisect_early_stop_threshold, bisect_func, threshold, *args, **kwargs)
    return bisection_numpy(max_step, bisect_early_stop_threshold, bisect_func, threshold, *args, **kwargs)


if __name__ == '__main__':
    import unittest

    import numpy as np
    import torch

    # Test function for bisection methods
    def test_func(x, prob_threshold):
        return x - prob_threshold

    class TestBisectionMethods(unittest.TestCase):

        def test_bisection_numpy_1(self):
            """Test basic bisection with simple function"""
            probability_threshold = np.array([0.5])
            result = bisection_numpy(100, 1e-6, test_func, probability_threshold, l_val=0, r_val=1)
            self.assertAlmostEqual(result[0], 0.5, places=6)

        def test_bisection_torch_1(self):
            """Test basic bisection with simple function"""
            probability_threshold = torch.tensor([0.5])
            result = bisection_torch(100, 1e-6, test_func, probability_threshold, l_val=0, r_val=1)
            self.assertAlmostEqual(result.item(), 0.5, places=6)

        def test_bisection_numpy_2(self):
            """Test with multiple values"""
            probability_threshold = np.array([0.1, 0.3, 0.7])
            result = bisection_numpy(100, 1e-6, test_func, probability_threshold, l_val=0, r_val=1)
            self.assertTrue(np.allclose(result, probability_threshold, atol=1e-6))

        def test_bisection_torch_2(self):
            """Test with multiple values"""
            probability_threshold = torch.tensor([0.1, 0.3, 0.7])
            result = bisection_torch(100, 1e-6, test_func, probability_threshold, l_val=0, r_val=1)
            self.assertTrue(torch.allclose(result, probability_threshold, atol=1e-6))

        def test_bisection_numpy_3(self):
            """Test with different range"""
            probability_threshold = np.array([0.9])
            result = bisection_numpy(100, 1e-6, test_func, probability_threshold, l_val=0.5, r_val=1)
            self.assertAlmostEqual(result[0], 0.9, places=6)

        def test_bisection_torch_3(self):
            """Test with different range"""
            probability_threshold = torch.tensor([0.9])
            result = bisection_torch(100, 1e-6, test_func, probability_threshold, l_val=0.5, r_val=1)
            self.assertAlmostEqual(result.item(), 0.9, places=6)

    unittest.main()