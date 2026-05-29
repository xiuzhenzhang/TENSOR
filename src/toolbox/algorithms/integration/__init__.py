import torch

from src.toolbox.algorithms.integration.integration_numpy import approximate_integration_numpy
from src.toolbox.algorithms.integration.integration_torch import approximate_integration_torch


def approximate_integration(expanded_func_value, expanded_x, dim, only_integral = False, func_val_x_having_same_shape = False):
    if torch.is_tensor(expanded_func_value) and torch.is_tensor(expanded_x):
        return approximate_integration_torch(expanded_func_value, expanded_x, dim, only_integral, func_val_x_having_same_shape)
    return approximate_integration_numpy(expanded_func_value, expanded_x, dim, only_integral, func_val_x_having_same_shape)


if __name__ == '__main__':
    import unittest

    import numpy as np
    import torch

    class TestIntegration(unittest.TestCase):

        def test_integration_numpy_1(self):
            """Test basic integration with numpy arrays"""
            resolution = 11
            x = np.arange(0, resolution)
            func1 = 2 * x
            func2 = 3 * x
            func3 = 4 * x

            # Test basic integration
            L1 = approximate_integration(func1, x, dim=-1, only_integral=True)
            self.assertAlmostEqual(L1, 100.0, places=6)

            # Test full integration
            L1 = approximate_integration(func1, x, dim=-1)
            expected = np.array([0., 1., 4., 9., 16., 25., 36., 49., 64., 81., 100.])
            self.assertTrue(np.allclose(L1, expected, atol=1e-6))

            # Test matrix integration
            func = np.stack([func1, func2, func3], axis=-1)
            l1_matrix = approximate_integration(func, x, dim=0)
            expected_matrix = np.array([[0., 0., 0.],
                                       [1., 1.5, 2.],
                                       [4., 6., 8.],
                                       [9., 13.5, 18.],
                                       [16., 24., 32.],
                                       [25., 37.5, 50.],
                                       [36., 54., 72.],
                                       [49., 73.5, 98.],
                                       [64., 96., 128.],
                                       [81., 121.5, 162.],
                                       [100., 150., 200.]])
            self.assertTrue(np.allclose(l1_matrix, expected_matrix, atol=1e-6))

        def test_integration_torch_1(self):
            """Test basic integration with torch tensors"""
            resolution = 11
            x = np.arange(0, resolution)
            func1 = 2 * x
            func2 = 3 * x
            func3 = 4 * x

            # Test basic integration
            L1 = approximate_integration(func1, x, dim=-1, only_integral=True)
            self.assertAlmostEqual(L1, 100.0, places=6)

            # Test full integration
            L1 = approximate_integration(func1, x, dim=-1)
            expected = np.array([0., 1., 4., 9., 16., 25., 36., 49., 64., 81., 100.])
            self.assertTrue(np.allclose(L1, expected, atol=1e-6))

            # Test matrix integration
            func = np.stack([func1, func2, func3], axis=-1)
            l1_matrix = approximate_integration(func, x, dim=0)
            expected_matrix = np.array([[0., 0., 0.],
                                       [1., 1.5, 2.],
                                       [4., 6., 8.],
                                       [9., 13.5, 18.],
                                       [16., 24., 32.],
                                       [25., 37.5, 50.],
                                       [36., 54., 72.],
                                       [49., 73.5, 98.],
                                       [64., 96., 128.],
                                       [81., 121.5, 162.],
                                       [100., 150., 200.]])
            self.assertTrue(np.allclose(l1_matrix, expected_matrix, atol=1e-6))

        def test_integration_numpy_2(self):
            """Test complex integration with numpy arrays"""
            resolution = 11
            x = np.arange(0, resolution)
            func1 = 2 * x
            func2 = 3 * x
            func3 = 4 * x

            # Test complex integration
            func = np.stack([func1, func2, func3], axis=-1)
            l1_matrix = approximate_integration(func, x, dim=0, only_integral=True)
            expected_values = [100., 150., 200.]
            self.assertTrue(np.allclose(l1_matrix, expected_values, atol=1e-6))

        def test_integration_torch_2(self):
            """Test complex integration with torch tensors"""
            resolution = 11
            x = np.arange(0, resolution)
            func1 = 2 * x
            func2 = 3 * x
            func3 = 4 * x

            # Test complex integration
            func = torch.stack([torch.from_numpy(func) for func in [func1, func2, func3]], axis=-1)
            l1_matrix = approximate_integration(func, x, dim=0, only_integral=True)
            expected_values = [100., 150., 200.]
            self.assertTrue(np.allclose(l1_matrix, expected_values, atol=1e-6))

        def test_integration_numpy_3(self):
            """Test edge cases with numpy arrays"""
            resolution = 5
            x = np.arange(0, resolution)
            func1 = 2 * x
            func2 = 3 * x

            # Test simple integration
            func = np.stack([func1, func2], axis=-1)
            l1_matrix = approximate_integration(func, x, dim=0)
            expected_matrix = np.array([[0., 0.],
                                       [1., 1.5],
                                       [4., 6.],
                                       [9., 13.5],
                                       [16., 24.]])
            self.assertTrue(np.allclose(l1_matrix, expected_matrix, atol=1e-6))

    unittest.main()
