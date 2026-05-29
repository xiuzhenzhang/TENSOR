import numpy as np


def break_batched_inputs_into_seqs(mask: np.ndarray, *args: list[np.ndarray]) -> list[list[np.ndarray]]:
    """break the batched input into sequences. In other words, break the input with batch_size=n to 1.

    Args:
        mask (np.ndarray): The mask tensor telling us which events are real and which are padding events.

    Returns:
        list[list[np.ndarray]]: The results.
    """
    results = [[] for _ in range(len(args))]
    for batch_idx in range(len(mask)):
        mask_row = mask[batch_idx]
        for arg_idx, arg_list in enumerate(args):
            item = arg_list[batch_idx]
            results[arg_idx].append(item[mask_row])

    return results

if __name__ == "__main__":
    import numpy as np

    def test_basic_functionality():
        # Test case 1: Basic functionality with two arrays
        mask = np.array([[True, False, True], [False, True, False]])
        arr1 = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        arr2 = [np.array([10, 20, 30]), np.array([40, 50, 60])]

        result = break_batched_inputs_into_seqs(mask, arr1, arr2)

        expected = [
            [np.array([1, 3]), np.array([5])],
            [np.array([10, 30]), np.array([50])]
        ]

        assert len(result) == 2
        assert len(result[0]) == 2
        assert len(result[1]) == 2
        assert np.array_equal(result[0][0], expected[0][0])
        assert np.array_equal(result[0][1], expected[0][1])
        assert np.array_equal(result[1][0], expected[1][0])
        assert np.array_equal(result[1][1], expected[1][1])
        print("Test case 1 passed")

    def test_single_batch():
        # Test case 2: Single batch
        mask = np.array([[True, False, True]])
        arr1 = [np.array([1, 2, 3])]
        arr2 = [np.array([10, 20, 30])]

        result = break_batched_inputs_into_seqs(mask, arr1, arr2)

        expected = [
            [np.array([1, 3])],
            [np.array([10, 30])]
        ]

        assert len(result) == 2
        assert len(result[0]) == 1
        assert np.array_equal(result[0][0], expected[0][0])
        assert np.array_equal(result[1][0], expected[1][0])
        print("Test case 2 passed")

    def test_no_padding():
        # Test case 3: No padding, all True
        mask = np.array([[True, True, True], [True, True, True]])
        arr1 = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        arr2 = [np.array([10, 20, 30]), np.array([40, 50, 60])]

        result = break_batched_inputs_into_seqs(mask, arr1, arr2)

        expected = [
            [np.array([1, 2, 3]), np.array([4, 5, 6])],
            [np.array([10, 20, 30]), np.array([40, 50, 60])]
        ]

        assert len(result) == 2
        assert np.array_equal(result[0][0], expected[0][0])
        assert np.array_equal(result[0][1], expected[0][1])
        assert np.array_equal(result[1][0], expected[1][0])
        assert np.array_equal(result[1][1], expected[1][1])
        print("Test case 3 passed")

    def test_all_padding():
        # Test case 4: All padding, all False
        mask = np.array([[False, False, False], [False, False, False]])
        arr1 = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        arr2 = [np.array([10, 20, 30]), np.array([40, 50, 60])]

        result = break_batched_inputs_into_seqs(mask, arr1, arr2)

        expected = [
            [np.array([]), np.array([])],
            [np.array([]), np.array([])]
        ]

        assert len(result) == 2
        assert len(result[0][0]) == 0
        assert len(result[0][1]) == 0
        assert len(result[1][0]) == 0
        assert len(result[1][1]) == 0
        print("Test case 4 passed")

    def test_multiple_args():
        # Test case 5: More than two args
        mask = np.array([[True, False], [False, True]])
        arr1 = [np.array([1, 2]), np.array([3, 4])]
        arr2 = [np.array([10, 20]), np.array([30, 40])]
        arr3 = [np.array([100, 200]), np.array([300, 400])]

        result = break_batched_inputs_into_seqs(mask, arr1, arr2, arr3)

        expected = [
            [np.array([1]), np.array([4])],
            [np.array([10]), np.array([40])],
            [np.array([100]), np.array([400])]
        ]

        assert len(result) == 3
        assert np.array_equal(result[0][0], expected[0][0])
        assert np.array_equal(result[0][1], expected[0][1])
        assert np.array_equal(result[1][0], expected[1][0])
        assert np.array_equal(result[1][1], expected[1][1])
        assert np.array_equal(result[2][0], expected[2][0])
        assert np.array_equal(result[2][1], expected[2][1])
        print("Test case 5 passed")

    # Run all tests
    test_basic_functionality()
    test_single_batch()
    test_no_padding()
    test_all_padding()
    test_multiple_args()

    print("All tests passed!")