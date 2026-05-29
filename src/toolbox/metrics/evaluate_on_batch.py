from collections.abc import Callable
from multiprocessing import Pool

import numpy as np
import torch
from einops import rearrange

from src.toolbox.metrics.evaluate_func import evaluate_func as func
from src.toolbox.misc import move_from_tensor_to_ndarray


def evaluate_on_one_batch(
    batched_input: np.ndarray | torch.Tensor,
    batched_target: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor,
    evaluate_func: list[Callable | str] | Callable | str,
    dim_input: int = -1,
    dim_target: int = -1,
    dim_mask: int = -1,
    multiprocessing=False,
    additional_inputs: list[np.ndarray | torch.Tensor] | None = None,
    **evaluate_kwargs,
) -> dict[str, np.ndarray] | np.ndarray:
    """Common evaluation functions provided in scikit-learn can only evaluate one sequence per run, while machine learning
    models outputs waiting for evaluation are batches of padded sequences marked by a bool tensor usually called a mask.

    A naive approach is to flatten the 2d model outputs into a 1d sequence with only the prediction
    of true events using the mask then evaluate the 1d sequence. However, this approach makes the evaluation result dependent on
    the batch size, which is undesired.

    To address this issue, this function evaluates each sequence in the batch with mask in mind and returns the
    result of all sequences in a numpy array.

    The evaluate_func is expected to take in a sequence and return a single value or an array with a consistent shape so
    numpy can handle the final result.

    Args:
        batched_input (Union[np.ndarray, torch.Tensor]): the estimated target returned by a model
        batched_target (Union[np.ndarray, torch.Tensor]): the ground truth
        mask (Union[np.ndarray, torch.Tensor]): the mask tensor marking which is a true event and which is a padding one
        evaluate_func (Union[Callable, str]): the evaluation function. It can have two or more inputs: first one is inputs, second one is targets, and additional ones from additional_inputs.
        dim_input (int): where is the seq_len dim of batched_input?
        dim_target (int): where is the seq_len dim of dim_target?
        dim_mask (int): where is the seq_len dim of mask?
        multiprocessing (bool): use multiprocessing or not.
        additional_inputs (Union[list[Union[np.ndarray, torch.Tensor]], None]): additional batched inputs for evaluate_func beyond input and target.
    Returns:
        np.array: the result.
    """
    if torch.is_tensor(batched_input):
        batched_input = move_from_tensor_to_ndarray(batched_input)

    if torch.is_tensor(batched_target):
        batched_target = move_from_tensor_to_ndarray(batched_target)

    if torch.is_tensor(mask):
        mask = move_from_tensor_to_ndarray(mask)

    if additional_inputs is not None:
        additional_inputs = [
            move_from_tensor_to_ndarray(inp) if torch.is_tensor(inp) else inp for inp in additional_inputs
        ]

    if isinstance(evaluate_func, list):
        results = {}
        for item in evaluate_func:
            results[item.__name__ if isinstance(item, Callable) else item] = evaluate_on_one_batch_numpy(
                batched_input,
                batched_target,
                mask,
                item,
                dim_input,
                dim_target,
                dim_mask,
                multiprocessing,
                additional_inputs=additional_inputs,
                **evaluate_kwargs,
            )
        return results

    return evaluate_on_one_batch_numpy(
        batched_input,
        batched_target,
        mask,
        evaluate_func,
        dim_input,
        dim_target,
        dim_mask,
        multiprocessing,
        additional_inputs=additional_inputs,
        **evaluate_kwargs,
    )


def job(
    single_input: np.ndarray,
    single_target: np.ndarray,
    single_mask: np.ndarray,
    evaluate_func: Callable,
    evaluate_kwargs: dict,
    additional_inputs_single: list[np.ndarray] | None = None,
) -> np.ndarray:
    single_input_true_event_picked_by_mask = single_input[single_mask]
    single_target_true_event_picked_by_mask = single_target[single_mask]
    if additional_inputs_single is not None:
        additional_masked = [inp[single_mask] for inp in additional_inputs_single]
        return evaluate_func(
            single_input_true_event_picked_by_mask,
            single_target_true_event_picked_by_mask,
            *additional_masked,
            **evaluate_kwargs,
        )
    return evaluate_func(
        single_input_true_event_picked_by_mask, single_target_true_event_picked_by_mask, **evaluate_kwargs
    )


def evaluate_on_one_batch_numpy(
    batched_input: np.ndarray,
    batched_target: np.ndarray,
    mask: np.ndarray,
    evaluate_func: Callable | str,
    dim_input: int = -1,
    dim_target: int = -1,
    dim_mask: int = -1,
    multiprocessing: bool = False,
    additional_inputs: list[np.ndarray] | None = None,
    **evaluate_kwargs,
) -> np.ndarray:
    """The worker function."""
    if isinstance(evaluate_func, str):
        evaluate_func = func(evaluate_func)

    # Move mask to bool if it is not.
    if mask.dtype != np.bool:
        mask = mask.astype(np.bool)

    batched_input_shape = batched_input.shape
    batched_target_shape = batched_target.shape
    mask_shape = mask.shape
    # The batch size should be the same.
    if not batched_input_shape[:dim_input] == batched_target_shape[:dim_target] == mask_shape[:dim_mask]:
        raise ValueError("Bad input shape.")

    batch_size = batched_input_shape[:dim_input]
    einop = f"{' '.join([f'a{index}' for index in range(len(batch_size))])} ... -> ({' '.join([f'a{index}' for index in range(len(batch_size))])}) ..."
    reversed_einop = f"({' '.join([f'a{index}' for index in range(len(batch_size))])}) ... -> {' '.join([f'a{index}' for index in range(len(batch_size))])} ..."
    batched_input = rearrange(batched_input, einop)  # [(...), seq_len, ...]
    batched_target = rearrange(batched_target, einop)  # [(...), seq_len, ...]
    mask = rearrange(mask, einop)  # [(...), seq_len, ...]

    if additional_inputs is not None:
        additional_inputs_rearranged = [rearrange(inp, einop) for inp in additional_inputs]

    result_list = []
    if multiprocessing:
        pool = Pool(4)
        if additional_inputs is not None:
            additional_single_list = list(zip(*additional_inputs_rearranged))
            result_list = pool.starmap(
                job,
                [
                    (single_input, single_target, single_mask, evaluate_func, evaluate_kwargs, additional_single)
                    for single_input, single_target, single_mask, additional_single in zip(
                        batched_input, batched_target, mask, additional_single_list
                    )
                ],
            )
        else:
            result_list = pool.starmap(
                job,
                [
                    (single_input, single_target, single_mask, evaluate_func, evaluate_kwargs)
                    for single_input, single_target, single_mask in zip(batched_input, batched_target, mask)
                ],
            )
    else:
        if additional_inputs is not None:
            for single_input, single_target, single_mask, *additional_single in zip(
                batched_input, batched_target, mask, *additional_inputs_rearranged
            ):
                single_input_true_event_picked_by_mask = single_input[single_mask]
                single_target_true_event_picked_by_mask = single_target[single_mask]
                additional_masked = [inp[single_mask] for inp in additional_single]
                result_list.append(
                    evaluate_func(
                        single_input_true_event_picked_by_mask,
                        single_target_true_event_picked_by_mask,
                        *additional_masked,
                        **evaluate_kwargs,
                    )
                )
        else:
            for single_input, single_target, single_mask in zip(batched_input, batched_target, mask):
                single_input_true_event_picked_by_mask = single_input[single_mask]
                single_target_true_event_picked_by_mask = single_target[single_mask]
                result_list.append(
                    evaluate_func(
                        single_input_true_event_picked_by_mask,
                        single_target_true_event_picked_by_mask,
                        **evaluate_kwargs,
                    )
                )

    result_list = np.array(result_list)  # [(...), ...]
    dim_prior_seq_len = {}
    for idx, item in enumerate(batched_input_shape[:dim_input]):
        dim_prior_seq_len[f"a{idx}"] = item
    return rearrange(result_list, reversed_einop, **dim_prior_seq_len)  # [..., ...]


if __name__ == "__main__":
    import time

    from sklearn.metrics import accuracy_score

    def acc1(input, target):
        return accuracy_score(y_pred=input, y_true=target)

    # case 1
    y_pred = np.array([0, 2, 1, 3])
    y_true = np.array([0, 1, 2, 3])
    mask = np.array([1, 1, 1, 1])
    result = evaluate_on_one_batch(
        y_pred,
        y_true,
        mask,
        [
            acc1,
        ],
    )
    print(f"case 1: {result}")

    # case 2
    y_pred = np.array([[0, 2, 1, 3], [0, 2, 1, 3]])
    y_true = np.array([[0, 1, 2, 3], [0, 1, 2, 3]])
    mask = np.array([[1, 1, 1, 1], [0, 1, 1, 1]])
    result = evaluate_on_one_batch(y_pred, y_true, mask, acc1)
    print(f"case 2: {result}")

    # case 3
    y_pred = np.array(
        [
            [[0, 2, 1, 3], [0, 2, 1, 3], [0, 2, 1, 3], [0, 2, 1, 3]],
            [[0, 2, 1, 3], [0, 2, 1, 3], [0, 2, 1, 3], [0, 2, 1, 3]],
        ]
    )
    y_true = np.array(
        [
            [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]],
            [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]],
        ]
    )
    mask = np.array(
        [
            [[1, 1, 1, 1], [0, 1, 1, 1], [1, 1, 1, 0], [0, 1, 1, 0]],
            [[1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 0, 0, 1]],
        ]
    )
    result = evaluate_on_one_batch(y_pred, y_true, mask, acc1)
    print(f"case 3: {result}")

    # case 4
    def acc(input, target, a):
        print(a)
        return accuracy_score(y_pred=input, y_true=target)

    y_pred = np.array([[[0, 2, 1, 3], [0, 2, 1, 3], [0, 2, 1, 3], [0, 2, 1, 3]] for _ in range(32)])
    y_true = np.array([[[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]] for _ in range(32)])
    mask = np.array([[[1, 1, 1, 1], [0, 1, 1, 1], [1, 1, 1, 0], [0, 1, 1, 0]] for _ in range(32)])
    start_time = time.time()
    result_slow = evaluate_on_one_batch(y_pred, y_true, mask, [acc, "acc", "micro-f1"], multiprocessing=False, a=12)
    exec_time_slow = time.time() - start_time
    print(f"exec_time_slow: {exec_time_slow}s")

    start_time = time.time()
    result_fast = evaluate_on_one_batch(y_pred, y_true, mask, [acc, "acc", "micro-f1"], multiprocessing=True, a=12)
    exec_time_fast = time.time() - start_time
    # assert (result_slow == result_fast).all()

    # print(f'case 4: {result_fast}')
    print(f"exec_time_fast: {exec_time_fast}s")

    # case 5: test with additional inputs for L1 metric
    from src.toolbox.metrics.l1 import L1_distance_between_two_funcs

    def l1_wrapper(input, target, timestamp):
        return L1_distance_between_two_funcs(input, target, timestamp)

    # Simple test data: seq_len first, then resolution
    seq_len = 4
    resolution = 3
    batch_size = 2

    # batched_input: [batch_size, seq_len, resolution]
    batched_input = np.random.rand(batch_size, seq_len, resolution)
    batched_target = np.random.rand(batch_size, seq_len, resolution)
    mask = np.ones((batch_size, seq_len), dtype=bool)
    # timestamp: [batch_size, seq_len, resolution]
    timestamp = np.random.rand(batch_size, seq_len, resolution)

    result_l1 = evaluate_on_one_batch(
        batched_input,
        batched_target,
        mask,
        l1_wrapper,
        dim_input=1,
        dim_target=1,
        dim_mask=1,
        additional_inputs=[timestamp],
    )
    print(f"case 5 L1 result shape: {result_l1.shape}")
    print(f"case 5 L1 result: {result_l1}")

    # Test with different batch dimensions
    batched_input_3d = np.random.rand(2, 3, seq_len, resolution)
    batched_target_3d = np.random.rand(2, 3, seq_len, resolution)
    mask_3d = np.ones((2, 3, seq_len), dtype=bool)
    timestamp_3d = np.random.rand(2, 3, seq_len, resolution)

    result_l1_3d = evaluate_on_one_batch(
        batched_input_3d,
        batched_target_3d,
        mask_3d,
        l1_wrapper,
        dim_input=2,
        dim_target=2,
        dim_mask=2,
        additional_inputs=[timestamp_3d],
    )
    print(f"case 5 L1 3D result shape: {result_l1_3d.shape}")
    print(f"case 5 L1 3D result: {result_l1_3d}")
