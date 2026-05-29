import torch


def bisection_torch(
    max_step,
    bisect_early_stop_threshold,
    bisect_func,
    threshold,
    *args,
    l_val=0.0001,
    r_val=1e6,
    **kwargs,
):
    """
    Bisection Method when the inputs are torch.tensors.
    """
    left = l_val * torch.ones_like(threshold)
    right = r_val * torch.ones_like(threshold)

    for _ in range(max_step):
        center = (left + right) / 2
        val = bisect_func(center, threshold, *args, **kwargs)
        left = torch.where(val < 0, center, left)
        right = torch.where(val > 0, center, right)
        if torch.allclose(right, left, atol=bisect_early_stop_threshold, rtol=0):
            break

    return (left + right) / 2
