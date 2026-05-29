def clamp_preserve_gradients(x, min_val=1e-7, max_val=1 - 1e-7):
    """Clamp the tensor while preserving gradients in the clamped region."""
    return x + (x.clamp(min_val, max_val) - x).detach()


def round_preserve_gradients(x, min_val=1e-7):
    """Clamp the tensor while preserving gradients in the clamped region."""
    x[x < min_val] = 0
    return x


if __name__ == "__main__":
    import torch

    a = torch.zeros(4, 5, 6)
    a = clamp_preserve_gradients(a, min=1e-7, max=1)
    print(a)

    a = torch.ones(4, 5, 6) * 1e-8
    a.grad = torch.ones(4, 5, 6)
    print(a)
    print(a.grad)

    a = round_preserve_gradients(a, min=1e-7)
    print(a)
    print(a.grad)
