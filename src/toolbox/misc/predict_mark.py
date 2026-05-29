import numpy as np
import torch


def predict_mark(probability, logits=False, sample=False):
    if torch.is_tensor(probability):
        return predict_mark_torch(probability, logits=logits, sample=sample)
    return predict_mark_numpy(probability, logits=logits, sample=sample)


def predict_mark_numpy(probability, logits=False, sample=False):
    """
    Sample event from a (unnormalized) probability distribution using numpy.
    """
    # The shape of the input probability is [..., num_events].
    if sample:
        if logits:
            distribution_of_marks = torch.distributions.categorical.Categorical(logits=torch.from_numpy(probability)) # [...]
        else:
            # https://github.com/pytorch/pytorch/issues/87468
            probability = probability / probability.sum(dim=-1, keepdim=True)  # [...]
            distribution_of_marks = torch.distributions.categorical.Categorical(probs=torch.from_numpy(probability), validate_args=False) # [...]
        sampled_marks = distribution_of_marks.sample().numpy()
    else:
        sampled_marks = np.argmax(probability, axis=-1)  # [...]

    return sampled_marks


def predict_mark_torch(probability, logits=False, sample=False):
    """
    Sample event from a (unnormalized) probability distribution using pytorch.
    """
    # The shape of the input probability is [..., num_events].
    if sample:
        if logits:
            distribution_of_marks = torch.distributions.categorical.Categorical(logits=probability)
        # [...]
        else:
            # https://github.com/pytorch/pytorch/issues/87468
            probability = probability / probability.sum(dim=-1, keepdim=True)  # [...]
            distribution_of_marks = torch.distributions.categorical.Categorical(probs=probability, validate_args=False)
            # [...]
        sampled_marks = distribution_of_marks.sample()
    else:
        sampled_marks = torch.argmax(probability, dim=-1)  # [...]

    return sampled_marks


if __name__ == "__main__":
    # test 1
    probability = np.array([[0.1, 0.2, 0.7], [0.3, 0.5, 0.2]])
    expected = np.array([2, 1])
    result = predict_mark(probability, sample=False)
    assert np.array_equal(result, expected), f"Expected {expected}, got {result}"
    print(result)

    # test 2
    probability = np.array(
        [[[0.1, 0.2, 0.7], [0.3, 0.5, 0.2]], [[0.1, 0.2, 0.7], [0.3, 0.5, 0.2]], [[0.1, 0.2, 0.7], [0.3, 0.5, 0.2]]]
    )
    result = predict_mark(probability, sample=True)
    assert result.shape == (3, 2), "Output shape should be (3, 2)"
    print(result)

    # test 3
    probability = torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.5, 0.2]], device='cuda')
    expected = torch.tensor([2, 1], device='cuda')
    result = predict_mark(probability, sample=False)
    assert torch.equal(result, expected), f"Expected {expected}, got {result}"
    print(result)

    # test 4
    probability = torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.5, 0.2]])
    result = predict_mark(probability, sample=True)
    assert result.shape == (2,), "Output shape should be (2,)"
    print(result)

    # test 5
    logits = np.array([[-np.inf, -np.inf, 0], [-np.inf, 0, -np.inf]])
    expected = np.array([2, 1])
    result = predict_mark(logits, logits=True, sample=False)
    assert np.array_equal(result, expected), f"Expected {expected}, got {result}"
    print(result)

    # test 6
    logits = torch.tensor([[-float("inf"), -float("inf"), 0], [-float("inf"), 0, -float("inf")]])
    expected = torch.tensor([2, 1])
    result = predict_mark(logits, logits=True, sample=False)
    assert torch.equal(result, expected), f"Expected {expected}, got {result}"
    print(result)
