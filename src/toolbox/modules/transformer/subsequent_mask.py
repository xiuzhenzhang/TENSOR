import torch


def get_subsequent_mask(seq_len, device):
    """For masking out the subsequent info, i.e., masked self-attention.
    In our case, 1 means item keeped, and 0 means item masked.
    """

    subsequent_mask = torch.triu(torch.ones((seq_len, seq_len), device=device, dtype=torch.uint8)).T
    # [seq_len, seq_len]
    return subsequent_mask.unsqueeze(dim=0)  # [1, seq_len, seq_len]
