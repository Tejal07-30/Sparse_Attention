import torch


def causal_mask(seq_len):
    """
    Create a lower-triangular causal mask.
    """

    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask.bool()