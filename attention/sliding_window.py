import torch
from attention.dense import dense_attention


def sliding_window_mask(seq_len, window):

    mask = torch.ones(seq_len, seq_len, dtype=torch.bool)

    for i in range(seq_len):

        # not to go outside the sequence
        left = max(0, i - window)
        right = min(seq_len, i + window + 1)

        mask[i, left:right] = False

    return mask


def sliding_window_attention(x, wq, wk, wv, window):

    mask = sliding_window_mask(x.size(1), window)

    out, probs = dense_attention(x, wq, wk, wv, mask)

    return out, probs