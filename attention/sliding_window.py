import math
import torch


def sliding_window_attention(x, wq, wk, wv, window):

    q = x @ wq
    k = x @ wk
    v = x @ wv

    batch, seq_len, d = q.shape

    out = torch.zeros_like(v)
    probs = torch.zeros(batch, seq_len, seq_len)

    for i in range(seq_len):

        left = max(0, i - window)
        right = min(seq_len, i + window + 1)

        q_now = q[:, i:i+1, :]
        k_now = k[:, left:right, :]
        v_now = v[:, left:right, :]

        scores = (q_now @ k_now.transpose(-2, -1)) / math.sqrt(d)

        attn = torch.softmax(scores, dim=-1)

        out[:, i:i+1, :] = attn @ v_now

        probs[:, i, left:right] = attn.squeeze(1)

    return out, probs