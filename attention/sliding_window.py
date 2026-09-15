import torch
from attention.dense import dense_attention

def sliding_window_mask(seq_len, window_size):
    """
    Create a sliding-window attention mask.

    True  -> masked
    False -> allowed
    """

    mask = torch.ones(seq_len, seq_len, dtype=torch.bool)

    for i in range(seq_len):
        left = max(0, i - window_size)
        right = min(seq_len, i + window_size + 1)

        mask[i, left:right] = False

    return mask


def sliding_window_attention(x, w_q, w_k, w_v, window_size):
    seq_len = x.size(1)

    mask = sliding_window_mask(seq_len, window_size)

    output, probs = dense_attention(
        x,
        w_q,
        w_k,
        w_v,
        mask=mask
    )

    return output, probs


torch.manual_seed(42)

batch = 1
seq_len = 8
d_model = 8
d_head = 4

x = torch.randn(batch, seq_len, d_model)

w_q = torch.randn(d_model, d_head)
w_k = torch.randn(d_model, d_head)
w_v = torch.randn(d_model, d_head)

output, probs = sliding_window_attention(
    x,
    w_q,
    w_k,
    w_v,
    window_size=2,
)

print(output.shape)
print(probs.shape)

mask = sliding_window_mask(seq_len,2)

masked_probs = probs[0][mask]

print(masked_probs.max())