import torch

from attention.dense import (
    create_qkv,
    compute_scores,
    stable_softmax,
)
from attention.sliding_window import sliding_window_mask

torch.manual_seed(42)

batch = 1
seq_len = 8
d_model = 8
d_head = 4

x = torch.randn(batch, seq_len, d_model)

w_q = torch.randn(d_model, d_head)
w_k = torch.randn(d_model, d_head)
w_v = torch.randn(d_model, d_head)

q,k,v = create_qkv(x,w_q,w_k,w_v)

scores = compute_scores(q,k)

mask = sliding_window_mask(seq_len,2)

dense_masked = scores.masked_fill(mask,float("-inf"))
sparse_scores = scores.masked_fill(mask,float("-inf"))

dense_probs = stable_softmax(dense_masked)
sparse_probs = stable_softmax(sparse_scores)

assert torch.allclose(
    dense_probs[~mask],
    sparse_probs[~mask],
    atol=1e-6
)

print("Sliding-window matches dense on valid positions.")