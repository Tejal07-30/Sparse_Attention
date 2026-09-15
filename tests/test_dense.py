import torch
from attention.dense import dense_attention
from attention.masks import causal_mask

torch.manual_seed(42)

batch = 1
seq_len = 4
d_model = 8
d_head = 4

x = torch.randn(batch, seq_len, d_model)

w_q = torch.randn(d_model, d_head)
w_k = torch.randn(d_model, d_head)
w_v = torch.randn(d_model, d_head)

mask = causal_mask(seq_len)

output, probs = dense_attention(x, w_q, w_k, w_v, mask)

print("Output shape:", output.shape)
print("Attention shape:", probs.shape)
print("Row sums:", probs.sum(dim=-1))

def compare_with_reference():
    q = x @ w_q
    k = x @ w_k
    v = x @ w_v

    ref_scores = q @ k.transpose(-2, -1)
    ref_scores /= d_head ** 0.5
    ref_scores = ref_scores.masked_fill(mask, float("-inf"))

    ref_probs = torch.softmax(ref_scores, dim=-1)
    ref_output = ref_probs @ v

    assert torch.allclose(output, ref_output, atol=1e-6)

compare_with_reference()
print("Dense attention matches reference.")