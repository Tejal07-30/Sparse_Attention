import torch
from attention.sliding_window import sliding_window_attention

torch.manual_seed(0)

x = torch.randn(1,8,8)

wq = torch.randn(8,4)
wk = torch.randn(8,4)
wv = torch.randn(8,4)

out, probs = sliding_window_attention(x,wq,wk,wv,2)

print(out.shape)
print(probs.shape)
print(probs[0].sum(dim=1))