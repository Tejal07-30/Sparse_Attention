import math
import torch


def create_qkv(x, wq, wk, wv):
   
    q = x @ wq
    k = x @ wk
    v = x @ wv

    return q, k, v

def compute_scores(q, k):
   
    dk = q.size(-1)

    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(dk)

    return scores

def apply_mask(scores, mask):
   
    return scores.masked_fill(mask, float("-inf"))

def stable_softmax(scores):

    max_scores = scores.max(dim=-1, keepdim=True).values

    # if the whole row is -inf, max becomes -inf
    max_scores[max_scores == float("-inf")] = 0

    scores = scores - max_scores

    exp_scores = torch.exp(scores)

    denom = exp_scores.sum(dim=-1, keepdim=True)

    # avoid 0/0
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)

    return exp_scores / denom

def dense_attention(x, wq, wk, wv, mask=None):
   
    q, k, v = create_qkv(x, wq, wk, wv)

    scores = compute_scores(q, k)

    if mask is not None:
        scores = apply_mask(scores, mask)

    probs = stable_softmax(scores)

    output = probs @ v

    return output, probs