import math
import torch


def create_qkv(x, w_q, w_k, w_v):
    """
    Create Query, Key and Value matrices.

    Args:
        x: Input embeddings (batch, seq_len, d_model)
        w_q, w_k, w_v: Weight matrices

    Returns:
        Q, K, V
    """

    q = x @ w_q
    k = x @ w_k
    v = x @ w_v

    return q, k, v

def compute_scores(q, k):
    """
    Compute scaled attention scores.

    Returns:
        (batch, seq_len, seq_len)
    """

    d_k = q.size(-1)

    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(d_k)

    return scores

def apply_mask(scores, mask):
    """
    Replace masked positions with -inf.
    """

    return scores.masked_fill(mask, float("-inf"))

def stable_softmax(scores):
    """
    Numerically stable softmax.
    """

    max_scores = scores.max(dim=-1, keepdim=True).values
    scores = scores - max_scores

    exp_scores = torch.exp(scores)
    probs = exp_scores / exp_scores.sum(dim=-1, keepdim=True)

    return probs

def dense_attention(x, w_q, w_k, w_v, mask=None):
    """
    Complete dense attention implementation.
    """

    q, k, v = create_qkv(x, w_q, w_k, w_v)

    scores = compute_scores(q, k)

    if mask is not None:
        scores = apply_mask(scores, mask)

    probs = stable_softmax(scores)

    output = probs @ v

    return output, probs