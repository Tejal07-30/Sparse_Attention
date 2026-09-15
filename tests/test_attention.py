import math
import torch

from attention.dense import dense_attention
from attention.sliding_window import sliding_window_attention
from attention.bigbird import bigbird_attention


torch.manual_seed(0)

x = torch.randn(1, 8, 8)

wq = torch.randn(8, 4)
wk = torch.randn(8, 4)
wv = torch.randn(8, 4)


def test_dense_shape():

    out, probs = dense_attention(x, wq, wk, wv)

    assert out.shape == (1, 8, 4)
    assert probs.shape == (1, 8, 8)


def test_sliding_window_shape():

    out, probs = sliding_window_attention(x, wq, wk, wv, 2)

    assert out.shape == (1, 8, 4)
    assert probs.shape == (1, 8, 8)


def test_bigbird_shape():

    out, probs = bigbird_attention(x, wq, wk, wv, 2)

    assert out.shape == (1, 8, 4)
    assert probs.shape == (1, 8, 8)


def test_sparse_matches_dense():

    dense_out, _ = dense_attention(x, wq, wk, wv)

    sparse_out, _ = sliding_window_attention(x, wq, wk, wv, 7)

    # with window=7 every token sees everything
    assert torch.allclose(dense_out, sparse_out, atol=1e-6)

def test_nan_case():

    scores = torch.tensor([
        [-float("inf"), -float("inf"), -float("inf")]
    ])

    probs = torch.softmax(scores, dim=-1)

    assert torch.isnan(probs).all()

from attention.dense import stable_softmax


def test_safe_softmax():

    scores = torch.tensor([
        [-float("inf"), -float("inf"), -float("inf")]
    ])

    probs = stable_softmax(scores)

    assert not torch.isnan(probs).any()