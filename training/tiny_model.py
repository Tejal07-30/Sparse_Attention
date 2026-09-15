import sys
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attention.dense import dense_attention


class AttentionHead(nn.Module):

    def __init__(self, d_model, head_size):

        super().__init__()

        self.wq = nn.Parameter(torch.randn(d_model, head_size) * 0.02)
        self.wk = nn.Parameter(torch.randn(d_model, head_size) * 0.02)
        self.wv = nn.Parameter(torch.randn(d_model, head_size) * 0.02)
        self.proj = nn.Linear(head_size, d_model)

    def forward(self, x):

        out, _ = dense_attention(
            x,
            self.wq,
            self.wk,
            self.wv,
        )

        return self.proj(out)

class FeedForward(nn.Module):

    def __init__(self, d_model):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):

    def __init__(self, d_model, head_size):

        super().__init__()

        self.attn = AttentionHead(d_model, head_size)
        self.ff = FeedForward(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
    def forward(self, x):

        # attention + skip connection
        x = x + self.attn(self.ln1(x))
        # feed forward + skip connection
        x = x + self.ff(self.ln2(x))

        return x

class TinyGPT(nn.Module):

    def __init__(self, vocab_size, block_size, d_model=64):

        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.blocks = nn.Sequential(
            TransformerBlock(d_model, 32),
            TransformerBlock(d_model, 32),
        )
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):

        B, T = idx.shape
        tok = self.token_embedding(idx)
        pos = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok + pos
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):

            idx_crop = idx[:, -64:]
            logits = self(idx_crop)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            next_char = torch.multinomial(probs, 1)
            idx = torch.cat((idx, next_char), dim=1)
        return idx