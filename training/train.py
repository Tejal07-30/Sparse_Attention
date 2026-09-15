import sys
from pathlib import Path

#For python can find attention and training
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
''' 
Imports used for testing :
from training.tiny_model import AttentionHead
from training.tiny_model import TransformerBlock
'''
from training.tiny_model import TinyGPT
import torch.nn.functional as F

text = Path("training/data/input.txt").read_text()

print(text[:300])
print()
print("characters:", len(text))

chars = sorted(set(text))

stoi = {}
itos = {}

for i, ch in enumerate(chars):
    stoi[ch] = i
    itos[i] = ch

def encode(s):
    return [stoi[c] for c in s]

def decode(nums):
    return "".join(itos[n] for n in nums)

print("vocab size:", len(chars))
print(encode("KING"))

block_size = 64
batch_size = 16
d_model = 64

data = torch.tensor(encode(text), dtype=torch.long)

def get_batch():

    starts = torch.randint(len(data) - block_size - 1, (batch_size,))

    x = []
    y = []

    for s in starts:
        x.append(data[s:s + block_size])
        y.append(data[s + 1:s + block_size + 1])

    return torch.stack(x), torch.stack(y)

#checking one sample before training

sample = data[:block_size]
print()
print("sample:")
print(decode(sample.tolist()))

#model

model = TinyGPT(len(chars), block_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

#first batch

x, y = get_batch()
logits = model(x)
print()
print("batch shape:", x.shape)
print("logits:", logits.shape)
loss = F.cross_entropy(
    logits.view(-1, len(chars)),
    y.view(-1),
)
print("starting loss:", loss.item())

#training 

steps = 500
for step in range(steps):

    x, y = get_batch()

    logits = model(x)

    loss = F.cross_entropy(
        logits.view(-1, len(chars)),
        y.view(-1),
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"step {step}: loss = {loss.item():.4f}")

print("\nGenerating...\n")
start = torch.zeros((1, 1), dtype=torch.long)
result = model.generate(start, 300)
print(decode(result[0].tolist()))