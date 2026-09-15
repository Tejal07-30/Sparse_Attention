import sys
import time
from pathlib import Path
#For python to see attention files
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import torch

from attention.dense import dense_attention
from attention.sliding_window import sliding_window_attention

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Running on: {device}")
print("-" * 40)

sizes = [512, 1024, 2048]

dense_times = []
sparse_times = []


def average_time(fn, runs=5):

    times = []

    # warm up once
    fn()

    for _ in range(runs):

        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        fn()

        if device == "cuda":
            torch.cuda.synchronize()

        times.append(time.perf_counter() - start)

    return sum(times) / len(times)


for n in sizes:

    x = torch.randn(1, n, 64, device=device)

    wq = torch.randn(64, 32, device=device)
    wk = torch.randn(64, 32, device=device)
    wv = torch.randn(64, 32, device=device)

    dense = average_time(lambda: dense_attention(x, wq, wk, wv))
    sparse = average_time(lambda: sliding_window_attention(x, wq, wk, wv, 16))

    dense_times.append(dense)
    sparse_times.append(sparse)

    print(f"{n} tokens")
    print(f"  dense  : {dense:.4f}s")
    print(f"  sparse : {sparse:.4f}s")

    if device == "cuda":
        print(f"  peak memory : {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
        torch.cuda.reset_peak_memory_stats()

    print()

plt.figure(figsize=(6,4))

plt.plot(sizes, dense_times, marker="o", label="Dense")
plt.plot(sizes, sparse_times, marker="o", label="Sliding Window")

plt.xlabel("Sequence length")
plt.ylabel("Average time (s)")
plt.title("Dense vs Sliding Window Attention")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(Path(__file__).parent / "timing.png")

print("Saved graph as benchmark/timing.png")