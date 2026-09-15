# Sparse Attention from Scratch in PyTorch

## Objective

The goal of this project was to understand how sparse attention works by implementing it myself instead of relying on PyTorch's built-in attention layers. 
I implemented dense attention first, then added two sparse patterns, tested them, benchmarked them, and finally used the implementation inside a small character-level GPT trained on TinyShakespeare.


## 1. Dense Attention

I started by implementing scaled dot-product attention from scratch.

The following steps were followed:

1. Project the input into Query, Key and Value.
2. Compute `QK`.
3. Scale by `√d`.
4. Apply softmax.
5. Multiply by the Value matrix.

This became the reference implementation that I compared everything else against.


## 2. Sliding Window Attention

Instead of letting every token attend to every other token, each token only looks at nearby tokens inside a fixed window.

For each token:

1. find the left boundary,
2. find the right boundary,
3. compute attention only inside that range.

This computes local attention directly instead of first creating a full attention matrix and then masking it.

---

## 3. BigBird-style Attention

After getting sliding-window attention working, I extended it to include two extra connection types.

Each token attends to:

* nearby tokens,
* one global token (token 0),
* two random tokens.

The random connections are generated with a fixed seed so the pattern stays reproducible.

---

## 4. Correctness Checks

Instead of creating many separate test files (which I previously made a mistake), I kept one test file with a few important checks.

The tests verified:

* output shapes,
* sliding-window behavior,
* BigBird output shapes,
* that sliding-window becomes equivalent to dense attention when the window covers the whole sequence.

---

## 5. NaN Issue

While testing, I ran into a numerical problem.

If every value in an attention row becomes `-inf`, softmax tries to compute `0/0`, producing `NaN`.

I fixed this by making the softmax more stable:

* replaced `-inf` maxima before subtraction,
* avoid division by zero in the denominator.

This prevented invalid values from appearing.

---

## 6. Benchmark

I compared dense attention with my sliding-window implementation on CPU.

| Sequence Length |    Dense | Sliding Window |
| --------------: | -------: | -------------: |
|             512 | 0.0026 s |       0.0522 s |
|            1024 | 0.0053 s |       0.0374 s |
|            2048 | 0.0109 s |       0.1066 s |


### Observation

I originally expected sparse attention to be faster.

But instead, dense attention finished sooner.

The reason is that my sliding-window implementation uses a Python loop for every token, while PyTorch performs dense matrix multiplication using highly optimized kernels. Even though fewer attention scores are computed, the Python overhead dominates on CPU.

---

## 7. TinyGPT Experiment

To verify that the attention implementation works inside an actual model, I built a small character-level GPT.

The training loss decreased from roughly 4.3 to 2.6 after a few hundred steps which showed that the model was learning meaningful patterns.

The generated text is still noisy, but it starts producing Shakespeare-like formatting and dialogue structure.


## Challenges

Some of the main problems I ran into were:

* circular imports while organizing the attention modules,
* Python path issues when running scripts from subfolders,
* shape mismatches during testing,
* NaNs caused by fully masked attention rows,
* benchmarking noise from first-run overhead.


## What I Learned

This project helped me understand attention much better.

* learned more into PyTorch 
* how attention is computed mathematically,
* why sparse attention reduces the attention neighborhood,
* why implementation details matter as much as theoretical complexity,
* how to debug tensor shape issues,
* how to integrate a custom attention implementation into a working Transformer model.

If I continue this project later, I'd like to replace the Python-loop implementation with block-wise sparse computation and compare GPU memory usage on much longer sequences.
