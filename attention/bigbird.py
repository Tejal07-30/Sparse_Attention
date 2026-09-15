import math
import torch


def bigbird_attention(x, wq, wk, wv, window):

    q = x @ wq
    k = x @ wk
    v = x @ wv

    batch, seq_len, d = q.shape

    out = torch.zeros_like(v)
    probs = torch.zeros(batch, seq_len, seq_len)

    for i in range(seq_len):

        left = max(0, i - window)
        right = min(seq_len, i + window + 1)

        # local tokens
        allowed = list(range(left, right))

        # made token 0 global
        if 0 not in allowed:
            allowed.append(0)

        # added two random tokens
        g = torch.Generator()
        g.manual_seed(i)

        while len(allowed) < (right - left) + 3:
            r = torch.randint(0, seq_len, (1,), generator=g).item()

            if r not in allowed:
                allowed.append(r)

        allowed.sort()

        q_now = q[:, i:i+1, :]
        k_now = k[:, allowed, :]
        v_now = v[:, allowed, :]

        scores = (q_now @ k_now.transpose(-2, -1)) / math.sqrt(d)

        attn = torch.softmax(scores, dim=-1)

        out[:, i:i+1, :] = attn @ v_now

        probs[:, i, allowed] = attn.squeeze(1)

    return out, probs