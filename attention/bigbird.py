import torch
from attention.dense import dense_attention


def bigbird_mask(
    seq_len,
    window_size=2,
    global_tokens=(0,),
    random_tokens=2,
    seed=42
):
    """
    BigBird-style attention mask.
    Returns:
        True  -> masked
        False -> allowed
    """

    torch.manual_seed(seed)

    mask = torch.ones(seq_len, seq_len, dtype=torch.bool)

    for i in range(seq_len):

        # Local window
        left = max(0, i - window_size)
        right = min(seq_len, i + window_size + 1)

        mask[i, left:right] = False

        # Global tokens
        for g in global_tokens:
            mask[i, g] = False
            mask[g, i] = False

        # Random connections
        candidates = torch.randperm(seq_len)

        count = 0
        for token in candidates:
            token = token.item()

            if mask[i, token]:
                mask[i, token] = False
                count += 1

            if count >= random_tokens:
                break

    return mask

def bigbird_attention(
    x,
    w_q,
    w_k,
    w_v,
    window_size=2,
    global_tokens=(0,),
    random_tokens=2,
):
    seq_len = x.size(1)

    mask = bigbird_mask(
        seq_len,
        window_size,
        global_tokens,
        random_tokens,
    )

    output, probs = dense_attention(
        x,
        w_q,
        w_k,
        w_v,
        mask=mask,
    )

    return output, probs