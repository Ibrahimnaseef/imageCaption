import random
import torch
import torch.nn.functional as F


def topk_decode(
    model,
    features,
    word2idx,
    idx2word,
    k=5,
    max_len=30,
):
    device = features.device
    sos = word2idx["<start>"]
    eos = word2idx["<end>"]
    pad = word2idx["<pad>"]

    tokens = [sos]

    with torch.no_grad():
        for _ in range(max_len):
            caps = torch.tensor(tokens, device=device).unsqueeze(0)
            outputs = model(features, caps)
            logits = outputs[0, -1]

            logits[sos] = -1e9
            logits[pad] = -1e9

            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_ids = torch.topk(probs, k)

            next_token = topk_ids[
                torch.multinomial(topk_probs, 1)
            ].item()

            if next_token == eos:
                break

            tokens.append(next_token)

    words = [
        idx2word[str(t)]
        for t in tokens
        if t not in (sos, eos, pad)
    ]

    return " ".join(words)
