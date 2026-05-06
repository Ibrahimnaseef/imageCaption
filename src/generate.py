# src/generate.py

import json
import torch
import torch.nn.functional as F

from src.dataset import Flickr8kDataset
from src.model import CaptionDecoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "checkpoints/model_epoch_20.pth"
MAX_LEN = 30

# Common high-frequency words that cause degeneration
STOPWORDS = {"of", "a", "the", "two"}


def beam_search_decode(
    model,
    features,
    word2idx,
    idx2word,
    stopword_ids,
    beam_size=3,
    max_len=30,
    alpha=0.7,
):
    """
    Beam search with:
    - length normalization
    - bigram blocking
    - stopword repetition control
    """

    model.eval()

    sos = word2idx["<start>"]
    eos = word2idx["<end>"]
    pad = word2idx["<pad>"]

    # Each beam: (token_list, log_prob)
    beams = [([sos], 0.0)]

    with torch.no_grad():
        for _ in range(max_len):
            new_beams = []

            for tokens, score in beams:
                # If already ended, keep it
                if tokens[-1] == eos:
                    new_beams.append((tokens, score))
                    continue

                captions = torch.tensor(
                    tokens, device=features.device).unsqueeze(0)
                outputs = model(features, captions)
                logits = outputs[0, -1]

                # 🚫 block invalid tokens
                logits[sos] = -1e9
                logits[pad] = -1e9

                log_probs = F.log_softmax(logits, dim=-1)

                # 🚫 bigram blocking (prevents loops like "of of", "two two")
                if len(tokens) >= 2:
                    prev_bigram = (tokens[-2], tokens[-1])
                    for v in range(log_probs.size(0)):
                        if (tokens[-1], v) == prev_bigram:
                            log_probs[v] = -1e9

                # 🚫 stopword repetition control
                if tokens[-1] in stopword_ids:
                    log_probs[tokens[-1]] = -1e9

                topk_log_probs, topk_ids = torch.topk(log_probs, beam_size)

                for k in range(beam_size):
                    next_token = topk_ids[k].item()
                    next_score = score + topk_log_probs[k].item()
                    new_beams.append((tokens + [next_token], next_score))

            # keep best beams (length-normalized)
            beams = sorted(
                new_beams,
                key=lambda x: x[1] / ((len(x[0]) - 1 + 1e-6) ** alpha),
                reverse=True,
            )[:beam_size]

        best_tokens = beams[0][0]

    words = [
        idx2word[str(idx)]
        for idx in best_tokens
        if idx not in (sos, eos, pad)
    ]

    return " ".join(words)


def main():
    # Load vocabulary
    with open("data/flickr8k/captions/word2idx.json") as f:
        word2idx = json.load(f)

    with open("data/flickr8k/captions/idx2word.json") as f:
        idx2word = json.load(f)

    vocab_size = len(word2idx)

    # Build stopword IDs AFTER vocab is loaded
    stopword_ids = {word2idx[w] for w in STOPWORDS if w in word2idx}

    # Dataset
    dataset = Flickr8kDataset()
    features, boxes, caption = dataset[0]

    features = features.unsqueeze(0).to(DEVICE)

    # Model
    model = CaptionDecoder(
        vocab_size=vocab_size,
        pad_idx=word2idx["<pad>"],
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    )

    # Generate caption
    generated = beam_search_decode(
        model,
        features,
        word2idx,
        idx2word,
        stopword_ids,
        beam_size=3,
        max_len=MAX_LEN,
    )

    print("Generated caption:")
    print(generated)


if __name__ == "__main__":
    main()
