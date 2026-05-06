# src/generate_coco_subset.py

import json
import torch
import torch.nn.functional as F

from src.coco_dataset import CocoSubsetDataset
from src.model import CaptionDecoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "checkpoints_coco_20k/model_coco_subset.pth"
MAX_LEN = 30


def beam_search_decode(
    model,
    features,
    word2idx,
    idx2word,
    beam_size=3,
    max_len=30,
    alpha=0.7,
):
    model.eval()

    sos = word2idx["<start>"]
    eos = word2idx["<end>"]
    pad = word2idx["<pad>"]

    beams = [([sos], 0.0)]

    with torch.no_grad():
        for _ in range(max_len):
            new_beams = []

            for tokens, score in beams:
                if tokens[-1] == eos:
                    new_beams.append((tokens, score))
                    continue

                captions = torch.tensor(
                    tokens, device=features.device).unsqueeze(0)
                outputs = model(features, captions)
                logits = outputs[0, -1]

                logits[sos] = -1e9
                logits[pad] = -1e9

                log_probs = F.log_softmax(logits, dim=-1)

                # prevent immediate repetition
                log_probs[tokens[-1]] = -1e9

                topk_log_probs, topk_ids = torch.topk(log_probs, beam_size)

                for k in range(beam_size):
                    next_token = topk_ids[k].item()
                    next_score = score + topk_log_probs[k].item()
                    new_beams.append((tokens + [next_token], next_score))

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
    # Load dataset
    dataset = CocoSubsetDataset("data/coco_subset")

    with open("data/coco_subset/captions/word2idx.json") as f:
        word2idx = json.load(f)

    with open("data/coco_subset/captions/idx2word.json") as f:
        idx2word = json.load(f)

    vocab_size = len(word2idx)

    # Model
    model = CaptionDecoder(
        vocab_size=vocab_size,
        pad_idx=word2idx["<pad>"],
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    )

    # Pick one image
    features, boxes, _ = dataset[0]
    features = features.unsqueeze(0).to(DEVICE)

    caption = beam_search_decode(
        model,
        features,
        word2idx,
        idx2word,
        beam_size=3,
        max_len=MAX_LEN,
    )

    print("Generated caption:")
    print(caption)


if __name__ == "__main__":
    main()
