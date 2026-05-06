# src/prepare_coco_captions.py

import json
import os
from collections import Counter
import nltk

nltk.download("punkt", quiet=True)

COCO_SUBSET_DIR = "data/coco_subset"
OUT_DIR = "data/coco_subset/captions"
MAX_VOCAB_SIZE = 10000
MIN_FREQ = 2

PAD = "<pad>"
SOS = "<start>"
EOS = "<end>"
UNK = "<unk>"


def tokenize(sentence):
    return nltk.word_tokenize(sentence.lower())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load raw captions
    with open(f"{COCO_SUBSET_DIR}/captions.json", "r") as f:
        raw_captions = json.load(f)

    # Count word frequencies
    counter = Counter()
    for caps in raw_captions.values():
        for c in caps:
            counter.update(tokenize(c))

    # Build vocabulary
    vocab = [PAD, SOS, EOS, UNK]
    for word, freq in counter.most_common():
        if freq >= MIN_FREQ and len(vocab) < MAX_VOCAB_SIZE:
            vocab.append(word)

    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}

    # Encode captions
    encoded = {}
    for img, caps in raw_captions.items():
        encoded_caps = []
        for c in caps:
            tokens = tokenize(c)
            ids = [word2idx.get(w, word2idx[UNK]) for w in tokens]
            ids = [word2idx[SOS]] + ids + [word2idx[EOS]]
            encoded_caps.append(ids)
        encoded[img] = encoded_caps

    # Save outputs
    with open(f"{OUT_DIR}/captions_encoded.json", "w") as f:
        json.dump(encoded, f)

    with open(f"{OUT_DIR}/word2idx.json", "w") as f:
        json.dump(word2idx, f)

    with open(f"{OUT_DIR}/idx2word.json", "w") as f:
        json.dump(idx2word, f)

    print("COCO captions preprocessing completed")
    print("Vocabulary size:", len(word2idx))
    print("Images with captions:", len(encoded))


if __name__ == "__main__":
    main()
