# src/prepare_captions.py

import os
import json
from collections import Counter

DATA_DIR = "data/flickr8k"
TOKEN_FILE = os.path.join(DATA_DIR, "Flickr8k.token.txt")
SAVE_DIR = os.path.join(DATA_DIR, "captions")
os.makedirs(SAVE_DIR, exist_ok=True)

MIN_WORD_FREQ = 5
SPECIAL_TOKENS = ["<pad>", "<start>", "<end>", "<unk>"]


def load_captions(token_file):
    captions = {}
    with open(token_file, "r", encoding="utf-8") as f:
        for line in f:
            img_id, caption = line.strip().split("\t")
            img_name = img_id.split("#")[0]
            caption = caption.lower().strip()

            if img_name not in captions:
                captions[img_name] = []
            captions[img_name].append(caption)
    return captions


def build_vocab(captions, min_freq):
    counter = Counter()
    for caps in captions.values():
        for c in caps:
            counter.update(c.split())

    words = [w for w, f in counter.items() if f >= min_freq]

    word2idx = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    idx = len(word2idx)

    for w in words:
        if w not in word2idx:
            word2idx[w] = idx
            idx += 1

    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word


def encode_captions(captions, word2idx):
    encoded = {}
    unk = word2idx["<unk>"]

    for img, caps in captions.items():
        encoded[img] = []
        for c in caps:
            tokens = ["<start>"] + c.split() + ["<end>"]
            seq = [word2idx.get(w, unk) for w in tokens]
            encoded[img].append(seq)
    return encoded


def main():
    captions = load_captions(TOKEN_FILE)
    print("Images with captions:", len(captions))

    word2idx, idx2word = build_vocab(captions, MIN_WORD_FREQ)
    print("Vocab size:", len(word2idx))

    encoded = encode_captions(captions, word2idx)

    with open(os.path.join(SAVE_DIR, "captions_encoded.json"), "w") as f:
        json.dump(encoded, f)

    with open(os.path.join(SAVE_DIR, "word2idx.json"), "w") as f:
        json.dump(word2idx, f)

    with open(os.path.join(SAVE_DIR, "idx2word.json"), "w") as f:
        json.dump(idx2word, f)

    print("Caption preprocessing completed.")


if __name__ == "__main__":
    main()
