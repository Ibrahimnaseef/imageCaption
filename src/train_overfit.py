# src/train_overfit.py
import json
import torch
import torch.nn as nn
from torch.optim import Adam
from src.model import CaptionDecoder
from src.dataset import Flickr8kDataset
import sys
import os
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# kjsg;k

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 200
LR = 1e-3


def main():
    with open("data/flickr8k/captions/word2idx.json") as f:
        word2idx = json.load(f)

    vocab_size = len(word2idx)
    pad_idx = word2idx["<pad>"]

    dataset = Flickr8kDataset()
    features, boxes, captions = dataset[0]

    features = features.unsqueeze(0).to(DEVICE)
    captions = captions.unsqueeze(0).to(DEVICE)

    model = CaptionDecoder(
        vocab_size=vocab_size,
        pad_idx=pad_idx
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = Adam(model.parameters(), lr=LR)

    model.train()

    for epoch in range(EPOCHS):
        optimizer.zero_grad()

        outputs = model(features, captions)
        loss = criterion(
            outputs.view(-1, vocab_size),
            captions.view(-1)
        )

        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    print("Overfit test completed.")


if __name__ == "__main__":
    main()
