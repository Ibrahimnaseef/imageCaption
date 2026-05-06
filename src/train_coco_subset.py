# src/train_coco_subset.py

import torch
import torch.nn as nn
from torch.optim import Adam

from src.coco_dataset import get_coco_loader
from src.model import CaptionDecoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = "data/coco_subset"
BATCH_SIZE = 16        # keep same (32 only if GPU allows)
EPOCHS = 30            # increased
LR = 5e-4              # lower LR for stability
CHECKPOINT_DIR = "checkpoints_coco_20k"


def main():
    # Load dataset
    loader, dataset = get_coco_loader(DATA_DIR, batch_size=BATCH_SIZE)

    vocab_size = len(dataset.word2idx)
    pad_idx = dataset.pad_idx

    print("Vocabulary size:", vocab_size)

    # Model
    model = CaptionDecoder(
        vocab_size=vocab_size,
        pad_idx=pad_idx,
    ).to(DEVICE)

    optimizer = Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0

        for features, boxes, captions in loader:
            features = features.to(DEVICE)
            captions = captions.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(features, captions[:, :-1])
            loss = criterion(
                outputs.reshape(-1, vocab_size),
                captions[:, 1:].reshape(-1),
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {avg_loss:.4f}")

    # Save checkpoint
    import os
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/model_coco_subset.pth")

    print("COCO subset training completed")


if __name__ == "__main__":
    main()
