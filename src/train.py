# src/train.py

import json
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from src.dataset import Flickr8kDataset
from src.model import CaptionDecoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#traning parameters
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
VAL_SPLIT = 0.1  #10% data used for validation


def main():
    # Load vocab
    with open("data/flickr8k/captions/word2idx.json") as f:
        word2idx = json.load(f) #This loads a dictionary mapping words to numbers.

    vocab_size = len(word2idx) #Total number of words
    pad_idx = word2idx["<pad>"]#index used for padding

    # Dataset load
    dataset = Flickr8kDataset()
 #Train / Validation Split
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False
    )

    # Model  Model Initialization
    model = CaptionDecoder(
        vocab_size=vocab_size,
        pad_idx=pad_idx
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx) #This measures how wrong the predictions are.
    optimizer = Adam(model.parameters(), lr=LR)  #Optimizer updates the model weights.

    for epoch in range(EPOCHS): #in each epochModel sees the entire dataset once
        model.train() #Enables training features like:dropout,gradient updates
        total_loss = 0.0

        for features, boxes, captions in train_loader:
            features = features.to(DEVICE)
            captions = captions.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(features, captions) #Forward Pass :The model predicts

            loss = criterion(   #Loss Calculation
                outputs.view(-1, vocab_size),
                captions.view(-1)
            )

            loss.backward() #Backpropagation:This calculates gradients using backpropagation
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():#Disables gradient computation to save memory and speed.
            for features, boxes, captions in val_loader:
                features = features.to(DEVICE)
                captions = captions.to(DEVICE)

                outputs = model(features, captions)
                loss = criterion(
                    outputs.view(-1, vocab_size),
                    captions.view(-1)
                )

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(  #Print Training Results
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}"
        )

        # Save checkpoint
        torch.save(
            model.state_dict(),
            f"checkpoints/model_epoch_{epoch+1}.pth"
        )


if __name__ == "__main__":
    main()
