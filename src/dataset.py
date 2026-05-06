# src/dataset.py

import json
import numpy as np
import torch
from torch.utils.data import Dataset

FEATURES_DIR = "data/flickr8k/features"
CAPTIONS_DIR = "data/flickr8k/captions"


class Flickr8kDataset(Dataset):     #Dataset Class
    def __init__(self, max_len=30):  #(Data loading)
        self.features = np.load(f"{FEATURES_DIR}/features.npy")
        self.boxes = np.load(f"{FEATURES_DIR}/boxes.npy")
        self.image_ids = np.load(f"{FEATURES_DIR}/image_ids.npy")

        with open(f"{CAPTIONS_DIR}/captions_encoded.json") as f:  #Load captions
            self.captions = json.load(f)

        with open(f"{CAPTIONS_DIR}/word2idx.json") as f:  #Load vocabulary
            self.word2idx = json.load(f)

        self.pad_idx = self.word2idx["<pad>"]   #If <pad> missing → crash
        self.max_len = max_len

        # Keep only images that actually have captions:-filter
        self.valid_ids = [
            img for img in self.image_ids
            if img in self.captions
        ]

    def __len__(self):
        return len(self.valid_ids)

    def pad_caption(self, seq):
        if len(seq) < self.max_len:
            seq = seq + [self.pad_idx] * (self.max_len - len(seq))
        else:
            seq = seq[:self.max_len]
        return seq

    def __getitem__(self, idx):
        img_id = self.valid_ids[idx]    #Get image ID

        feat_idx = np.where(self.image_ids == img_id)[0][0] #Find feature index

        features = torch.tensor(self.features[feat_idx], dtype=torch.float32)   #Load features & boxes
        boxes = torch.tensor(self.boxes[feat_idx], dtype=torch.float32)

        caption = self.captions[img_id][0]  #Get caption
        caption = torch.tensor(self.pad_caption(caption), dtype=torch.long) #Converts to tensor for embedding layerS

        return features, boxes, caption
