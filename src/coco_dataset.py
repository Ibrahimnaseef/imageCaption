# src/coco_dataset.py

import json
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class CocoSubsetDataset(Dataset):
    def __init__(self, data_dir, max_len=30):
        self.data_dir = data_dir
        self.max_len = max_len

        # Load features & boxes
        self.features = np.load(os.path.join(
            data_dir, "features", "features.npy"))
        self.boxes = np.load(os.path.join(data_dir, "features", "boxes.npy"))

        # Load image ids
        image_ids = np.load(
            os.path.join(data_dir, "features", "image_ids.npy"),
            allow_pickle=True,
        ).tolist()

        # Load encoded captions
        with open(os.path.join(data_dir, "captions", "captions_encoded.json")) as f:
            captions = json.load(f)

        # Load vocab
        with open(os.path.join(data_dir, "captions", "word2idx.json")) as f:
            self.word2idx = json.load(f)

        self.pad_idx = self.word2idx["<pad>"]

        # 🔑 FILTER: keep only images that HAVE captions
        self.image_ids = []
        self.features_filt = []
        self.boxes_filt = []

        for i, img in enumerate(image_ids):
            if img in captions:
                self.image_ids.append(img)
                self.features_filt.append(self.features[i])
                self.boxes_filt.append(self.boxes[i])

        self.features = np.array(self.features_filt)
        self.boxes = np.array(self.boxes_filt)
        self.captions = captions

        print(f"Filtered dataset size: {len(self.image_ids)}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_name = self.image_ids[idx]

        feat = torch.tensor(self.features[idx], dtype=torch.float32)
        box = torch.tensor(self.boxes[idx], dtype=torch.float32)

        caps = self.captions[img_name]
        cap = random.choice(caps)

        if len(cap) < self.max_len:
            cap = cap + [self.pad_idx] * (self.max_len - len(cap))
        else:
            cap = cap[: self.max_len]

        cap = torch.tensor(cap, dtype=torch.long)

        return feat, box, cap


def get_coco_loader(data_dir, batch_size=16, shuffle=True):
    dataset = CocoSubsetDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader, dataset
