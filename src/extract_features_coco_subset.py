# src/extract_features_coco_subset.py

import os
import numpy as np
from tqdm import tqdm
from src.extract_features import extract_from_image, NUM_REGIONS

IMAGE_DIR = "data/coco_subset/Images"
SAVE_DIR = "data/coco_subset/features"


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    image_files = sorted(os.listdir(IMAGE_DIR))

    features_all = []
    boxes_all = []
    image_ids = []

    for img_name in tqdm(image_files):
        img_path = os.path.join(IMAGE_DIR, img_name)

        feats, boxes = extract_from_image(img_path)

        features_all.append(feats)
        boxes_all.append(boxes)
        image_ids.append(img_name)

    np.save(os.path.join(SAVE_DIR, "features.npy"), np.array(features_all))
    np.save(os.path.join(SAVE_DIR, "boxes.npy"), np.array(boxes_all))
    np.save(os.path.join(SAVE_DIR, "image_ids.npy"), np.array(image_ids))

    print("COCO subset feature extraction completed")
    print("Features shape:", np.array(features_all).shape)


if __name__ == "__main__":
    main()
