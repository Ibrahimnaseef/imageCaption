# src/prepare_coco_subset.py

import json
import os
import random
from collections import defaultdict

COCO_DIR = "data/coco"
OUT_DIR = "data/coco_subset"
NUM_IMAGES = 20000


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load COCO captions
    with open(f"{COCO_DIR}/annotations/captions_train2017.json", "r") as f:
        coco = json.load(f)

    # Map image_id -> filename
    id2file = {img["id"]: img["file_name"] for img in coco["images"]}

    # Collect captions per image
    image_captions = defaultdict(list)
    for ann in coco["annotations"]:
        image_captions[ann["image_id"]].append(ann["caption"])

    image_ids = list(image_captions.keys())
    random.shuffle(image_ids)
    selected_ids = image_ids[:NUM_IMAGES]

    subset = {}
    for img_id in selected_ids:
        subset[id2file[img_id]] = image_captions[img_id]

    # Save captions
    with open(f"{OUT_DIR}/captions.json", "w") as f:
        json.dump(subset, f, indent=2)

    # Save image list
    with open(f"{OUT_DIR}/images.txt", "w") as f:
        for name in subset.keys():
            f.write(name + "\n")

    print(f"COCO subset created with {len(subset)} images")


if __name__ == "__main__":
    main()
