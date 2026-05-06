# src/extract_features.py

import os
import torch
import torchvision
import numpy as np
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_REGIONS = 36
image_ids = []

# Image preprocessing
transform = transforms.Compose([        #Converts image → tensor [0,1]
    transforms.ToTensor(),
])

# Load pretrained Faster R-CNN
detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(        #Pretrained object detector (COCO dataset)
    weights="DEFAULT")
detector.to(DEVICE)
detector.eval()

backbone = detector.backbone    #backbone: CNN feature extractor
roi_align = torchvision.ops.roi_align   #roi_align: extracts region features
# Project ROI features (256-dim) to 2048-dim
roi_projector = torch.nn.Linear(256, 2048).to(DEVICE)
roi_projector.eval()
#extract_from_image() (CORE)

def extract_from_image(img_path):
    img = Image.open(img_path).convert("RGB")   #Load image
    img_tensor = transform(img).to(DEVICE)

    with torch.no_grad():
        backbone_feats = backbone(img_tensor.unsqueeze(0))  #Run model
        outputs = detector([img_tensor])[0]

    # STEP 3: Get boxes and select top-K by confidence
    boxes = outputs["boxes"]          # (N, 4)
    scores = outputs["scores"]        # (N,)

    scores, idx = scores.sort(descending=True)      #Select top-K regions
    boxes = boxes[idx]

    num = min(len(boxes), NUM_REGIONS)
    boxes = boxes[:num]

    # Initialize fixed-size arrays
    appearance = np.zeros((NUM_REGIONS, 2048), dtype="float32")
    geom = np.zeros((NUM_REGIONS, 4), dtype="float32")

    if num > 0:
        # Normalize box coordinates
        selected_boxes = boxes.cpu().numpy()
        w, h = img.size
        selected_boxes[:, 0] /= w
        selected_boxes[:, 1] /= h
        selected_boxes[:, 2] /= w
        selected_boxes[:, 3] /= h
        geom[:num] = selected_boxes

        # -------- STEP 4 starts here --------

        # Prepare ROIs: [batch_idx, x1, y1, x2, y2]
        rois = torch.cat([
            torch.zeros((num, 1), device=DEVICE),
            boxes
        ], dim=1)

        # Compute spatial scale correctly
        _, _, H, W = backbone_feats["0"].shape
        img_w, img_h = img.size
        spatial_scale = W / img_w

        # ROI Align
        roi_feats = roi_align(      #Extracts region-specific features
            input=backbone_feats["0"],
            boxes=rois,
            output_size=(7, 7),
            spatial_scale=spatial_scale,
            aligned=True
        )

        # Pool + project
        roi_feats = roi_feats.mean(dim=(2, 3))      # (num, 256)
        roi_feats = roi_projector(roi_feats)         # (num, 2048)

        appearance[:num] = roi_feats.detach().cpu().numpy()

    return appearance, geom


def main():
    image_dir = "data/flickr8k/Images"
    save_dir = "data/flickr8k/features"
    os.makedirs(save_dir, exist_ok=True)

    features_all = []
    boxes_all = []

    image_files = sorted(os.listdir(image_dir))
    # 🔹 TEMPORARY: only 1 image
    # image_files = sorted(os.listdir(image_dir))[:10]

    for img_name in tqdm(image_files):
        img_path = os.path.join(image_dir, img_name)
        feats, boxes = extract_from_image(img_path)

        features_all.append(feats)
        boxes_all.append(boxes)
        image_ids.append(img_name)

    np.save(os.path.join(save_dir, "features.npy"), np.array(features_all))
    np.save(os.path.join(save_dir, "boxes.npy"), np.array(boxes_all))
    np.save(os.path.join(save_dir, "image_ids.npy"), np.array(image_ids))

    print("Feature extraction completed.")
    print("Features shape:", np.array(features_all).shape)
    print("Boxes shape:", np.array(boxes_all).shape)
    print("Image IDs:", len(image_ids))


if __name__ == "__main__":
    main()
