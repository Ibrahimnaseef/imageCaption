import torch
import torchvision
from torchvision import transforms

NUM_REGIONS = 36

# Image transform
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load detector ONCE
detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    weights="DEFAULT"
)
detector.eval()

backbone = detector.backbone
roi_align = torchvision.ops.roi_align

# Project 256-dim ROI features to 2048-dim
roi_projector = torch.nn.Linear(256, 2048)
roi_projector.eval()


def extract_single_image_features(image, device="cpu"):
    """
    image: PIL Image
    returns:
        features: (36, 2048)
        boxes:    (36, 4) normalized
    """

    image_tensor = transform(image).to(device)

    detector.to(device)
    roi_projector.to(device)

    with torch.no_grad():
        backbone_feats = backbone(image_tensor.unsqueeze(0))
        outputs = detector([image_tensor])[0]

    boxes = outputs["boxes"]
    scores = outputs["scores"]

    # sort by confidence
    scores, idx = scores.sort(descending=True)
    boxes = boxes[idx]

    num = min(len(boxes), NUM_REGIONS)
    boxes = boxes[:num]

    features = torch.zeros((NUM_REGIONS, 2048), device=device)
    geom = torch.zeros((NUM_REGIONS, 4), device=device)

    if num > 0:
        # normalize boxes
        w, h = image.size
        geom[:num] = boxes / torch.tensor([w, h, w, h], device=device)

        # prepare ROIs
        rois = torch.cat([
            torch.zeros((num, 1), device=device),
            boxes
        ], dim=1)

        _, _, H, W = backbone_feats["0"].shape
        spatial_scale = W / w

        roi_feats = roi_align(
            backbone_feats["0"],
            rois,
            output_size=(7, 7),
            spatial_scale=spatial_scale,
            aligned=True
        )

        roi_feats = roi_feats.mean(dim=(2, 3))
        roi_feats = roi_projector(roi_feats)

        features[:num] = roi_feats

    return features, geom
