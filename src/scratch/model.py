import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


class SafetyNetDetector(nn.Module):
    """Single-scale detector that predicts one box per 26 x 26 grid cell."""

    def __init__(self, num_classes=7):
        super().__init__()
        self.num_classes = num_classes

        # 416 -> 208 -> 104 -> 52 -> 26, followed by extraction at 26 x 26.
        self.backbone = nn.Sequential(
            ConvBlock(3, 32, stride=2),
            ConvBlock(32, 64, stride=2),
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, 256, stride=2),
            ConvBlock(256, 256),
        )
        # Each cell predicts xywh, objectness, and class logits.
        self.head = nn.Conv2d(256, 5 + num_classes, kernel_size=1)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        output = self.head(self.backbone(x))
        return output.permute(0, 2, 3, 1).contiguous()


def detection_loss(
    predictions,
    targets,
    lambda_box=5.0,
    lambda_obj=1.0,
    lambda_cls=1.0,
):
    """Calculate the SafetyNet v2 box, objectness, and class losses."""
    target_boxes = targets[..., :4]
    target_objectness = targets[..., 4]
    target_classes = targets[..., 5:]
    object_mask = target_objectness > 0.5
    background_mask = ~object_mask

    predicted_boxes = torch.sigmoid(predictions[..., :4])
    predicted_objectness = predictions[..., 4]
    predicted_classes = predictions[..., 5:]
    zero = predictions.sum() * 0.0

    if object_mask.any():
        predicted_positive_boxes = predicted_boxes[object_mask]
        target_positive_boxes = target_boxes[object_mask]
        xy_loss = F.smooth_l1_loss(
            predicted_positive_boxes[..., :2],
            target_positive_boxes[..., :2],
            beta=0.1,
        )

        # sqrt(w/h) gives size errors on small boxes more influence.
        predicted_wh = predicted_positive_boxes[..., 2:4].clamp(min=1e-6)
        target_wh = target_positive_boxes[..., 2:4].clamp(min=1e-6)
        wh_loss = F.smooth_l1_loss(
            torch.sqrt(predicted_wh),
            torch.sqrt(target_wh),
            beta=0.1,
        )
        box_loss = xy_loss + wh_loss
    else:
        box_loss = zero

    # Objectness is evaluated over the full grid. Dynamic positive weighting
    # balances sparse object cells without allowing extreme weights.
    positive_count = object_mask.sum().float()
    negative_count = background_mask.sum().float()
    positive_weight = (negative_count / positive_count.clamp(min=1.0)).clamp(
        min=1.0,
        max=100.0,
    )
    objectness_loss = F.binary_cross_entropy_with_logits(
        predicted_objectness,
        target_objectness,
        pos_weight=positive_weight,
    )

    if object_mask.any():
        target_class_indices = target_classes[object_mask].argmax(dim=-1)
        classification_loss = F.cross_entropy(
            predicted_classes[object_mask],
            target_class_indices,
        )
    else:
        classification_loss = zero

    total_loss = (
        lambda_box * box_loss
        + lambda_obj * objectness_loss
        + lambda_cls * classification_loss
    )
    return {
        "total": total_loss,
        "box": box_loss,
        "objectness": objectness_loss,
        "classification": classification_loss,
    }
