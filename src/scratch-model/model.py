import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CONVOLUTION BLOCK
# ============================================================

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


# ============================================================
# CUSTOM OBJECT DETECTOR
# ============================================================

class SafetyNetDetector(nn.Module):
    """
    Simplified single-scale grid-based object detector.

    Input:
        B x 3 x 416 x 416

    Output:
        B x 26 x 26 x (5 + num_classes)

    Each grid cell predicts:
        x_cell
        y_cell
        width
        height
        objectness
        class logits
    """

    def __init__(self, num_classes=7):
        super().__init__()

        self.num_classes = num_classes

        # 416 -> 208 -> 104 -> 52 -> 26
        self.backbone = nn.Sequential(
            ConvBlock(3, 32, stride=2),
            ConvBlock(32, 64, stride=2),
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, 256, stride=2),

            # Additional feature extraction at 26x26.
            ConvBlock(256, 256, stride=1),
        )

        # Per cell:
        # 4 box values + 1 objectness + class logits
        self.head = nn.Conv2d(
            256,
            5 + num_classes,
            kernel_size=1,
        )

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
        features = self.backbone(x)

        output = self.head(features)

        # B x C x H x W
        # ->
        # B x H x W x C
        output = output.permute(
            0, 2, 3, 1
        ).contiguous()

        return output


# ============================================================
# CUSTOM DETECTION LOSS — V2
# ============================================================

def detection_loss(
    predictions,
    targets,
    lambda_box=5.0,
    lambda_obj=1.0,
    lambda_cls=1.0,
):
    """
    Detection loss for SafetyNet v2.

    Main differences from v1:
    1. Objectness BCE is calculated across the complete grid.
    2. Positive cells receive a dynamically computed pos_weight.
    3. Width/height regression uses sqrt(w/h), improving emphasis
       on smaller bounding boxes.
    """

    # --------------------------------------------------------
    # TARGETS
    # --------------------------------------------------------

    target_boxes = targets[..., 0:4]
    target_objectness = targets[..., 4]
    target_classes = targets[..., 5:]

    object_mask = target_objectness > 0.5
    background_mask = ~object_mask


    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    # Our simple representation constrains xywh to [0, 1].
    predicted_boxes = torch.sigmoid(
        predictions[..., 0:4]
    )

    # Leave objectness as logits for BCEWithLogits.
    predicted_objectness = predictions[..., 4]

    # Leave classes as logits for cross entropy.
    predicted_classes = predictions[..., 5:]

    zero = predictions.sum() * 0.0


    # ========================================================
    # 1. BOX REGRESSION
    # ========================================================

    if object_mask.any():

        predicted_positive_boxes = predicted_boxes[
            object_mask
        ]

        target_positive_boxes = target_boxes[
            object_mask
        ]


        # ----------------------------------------------------
        # CENTER POSITION LOSS
        # ----------------------------------------------------

        xy_loss = F.smooth_l1_loss(
            predicted_positive_boxes[..., 0:2],
            target_positive_boxes[..., 0:2],
            beta=0.1,
        )


        # ----------------------------------------------------
        # WIDTH / HEIGHT LOSS
        #
        # sqrt transformation makes an absolute size error
        # more important for small boxes than huge boxes.
        # ----------------------------------------------------

        predicted_wh = predicted_positive_boxes[
            ..., 2:4
        ].clamp(min=1e-6)

        target_wh = target_positive_boxes[
            ..., 2:4
        ].clamp(min=1e-6)


        wh_loss = F.smooth_l1_loss(
            torch.sqrt(predicted_wh),
            torch.sqrt(target_wh),
            beta=0.1,
        )


        box_loss = xy_loss + wh_loss

    else:
        box_loss = zero


    # ========================================================
    # 2. OBJECTNESS LOSS
    # ========================================================

    positive_count = object_mask.sum().float()
    negative_count = background_mask.sum().float()


    # Balance sparse foreground cells against the much larger
    # number of background cells.
    #
    # Clamp prevents an extreme positive weight from producing
    # unstable gradients.
    positive_weight = (
        negative_count
        /
        positive_count.clamp(min=1.0)
    ).clamp(
        min=1.0,
        max=100.0,
    )


    objectness_loss = F.binary_cross_entropy_with_logits(
        predicted_objectness,
        target_objectness,
        pos_weight=positive_weight,
    )


    # ========================================================
    # 3. CLASSIFICATION LOSS
    # ========================================================

    if object_mask.any():

        target_class_indices = target_classes[
            object_mask
        ].argmax(dim=-1)


        classification_loss = F.cross_entropy(
            predicted_classes[object_mask],
            target_class_indices,
        )

    else:
        classification_loss = zero


    # ========================================================
    # TOTAL
    # ========================================================

    total_loss = (
        lambda_box * box_loss
        +
        lambda_obj * objectness_loss
        +
        lambda_cls * classification_loss
    )


    return {
        "total": total_loss,
        "box": box_loss,
        "objectness": objectness_loss,
        "classification": classification_loss,
    }