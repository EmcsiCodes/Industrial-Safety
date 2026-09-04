from pathlib import Path
import random

import torch
from torch.utils.data import Dataset

from PIL import Image, ImageOps
from torchvision.transforms import ColorJitter
import torchvision.transforms.functional as TF


CLASS_NAMES = [
    "person",
    "tool",
    "helmet",
    "safety-vest",
    "gloves",
    "glasses",
    "face-mask",
]


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


class SH17GridDataset(Dataset):

    def __init__(
        self,
        root,
        split="train",
        image_size=416,
        grid_size=26,
        augment=False,
        limit=None,
    ):

        self.root = Path(root)

        self.image_dir = (
            self.root /
            "images" /
            split
        )

        self.label_dir = (
            self.root /
            "labels" /
            split
        )

        self.image_size = image_size
        self.grid_size = grid_size

        self.augment = augment

        self.image_paths = sorted([
            path
            for path in self.image_dir.iterdir()
            if path.suffix.lower()
            in IMAGE_EXTENSIONS
        ])

        if limit is not None:
            self.image_paths = (
                self.image_paths[:limit]
            )

        if len(self.image_paths) == 0:

            raise RuntimeError(
                f"No images found in:\n"
                f"{self.image_dir}"
            )

        self.color_jitter = ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
            hue=0.02,
        )

        print(
            f"{split}: "
            f"{len(self.image_paths)} images"
        )

        self._report_grid_collisions()


    # ========================================================
    # LABEL READING
    # ========================================================

    def _read_labels(
        self,
        image_path
    ):

        label_path = (
            self.label_dir /
            f"{image_path.stem}.txt"
        )

        labels = []

        if not label_path.exists():
            return labels

        for line in (
            label_path
            .read_text(
                encoding="utf-8"
            )
            .splitlines()
        ):

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                continue

            class_id = int(
                parts[0]
            )

            x = float(
                parts[1]
            )

            y = float(
                parts[2]
            )

            width = float(
                parts[3]
            )

            height = float(
                parts[4]
            )

            labels.append([
                class_id,
                x,
                y,
                width,
                height,
            ])

        return labels


    # ========================================================
    # GRID TARGET
    # ========================================================

    def _create_target(
        self,
        labels
    ):

        S = self.grid_size
        C = len(CLASS_NAMES)

        # [
        #   x_cell,
        #   y_cell,
        #   width,
        #   height,
        #   objectness,
        #   class probabilities...
        # ]

        target = torch.zeros(
            (
                S,
                S,
                5 + C
            ),
            dtype=torch.float32
        )

        for (
            class_id,
            x,
            y,
            width,
            height
        ) in labels:

            grid_x = min(
                int(x * S),
                S - 1
            )

            grid_y = min(
                int(y * S),
                S - 1
            )

            # Position relative to cell
            x_cell = (
                x * S -
                grid_x
            )

            y_cell = (
                y * S -
                grid_y
            )

            # -----------------------------------------------
            # One object per grid cell.
            #
            # If multiple objects share a cell, keep the
            # largest one.
            # -----------------------------------------------

            if (
                target[
                    grid_y,
                    grid_x,
                    4
                ] == 1
            ):

                existing_area = (
                    target[
                        grid_y,
                        grid_x,
                        2
                    ]
                    *
                    target[
                        grid_y,
                        grid_x,
                        3
                    ]
                )

                new_area = (
                    width *
                    height
                )

                if new_area <= existing_area:
                    continue


            target[
                grid_y,
                grid_x,
                :
            ] = 0


            target[
                grid_y,
                grid_x,
                0
            ] = x_cell

            target[
                grid_y,
                grid_x,
                1
            ] = y_cell

            # Width/height remain normalized
            # relative to entire image.

            target[
                grid_y,
                grid_x,
                2
            ] = width

            target[
                grid_y,
                grid_x,
                3
            ] = height

            target[
                grid_y,
                grid_x,
                4
            ] = 1.0

            target[
                grid_y,
                grid_x,
                5 + class_id
            ] = 1.0


        return target


    # ========================================================
    # COLLISION ANALYSIS
    # ========================================================

    def _report_grid_collisions(
        self
    ):

        total_objects = 0
        collisions = 0

        S = self.grid_size

        for image_path in (
            self.image_paths
        ):

            labels = (
                self._read_labels(
                    image_path
                )
            )

            occupied = set()

            for label in labels:

                _, x, y, _, _ = label

                gx = min(
                    int(x * S),
                    S - 1
                )

                gy = min(
                    int(y * S),
                    S - 1
                )

                cell = (
                    gx,
                    gy
                )

                total_objects += 1

                if cell in occupied:
                    collisions += 1
                else:
                    occupied.add(
                        cell
                    )

        percentage = (
            collisions /
            total_objects *
            100
            if total_objects > 0
            else 0
        )

        print(
            f"Grid collisions: "
            f"{collisions}/{total_objects} "
            f"({percentage:.2f}%)"
        )


    # ========================================================
    # DATASET
    # ========================================================

    def __len__(
        self
    ):

        return len(
            self.image_paths
        )


    def __getitem__(
        self,
        index
    ):

        image_path = (
            self.image_paths[
                index
            ]
        )

        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )

        labels = (
            self._read_labels(
                image_path
            )
        )


        # ----------------------------------------------------
        # SIMPLE TRAINING AUGMENTATION
        # ----------------------------------------------------

        if self.augment:

            if random.random() < 0.5:

                image = (
                    ImageOps.mirror(
                        image
                    )
                )

                for label in labels:

                    # x -> 1 - x
                    label[1] = (
                        1.0 -
                        label[1]
                    )


            image = (
                self.color_jitter(
                    image
                )
            )


        # ----------------------------------------------------
        # RESIZE
        # ----------------------------------------------------

        # Direct resize is deliberately kept simple.
        #
        # Since YOLO coordinates are normalized,
        # the bounding boxes remain valid.

        image = image.resize(
            (
                self.image_size,
                self.image_size
            ),
            Image.Resampling.BILINEAR
        )


        image = TF.to_tensor(
            image
        )


        # Normalize approximately to [-1, 1]
        image = (
            image - 0.5
        ) / 0.5


        target = (
            self._create_target(
                labels
            )
        )


        return (
            image,
            target
        )