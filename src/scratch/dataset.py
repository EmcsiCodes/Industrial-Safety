import random
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter


CLASS_NAMES = [
    "person",
    "tool",
    "helmet",
    "safety-vest",
    "gloves",
    "glasses",
    "face-mask",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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
        self.image_dir = self.root / "images" / split
        self.label_dir = self.root / "labels" / split
        self.image_size = image_size
        self.grid_size = grid_size
        self.augment = augment
        self.image_paths = sorted(
            path
            for path in self.image_dir.iterdir()
            if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if limit is not None:
            self.image_paths = self.image_paths[:limit]
        if not self.image_paths:
            raise RuntimeError(f"No images found in:\n{self.image_dir}")

        self.color_jitter = ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
            hue=0.02,
        )
        print(f"{split}: {len(self.image_paths)} images")
        self._report_grid_collisions()

    def _read_labels(self, image_path):
        label_path = self.label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            return []

        labels = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            x, y, width, height = map(float, parts[1:])
            labels.append([class_id, x, y, width, height])
        return labels

    def _create_target(self, labels):
        """Encode labels on a grid that can retain one object in each cell."""
        grid_size = self.grid_size
        target = torch.zeros(
            (grid_size, grid_size, 5 + len(CLASS_NAMES)),
            dtype=torch.float32,
        )

        for class_id, x, y, width, height in labels:
            grid_x = min(int(x * grid_size), grid_size - 1)
            grid_y = min(int(y * grid_size), grid_size - 1)
            x_cell = x * grid_size - grid_x
            y_cell = y * grid_size - grid_y

            # This educational detector predicts one object per cell. When two
            # centers share a cell, the larger box is retained.
            if target[grid_y, grid_x, 4] == 1:
                existing_area = target[grid_y, grid_x, 2] * target[grid_y, grid_x, 3]
                if width * height <= existing_area:
                    continue

            target[grid_y, grid_x] = 0
            target[grid_y, grid_x, 0] = x_cell
            target[grid_y, grid_x, 1] = y_cell
            # Width and height stay normalized to the complete image.
            target[grid_y, grid_x, 2] = width
            target[grid_y, grid_x, 3] = height
            target[grid_y, grid_x, 4] = 1.0
            target[grid_y, grid_x, 5 + class_id] = 1.0

        return target

    def _report_grid_collisions(self):
        total_objects = 0
        collisions = 0

        for image_path in self.image_paths:
            occupied = set()
            for _, x, y, _, _ in self._read_labels(image_path):
                cell = (
                    min(int(x * self.grid_size), self.grid_size - 1),
                    min(int(y * self.grid_size), self.grid_size - 1),
                )
                total_objects += 1
                if cell in occupied:
                    collisions += 1
                else:
                    occupied.add(cell)

        percentage = collisions / total_objects * 100 if total_objects else 0
        print(f"Grid collisions: {collisions}/{total_objects} ({percentage:.2f}%)")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        labels = self._read_labels(image_path)

        if self.augment:
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
                for label in labels:
                    label[1] = 1.0 - label[1]
            image = self.color_jitter(image)

        # Direct resizing keeps normalized YOLO coordinates valid.
        image = image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR,
        )
        image = (TF.to_tensor(image) - 0.5) / 0.5
        return image, self._create_target(labels)
