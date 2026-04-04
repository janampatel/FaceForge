# data_pipeline.py — Phase 0: Environment and Data Pipeline
#
# What this builds and why:
# CelebA has 202,599 face images with 40 binary attribute labels. We select 18
# attributes that cover the most visually meaningful traits for face editing.
# Images are resized to 64x64 and normalized to [-1, 1] (standard for generative
# models using Tanh output). The DataLoader is configured for Colab's T4 GPU.

import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Attribute configuration
# ---------------------------------------------------------------------------

# Full list of CelebA attributes in order (index 0–39)
CELEBA_ALL_ATTRS = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks",
    "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
    "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie",
    "Young"
]

# The 18 attributes we actually use, in the exact order we want them
SELECTED_ATTRS = [
    "5_o_Clock_Shadow",   # 0
    "Arched_Eyebrows",    # 1
    "Attractive",         # 2
    "Bags_Under_Eyes",    # 3
    "Bald",               # 4
    "Bangs",              # 5
    "Black_Hair",         # 6
    "Blond_Hair",         # 7
    "Brown_Hair",         # 8
    "Bushy_Eyebrows",     # 9
    "Eyeglasses",         # 10
    "Male",               # 11
    "Mouth_Slightly_Open",# 12
    "Mustache",           # 13
    "No_Beard",           # 14
    "Pale_Skin",          # 15
    "Smiling",            # 16
    "Young",              # 17
]
NUM_ATTRS = len(SELECTED_ATTRS)  # 18

# Pre-compute the indices of our 18 attrs in the full 40-attr vector
ATTR_INDICES = [CELEBA_ALL_ATTRS.index(a) for a in SELECTED_ATTRS]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CelebADataset(Dataset):
    """
    Wraps torchvision.datasets.CelebA to:
    - Return only the 18 selected attributes
    - Return attributes as float32 in {0.0, 1.0} (CelebA stores them as {0, 1})
    - Apply consistent image transforms (resize, center-crop, normalize to [-1,1])

    Args:
        root:      Directory that contains (or will contain) the celeba/ folder.
        split:     'train', 'valid', or 'test'.
        download:  Whether to attempt automatic download (may fail due to GDrive limits).
        transform: Optional override for image transforms.
    """

    IMAGE_SIZE = 64  # starting resolution

    def __init__(self, root: str, split: str = "train",
                 download: bool = False, transform=None):
        self.transform = transform or self._default_transform()

        # NOTE: torchvision CelebA download can fail with "Too many requests" from
        # Google Drive. If that happens, download manually from Kaggle
        # (jessicali9530/celeba-dataset) and place the folder at root/celeba/.
        self._celeba = torchvision.datasets.CelebA(
            root=root,
            split=split,
            target_type="attr",
            transform=self.transform,
            download=download,
        )

    @staticmethod
    def _default_transform():
        return transforms.Compose([
            transforms.Resize(72),           # slightly larger before crop
            transforms.CenterCrop(64),       # 64x64 center crop
            transforms.ToTensor(),           # [0,1]
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]          # maps [0,1] -> [-1,1]
            ),
        ])

    def __len__(self):
        return len(self._celeba)

    def __getitem__(self, idx):
        image, full_attrs = self._celeba[idx]
        # full_attrs is an int tensor of shape [40] with values {0, 1}
        selected = full_attrs[ATTR_INDICES].float()  # [18] float32
        return image, selected


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_dataloader(root: str, split: str = "train",
                    download: bool = False,
                    batch_size: int = 64,
                    num_workers: int = 2) -> DataLoader:
    """
    Returns a DataLoader for the requested split.

    pin_memory=True accelerates host→GPU transfers on Colab's T4.
    persistent_workers=True avoids re-spawning workers each epoch.
    """
    dataset = CelebADataset(root=root, split=split, download=download)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=True,   # keeps batch size constant; important for BatchNorm
    )
    return loader


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """[-1,1] tensor (C,H,W) -> uint8 numpy (H,W,C)."""
    img = (tensor * 0.5 + 0.5).clamp(0, 1)
    return (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def visualize_batch(images: torch.Tensor, attrs: torch.Tensor,
                    n_rows: int = 4, n_cols: int = 4,
                    save_path: str = None):
    """
    Display a grid of face images with their attribute vectors.

    Args:
        images:    Batch tensor [B, 3, 64, 64] in [-1, 1].
        attrs:     Batch tensor [B, 18] float32.
        n_rows:    Grid rows.
        n_cols:    Grid columns.
        save_path: If given, saves figure instead of showing.
    """
    n = min(n_rows * n_cols, images.size(0))
    fig = plt.figure(figsize=(n_cols * 3, n_rows * 4))
    gs = gridspec.GridSpec(n_rows, n_cols, hspace=0.6, wspace=0.3)

    for i in range(n):
        row, col = divmod(i, n_cols)
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(denormalize(images[i]))
        ax.axis("off")

        # Print only the active attributes to keep the label readable
        active = [SELECTED_ATTRS[j] for j in range(NUM_ATTRS)
                  if attrs[i, j].item() > 0.5]
        label = "\n".join(active) if active else "(none)"
        ax.set_title(label, fontsize=5, loc="left")

    plt.suptitle("CelebA batch — active attributes shown", fontsize=9)
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved grid to {save_path}")
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------

def compute_attr_statistics(loader: DataLoader) -> dict:
    """
    Compute mean and std of the 18-dim attribute vector over the full dataset.

    Iterates the entire DataLoader once. On a T4 this takes ~1 minute for the
    training split (162,770 images).
    """
    print("Computing attribute statistics (this may take a minute)...")
    all_attrs = []
    for _, attrs in loader:
        all_attrs.append(attrs)

    all_attrs = torch.cat(all_attrs, dim=0)  # [N, 18]
    stats = {
        "mean": all_attrs.mean(dim=0),   # [18] — fraction of images with attr=1
        "std":  all_attrs.std(dim=0),    # [18]
        "n_samples": all_attrs.size(0),
    }
    return stats


def print_attr_statistics(stats: dict):
    print(f"\nAttribute statistics over {stats['n_samples']:,} samples:\n")
    print(f"{'Attribute':<25} {'Mean':>8} {'Std':>8}")
    print("-" * 45)
    for i, name in enumerate(SELECTED_ATTRS):
        m = stats["mean"][i].item()
        s = stats["std"][i].item()
        print(f"{name:<25} {m:>8.4f} {s:>8.4f}")


# ---------------------------------------------------------------------------
# Colab helper: check if CelebA already exists
# ---------------------------------------------------------------------------

def celeba_is_downloaded(root: str) -> bool:
    """
    Returns True if the CelebA img_align_celeba folder and attr file exist.
    Prevents re-downloading on session resume.
    """
    img_dir  = os.path.join(root, "celeba", "img_align_celeba")
    attr_file = os.path.join(root, "celeba", "list_attr_celeba.txt")
    return os.path.isdir(img_dir) and os.path.isfile(attr_file)


# ---------------------------------------------------------------------------
# Quick sanity check (run this module directly to test)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./data",
                        help="Root directory containing celeba/")
    parser.add_argument("--download", action="store_true",
                        help="Attempt to download CelebA (may fail on GDrive limits)")
    parser.add_argument("--stats", action="store_true",
                        help="Compute and print attribute statistics")
    parser.add_argument("--save-grid", type=str, default=None,
                        help="Path to save the visualization grid (e.g. grid.png)")
    args = parser.parse_args()

    if not celeba_is_downloaded(args.root) and not args.download:
        print(f"CelebA not found at {args.root}. Pass --download to download it.")
        exit(1)

    print(f"Building DataLoader from {args.root} ...")
    loader = make_dataloader(root=args.root, split="train",
                             download=args.download, batch_size=64)

    images, attrs = next(iter(loader))
    print(f"Image batch shape : {images.shape}  dtype: {images.dtype}")
    print(f"Attr  batch shape : {attrs.shape}   dtype: {attrs.dtype}")
    assert images.shape == (64, 3, 64, 64), "Unexpected image shape"
    assert attrs.shape  == (64, 18),        "Unexpected attribute shape"
    assert attrs.dtype  == torch.float32,   "Attributes should be float32"

    visualize_batch(images, attrs, save_path=args.save_grid)

    if args.stats:
        stats = compute_attr_statistics(loader)
        print_attr_statistics(stats)

    print("\nPhase 0 sanity check passed.")
