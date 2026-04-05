# clip_probe.py — Phase 3: CLIP text conditioning via linear probe
#
# What this builds and why:
# CLIP (ViT-B/32) maps both images and text into a shared 512-dim embedding
# space. A linear probe — just Linear(512, 18) — is trained to predict the
# 18 binary CelebA attributes from CLIP IMAGE features. Because CLIP aligns
# text and image embeddings, the same probe generalises to CLIP TEXT features
# at inference time, giving us "text prompt → attribute vector → cVAE".
#
# Three-step pipeline at inference:
#   text prompt → CLIP text encoder → L2-norm → probe → sigmoid → attr vector
#   attr vector + reference photo → FullCVAE → generated face

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as T

from data_pipeline import NUM_ATTRS, ATTR_INDICES, SELECTED_ATTRS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLIP_DIM   = 512    # ViT-B/32 embedding dimension
CLIP_IMG_SIZE = 224


# ---------------------------------------------------------------------------
# Load frozen CLIP
# ---------------------------------------------------------------------------

def load_clip(device: torch.device):
    """
    Load CLIP ViT-B/32 with all weights frozen.
    Returns (clip_model, clip_preprocess).
    clip_preprocess is a torchvision transform pipeline for PIL images.
    """
    import clip
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, preprocess


# ---------------------------------------------------------------------------
# CelebA dataset with CLIP preprocessing (PIL → CLIP input)
# ---------------------------------------------------------------------------

class CelebACLIPDataset(torch.utils.data.Dataset):
    """
    CelebA split with CLIP image preprocessing.
    Returns (clip_input_tensor, selected_attrs [18]).
    """

    def __init__(self, root: str, split: str, clip_preprocess):
        # clip_preprocess expects PIL → tensor in CLIP's normalisation
        self._celeba = torchvision.datasets.CelebA(
            root=root, split=split, target_type='attr',
            transform=clip_preprocess, download=False,
        )

    def __len__(self):
        return len(self._celeba)

    def __getitem__(self, idx):
        img, full_attrs = self._celeba[idx]
        selected = full_attrs[ATTR_INDICES].float()
        return img, selected


# ---------------------------------------------------------------------------
# Feature extraction + caching
# ---------------------------------------------------------------------------

def extract_and_cache_features(clip_model, data_root: str, split: str,
                                cache_path: str, device: torch.device,
                                batch_size: int = 256):
    """
    Extract L2-normalised CLIP image features for an entire CelebA split and
    save them to cache_path.  Skips extraction if cache already exists.

    Cache format: {'features': [N, 512], 'attrs': [N, 18], 'split': split}
    Approximate sizes:
        train  (162 770 images) ≈ 345 MB
        valid  ( 19 867 images) ≈  42 MB
    """
    if os.path.exists(cache_path):
        print(f'Cache exists — loading from {cache_path}')
        return torch.load(cache_path, map_location='cpu', weights_only=False)

    import clip
    _, clip_preprocess = clip.load("ViT-B/32", device=device)
    dataset = CelebACLIPDataset(data_root, split, clip_preprocess)
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=False, num_workers=2, pin_memory=True)

    all_feats, all_attrs = [], []
    n_batches = len(loader)
    print(f'Extracting CLIP features for {split} split '
          f'({len(dataset):,} images, {n_batches} batches)…')

    clip_model.eval()
    with torch.no_grad():
        for i, (imgs, attrs) in enumerate(loader):
            imgs = imgs.to(device)
            feats = clip_model.encode_image(imgs).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)   # L2-norm
            all_feats.append(feats.cpu())
            all_attrs.append(attrs)
            if (i + 1) % 50 == 0 or (i + 1) == n_batches:
                print(f'  {i+1}/{n_batches} batches done', end='\r')

    cache = {
        'features': torch.cat(all_feats, dim=0),   # [N, 512]
        'attrs':    torch.cat(all_attrs, dim=0),    # [N, 18]
        'split':    split,
    }
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(cache, cache_path)
    print(f'\nSaved cache → {cache_path}  '
          f'({cache["features"].shape[0]:,} × {CLIP_DIM})')
    return cache


# ---------------------------------------------------------------------------
# Linear probe
# ---------------------------------------------------------------------------

class CLIPProbe(nn.Module):
    """
    Linear(512 → 18) probe over frozen CLIP image features.
    Outputs raw logits — pass through sigmoid for probabilities.
    """

    def __init__(self, clip_dim: int = CLIP_DIM, n_attrs: int = NUM_ATTRS):
        super().__init__()
        self.linear = nn.Linear(clip_dim, n_attrs)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: [B, 512] L2-normalised → logits [B, 18]"""
        return self.linear(features)


# ---------------------------------------------------------------------------
# Probe training
# ---------------------------------------------------------------------------

def train_probe(probe: CLIPProbe, train_cache: dict,
                device: torch.device,
                n_epochs: int = 15,
                lr: float = 1e-3,
                batch_size: int = 512) -> list:
    """
    Train the linear probe on cached CLIP features.
    Returns per-epoch BCE loss history.
    """
    features = train_cache['features']   # [N, 512] on CPU
    attrs    = train_cache['attrs']      # [N, 18]  on CPU

    dataset  = TensorDataset(features, attrs)
    loader   = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    history = []
    for epoch in range(n_epochs):
        probe.train()
        total, n = 0., 0
        for feat_b, attr_b in loader:
            feat_b = feat_b.to(device)
            attr_b = attr_b.to(device)
            optimizer.zero_grad()
            loss = criterion(probe(feat_b), attr_b)
            loss.backward()
            optimizer.step()
            total += loss.item()
            n += 1
        avg = total / n
        history.append(avg)
        print(f'Epoch [{epoch+1:02d}/{n_epochs}]  BCE={avg:.4f}')

    return history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_probe(probe: CLIPProbe, cache: dict,
                   device: torch.device,
                   threshold: float = 0.5) -> dict:
    """
    Compute per-attribute binary accuracy on cached features.
    Returns dict: {attr_name: accuracy_float, 'mean': float}
    """
    features = cache['features'].to(device)
    attrs    = cache['attrs'].to(device)

    probe.eval()
    with torch.no_grad():
        logits = probe(features)           # [N, 18]
        preds  = (torch.sigmoid(logits) > threshold).float()

    per_attr_acc = (preds == attrs).float().mean(dim=0)   # [18]
    results = {name: per_attr_acc[i].item()
               for i, name in enumerate(SELECTED_ATTRS)}
    results['mean'] = per_attr_acc.mean().item()
    return results


def print_probe_accuracy(results: dict):
    print(f'\n{"Attribute":<25} {"Accuracy":>9}')
    print('-' * 36)
    for name in SELECTED_ATTRS:
        bar = '█' * int(results[name] * 20)
        print(f'{name:<25} {results[name]:>8.1%}  {bar}')
    print('-' * 36)
    print(f'{"Mean accuracy":<25} {results["mean"]:>8.1%}')


# ---------------------------------------------------------------------------
# Text → attribute vector
# ---------------------------------------------------------------------------

def text_to_attrs(clip_model, probe: CLIPProbe,
                  prompts: list, device: torch.device,
                  threshold: float = 0.5):
    """
    Convert a list of text prompts to binary attribute vectors.

    Returns:
        attrs_binary : [N, 18] float32 — thresholded binary vector
        attrs_prob   : [N, 18] float32 — soft probabilities (for inspection)
    """
    import clip
    tokens = clip.tokenize(prompts, truncate=True).to(device)
    probe.eval()
    with torch.no_grad():
        text_feats = clip_model.encode_text(tokens).float()
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        logits     = probe(text_feats)
        probs      = torch.sigmoid(logits)
        binary     = (probs > threshold).float()
    return binary, probs


def print_activated_attrs(attrs_binary: torch.Tensor, prompts: list):
    """Pretty-print which attributes each prompt activated."""
    for i, prompt in enumerate(prompts):
        active = [SELECTED_ATTRS[j] for j in range(NUM_ATTRS)
                  if attrs_binary[i, j].item() > 0.5]
        print(f'  "{prompt}"')
        print(f'    → {active if active else "(none)"}')


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_probe(probe: CLIPProbe, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(probe.state_dict(), path)


def load_probe(path: str, device: torch.device) -> CLIPProbe:
    probe = CLIPProbe()
    probe.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return probe.to(device)
