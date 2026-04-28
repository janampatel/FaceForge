import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.metrics import (structural_similarity as ssim_fn,
                             peak_signal_noise_ratio as psnr_fn)
from torchvision.utils import save_image

from full_cvae_v3 import LATENT_DIM
from data_pipeline import SELECTED_ATTRS, NUM_ATTRS, denormalize


_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275,  0.40821073]).view(1, 3, 1, 1)
_CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def reconstruction_quality(model, val_loader, device, n=500):
    """SSIM and PSNR on n val images. Returns {'ssim': float, 'psnr': float}."""
    model.eval()
    ssims, psnrs, count = [], [], 0

    with torch.no_grad():
        for imgs, attrs in val_loader:
            imgs, attrs = imgs.to(device), attrs.to(device)
            recon, _, _ = model(imgs, attrs)

            real_np = ((imgs.cpu().permute(0, 2, 3, 1).numpy() + 1) / 2).clip(0, 1)
            fake_np = ((recon.cpu().permute(0, 2, 3, 1).numpy() + 1) / 2).clip(0, 1)

            for r, f in zip(real_np, fake_np):
                ssims.append(ssim_fn(r, f, data_range=1.0, channel_axis=2))
                psnrs.append(psnr_fn(r, f, data_range=1.0))
                count += 1
                if count >= n:
                    break
            if count >= n:
                break

    return {"ssim": float(np.mean(ssims)), "psnr": float(np.mean(psnrs)), "n": count}


def compute_fid(model, val_loader, device, n=5000,
                real_dir="/content/fid_real", fake_dir="/content/fid_fake"):
    """Generate n real + fake images and return FID score."""
    from pytorch_fid import fid_score as fid_lib

    real_dir = Path(real_dir); real_dir.mkdir(exist_ok=True)
    fake_dir = Path(fake_dir); fake_dir.mkdir(exist_ok=True)

    model.eval()
    saved_real = saved_fake = 0

    with torch.no_grad():
        for imgs, attrs in val_loader:
            imgs, attrs = imgs.to(device), attrs.to(device)
            B = imgs.size(0)

            for i in range(B):
                if saved_real < n:
                    save_image((imgs[i] + 1) / 2, real_dir / f"r{saved_real:05d}.png")
                    saved_real += 1

            z    = torch.randn(B, LATENT_DIM, device=device)
            id_e = model.photo_enc(imgs)
            fake = model.decode(z, id_e, attrs)

            for i in range(B):
                if saved_fake < n:
                    save_image((fake[i] + 1) / 2, fake_dir / f"f{saved_fake:05d}.png")
                    saved_fake += 1

            if saved_real >= n and saved_fake >= n:
                break

    return fid_lib.calculate_fid_given_paths(
        [str(real_dir), str(fake_dir)],
        batch_size=64, device=str(device), dims=2048,
    )


def attribute_compliance(model, clip_model, probe, val_loader, device, n=500):
    """Per-attribute accuracy using CLIP probe on reconstructions.
    Returns {'per_attr': tensor[18], 'mean': float}."""
    model.eval()
    mean = _CLIP_MEAN.to(device)
    std  = _CLIP_STD.to(device)
    all_pred, all_gt, count = [], [], 0

    with torch.no_grad():
        for imgs, attrs in val_loader:
            imgs, attrs = imgs.to(device), attrs.to(device)
            recon, _, _ = model(imgs, attrs)

            clip_in = F.interpolate((recon + 1) / 2, size=224,
                                    mode="bilinear", align_corners=False)
            clip_in = (clip_in - mean) / std
            feats   = clip_model.encode_image(clip_in).float()
            feats   = feats / feats.norm(dim=-1, keepdim=True)
            preds   = (torch.sigmoid(probe(feats)) > 0.5).float()

            all_pred.append(preds.cpu())
            all_gt.append(attrs.cpu())
            count += imgs.size(0)
            if count >= n:
                break

    all_pred = torch.cat(all_pred)[:n]
    all_gt   = torch.cat(all_gt)[:n]
    per_attr = (all_pred == all_gt).float().mean(0)
    return {"per_attr": per_attr, "mean": per_attr.mean().item()}


def latent_traversal_figure(model, val_loader, device,
                            attr_names=None, steps=7, save_path=None):
    """Latent traversal: z fixed, each attr swept from -1 to +1.
    Returns matplotlib figure."""
    if attr_names is None:
        attr_names = ["Smiling", "Eyeglasses", "Bangs", "Young", "Blond_Hair"]

    sweep = torch.linspace(-1.0, 1.0, steps)
    model.eval()

    test_img, test_attr = next(iter(val_loader))
    test_img  = test_img[:1].to(device)
    test_attr = test_attr[:1].to(device)

    rows = []
    with torch.no_grad():
        id_emb     = model.photo_enc(test_img)
        attr_emb_b = model.attr_emb(test_attr)
        mu, _      = model.encoder(test_img, attr_emb_b)
        z = mu

        for attr_name in attr_names:
            attr_idx = SELECTED_ATTRS.index(attr_name)
            row = []
            for v in sweep:
                a = test_attr.clone()
                a[0, attr_idx] = v.item()
                row.append(model.decode(z, id_emb, a).squeeze(0).cpu())
            rows.append(torch.stack(row))

    fig, axes = plt.subplots(len(attr_names), steps,
                             figsize=(steps * 2.2, len(attr_names) * 2.4))
    for r, (name, row) in enumerate(zip(attr_names, rows)):
        for c, img in enumerate(row):
            ax = axes[r, c]
            ax.imshow(denormalize(img)); ax.axis("off")
            if c == 0:
                ax.set_ylabel(name, fontsize=9, rotation=0, labelpad=70, va="center")
            if r == 0:
                ax.set_title(f"{sweep[c]:.1f}", fontsize=8)

    plt.suptitle("Latent traversal — z fixed, attribute swept −1 → +1", fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig
