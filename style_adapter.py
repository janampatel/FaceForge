import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pathlib import Path


# ---------------------------------------------------------------------------
# Gram-matrix style loss
# ---------------------------------------------------------------------------

def _gram(feat: torch.Tensor) -> torch.Tensor:
    B, C, H, W = feat.shape
    f = feat.view(B, C, H * W)
    return torch.bmm(f, f.transpose(1, 2)) / (C * H * W)


class GramStyleLoss(nn.Module):
    """
    Multi-scale Gram-matrix style loss using VGG16 relu1_2 and relu2_2.
    Compares texture/color statistics rather than spatial content.
    """

    _MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    _STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(self):
        super().__init__()
        vgg = torchvision.models.vgg16(weights="IMAGENET1K_V1").features
        self.slice1 = vgg[:4].eval()   # relu1_2
        self.slice2 = vgg[:9].eval()   # relu2_2
        for p in self.parameters():
            p.requires_grad = False

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = self._MEAN.to(x.device)
        std  = self._STD.to(x.device)
        return ((x + 1) / 2 - mean) / std

    def forward(self, out: torch.Tensor, style_tgt: torch.Tensor) -> torch.Tensor:
        out_n = self._normalize(out)
        tgt_n = self._normalize(style_tgt)
        loss = 0.0
        for slicer in [self.slice1, self.slice2]:
            f_out = slicer(out_n)
            f_tgt = slicer(tgt_n)
            g_tgt = _gram(f_tgt).mean(0, keepdim=True)
            g_out = _gram(f_out)
            loss  = loss + F.mse_loss(g_out, g_tgt.expand_as(g_out))
        return loss


# ---------------------------------------------------------------------------
# Style adapter network
# ---------------------------------------------------------------------------

class StyleAdapter(nn.Module):
    """
    Residual correction network placed after the frozen cartoon generator.
    ~38K parameters. Input/output: [-1, 1] tensors.
    InstanceNorm is used because it normalises per-image statistics,
    the right inductive bias for style correction.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x + self.net(x)).tanh()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_adapter(adapter: StyleAdapter,
                  anime_gen,
                  p6r_model,
                  val_loader,
                  style_tensor: torch.Tensor,
                  device: torch.device,
                  epochs: int = 50,
                  base_lr: float = 3e-4,
                  ckpt_path=None,
                  best_ckpt_path=None) -> list:
    """
    Train the StyleAdapter to refine AnimeGANv2 output toward a small set of
    style reference images using a Gram-matrix style loss.

    Args:
        adapter:        StyleAdapter instance (on device)
        anime_gen:      Frozen AnimeGANv2 generator (on device)
        p6r_model:      Frozen cVAE (on device) — supplies face reconstructions
        val_loader:     DataLoader for real CelebA images
        style_tensor:   [N, 3, H, W] style reference images in [-1, 1]
        device:         torch.device
        epochs:         Training epochs (default 50)
        base_lr:        Initial learning rate (default 3e-4, decayed via cosine)
        ckpt_path:      Path to save final adapter weights
        best_ckpt_path: Path to save best-loss adapter weights

    Returns:
        List of per-epoch total loss values.
    """
    gram_fn = GramStyleLoss().to(device)
    opt     = torch.optim.Adam(adapter.parameters(), lr=base_lr)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(
                  opt, T_max=epochs, eta_min=1e-6)

    batch     = min(8, style_tensor.size(0))
    best_loss = float("inf")
    history   = []

    p6r_model.eval()
    anime_gen.eval()
    adapter.train()

    val_iter = iter(val_loader)

    for epoch in range(epochs):
        # Random style sample each epoch for diversity
        idx       = torch.randperm(style_tensor.size(0))[:batch]
        style_tgt = style_tensor[idx].to(device)

        try:
            imgs_b, attrs_b = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            imgs_b, attrs_b = next(val_iter)

        imgs_b  = imgs_b[:batch].to(device)
        attrs_b = attrs_b[:batch].to(device)

        # Build base cartoonised image — no gradients needed here
        with torch.no_grad():
            recon, _, _ = p6r_model(imgs_b, attrs_b)
            up   = F.interpolate(recon, size=512, mode="bilinear", align_corners=False)
            base = anime_gen(up)                      # [-1, 1] at 512×512

        opt.zero_grad()
        out = adapter(base)

        loss_style = gram_fn(out, style_tgt)
        loss_id    = F.mse_loss(out, base.detach())
        loss       = loss_style + 0.05 * loss_id

        loss.backward()
        nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=5.0)
        opt.step()
        sched.step()

        total_v = loss.item()
        history.append(total_v)

        if total_v < best_loss:
            best_loss = total_v
            if best_ckpt_path:
                torch.save(adapter.state_dict(), str(best_ckpt_path))

        print(f"Epoch {epoch+1:02d}/{epochs}  "
              f"loss={total_v:.4f}  "
              f"style={loss_style.item():.4f}  "
              f"id={loss_id.item():.4f}  "
              f"lr={sched.get_last_lr()[0]:.2e}")

    if ckpt_path:
        torch.save(adapter.state_dict(), str(ckpt_path))
        print(f"Adapter saved → {ckpt_path}  (best={best_loss:.4f})")

    return history
