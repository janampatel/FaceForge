import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from losses import PerceptualLoss


class StyleAdapter(nn.Module):
    """Residual correction network placed after the frozen cartoon generator.
    ~883 parameters. Input/output: [-1, 1] at 256×256.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


def compute_uncertainty(x: torch.Tensor) -> torch.Tensor:
    """Proxy uncertainty via softmax entropy over flattened pixel values.
    Returns scalar tensor."""
    prob = torch.softmax(x.flatten(1), dim=1)
    return (-(prob * torch.log(prob + 1e-8)).sum(dim=1)).mean()


def train_adapter(adapter, anime_gen, p6r_model, val_loader,
                  style_tensor, device,
                  epochs=10, base_lr=1e-4,
                  ckpt_path=None, best_ckpt_path=None):
    """Uncertainty-aware adapter training.

    Args:
        adapter:       StyleAdapter instance (on device)
        anime_gen:     Frozen AnimeGANv2 generator (on device)
        p6r_model:     Frozen cVAE (on device)
        val_loader:    DataLoader for real images
        style_tensor:  [N, 3, 256, 256] style reference images in [-1,1]
        device:        torch.device
        epochs:        Number of training epochs (default 10)
        base_lr:       Base learning rate (default 1e-4)
        ckpt_path:     Path to save final adapter weights
        best_ckpt_path: Path to save best adapter weights

    Returns:
        loss history list
    """
    perc_fn   = PerceptualLoss(weight=1.0).to(device).eval()
    opt       = torch.optim.Adam(adapter.parameters(), lr=base_lr)
    batch     = min(8, style_tensor.size(0))
    style_tgt = style_tensor[:batch]
    best_loss = float("inf")
    history   = []

    p6r_model.eval(); anime_gen.eval(); adapter.train()
    val_iter = iter(val_loader)

    for epoch in range(epochs):
        try:
            imgs_b, attrs_b = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            imgs_b, attrs_b = next(val_iter)

        imgs_b  = imgs_b[:batch].to(device)
        attrs_b = attrs_b[:batch].to(device)

        with torch.no_grad():
            recon, _, _ = p6r_model(imgs_b, attrs_b)
            up   = F.interpolate(recon, size=256, mode="bilinear", align_corners=False)
            base = anime_gen(up)

        opt.zero_grad()
        base_g  = base.detach().requires_grad_(True)
        out     = adapter(base_g)

        loss_p  = perc_fn(out, style_tgt)
        loss_id = F.mse_loss(out, base_g.detach())
        loss    = loss_p + 0.1 * loss_id
        loss.backward(retain_graph=True)

        with torch.no_grad():
            unc = compute_uncertainty(out.detach())

        total = loss
        if base_g.grad is not None:
            eps   = 0.01 * unc.item()
            x_adv = (base_g + eps * base_g.grad.sign()).detach()
            loss_adv = perc_fn(adapter(x_adv), style_tgt)
            total    = loss + 0.5 * loss_adv
            total.backward()

        lr = base_lr * (1 + 0.5 * unc.item())
        for g in opt.param_groups:
            g["lr"] = lr
        opt.step()

        total_v = total.item()
        history.append(total_v)

        if total_v < best_loss:
            best_loss = total_v
            if best_ckpt_path:
                torch.save(adapter.state_dict(), str(best_ckpt_path))

        print(f"Epoch {epoch+1:02d}/{epochs}  loss={total_v:.4f}  "
              f"perc={loss_p.item():.4f}  id={loss_id.item():.4f}  "
              f"unc={unc.item():.4f}  lr={lr:.2e}")

    if ckpt_path:
        torch.save(adapter.state_dict(), str(ckpt_path))
        print(f"Adapter saved → {ckpt_path}  (best={best_loss:.4f})")

    return history
