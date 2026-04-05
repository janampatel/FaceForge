# baseline_cvae.py — Phase 1: Baseline cVAE (attribute conditioning only)
#
# Architecture:
#   AttrEmbedding : Linear(18→64)→ReLU→Linear(64→64)→ReLU  →  [B, 64]
#   Encoder       : 4× Conv2d(BN+LeakyReLU) → flatten → concat(attr_emb)
#                   → Linear(512) → μ [128], logvar [128]
#   Decoder       : concat(z[128], attr_emb[64]) → Linear → reshape
#                   → 4× ConvTranspose2d → Tanh  →  [B, 3, 64, 64]
#   Loss          : MSE recon + β·KL   (β=1.0 default)

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_pipeline import NUM_ATTRS

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------

LATENT_DIM   = 128
ATTR_EMB_DIM = 64
IMG_CHANNELS = 3
IMG_SIZE     = 64

# Encoder/decoder channel progression
ENC_CHANNELS = [IMG_CHANNELS, 32, 64, 128, 256]

# Spatial size after 4× stride-2 convs on 64×64 input: 64 / 2^4 = 4
ENC_SPATIAL = IMG_SIZE // (2 ** len(ENC_CHANNELS[1:]))   # 4
ENC_FLAT    = ENC_CHANNELS[-1] * ENC_SPATIAL * ENC_SPATIAL   # 256*4*4 = 4096


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _enc_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Conv2d 4×4, stride 2 + BatchNorm + LeakyReLU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


def _dec_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """ConvTranspose2d 4×4, stride 2 + BatchNorm + ReLU."""
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class AttrEmbedding(nn.Module):
    """Binary attribute vector [B, 18] → dense embedding [B, 64]."""

    def __init__(self, n_attrs: int = NUM_ATTRS, emb_dim: int = ATTR_EMB_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_attrs, emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, attrs: torch.Tensor) -> torch.Tensor:
        return self.net(attrs)


class Encoder(nn.Module):
    """
    Image [B, 3, 64, 64] + attr_emb [B, 64]
    → mu [B, 128], logvar [B, 128]
    """

    def __init__(self):
        super().__init__()
        chs = ENC_CHANNELS
        self.conv = nn.Sequential(
            _enc_block(chs[0], chs[1]),   # 64→32
            _enc_block(chs[1], chs[2]),   # 32→16
            _enc_block(chs[2], chs[3]),   # 16→8
            _enc_block(chs[3], chs[4]),   # 8→4
        )
        self.fc     = nn.Linear(ENC_FLAT + ATTR_EMB_DIM, 512)
        self.fc_mu  = nn.Linear(512, LATENT_DIM)
        self.fc_lv  = nn.Linear(512, LATENT_DIM)

    def forward(self, x: torch.Tensor, attr_emb: torch.Tensor):
        h = self.conv(x)                                    # [B, 256, 4, 4]
        h = h.view(h.size(0), -1)                           # [B, 4096]
        h = F.relu(self.fc(torch.cat([h, attr_emb], dim=1)))  # [B, 512]
        return self.fc_mu(h), self.fc_lv(h)                 # [B, 128] each


class Decoder(nn.Module):
    """
    z [B, 128] + attr_emb [B, 64]
    → reconstructed image [B, 3, 64, 64] in [-1, 1]
    """

    def __init__(self):
        super().__init__()
        chs = ENC_CHANNELS
        self.fc = nn.Linear(LATENT_DIM + ATTR_EMB_DIM, ENC_FLAT)
        self.deconv = nn.Sequential(
            _dec_block(chs[4], chs[3]),   # 4→8
            _dec_block(chs[3], chs[2]),   # 8→16
            _dec_block(chs[2], chs[1]),   # 16→32
            # Final layer: no BatchNorm, Tanh activation
            nn.ConvTranspose2d(chs[1], chs[0],
                               kernel_size=4, stride=2, padding=1),  # 32→64
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, attr_emb: torch.Tensor):
        h = F.relu(self.fc(torch.cat([z, attr_emb], dim=1)))          # [B, 4096]
        h = h.view(h.size(0), ENC_CHANNELS[-1], ENC_SPATIAL, ENC_SPATIAL)  # [B, 256, 4, 4]
        return self.deconv(h)                                          # [B, 3, 64, 64]


# ---------------------------------------------------------------------------
# Full cVAE
# ---------------------------------------------------------------------------

class BaselineCVAE(nn.Module):
    """
    Conditional VAE conditioned on 18-dim binary attribute labels.
    Phase 1 of FaceForge — no photo encoder yet.

    Args:
        beta: Weight on the KL term. Start at 1.0; reduce to 0.5 if
              posterior collapse is observed (all KL → 0, blurry outputs).
    """

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta     = beta
        self.attr_emb = AttrEmbedding()
        self.encoder  = Encoder()
        self.decoder  = Decoder()

    def encode(self, x: torch.Tensor, attrs: torch.Tensor):
        return self.encoder(x, self.attr_emb(attrs))

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return mu   # deterministic at eval time

    def decode(self, z: torch.Tensor, attrs: torch.Tensor) -> torch.Tensor:
        return self.decoder(z, self.attr_emb(attrs))

    def forward(self, x: torch.Tensor, attrs: torch.Tensor):
        mu, logvar = self.encode(x, attrs)
        z          = self.reparameterize(mu, logvar)
        recon      = self.decode(z, attrs)
        return recon, mu, logvar

    def loss(self, x: torch.Tensor, recon: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor) -> dict:
        """
        Returns dict with 'total', 'recon', 'kl' (all scalar means).

        KL is averaged over both the batch and latent dimensions so it
        stays on a comparable scale to the per-pixel MSE.
        """
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        kl_loss    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total      = recon_loss + self.beta * kl_loss
        return {'total': total, 'recon': recon_loss, 'kl': kl_loss}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model: BaselineCVAE, optimizer: torch.optim.Optimizer,
                    epoch: int, history: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch':     epoch,
        'model':     model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'history':   history,
        'beta':      model.beta,
    }, path)


def load_checkpoint(path: str, model: BaselineCVAE,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device) -> tuple:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt['epoch'], ckpt['history']


# ---------------------------------------------------------------------------
# Training (one epoch)
# ---------------------------------------------------------------------------

def train_one_epoch(model: BaselineCVAE, loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device) -> dict:
    model.train()
    sums = {'total': 0., 'recon': 0., 'kl': 0.}
    n = 0

    for images, attrs in loader:
        images = images.to(device, non_blocking=True)
        attrs  = attrs.to(device,  non_blocking=True)

        optimizer.zero_grad()
        recon, mu, logvar = model(images, attrs)
        losses = model.loss(images, recon, mu, logvar)
        losses['total'].backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        for k in sums:
            sums[k] += losses[k].item()
        n += 1

    return {k: v / n for k, v in sums.items()}
