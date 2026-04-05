# full_cvae_v2.py — Phase 4: InstanceNorm + Aux Attribute Loss
#
# Phase 4 change:  BatchNorm2d → InstanceNorm2d(affine=True) everywhere.
#   InstanceNorm normalises per-image per-channel, eliminating cross-sample
#   dependencies that conflict with per-sample KL loss in VAEs.
#
# Phase 4.5 change: AuxAttrLoss wraps the frozen CLIP probe from Phase 3.
#   After decoding, the recon is fed through CLIP → probe → BCE loss against
#   the target attribute vector.  Gradients flow through CLIP back to the
#   decoder, forcing it to actually produce the requested attributes.
#   total_loss = recon_loss + β·KL + λ·attr_bce   (λ=0.5 default)
#
# Everything else (architecture shape, latent dim, attr embedding) is
# identical to full_cvae.py so Phase 2 weights can warm-start this model.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from baseline_cvae import (
    ENC_CHANNELS, ENC_SPATIAL, ENC_FLAT, LATENT_DIM, ATTR_EMB_DIM, AttrEmbedding,
)
from full_cvae import PHOTO_EMB_DIM, DEC_IN_DIM, attribute_swap_grid, SWAP_ATTRS
from data_pipeline import NUM_ATTRS

# ---------------------------------------------------------------------------
# InstanceNorm building blocks
# ---------------------------------------------------------------------------

def _enc_block_in(in_ch: int, out_ch: int) -> nn.Sequential:
    """Conv2d 4×4 stride-2 + InstanceNorm + LeakyReLU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=True),
        nn.InstanceNorm2d(out_ch, affine=True),
        nn.LeakyReLU(0.2, inplace=True),
    )


def _dec_block_in(in_ch: int, out_ch: int) -> nn.Sequential:
    """ConvTranspose2d 4×4 stride-2 + InstanceNorm + ReLU."""
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=True),
        nn.InstanceNorm2d(out_ch, affine=True),
        nn.ReLU(inplace=True),
    )


# ---------------------------------------------------------------------------
# Sub-modules (identical structure to Phase 2, IN instead of BN)
# ---------------------------------------------------------------------------

class PhotoEncoderV2(nn.Module):
    def __init__(self):
        super().__init__()
        chs = ENC_CHANNELS
        self.conv = nn.Sequential(
            _enc_block_in(chs[0], chs[1]),
            _enc_block_in(chs[1], chs[2]),
            _enc_block_in(chs[2], chs[3]),
            _enc_block_in(chs[3], chs[4]),
        )
        self.fc = nn.Sequential(nn.Linear(ENC_FLAT, PHOTO_EMB_DIM), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.fc(self.conv(x).view(x.size(0), -1))


class EncoderV2(nn.Module):
    def __init__(self):
        super().__init__()
        chs = ENC_CHANNELS
        self.conv = nn.Sequential(
            _enc_block_in(chs[0], chs[1]),
            _enc_block_in(chs[1], chs[2]),
            _enc_block_in(chs[2], chs[3]),
            _enc_block_in(chs[3], chs[4]),
        )
        self.fc    = nn.Linear(ENC_FLAT + ATTR_EMB_DIM, 512)
        self.fc_mu = nn.Linear(512, LATENT_DIM)
        self.fc_lv = nn.Linear(512, LATENT_DIM)

    def forward(self, x, attr_emb):
        h = self.conv(x).view(x.size(0), -1)
        h = F.relu(self.fc(torch.cat([h, attr_emb], dim=1)))
        return self.fc_mu(h), self.fc_lv(h)


class FullDecoderV2(nn.Module):
    def __init__(self):
        super().__init__()
        chs = ENC_CHANNELS
        self.fc = nn.Linear(DEC_IN_DIM, ENC_FLAT)
        self.deconv = nn.Sequential(
            _dec_block_in(chs[4], chs[3]),
            _dec_block_in(chs[3], chs[2]),
            _dec_block_in(chs[2], chs[1]),
            nn.ConvTranspose2d(chs[1], chs[0], 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z, id_emb, attr_emb):
        h = F.relu(self.fc(torch.cat([z, id_emb, attr_emb], dim=1)))
        return self.deconv(h.view(h.size(0), ENC_CHANNELS[-1], ENC_SPATIAL, ENC_SPATIAL))


# ---------------------------------------------------------------------------
# Auxiliary attribute loss (Phase 4.5)
# ---------------------------------------------------------------------------

# CLIP ViT-B/32 normalisation constants
_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275,  0.40821073])
_CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711])


class AuxAttrLoss(nn.Module):
    """
    Evaluates the generated face with the frozen CLIP probe from Phase 3.
    Gradients flow through CLIP back to the decoder, forcing it to produce
    faces that actually match the target attribute vector.

    Both clip_model and probe must be frozen before passing here.
    lam: weight on the BCE term (0.5 = roughly same scale as recon_loss).
    """

    def __init__(self, clip_model, probe, lam: float = 0.5):
        super().__init__()
        self.clip_model = clip_model
        self.probe      = probe
        self.lam        = lam

    def _to_clip(self, x: torch.Tensor) -> torch.Tensor:
        """[-1,1] 64×64 tensor → CLIP-normalised 224×224."""
        x    = (x + 1.0) / 2.0
        x    = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        mean = _CLIP_MEAN.view(1, 3, 1, 1).to(x.device)
        std  = _CLIP_STD.view(1, 3, 1, 1).to(x.device)
        return (x - mean) / std

    def forward(self, recon: torch.Tensor,
                target_attrs: torch.Tensor) -> torch.Tensor:
        clip_in = self._to_clip(recon.float())
        # CLIP params are frozen (requires_grad=False) but activations are
        # tracked so gradients flow back to recon / decoder weights.
        feats   = self.clip_model.encode_image(clip_in).float()
        feats   = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
        logits  = self.probe(feats)
        return self.lam * F.binary_cross_entropy_with_logits(logits, target_attrs)


# ---------------------------------------------------------------------------
# Full cVAE v2
# ---------------------------------------------------------------------------

class FullCVAEv2(nn.Module):
    """
    Full cVAE with InstanceNorm (Phase 4) and auxiliary attribute loss (Phase 4.5).
    Interface identical to FullCVAE — drop-in replacement.
    """

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta      = beta
        self.photo_enc = PhotoEncoderV2()
        self.attr_emb  = AttrEmbedding()
        self.encoder   = EncoderV2()
        self.decoder   = FullDecoderV2()

    def encode(self, x, attrs):
        return self.encoder(x, self.attr_emb(attrs))

    def reparameterize(self, mu, logvar):
        if self.training:
            return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return mu

    def decode(self, z, id_emb, attrs):
        return self.decoder(z, id_emb, self.attr_emb(attrs))

    def forward(self, x, attrs, photo=None):
        if photo is None:
            photo = x
        id_emb     = self.photo_enc(photo)
        mu, logvar = self.encode(x, attrs)
        z          = self.reparameterize(mu, logvar)
        recon      = self.decode(z, id_emb, attrs)
        return recon, mu, logvar

    def vae_loss(self, x, recon, mu, logvar) -> dict:
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        kl_loss    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return {'recon': recon_loss, 'kl': kl_loss}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model: FullCVAEv2, optimizer, epoch: int,
                    history: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({'epoch': epoch, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'history': history, 'beta': model.beta}, path)


def load_checkpoint(path: str, model: FullCVAEv2, optimizer,
                    device: torch.device) -> tuple:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt['epoch'], ckpt['history']


def warmstart_from_phase2(model: FullCVAEv2, phase2_ckpt: str,
                           device: torch.device) -> int:
    """
    Transfer weights from Phase 2 checkpoint (strict=False — skips BatchNorm
    running stats that don't exist in InstanceNorm).
    Returns number of parameter tensors transferred.
    """
    p2 = torch.load(phase2_ckpt, map_location=device, weights_only=False)['model']
    cur = model.state_dict()
    transferred = 0
    for k, v in p2.items():
        if k in cur and cur[k].shape == v.shape:
            cur[k] = v
            transferred += 1
    model.load_state_dict(cur, strict=False)
    return transferred


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model: FullCVAEv2, loader: DataLoader,
                    optimizer, device: torch.device,
                    aux_loss_fn=None, scaler=None) -> dict:
    """
    aux_loss_fn : AuxAttrLoss instance, or None to skip (Phase 4 only).
    scaler      : GradScaler for AMP, or None for fp32.
    """
    model.train()
    sums = {'total': 0., 'recon': 0., 'kl': 0., 'aux': 0.}
    n = 0

    for images, attrs in loader:
        images = images.to(device, non_blocking=True)
        attrs  = attrs.to(device,  non_blocking=True)

        optimizer.zero_grad()
        with torch.autocast('cuda', enabled=(scaler is not None)):
            recon, mu, logvar = model(images, attrs)
            losses = model.vae_loss(images, recon, mu, logvar)
            total  = losses['recon'] + model.beta * losses['kl']
            if aux_loss_fn is not None:
                a_loss = aux_loss_fn(recon, attrs)
                total  = total + a_loss
                losses['aux'] = a_loss
            else:
                losses['aux'] = torch.tensor(0.0)

        if scaler is not None:
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        sums['total'] += total.item()
        for k in ('recon', 'kl', 'aux'):
            sums[k] += losses[k].item()
        n += 1

    return {k: v / n for k, v in sums.items()}
