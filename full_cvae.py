# full_cvae.py — Phase 2: Full cVAE (photo identity encoder + attribute labels)
#
# What's new vs Phase 1:
#   PhotoEncoder: a separate CNN that extracts a 256-dim identity embedding
#                 from the reference photo. During training, photo == input image
#                 (self-reconstruction). At test time, use any reference photo.
#   FullDecoder:  conditioned on concat(z[128], id_emb[256], attr_emb[64]) = [448]
#                 instead of concat(z[128], attr_emb[64]) = [192] in Phase 1.
#   Encoder:      unchanged from Phase 1 — takes (image, attr_emb), NOT id_emb,
#                 so z captures only residual variation (lighting, expression)
#                 that identity + attributes don't explain.
#
# Attribute swap demo:
#   1. Encode reference photo → id_emb (identity stays fixed)
#   2. Encode same photo → z (residual style stays fixed)
#   3. Decode with DIFFERENT attr vector → swapped face

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from baseline_cvae import (
    _enc_block, _dec_block, AttrEmbedding, Encoder,
    ENC_CHANNELS, ENC_SPATIAL, ENC_FLAT, LATENT_DIM, ATTR_EMB_DIM,
)
from data_pipeline import NUM_ATTRS, SELECTED_ATTRS

# ---------------------------------------------------------------------------
# New constant
# ---------------------------------------------------------------------------

PHOTO_EMB_DIM = 256   # identity embedding size
# Decoder input: z + id_emb + attr_emb
DEC_IN_DIM = LATENT_DIM + PHOTO_EMB_DIM + ATTR_EMB_DIM   # 128+256+64 = 448


# ---------------------------------------------------------------------------
# Photo (identity) encoder
# ---------------------------------------------------------------------------

class PhotoEncoder(nn.Module):
    """
    Reference photo [B, 3, 64, 64] → identity embedding [B, 256].

    Shares the same conv architecture as the VAE encoder but has its own
    independent weights — the two encoders specialise for different things.
    """

    def __init__(self):
        super().__init__()
        chs = ENC_CHANNELS
        self.conv = nn.Sequential(
            _enc_block(chs[0], chs[1]),
            _enc_block(chs[1], chs[2]),
            _enc_block(chs[2], chs[3]),
            _enc_block(chs[3], chs[4]),
        )
        self.fc = nn.Sequential(
            nn.Linear(ENC_FLAT, PHOTO_EMB_DIM),
            nn.ReLU(inplace=True),
        )

    def forward(self, photo: torch.Tensor) -> torch.Tensor:
        h = self.conv(photo)          # [B, 256, 4, 4]
        h = h.view(h.size(0), -1)     # [B, 4096]
        return self.fc(h)             # [B, 256]


# ---------------------------------------------------------------------------
# Full decoder (takes id_emb in addition to z and attr_emb)
# ---------------------------------------------------------------------------

class FullDecoder(nn.Module):
    """
    concat(z[128], id_emb[256], attr_emb[64]) = [448]
    → reconstructed image [B, 3, 64, 64] in [-1, 1]
    """

    def __init__(self):
        super().__init__()
        chs = ENC_CHANNELS
        self.fc = nn.Linear(DEC_IN_DIM, ENC_FLAT)
        self.deconv = nn.Sequential(
            _dec_block(chs[4], chs[3]),   # 4→8
            _dec_block(chs[3], chs[2]),   # 8→16
            _dec_block(chs[2], chs[1]),   # 16→32
            nn.ConvTranspose2d(chs[1], chs[0],
                               kernel_size=4, stride=2, padding=1),  # 32→64
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor,
                id_emb: torch.Tensor,
                attr_emb: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc(torch.cat([z, id_emb, attr_emb], dim=1)))
        h = h.view(h.size(0), ENC_CHANNELS[-1], ENC_SPATIAL, ENC_SPATIAL)
        return self.deconv(h)


# ---------------------------------------------------------------------------
# Full cVAE
# ---------------------------------------------------------------------------

class FullCVAE(nn.Module):
    """
    Conditional VAE with a photo identity encoder.
    Phase 2 of FaceForge.

    During training  : photo == x  (self-reconstruction)
    At inference time: pass any reference photo to preserve its identity
                       while decoding with different attributes.
    """

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta       = beta
        self.photo_enc  = PhotoEncoder()
        self.attr_emb   = AttrEmbedding()
        self.encoder    = Encoder()         # same as Phase 1
        self.decoder    = FullDecoder()

    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor, attrs: torch.Tensor):
        """Returns mu, logvar — does NOT use photo_emb."""
        return self.encoder(x, self.attr_emb(attrs))

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return mu

    def decode(self, z: torch.Tensor,
               id_emb: torch.Tensor,
               attrs: torch.Tensor) -> torch.Tensor:
        return self.decoder(z, id_emb, self.attr_emb(attrs))

    def forward(self, x: torch.Tensor, attrs: torch.Tensor,
                photo: torch.Tensor = None):
        if photo is None:
            photo = x                          # self-reconstruction during training
        id_emb         = self.photo_enc(photo)
        mu, logvar     = self.encode(x, attrs)
        z              = self.reparameterize(mu, logvar)
        recon          = self.decode(z, id_emb, attrs)
        return recon, mu, logvar

    # ------------------------------------------------------------------
    def loss(self, x: torch.Tensor, recon: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor) -> dict:
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        kl_loss    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total      = recon_loss + self.beta * kl_loss
        return {'total': total, 'recon': recon_loss, 'kl': kl_loss}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model: FullCVAE, optimizer: torch.optim.Optimizer,
                    epoch: int, history: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch':     epoch,
        'model':     model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'history':   history,
        'beta':      model.beta,
    }, path)


def load_checkpoint(path: str, model: FullCVAE,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device) -> tuple:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt['epoch'], ckpt['history']


# ---------------------------------------------------------------------------
# Warm-start from Phase 1
# ---------------------------------------------------------------------------

def warmstart_from_phase1(model: FullCVAE, phase1_ckpt_path: str,
                           device: torch.device) -> int:
    """
    Transfer encoder + attr_emb weights from a Phase 1 checkpoint.
    The decoder cannot be transferred (input dim changed: 192 → 448).
    Returns the number of parameter tensors transferred.
    """
    ckpt       = torch.load(phase1_ckpt_path, map_location=device, weights_only=False)
    p1_state   = ckpt['model']
    cur_state  = model.state_dict()
    transferred = 0

    for k, v in p1_state.items():
        if k.startswith('encoder.') or k.startswith('attr_emb.'):
            if k in cur_state and cur_state[k].shape == v.shape:
                cur_state[k] = v
                transferred += 1

    model.load_state_dict(cur_state)
    return transferred


# ---------------------------------------------------------------------------
# Training (one epoch)
# ---------------------------------------------------------------------------

def train_one_epoch(model: FullCVAE, loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    scaler=None) -> dict:
    """
    scaler: pass a torch.cuda.amp.GradScaler instance to enable AMP (fp16).
            Pass None to use full fp32.
    """
    model.train()
    sums = {'total': 0., 'recon': 0., 'kl': 0.}
    n = 0

    for images, attrs in loader:
        images = images.to(device, non_blocking=True)
        attrs  = attrs.to(device,  non_blocking=True)

        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', enabled=(scaler is not None)):
            recon, mu, logvar = model(images, attrs)
            losses = model.loss(images, recon, mu, logvar)

        if scaler is not None:
            scaler.scale(losses['total']).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses['total'].backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        for k in sums:
            sums[k] += losses[k].item()
        n += 1

    return {k: v / n for k, v in sums.items()}


# ---------------------------------------------------------------------------
# Attribute swap demo helper
# ---------------------------------------------------------------------------

SWAP_ATTRS = {
    'Smiling':    SELECTED_ATTRS.index('Smiling'),
    'Eyeglasses': SELECTED_ATTRS.index('Eyeglasses'),
    'Young':      SELECTED_ATTRS.index('Young'),
    'Bangs':      SELECTED_ATTRS.index('Bangs'),
    'Blond_Hair': SELECTED_ATTRS.index('Blond_Hair'),
}


def attribute_swap_grid(model: FullCVAE,
                        photos: torch.Tensor,
                        attrs: torch.Tensor,
                        swap_list: list,
                        device: torch.device) -> torch.Tensor:
    """
    For each photo, produce one reconstruction per swap variant.

    Args:
        photos    : [N, 3, 64, 64]
        attrs     : [N, 18]  — original attributes
        swap_list : list of (label, attr_idx, new_value)
                    e.g. [('+ Smile', 16, 1.0), ('- Smile', 16, 0.0)]
        device    : torch device

    Returns:
        grid tensor [N, 1+len(swap_list), 3, 64, 64]
          column 0 = original reconstruction, rest = swapped variants
    """
    model.eval()
    photos = photos.to(device)
    attrs  = attrs.to(device)

    rows = []
    with torch.no_grad():
        id_emb     = model.photo_enc(photos)               # [N, 256]
        attr_e_src = model.attr_emb(attrs)
        mu, logvar = model.encoder(photos, attr_e_src)
        z          = mu                                     # deterministic

        # Column 0: reconstruction with original attrs
        recon = model.decoder(z, id_emb, attr_e_src)
        rows.append(recon.unsqueeze(1))                    # [N, 1, 3, 64, 64]

        for _, attr_idx, new_val in swap_list:
            swapped_attrs = attrs.clone()
            swapped_attrs[:, attr_idx] = new_val
            attr_e_swapped = model.attr_emb(swapped_attrs)
            out = model.decoder(z, id_emb, attr_e_swapped)
            rows.append(out.unsqueeze(1))

    return torch.cat(rows, dim=1)   # [N, 1+len(swap_list), 3, 64, 64]
