# full_cvae_v3.py — Phase 5: Resolution upgrade to 128×128
#
# What changes from Phase 4 (full_cvae_v2.py):
#   - Input/output resolution: 64×64 → 128×128
#   - 5 Conv/ConvTranspose strides instead of 4 (one extra layer at each end)
#   - Channel progression: [3, 32, 64, 128, 256, 512]  — new 512-ch layer
#   - ENC_FLAT: 256*4*4=4096 → 512*4*4=8192
#   - KL: standard with fixed β=0.1 — no free-bits (gradient zeroing in Phase 4
#     left the encoder passive; low fixed β is simpler and keeps real gradients)
#   - Warm-start from Phase 4 checkpoint: inner layers transfer, new 512-ch
#     layers at encoder top / decoder bottom initialise fresh.
#   - Everything else (attr embedding, LATENT_DIM, AuxAttrLoss) unchanged.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_pipeline import NUM_ATTRS, SELECTED_ATTRS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LATENT_DIM    = 128
ATTR_EMB_DIM  = 64
PHOTO_EMB_DIM = 256
IMG_SIZE      = 128

# 5-stride channel progression: input + 5 output channel counts
ENC_CHANNELS = [3, 32, 64, 128, 256, 512]
# After 5 stride-2 convs on 128×128:  128 / 2^5 = 4
ENC_SPATIAL  = 4
ENC_FLAT     = ENC_CHANNELS[-1] * ENC_SPATIAL * ENC_SPATIAL  # 512*4*4 = 8192
DEC_IN_DIM   = LATENT_DIM + PHOTO_EMB_DIM + ATTR_EMB_DIM     # 128+256+64 = 448

SWAP_ATTRS = {
    'Smiling':    SELECTED_ATTRS.index('Smiling'),
    'Eyeglasses': SELECTED_ATTRS.index('Eyeglasses'),
    'Young':      SELECTED_ATTRS.index('Young'),
    'Bangs':      SELECTED_ATTRS.index('Bangs'),
    'Blond_Hair': SELECTED_ATTRS.index('Blond_Hair'),
}

# ---------------------------------------------------------------------------
# Building blocks — InstanceNorm throughout
# ---------------------------------------------------------------------------

def _enc_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Conv2d 4×4 stride-2 + InstanceNorm + LeakyReLU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=True),
        nn.InstanceNorm2d(out_ch, affine=True),
        nn.LeakyReLU(0.2, inplace=True),
    )


def _dec_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """ConvTranspose2d 4×4 stride-2 + InstanceNorm + ReLU."""
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=True),
        nn.InstanceNorm2d(out_ch, affine=True),
        nn.ReLU(inplace=True),
    )


# ---------------------------------------------------------------------------
# Attribute embedding (identical to all prior phases)
# ---------------------------------------------------------------------------

class AttrEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NUM_ATTRS, ATTR_EMB_DIM),
            nn.ReLU(inplace=True),
            nn.Linear(ATTR_EMB_DIM, ATTR_EMB_DIM),
            nn.ReLU(inplace=True),
        )

    def forward(self, attrs: torch.Tensor) -> torch.Tensor:
        return self.net(attrs)


# ---------------------------------------------------------------------------
# Photo (identity) encoder — 5 conv strides for 128×128 input
# ---------------------------------------------------------------------------

class PhotoEncoderV3(nn.Module):
    """[B, 3, 128, 128] → [B, 256] identity embedding."""

    def __init__(self):
        super().__init__()
        chs = ENC_CHANNELS
        self.conv = nn.Sequential(
            _enc_block(chs[0], chs[1]),   # 128 → 64
            _enc_block(chs[1], chs[2]),   # 64  → 32
            _enc_block(chs[2], chs[3]),   # 32  → 16
            _enc_block(chs[3], chs[4]),   # 16  → 8
            _enc_block(chs[4], chs[5]),   # 8   → 4
        )
        self.fc = nn.Sequential(
            nn.Linear(ENC_FLAT, PHOTO_EMB_DIM),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x).view(x.size(0), -1))


# ---------------------------------------------------------------------------
# VAE encoder — 5 conv strides for 128×128 input
# ---------------------------------------------------------------------------

class EncoderV3(nn.Module):
    """[B, 3, 128, 128] + attr_emb[64] → mu[128], logvar[128]."""

    def __init__(self):
        super().__init__()
        chs = ENC_CHANNELS
        self.conv = nn.Sequential(
            _enc_block(chs[0], chs[1]),
            _enc_block(chs[1], chs[2]),
            _enc_block(chs[2], chs[3]),
            _enc_block(chs[3], chs[4]),
            _enc_block(chs[4], chs[5]),
        )
        self.fc    = nn.Linear(ENC_FLAT + ATTR_EMB_DIM, 512)
        self.fc_mu = nn.Linear(512, LATENT_DIM)
        self.fc_lv = nn.Linear(512, LATENT_DIM)

    def forward(self, x: torch.Tensor, attr_emb: torch.Tensor):
        h = self.conv(x).view(x.size(0), -1)
        h = F.relu(self.fc(torch.cat([h, attr_emb], dim=1)))
        return self.fc_mu(h), self.fc_lv(h)


# ---------------------------------------------------------------------------
# Decoder — 5 ConvTranspose strides → 128×128 output
# ---------------------------------------------------------------------------

class FullDecoderV3(nn.Module):
    """concat(z[128], id_emb[256], attr_emb[64]) → [B, 3, 128, 128]."""

    def __init__(self):
        super().__init__()
        chs = ENC_CHANNELS
        self.fc = nn.Linear(DEC_IN_DIM, ENC_FLAT)
        self.deconv = nn.Sequential(
            _dec_block(chs[5], chs[4]),                           # 4  → 8
            _dec_block(chs[4], chs[3]),                           # 8  → 16
            _dec_block(chs[3], chs[2]),                           # 16 → 32
            _dec_block(chs[2], chs[1]),                           # 32 → 64
            nn.ConvTranspose2d(chs[1], chs[0], 4, 2, 1),         # 64 → 128
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor,
                id_emb: torch.Tensor,
                attr_emb: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc(torch.cat([z, id_emb, attr_emb], dim=1)))
        return self.deconv(h.view(h.size(0), ENC_CHANNELS[-1], ENC_SPATIAL, ENC_SPATIAL))


# ---------------------------------------------------------------------------
# Full cVAE v3
# ---------------------------------------------------------------------------

class FullCVAEv3(nn.Module):
    """
    Full cVAE at 128×128 with InstanceNorm and auxiliary attribute loss.
    Drop-in interface replacement for FullCVAEv2 — only input/output size differs.
    """

    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta      = beta
        self.photo_enc = PhotoEncoderV3()
        self.attr_emb  = AttrEmbedding()
        self.encoder   = EncoderV3()
        self.decoder   = FullDecoderV3()

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


# ---------------------------------------------------------------------------
# Auxiliary attribute loss (identical to v2)
# ---------------------------------------------------------------------------

_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275,  0.40821073])
_CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711])


class AuxAttrLoss(nn.Module):
    """
    Passes the decoded image through frozen CLIP → probe → BCE against target attrs.
    Gradients flow back to the decoder, enforcing attribute compliance.
    """

    def __init__(self, clip_model, probe, lam: float = 0.25):
        super().__init__()
        self.clip_model = clip_model
        self.probe      = probe
        self.lam        = lam

    def _to_clip(self, x: torch.Tensor) -> torch.Tensor:
        """[-1,1] 128×128 tensor → CLIP-normalised 224×224."""
        x    = (x + 1.0) / 2.0
        x    = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        mean = _CLIP_MEAN.view(1, 3, 1, 1).to(x.device)
        std  = _CLIP_STD.view(1, 3, 1, 1).to(x.device)
        return (x - mean) / std

    def forward(self, recon: torch.Tensor, target_attrs: torch.Tensor) -> torch.Tensor:
        clip_in = self._to_clip(recon.float())
        feats   = self.clip_model.encode_image(clip_in).float()
        feats   = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
        logits  = self.probe(feats)
        return self.lam * F.binary_cross_entropy_with_logits(logits, target_attrs)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model: FullCVAEv3, optimizer, epoch: int,
                    history: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({'epoch': epoch, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'history': history, 'beta': model.beta}, path)


def load_checkpoint(path: str, model: FullCVAEv3, optimizer,
                    device: torch.device) -> tuple:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt['epoch'], ckpt['history']


def warmstart_from_phase4(model: FullCVAEv3, phase4_ckpt: str,
                           device: torch.device) -> int:
    """
    Transfer shape-matching weights from the Phase 4 (64×64) checkpoint.

    What transfers: attr_emb, inner encoder conv layers (32/64/128/256 ch),
    inner decoder ConvTranspose layers (256/128/64/32 ch), fc_mu, fc_lv.

    What doesn't transfer (shape changed):
      - encoder/photo_enc top conv (new 512-ch layer)
      - encoder.fc  (ENC_FLAT changed 4096→8192)
      - decoder.fc  (output changed 4096→8192)
      - decoder first ConvTranspose (512→256 vs old 256→128)

    Returns number of parameter tensors transferred.
    """
    p4 = torch.load(phase4_ckpt, map_location=device, weights_only=False)['model']
    cur = model.state_dict()
    n = 0
    for k, v in p4.items():
        if k in cur and cur[k].shape == v.shape:
            cur[k] = v
            n += 1
    model.load_state_dict(cur, strict=False)
    return n


# ---------------------------------------------------------------------------
# Training loop — standard KL with fixed beta
# ---------------------------------------------------------------------------

def train_one_epoch(model: FullCVAEv3, loader: DataLoader,
                    optimizer, device: torch.device,
                    aux_loss_fn=None, scaler=None) -> dict:
    """
    Standard KL with model.beta (fixed, no annealing).
    Free-bits approach from Phase 4 zeroed encoder gradients entirely — this
    keeps real gradients flowing with a small beta to slow collapse.
    """
    model.train()
    sums = {'total': 0., 'recon': 0., 'kl': 0., 'aux': 0.}
    n = 0

    for imgs, attrs in loader:
        imgs  = imgs.to(device, non_blocking=True)
        attrs = attrs.to(device, non_blocking=True)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            recon, mu, logvar = model(imgs, attrs)
            r_loss = F.mse_loss(recon, imgs, reduction='mean')
            kl     = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
            a_loss = (aux_loss_fn(recon, attrs)
                      if aux_loss_fn is not None
                      else torch.tensor(0., device=device))
            loss   = r_loss + model.beta * kl + a_loss

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        sums['total'] += loss.item()
        sums['recon'] += r_loss.item()
        sums['kl']    += kl.item()
        sums['aux']   += a_loss.item()
        n += 1

    return {k: v / n for k, v in sums.items()}


# ---------------------------------------------------------------------------
# Attribute swap helper
# ---------------------------------------------------------------------------

def attribute_swap_grid(model: FullCVAEv3,
                        photos: torch.Tensor,
                        attrs: torch.Tensor,
                        swap_list: list,
                        device: torch.device) -> torch.Tensor:
    """
    photos    : [N, 3, 128, 128]
    attrs     : [N, 18]
    swap_list : [(label, attr_idx, new_val), ...]
    Returns   : [N, 1+len(swap_list), 3, 128, 128]
                col 0 = reconstruction with original attrs
    """
    model.eval()
    photos = photos.to(device)
    attrs  = attrs.to(device)
    rows = []

    with torch.no_grad():
        id_emb = model.photo_enc(photos)
        attr_e = model.attr_emb(attrs)
        mu, _  = model.encoder(photos, attr_e)
        z      = mu  # deterministic at inference

        rows.append(model.decoder(z, id_emb, attr_e).unsqueeze(1))
        for _, attr_idx, new_val in swap_list:
            swapped = attrs.clone()
            swapped[:, attr_idx] = new_val
            rows.append(model.decoder(z, id_emb, model.attr_emb(swapped)).unsqueeze(1))

    return torch.cat(rows, dim=1)
