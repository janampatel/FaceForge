# full_cvae_v4.py — Architecture improvements over v3
#
# Changes from v3:
#   1. SharedConvBase: first 3 encoder conv strides shared between photo_enc and VAE
#      encoder — ~30% fewer parameters, one coherent low-level feature space,
#      single forward pass through the shared layers during training (photo==x).
#
#   2. U-Net skip connections: photo_enc intermediate activations (pe1..pe4) are
#      concatenated into the decoder at matching spatial resolutions.
#      Skips come from the PHOTO encoder (not the VAE encoder) so they carry identity
#      texture without locking in attribute patterns — attribute changes are still
#      fully driven by attr_emb at the bottleneck.
#
#   3. SelfAttention2d bottleneck: multi-head self-attention at 4×4 (16 tokens, cheap).
#      Captures long-range facial structure and symmetric feature relationships that
#      stride-2 convolutions miss.
#
#   4. FREE_BITS = 0.1 baked in as the default (was 0.5 in v3 — the Phase 6R fix).
#
#   5. AuxAttrLoss default lam = 2.0 (was 0.25 in v3 — Phase 5 target value).
#
# Warm-start: use warmstart_from_v3() to transfer as many v3 weights as possible.
# The shared.b1-b3 weights are initialised from v3's encoder.conv.0-2.
# decoder.up1 transfers from v3's decoder.deconv.0 (same 512→256 shape).
# New skip-concat layers (up2-up5) and the attention block initialise fresh.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_pipeline import NUM_ATTRS, SELECTED_ATTRS

# ---------------------------------------------------------------------------
# Constants  (same as v3 except FREE_BITS)
# ---------------------------------------------------------------------------

LATENT_DIM    = 128
ATTR_EMB_DIM  = 64
PHOTO_EMB_DIM = 256
IMG_SIZE      = 128
FREE_BITS     = 0.1   # Phase 6R fix baked in; 0.5 caused KL floor=64 → posterior collapse

ENC_CHANNELS = [3, 32, 64, 128, 256, 512]
ENC_SPATIAL  = 4                                                 # 128 / 2^5
ENC_FLAT     = ENC_CHANNELS[-1] * ENC_SPATIAL * ENC_SPATIAL     # 512*4*4 = 8192
DEC_IN_DIM   = LATENT_DIM + PHOTO_EMB_DIM + ATTR_EMB_DIM        # 448

SWAP_ATTRS = {
    'Smiling':    SELECTED_ATTRS.index('Smiling'),
    'Eyeglasses': SELECTED_ATTRS.index('Eyeglasses'),
    'Young':      SELECTED_ATTRS.index('Young'),
    'Bangs':      SELECTED_ATTRS.index('Bangs'),
    'Blond_Hair': SELECTED_ATTRS.index('Blond_Hair'),
}

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _enc_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=True),
        nn.InstanceNorm2d(out_ch, affine=True),
        nn.LeakyReLU(0.2, inplace=True),
    )


def _dec_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=True),
        nn.InstanceNorm2d(out_ch, affine=True),
        nn.ReLU(inplace=True),
    )


# ---------------------------------------------------------------------------
# Self-attention (cheap at 4×4 = 16 tokens)
# ---------------------------------------------------------------------------

class SelfAttention2d(nn.Module):
    """
    Multi-head self-attention for 2-D feature maps.
    Implemented as q/k/v projections with 1×1 convolutions + residual.
    GroupNorm (instead of LayerNorm) keeps it compatible with InstanceNorm elsewhere.

    At 4×4 spatial, HW=16 and the attention matrix is [B, heads, 16, 16] — negligible cost.
    """

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.nh    = num_heads
        self.scale = (channels // num_heads) ** -0.5
        self.norm  = nn.GroupNorm(min(32, channels), channels)
        self.qkv   = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj  = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        hd = C // self.nh

        h   = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.nh, hd, H * W)
        q, k, v = qkv.unbind(1)                                    # each [B, nh, hd, HW]

        attn = torch.einsum('bhdi,bhdj->bhij', q, k) * self.scale  # [B, nh, HW, HW]
        attn = attn.softmax(dim=-1)

        out = torch.einsum('bhij,bhdj->bhdi', attn, v)             # [B, nh, hd, HW]
        out = out.reshape(B, C, H, W)
        return x + self.proj(out)                                    # residual


# ---------------------------------------------------------------------------
# Attribute embedding  (identical to all prior phases)
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
# Shared conv base — first 3 strides, reused by both photo_enc and encoder
# ---------------------------------------------------------------------------

class SharedConvBase(nn.Module):
    """
    [B, 3, 128, 128] → e3 [B, 128, 16, 16]
    Also returns (e1, e2) as shallow skip features for the decoder.

    Both PhotoEncoderV4 and EncoderV4 call this module.  During training
    when photo==x, FullCVAEv4.forward() calls it once and reuses e3 for both
    — avoiding the redundant double forward pass that independent encoders would cause.
    """

    def __init__(self):
        super().__init__()
        chs = ENC_CHANNELS
        self.b1 = _enc_block(chs[0], chs[1])   # 128 → 64:  [B, 32,  64, 64]
        self.b2 = _enc_block(chs[1], chs[2])   # 64  → 32:  [B, 64,  32, 32]
        self.b3 = _enc_block(chs[2], chs[3])   # 32  → 16:  [B, 128, 16, 16]

    def forward(self, x: torch.Tensor):
        e1 = self.b1(x)
        e2 = self.b2(e1)
        e3 = self.b3(e2)
        return e3, (e1, e2)


# ---------------------------------------------------------------------------
# Photo (identity) encoder — uses shared base + 2 private strides
# ---------------------------------------------------------------------------

class PhotoEncoderV4(nn.Module):
    """
    [B, 3, 128, 128] → id_emb [B, 256]  +  photo_skips (pe1..pe4)

    photo_skips are the intermediate activations fed to the decoder via skip
    connections.  They carry fine identity texture (hair, skin tone, eye shape)
    without encoding attribute patterns — the attr_emb handles attributes.
    """

    def __init__(self):
        super().__init__()
        chs = ENC_CHANNELS
        self.b4 = _enc_block(chs[3], chs[4])   # 16 → 8:  [B, 256, 8, 8]
        self.b5 = _enc_block(chs[4], chs[5])   # 8  → 4:  [B, 512, 4, 4]
        self.fc = nn.Sequential(
            nn.Linear(ENC_FLAT, PHOTO_EMB_DIM),
            nn.ReLU(inplace=True),
        )

    def encode_from_base(self, e3: torch.Tensor, base_skips: tuple):
        """
        Called by FullCVAEv4.forward() with pre-computed shared base output.
        Returns id_emb and photo_skips = (pe1, pe2, pe3, pe4).
        """
        pe1, pe2 = base_skips
        pe4 = self.b4(e3)                                    # [B, 256, 8, 8]
        e5  = self.b5(pe4)                                   # [B, 512, 4, 4]
        id_emb = self.fc(e5.view(e5.size(0), -1))
        return id_emb, (pe1, pe2, e3, pe4)                   # e3 serves as pe3


# ---------------------------------------------------------------------------
# VAE encoder — uses shared base + 2 private strides
# ---------------------------------------------------------------------------

class EncoderV4(nn.Module):
    """[B, 3, 128, 128] + attr_emb[64] → mu[128], logvar[128]."""

    def __init__(self):
        super().__init__()
        chs = ENC_CHANNELS
        self.b4    = _enc_block(chs[3], chs[4])
        self.b5    = _enc_block(chs[4], chs[5])
        self.fc    = nn.Linear(ENC_FLAT + ATTR_EMB_DIM, 512)
        self.fc_mu = nn.Linear(512, LATENT_DIM)
        self.fc_lv = nn.Linear(512, LATENT_DIM)

    def encode_from_base(self, e3: torch.Tensor, attr_emb: torch.Tensor):
        """Called with pre-computed shared base e3."""
        e4 = self.b4(e3)
        e5 = self.b5(e4)
        h  = F.relu(self.fc(torch.cat([e5.view(e5.size(0), -1), attr_emb], dim=1)))
        return self.fc_mu(h), self.fc_lv(h)


# ---------------------------------------------------------------------------
# Decoder with U-Net skip connections from photo encoder
# ---------------------------------------------------------------------------

class FullDecoderV4(nn.Module):
    """
    concat(z[128], id_emb[256], attr_emb[64]) → [B, 3, 128, 128]

    Skip connections from photo_enc (pe1..pe4) are concatenated at each upsampling
    step.  Skip in-channels:
        up2: 256 decoder + 256 pe4  = 512 in
        up3: 128 decoder + 128 pe3  = 256 in
        up4:  64 decoder +  64 pe2  = 128 in
        up5:  32 decoder +  32 pe1  =  64 in (final ConvTranspose)
    """

    def __init__(self):
        super().__init__()
        chs = ENC_CHANNELS   # [3, 32, 64, 128, 256, 512]

        self.fc            = nn.Linear(DEC_IN_DIM, ENC_FLAT)
        self.bottleneck_sa = SelfAttention2d(chs[5])             # attention at 4×4

        self.up1 = _dec_block(chs[5],              chs[4])       # 512     → 256,  4→8
        self.up2 = _dec_block(chs[4] + chs[4],     chs[3])       # 256+256 → 128,  8→16
        self.up3 = _dec_block(chs[3] + chs[3],     chs[2])       # 128+128 →  64, 16→32
        self.up4 = _dec_block(chs[2] + chs[2],     chs[1])       #  64+64  →  32, 32→64
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(chs[1] + chs[1], chs[0], 4, 2, 1),  # 32+32 → 3,  64→128
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, id_emb: torch.Tensor,
                attr_emb: torch.Tensor, photo_skips: tuple) -> torch.Tensor:
        pe1, pe2, pe3, pe4 = photo_skips

        h  = F.relu(self.fc(torch.cat([z, id_emb, attr_emb], dim=1)))
        h  = h.view(h.size(0), ENC_CHANNELS[-1], ENC_SPATIAL, ENC_SPATIAL)
        h  = self.bottleneck_sa(h)

        d1 = self.up1(h)
        d2 = self.up2(torch.cat([d1, pe4], dim=1))
        d3 = self.up3(torch.cat([d2, pe3], dim=1))
        d4 = self.up4(torch.cat([d3, pe2], dim=1))
        d5 = self.up5(torch.cat([d4, pe1], dim=1))
        return d5


# ---------------------------------------------------------------------------
# Full cVAE v4
# ---------------------------------------------------------------------------

class FullCVAEv4(nn.Module):
    """
    Full cVAE at 128×128 with:
      - Shared conv base (photo_enc and VAE encoder share first 3 strides)
      - U-Net skip connections from photo encoder into decoder
      - Self-attention bottleneck at 4×4
      - FREE_BITS = 0.1 (Phase 6R fix baked in as default)

    Public interface is identical to FullCVAEv3 — model(x, attrs) returns
    (recon, mu, logvar).  attribute_swap_grid uses _encode_photo() to get
    photo_skips separately.
    """

    def __init__(self, beta: float = 0.05):
        super().__init__()
        self.beta      = beta
        self.shared    = SharedConvBase()
        self.photo_enc = PhotoEncoderV4()
        self.attr_emb  = AttrEmbedding()
        self.encoder   = EncoderV4()
        self.decoder   = FullDecoderV4()

    # ------------------------------------------------------------------
    def _encode_photo(self, photo: torch.Tensor):
        """Returns id_emb, photo_skips.  Used at inference."""
        e3, base_skips = self.shared(photo)
        return self.photo_enc.encode_from_base(e3, base_skips)

    def encode(self, x: torch.Tensor, attrs: torch.Tensor):
        e3, _ = self.shared(x)
        return self.encoder.encode_from_base(e3, self.attr_emb(attrs))

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return mu

    def decode(self, z: torch.Tensor, id_emb: torch.Tensor,
               attrs: torch.Tensor, photo_skips: tuple) -> torch.Tensor:
        return self.decoder(z, id_emb, self.attr_emb(attrs), photo_skips)

    def forward(self, x: torch.Tensor, attrs: torch.Tensor,
                photo: torch.Tensor = None):
        same_input = (photo is None)
        if same_input:
            photo = x

        # Run shared base once on photo
        e3, base_skips     = self.shared(photo)
        id_emb, photo_skips = self.photo_enc.encode_from_base(e3, base_skips)

        attr_emb = self.attr_emb(attrs)

        # Reuse e3 when photo==x (avoids running shared base twice)
        if same_input:
            mu, logvar = self.encoder.encode_from_base(e3, attr_emb)
        else:
            e3_vae, _ = self.shared(x)
            mu, logvar = self.encoder.encode_from_base(e3_vae, attr_emb)

        z     = self.reparameterize(mu, logvar)
        recon = self.decoder(z, id_emb, attr_emb, photo_skips)
        return recon, mu, logvar


# ---------------------------------------------------------------------------
# Auxiliary attribute loss  (identical to v3 — lambda default raised to 2.0)
# ---------------------------------------------------------------------------

_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275,  0.40821073])
_CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711])


class AuxAttrLoss(nn.Module):
    """
    Frozen CLIP → probe → BCE against target attrs.
    Gradients flow back to the decoder, enforcing attribute compliance.
    lam=2.0 is the Phase 5 target; Phase 6R training used lam=8.0.
    """

    def __init__(self, clip_model, probe, lam: float = 2.0):
        super().__init__()
        self.clip_model = clip_model
        self.probe      = probe
        self.lam        = lam

    def _to_clip(self, x: torch.Tensor) -> torch.Tensor:
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
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model: FullCVAEv4, loader: DataLoader,
                    optimizer, device: torch.device,
                    aux_loss_fn=None,
                    perc_loss_fn=None,
                    beta: float = None,
                    free_bits: float = FREE_BITS,
                    scaler=None) -> dict:
    """
    Same interface as v3.train_one_epoch.
    Loss formula: 0.5*MSE + 1.0*perceptual + beta_annealed*KL_free_bits + aux
    """
    model.train()
    effective_beta = beta if beta is not None else model.beta
    sums = {'total': 0., 'recon': 0., 'kl': 0., 'aux': 0., 'perc': 0.}
    n = 0

    for imgs, attrs in loader:
        imgs  = imgs.to(device, non_blocking=True)
        attrs = attrs.to(device, non_blocking=True)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            recon, mu, logvar = model(imgs, attrs)

            mse_loss = F.mse_loss(recon, imgs, reduction='mean')
            if perc_loss_fn is not None:
                p_loss = perc_loss_fn(recon, imgs)
                r_loss = 0.5 * mse_loss + p_loss
            else:
                p_loss = torch.tensor(0., device=device)
                r_loss = mse_loss

            kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
            kl         = kl_per_dim.sum(dim=1).mean()

            a_loss = (aux_loss_fn(recon, attrs)
                      if aux_loss_fn is not None
                      else torch.tensor(0., device=device))

            loss = r_loss + effective_beta * kl + a_loss

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
        sums['recon'] += mse_loss.item()
        sums['kl']    += kl.item()
        sums['aux']   += a_loss.item()
        sums['perc']  += p_loss.item()
        n += 1

    return {k: v / n for k, v in sums.items()}


# ---------------------------------------------------------------------------
# Attribute swap helper
# ---------------------------------------------------------------------------

def attribute_swap_grid(model: FullCVAEv4,
                        photos: torch.Tensor,
                        attrs: torch.Tensor,
                        swap_list: list,
                        device: torch.device) -> torch.Tensor:
    """
    photos    : [N, 3, 128, 128]
    attrs     : [N, 18]
    swap_list : [(label, attr_idx, new_val), ...]
    Returns   : [N, 1+len(swap_list), 3, 128, 128]  — col 0 = original reconstruction
    """
    model.eval()
    photos = photos.to(device)
    attrs  = attrs.to(device)
    rows   = []

    with torch.no_grad():
        # Identity embedding + skip features (fixed for all swap variants)
        id_emb, photo_skips = model._encode_photo(photos)

        # z: deterministic at inference
        attr_emb = model.attr_emb(attrs)
        e3, _    = model.shared(photos)
        mu, _    = model.encoder.encode_from_base(e3, attr_emb)
        z        = mu

        rows.append(model.decoder(z, id_emb, attr_emb, photo_skips).unsqueeze(1))

        for _, attr_idx, new_val in swap_list:
            swapped                = attrs.clone()
            swapped[:, attr_idx]   = new_val
            attr_emb_s             = model.attr_emb(swapped)
            rows.append(model.decoder(z, id_emb, attr_emb_s, photo_skips).unsqueeze(1))

    return torch.cat(rows, dim=1)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model: FullCVAEv4, optimizer, epoch: int,
                    history: dict, path: str, extra: dict = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {'epoch': epoch, 'model': model.state_dict(),
               'optimizer': optimizer.state_dict(),
               'history': history, 'beta': model.beta}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: str, model: FullCVAEv4, optimizer,
                    device: torch.device) -> tuple:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt['epoch'], ckpt['history']


def warmstart_from_v3(model: FullCVAEv4, v3_ckpt: str,
                      device: torch.device) -> int:
    """
    Transfer shape-matching weights from a v3 checkpoint into v4.

    Key remapping:
        v3 encoder.conv.0-2  → v4 shared.b1-b3  (shared base initialised from encoder)
        v3 encoder.conv.3-4  → v4 encoder.b4-b5
        v3 encoder.fc/mu/lv  → v4 encoder.fc/mu/lv
        v3 photo_enc.conv.3-4→ v4 photo_enc.b4-b5
        v3 photo_enc.fc      → v4 photo_enc.fc
        v3 attr_emb          → v4 attr_emb
        v3 decoder.fc        → v4 decoder.fc  (shape unchanged)
        v3 decoder.deconv.0  → v4 decoder.up1 (both 512→256)

    Returns number of parameter tensors transferred.
    """
    v3  = torch.load(v3_ckpt, map_location=device, weights_only=False)['model']
    cur = model.state_dict()
    n   = 0

    # Prefix remapping table: (v3_prefix, v4_prefix)
    prefix_map = [
        ('encoder.conv.0', 'shared.b1'),
        ('encoder.conv.1', 'shared.b2'),
        ('encoder.conv.2', 'shared.b3'),
        ('encoder.conv.3', 'encoder.b4'),
        ('encoder.conv.4', 'encoder.b5'),
        ('photo_enc.conv.3', 'photo_enc.b4'),
        ('photo_enc.conv.4', 'photo_enc.b5'),
        ('decoder.deconv.0', 'decoder.up1'),
    ]

    for v3_pfx, v4_pfx in prefix_map:
        for v3_k, v3_v in v3.items():
            if v3_k.startswith(v3_pfx + '.'):
                v4_k = v4_pfx + v3_k[len(v3_pfx):]
                if v4_k in cur and cur[v4_k].shape == v3_v.shape:
                    cur[v4_k] = v3_v
                    n += 1

    # Direct key matches (attr_emb, encoder.fc*, photo_enc.fc*, decoder.fc)
    direct_prefixes = (
        'attr_emb.',
        'encoder.fc.', 'encoder.fc_mu.', 'encoder.fc_lv.',
        'photo_enc.fc.',
        'decoder.fc.',
    )
    for v3_k, v3_v in v3.items():
        if any(v3_k.startswith(p) for p in direct_prefixes):
            if v3_k in cur and cur[v3_k].shape == v3_v.shape:
                cur[v3_k] = v3_v
                n += 1

    model.load_state_dict(cur, strict=False)
    return n
