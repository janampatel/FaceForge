# losses.py — Phase 6: Multi-Scale VGG Perceptual Loss
#
# What this builds and why:
# MSE reconstruction loss is mean-field — it averages over uncertainty and produces
# blurry outputs. This is severe at 128×128. VGG perceptual loss compares feature
# activations instead of raw pixels. Using three VGG layers at different depths:
#
#   relu1_2 (low-level):  edges, fine texture — catches hair strands, pores
#   relu2_2 (mid-level):  facial structure, hair patterns — main sharpness driver
#   relu3_3 (high-level): semantic features, attribute-level info — identity/expression
#
# Combined loss (weights 0.25 / 0.5 / 1.0) gives coverage across all frequency bands,
# yielding sharper and more attribute-faithful reconstructions than single-scale loss.
#
# Total Phase 5+6 loss:
#   0.5 * mse + 1.0 * perceptual_multiscale + beta_annealed * kl_free_bits + aux

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PerceptualLoss(nn.Module):
    """
    Multi-scale VGG16 perceptual loss at relu1_2, relu2_2, relu3_3.

    VGG16 features layer indices:
      0-3:  Conv(3,64) ReLU Conv(64,64) ReLU(relu1_2)
      4-8:  MaxPool Conv(64,128) ReLU Conv(128,128) ReLU(relu2_2)
      9-15: MaxPool Conv(128,256) ReLU Conv(256,256) ReLU Conv(256,256) ReLU(relu3_3)

    A single forward pass is used — each block feeds into the next, so we compute
    all three loss terms in one pass per image (recon and target).

    Input convention: [-1, 1] tensors of any resolution. Internally resized to
    224×224 and ImageNet-normalised before passing through VGG. No gradients flow
    through VGG (all parameters frozen).

    Args:
        weight: outer scalar multiplier on the total multi-scale loss. Default 1.0.
    """

    # Slices of vgg.features for the three blocks
    _BLOCK1_END = 4    # 0..3  → relu1_2
    _BLOCK2_END = 9    # 4..8  → relu2_2
    _BLOCK3_END = 16   # 9..15 → relu3_3

    # Per-scale loss weights (low → high frequency)
    _SCALE_WEIGHTS = [0.25, 0.5, 1.0]

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        feats = list(vgg.features.children())

        # Three sequential blocks — each block takes output of the previous
        self.block1 = nn.Sequential(*feats[:self._BLOCK1_END])    # input → relu1_2
        self.block2 = nn.Sequential(*feats[self._BLOCK1_END:self._BLOCK2_END])  # → relu2_2
        self.block3 = nn.Sequential(*feats[self._BLOCK2_END:self._BLOCK3_END])  # → relu3_3

        for p in self.parameters():
            p.requires_grad = False

        # ImageNet normalisation — buffers move automatically with .to(device)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode=True):
        # Never put VGG in train mode — BN/pooling stats must stay frozen
        return super().train(False)

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """[-1,1] float → ImageNet-normalised 224×224."""
        x = (x + 1.0) / 2.0
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return (x - self.mean) / self.std

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            recon:  [B, 3, H, W] in [-1, 1] — decoder output, gradients flow here
            target: [B, 3, H, W] in [-1, 1] — real image, detached inside
        Returns:
            Weighted sum of three-scale feature MSE losses × self.weight
        """
        r = self._preprocess(recon.float())
        t = self._preprocess(target.float())

        loss = torch.tensor(0.0, device=recon.device)
        w = self._SCALE_WEIGHTS

        # Block 1 → relu1_2
        r1 = self.block1(r);  t1 = self.block1(t).detach()
        loss = loss + w[0] * F.mse_loss(r1, t1)

        # Block 2 → relu2_2  (feeds from block1 output)
        r2 = self.block2(r1); t2 = self.block2(t1).detach()
        loss = loss + w[1] * F.mse_loss(r2, t2)

        # Block 3 → relu3_3  (feeds from block2 output)
        r3 = self.block3(r2); t3 = self.block3(t2).detach()
        loss = loss + w[2] * F.mse_loss(r3, t3)

        return self.weight * loss
