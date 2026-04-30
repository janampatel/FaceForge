# FaceForge

A Conditional Variational Autoencoder (cVAE) that generates and edits celebrity face images conditioned on binary attributes, identity photos, and natural language prompts via CLIP.

Built on the CelebA dataset (202,599 face images, 40 binary attribute labels) and designed to run on Google Colab free tier (T4 GPU).

---

## What It Does

1. **Attribute-conditioned generation** — generate a face from noise + a binary attribute vector (e.g., Smiling=1, Eyeglasses=1)
2. **Identity-preserving attribute swap** — encode a real photo, then decode with different attributes applied while preserving identity
3. **Natural language control** — type a prompt ("add glasses and make me look older") and CLIP maps it to the attribute vector automatically
4. **Cartoon style transfer** — apply a pretrained anime/cartoon style on top of the cVAE output with a lightweight few-shot adapter

---

## Architecture Overview

```
Text prompt
    └─► CLIP encoder (frozen) ──► Linear probe (18 neurons) ──► attr vector [18]
                                                                         │
Real photo ──► Photo Encoder ──► id_emb [256]                           │
                    └─────────────────────────────────────────────┐      │
                                                                  ▼      ▼
Noise z [128] ──────────────────────────────────────────────► Decoder ──► Face [3, 128, 128]
                                                                  ▲
                                                             concat(z, id_emb, attr_emb)
```

The model progressed through 8 development phases, each adding one capability while keeping prior phases intact.

---

## File Structure

| File | Phase | What it contains |
|---|---|---|
| `data_pipeline.py` | 0 | CelebA dataset class, 18-attribute selection, DataLoader |
| `baseline_cvae.py` | 1 | cVAE conditioned on labels only (no photo), 64×64 |
| `full_cvae.py` | 2 | cVAE with photo encoder for identity preservation, 64×64 |
| `clip_probe.py` | 3 | Frozen CLIP + trained linear probe for text-to-attribute mapping |
| `full_cvae_v2.py` | 4 | Phase 2 rebuilt with InstanceNorm (replaces BatchNorm), 64×64 |
| `full_cvae_v3.py` | 5+6 | Final model: 128×128, InstanceNorm, perceptual loss, KL annealing |
| `losses.py` | 5 | Multi-scale VGG16 perceptual loss + auxiliary attribute classification loss |
| `style_transfer.py` | 8A | Frozen AnimeGANv2 cartoonizer wrapper |
| `style_adapter.py` | 8B–D | Residual adapter (~880 params) for few-shot style personalization |
| `evaluate.py` | 7 | FID, SSIM/PSNR, per-attribute accuracy, latent traversal figures |
| `train.ipynb` | — | Main Colab training notebook (all phases) |
| `train_baseline.ipynb` | — | Baseline-only training notebook |

---

## Selected Attributes (18 of 40)

`5_o_Clock_Shadow`, `Arched_Eyebrows`, `Attractive`, `Bags_Under_Eyes`, `Bald`, `Bangs`, `Black_Hair`, `Blond_Hair`, `Brown_Hair`, `Bushy_Eyebrows`, `Eyeglasses`, `Male`, `Mouth_Slightly_Open`, `Mustache`, `No_Beard`, `Pale_Skin`, `Smiling`, `Young`

---

## Model Evolution

### Phase 1 — Baseline cVAE (64×64, labels only)
- 4-layer Conv encoder/decoder, BatchNorm, β-VAE loss
- Validates attribute-controlled generation before adding identity complexity

### Phase 2 — Full cVAE (64×64, photo + labels)
- Photo encoder encodes a real face to latent z
- Decoder receives `concat(z, id_emb, attr_emb)` for identity-preserving attribute swap

### Phase 3 — CLIP Text Conditioning
- Frozen CLIP ViT-B/32 maps text to 512-dim embedding
- Trained 18-neuron linear probe converts embedding to soft attribute logits
- Enables prompts like "add a smile" or "make me look older"

### Phase 4 — InstanceNorm
- Replaces every BatchNorm2d with InstanceNorm2d
- Eliminates cross-sample dependencies that conflict with per-sample KL divergence

### Phase 5+6 — 128×128 + Perceptual Loss + Attribute Compliance Fix
Final training config:
```
model:         full_cvae_v3.py (5-layer InstanceNorm encoder/decoder)
optimizer:     Adam, G_LR=1e-4, D_LR=5e-6, betas=(0.5, 0.999)
lr_schedule:   CosineAnnealingWarmRestarts(T_0=10)
epochs:        30 (Phase 6R fine-tune from Phase 5 checkpoint)
batch_size:    128
latent_dim:    128
KL:            beta annealed 0.0 → 0.1 over 15 epochs, free_bits=0.1 nats/dim
loss:          0.5·MSE + 1.0·perceptual(VGG multi-scale) + 0.05·KL + 8.0·aux + 0.5·adv + 10.0·feature_matching
discriminator: PatchDiscriminator with spectral norm, LSGAN, G:D update ratio 5:1
```

### Phase 7 — Evaluation Suite
- **FID**: 10K generated vs 10K real CelebA images via `pytorch-fid`
- **Attribute compliance**: CLIP probe accuracy on generated images (target >85% per attribute)
- **Reconstruction quality**: SSIM and PSNR on 500 test images
- **Latent traversal**: attribute swept −1 → +1 with z fixed

### Phase 8 — Cartoon Style Transfer
- Frozen AnimeGANv2 (pretrained) as base cartoonizer
- Residual `StyleAdapter` (880 parameters) for few-shot personalization with 5–20 images
- Uncertainty-aware training: adaptive LR and FGSM perturbation scaled by output entropy

---

## Loss Function (Phase 5+)

**Multi-scale VGG16 perceptual loss** uses three layers for full frequency coverage:
- `relu1_2` (weight 0.25) — edges, fine texture
- `relu2_2` (weight 0.5) — facial structure, hair
- `relu3_3` (weight 1.0) — semantic/attribute-level features

**Auxiliary attribute classification loss** forces the decoder to actually respond to attribute conditioning by penalizing reconstructions that don't satisfy the target attribute vector. This closes the shortcut where the decoder ignores `attr_emb` and copies from `id_emb`.

---

## Setup

```bash
pip install torch torchvision clip-by-openai scikit-image matplotlib pytorch-fid
```

For Google Colab, mount Drive and use the provided notebooks. Checkpoints are saved to Drive every 5 epochs to survive session timeouts.

**CelebA download fallback:** if `torchvision.datasets.CelebA` fails due to Google Drive limits, download manually from Kaggle (`jessicali9530/celeba-dataset`) and load from `/content/drive/MyDrive/celeba/`.

---

## Checkpoints

Saved to Google Drive under `checkpoints/`:
```
baseline_epoch_*.pt          Phase 1
full_cvae_v2_best.pt         Phase 4 (InstanceNorm, 64×64)
full_cvae_v3_best.pt         Phase 5 (128×128 — sharp, weak attribute swap)
full_cvae_v3_p6r_best.pt     Phase 6R (128×128 — sharp + attribute compliant)
clip_probe.pt                Phase 3
```

---

## Known Issues and Lessons Learned

- **BatchNorm in VAEs** causes cross-sample KL interference. InstanceNorm is the correct choice for all phases after the baseline.
- **KL collapse with free_bits=0.5**: sets the KL floor at `0.5 × 128 = 64 nats`, effectively preventing z from encoding any information. Lowered to 0.1 in Phase 6.
- **VGG at non-standard resolution**: running VGG16 at 128×128 instead of 224×224 produces miscalibrated feature magnitudes (~12 vs ~6). VGG must always run at 224×224.
- **Discriminator dominance**: with D_LR=4e-5 and G_LR=1e-5, D_loss dropped to near zero by epoch 1. Fix: D_LR=5e-6, G:D update ratio 5:1, feature matching loss.
- **Aux loss swamped by perceptual loss**: with `lambda_aux=2.0` and `perc_weight=1.0`, aux was ~4% of the combined loss and ignored by the decoder. Raised to `lambda_aux=8.0` in Phase 6.
