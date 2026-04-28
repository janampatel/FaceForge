import torch
import torch.nn.functional as F
from pathlib import Path


class Cartoonizer:
    """Wraps frozen AnimeGANv2 generator.

    Input/output: [-1, 1] tensors at any resolution.
    Internally upscales to 256×256 before the generator.
    """

    def __init__(self, device, ckpt_path=None):
        self.device = device
        if ckpt_path and Path(ckpt_path).exists():
            model = torch.hub.load(
                "bryandlee/animegan2-pytorch:main", "generator",
                pretrained=False, progress=False,
            )
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        else:
            model = torch.hub.load(
                "bryandlee/animegan2-pytorch:main", "generator",
                pretrained="face_paint_512_v2", progress=True,
            )
            if ckpt_path:
                Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ckpt_path)

        self.model = model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def stylize(self, x: torch.Tensor, out_size: int = 512) -> torch.Tensor:
        """x: [B, 3, H, W] in [-1, 1] → cartoon [B, 3, out_size, out_size] in [-1, 1].

        Default is 512 because face_paint_512_v2 was trained at that resolution;
        running at 256 discards detail the generator was trained to produce.
        """
        up = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return self.model(up)
