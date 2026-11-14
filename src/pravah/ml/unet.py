"""Small trainable 6-band U-Net for Sen1Floods11; NOT pretrained Prithvi."""
import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Sequential):
    def __init__(self, ic: int, oc: int):
        super().__init__(nn.Conv2d(ic, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(True),
                         nn.Conv2d(oc, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(True))


class MiniUNet(nn.Module):
    def __init__(self, in_channels: int = 6, classes: int = 2, base: int = 16):
        super().__init__()
        self.e1 = Block(in_channels, base)
        self.e2 = Block(base, base * 2)
        self.e3 = Block(base * 2, base * 4)
        self.bottleneck = Block(base * 4, base * 8)
        self.d3 = Block(base * 12, base * 4)
        self.d2 = Block(base * 6, base * 2)
        self.d1 = Block(base * 3, base)
        self.out = nn.Conv2d(base, classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.e1(x)
        b = self.e2(F.max_pool2d(a, 2))
        c = self.e3(F.max_pool2d(b, 2))
        d = self.bottleneck(F.max_pool2d(c, 2))
        def up(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
            return F.interpolate(t, size=ref.shape[-2:], mode="bilinear", align_corners=False)

        c_decoded = self.d3(torch.cat([c, up(d, c)], dim=1))
        b_decoded = self.d2(torch.cat([b, up(c_decoded, b)], dim=1))
        a_decoded = self.d1(torch.cat([a, up(b_decoded, a)], dim=1))
        return self.out(a_decoded)
