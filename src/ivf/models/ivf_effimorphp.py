import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0

try:
    from torchvision.models import EfficientNet_B0_Weights
except ImportError:  # pragma: no cover - older torchvision
    EfficientNet_B0_Weights = None


class SimAM(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = (x - mean).pow(2).mean(dim=(2, 3), keepdim=True)
        energy = (x - mean).pow(2) / (4 * (var + self.eps)) + 0.5
        return x * torch.sigmoid(energy)


class MSMABlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dw3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.dw5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.dw7 = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.pw = nn.Conv2d(channels * 3, channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.cat([self.dw3(x), self.dw5(x), self.dw7(x)], dim=1)
        y = self.pw(y)
        y = self.bn(y)
        y = self.act(y)
        return x + y


class MCMBlock(nn.Module):
    def __init__(self, channels: int, expand: int = 2):
        super().__init__()
        self.dw = nn.Conv2d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.SiLU()
        self.pw_expand = nn.Conv2d(channels, channels * expand, kernel_size=1)
        self.pw_project = nn.Conv2d(channels * expand, channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dw(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pw_expand(x)
        x = self.act(x)
        x = self.pw_project(x)
        x = self.bn2(x)
        return residual + x


class ECA(nn.Module):
    def __init__(self, channels: int, kernel_size: int | None = None, gamma: int = 2, b: int = 1):
        super().__init__()
        if kernel_size is None:
            k = int(abs((math.log2(channels) / gamma) + b))
            kernel_size = k if k % 2 == 1 else k + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(1, 2))
        y = torch.sigmoid(y.transpose(1, 2).unsqueeze(-1))
        return x * y


class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.pow(1.0 / self.p)
        return x.flatten(1)


class IVF_EffiMorphPP(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        use_simam: bool = True,
        use_msma: bool = True,
        use_mcm: bool = True,
        mcm_depth: int = 1,
        mcm_expand: int = 2,
        use_eca: bool = True,
        pooling: str = "gem",
        dropout: float = 0.3,
        gem_p: float = 3.0,
        eca_kernel: int | None = None,
        reduce_channels: int | None = 640,
    ):
        super().__init__()
        if EfficientNet_B0_Weights is not None:
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            backbone = efficientnet_b0(weights=weights)
        else:
            backbone = efficientnet_b0(pretrained=pretrained)

        self.trunk = backbone.features
        channels = backbone.classifier[-1].in_features
        if reduce_channels is not None and reduce_channels > 0 and reduce_channels != channels:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(channels, reduce_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduce_channels),
                nn.SiLU(),
            )
            channels = reduce_channels
        else:
            self.bottleneck = nn.Identity()

        self.simam = SimAM() if use_simam else nn.Identity()
        self.msma = MSMABlock(channels) if use_msma else nn.Identity()
        if use_mcm and mcm_depth > 0:
            self.mcm = nn.Sequential(*[MCMBlock(channels, expand=mcm_expand) for _ in range(mcm_depth)])
        else:
            self.mcm = nn.Identity()
        self.eca = ECA(channels, kernel_size=eca_kernel) if use_eca else nn.Identity()

        if pooling.lower() == "gem":
            self.pool = GeM(p=gem_p)
            self._pool_is_gem = True
        elif pooling.lower() == "gap":
            self.pool = nn.AdaptiveAvgPool2d(1)
            self._pool_is_gem = False
        else:
            raise ValueError(f"Unknown pooling type: {pooling}")

        self.dropout = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()
        self.head = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        x = self.trunk(x)
        x = self.bottleneck(x)
        x = self.simam(x)
        x = self.msma(x)
        x = self.mcm(x)
        x = self.eca(x)
        if self._pool_is_gem:
            feats = self.pool(x)
        else:
            feats = self.pool(x).flatten(1)
        feats = self.dropout(feats)
        logits = self.head(feats).squeeze(1)
        if return_features:
            return logits, feats
        return logits


class EffiMorphPPEncoder(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        use_simam: bool = True,
        use_msma: bool = True,
        use_mcm: bool = True,
        mcm_depth: int = 1,
        mcm_expand: int = 2,
        use_eca: bool = True,
        eca_kernel: int | None = None,
        reduce_channels: int | None = 640,
    ):
        super().__init__()
        if EfficientNet_B0_Weights is not None:
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            backbone = efficientnet_b0(weights=weights)
        else:
            backbone = efficientnet_b0(pretrained=pretrained)

        self.trunk = backbone.features
        channels = backbone.classifier[-1].in_features
        if reduce_channels is not None and reduce_channels > 0 and reduce_channels != channels:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(channels, reduce_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduce_channels),
                nn.SiLU(),
            )
            channels = reduce_channels
        else:
            self.bottleneck = nn.Identity()

        self.simam = SimAM() if use_simam else nn.Identity()
        self.msma = MSMABlock(channels) if use_msma else nn.Identity()
        if use_mcm and mcm_depth > 0:
            self.mcm = nn.Sequential(*[MCMBlock(channels, expand=mcm_expand) for _ in range(mcm_depth)])
        else:
            self.mcm = nn.Identity()
        self.eca = ECA(channels, kernel_size=eca_kernel) if use_eca else nn.Identity()

        self.feature_dim = channels
        self.returns_feature_map = True

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.trunk(x)
        x = self.bottleneck(x)
        x = self.simam(x)
        x = self.msma(x)
        x = self.mcm(x)
        x = self.eca(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)
