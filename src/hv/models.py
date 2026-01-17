# models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, efficientnet_b0

try:
    from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights
except ImportError:  # pragma: no cover - older torchvision
    ResNet50_Weights = None
    EfficientNet_B0_Weights = None


# -------------------------
# Utilities
# -------------------------

class LayerNorm2d(nn.Module):
    """
    LayerNorm for channels-first tensors: (N, C, H, W).
    Normalizes over channel dimension per spatial location.
    """
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, H, W]
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    """
    Stochastic depth per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


def _trunc_normal_(tensor: torch.Tensor, mean=0.0, std=0.02):
    """
    Lightweight trunc_normal init (good enough for ConvNeXt-style).
    """
    with torch.no_grad():
        return tensor.normal_(mean, std).clamp_(mean - 2 * std, mean + 2 * std)


# -------------------------
# ConvNeXt-Lite blocks
# -------------------------

class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt-style block:
      - Depthwise 7x7 conv
      - LayerNorm (channels-first)
      - Pointwise MLP (1x1 convs): C -> 4C -> C
      - Residual + DropPath
    """
    def __init__(
        self,
        dim: int,
        mlp_ratio: int = 4,
        drop_path_prob: float = 0.0,
        layer_scale_init: float = 1e-6,
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.pwconv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)

        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim)) if layer_scale_init > 0 else None
        self.drop_path = DropPath(drop_path_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = x * self.gamma[:, None, None]

        x = identity + self.drop_path(x)
        return x


class Downsample(nn.Module):
    """
    Downsample between stages: LayerNorm2d + 2x2 stride-2 conv
    (ConvNeXt-like downsample).
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm = LayerNorm2d(in_dim)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.conv(x)
        return x


# -------------------------
# IVFConvNeXtLite (Recommended)
# -------------------------

class IVFConvNeXtLite(nn.Module):
    """
    Strong in-domain backbone for embryo quality classification.
    - Modern CNN (ConvNeXt-style)
    - Works well on texture/morphology
    - No hard-coded flatten sizes
    - Output: logits shape [B]
    """
    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 1,          # binary logits
        dims: Tuple[int, int, int, int] = (64, 128, 256, 512),
        depths: Tuple[int, int, int, int] = (2, 2, 6, 2),
        mlp_ratio: int = 4,
        drop_path_rate: float = 0.1,
        head_dropout: float = 0.2,
        layer_scale_init: float = 1e-6,
        stem_stride: int = 4,          # 4 is standard ConvNeXt; 2 preserves more detail
    ):
        super().__init__()
        assert len(dims) == 4 and len(depths) == 4

        # Stem: patchify-like conv
        # stride=4 gives speed; stride=2 preserves detail (often beneficial for IVF).
        k = stem_stride
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=k, stride=stem_stride, padding=0),
            LayerNorm2d(dims[0]),
        )

        # Stochastic depth schedule
        total_blocks = sum(depths)
        dp_rates = torch.linspace(0, drop_path_rate, total_blocks).tolist()

        cur = 0
        self.stages = nn.ModuleList()
        for i in range(4):
            blocks = []
            for j in range(depths[i]):
                blocks.append(
                    ConvNeXtBlock(
                        dim=dims[i],
                        mlp_ratio=mlp_ratio,
                        drop_path_prob=dp_rates[cur + j],
                        layer_scale_init=layer_scale_init,
                    )
                )
            cur += depths[i]
            self.stages.append(nn.Sequential(*blocks))
            if i < 3:
                self.stages.append(Downsample(dims[i], dims[i + 1]))

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head_dropout = nn.Dropout(head_dropout) if head_dropout and head_dropout > 0 else nn.Identity()
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            _trunc_normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for layer in self.stages:
            x = layer(x)
        x = self.pool(x).flatten(1)         # [B, C]
        x = self.head_dropout(x)
        x = self.head(x)                    # [B, 1]
        return x.squeeze(1)                 # [B] logits


class BinaryLogitsWrapper(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(x)
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        return logits


def build_resnet50_baseline(pretrained: bool, dropout: float) -> nn.Module:
    if ResNet50_Weights is not None:
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
    else:
        model = resnet50(pretrained=pretrained)
    in_features = model.fc.in_features
    if dropout and dropout > 0:
        model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, 1))
    else:
        model.fc = nn.Linear(in_features, 1)
    return BinaryLogitsWrapper(model)


def build_efficientnet_b0_baseline(pretrained: bool, dropout: float) -> nn.Module:
    if EfficientNet_B0_Weights is not None:
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
    else:
        model = efficientnet_b0(pretrained=pretrained)
    in_features = model.classifier[-1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, 1))
    return BinaryLogitsWrapper(model)


# -------------------------
# build_model(cfg)
# -------------------------

def _cfg_get(cfg, key: str, default=None):
    """
    Works with OmegaConf DictConfig-like objects and plain dicts.
    """
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def build_model(cfg):
    """
    Expected cfg structure (minimal):
      cfg.model.name: "ivf_convnext_lite" (default)
        - "resnet50_baseline"
        - "efficientnet_b0_baseline"
      cfg.model.dropout: head dropout
      cfg.model.pretrained: false
      cfg.model.stem_stride: 2 or 4
      cfg.model.dims: [64,128,256,512]
      cfg.model.depths: [2,2,6,2]
      cfg.model.drop_path_rate: 0.1

    Baseline examples:
      model:
        name: resnet50_baseline
        dropout: 0.2
        pretrained: false

      model:
        name: efficientnet_b0_baseline
        dropout: 0.2
        pretrained: false
    """
    model_cfg = _cfg_get(cfg, "model", cfg)

    name = str(_cfg_get(model_cfg, "name", "ivf_convnext_lite")).lower()
    head_dropout = float(_cfg_get(model_cfg, "dropout", 0.2))
    pretrained = bool(_cfg_get(model_cfg, "pretrained", False))
    stem_stride = int(_cfg_get(model_cfg, "stem_stride", 2))  # for IVF, 2 often better than 4
    drop_path_rate = float(_cfg_get(model_cfg, "drop_path_rate", 0.1))

    dims = tuple(_cfg_get(model_cfg, "dims", (64, 128, 256, 512)))
    depths = tuple(_cfg_get(model_cfg, "depths", (2, 2, 6, 2)))
    mlp_ratio = int(_cfg_get(model_cfg, "mlp_ratio", 4))
    layer_scale_init = float(_cfg_get(model_cfg, "layer_scale_init", 1e-6))

    if name in ("ivf_convnext_lite", "convnext_lite", "ivf_cnn_best"):
        return IVFConvNeXtLite(
            dims=dims,
            depths=depths,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            head_dropout=head_dropout,
            layer_scale_init=layer_scale_init,
            stem_stride=stem_stride,
        )
    if name == "resnet50_baseline":
        return build_resnet50_baseline(pretrained=pretrained, dropout=head_dropout)
    if name == "efficientnet_b0_baseline":
        return build_efficientnet_b0_baseline(pretrained=pretrained, dropout=head_dropout)

    raise ValueError(f"Unknown model name: {name}")
