# models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, efficientnet_b0, mobilenet_v3_large

from ivf.models.ivf_effimorphp import IVF_EffiMorphPP, EffiMorphPPEncoder, GeM

try:
    from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights, MobileNet_V3_Large_Weights
except ImportError:  # pragma: no cover - older torchvision
    ResNet50_Weights = None
    EfficientNet_B0_Weights = None
    MobileNet_V3_Large_Weights = None


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
        x = self.forward_features(x)
        x = self.head_dropout(x)
        x = self.head(x)                    # [B, 1]
        return x.squeeze(1)                 # [B] logits

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for layer in self.stages:
            x = layer(x)
        x = self.pool(x).flatten(1)         # [B, C]
        return x


class IVFConvNeXtLiteEncoder(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        dims: Tuple[int, int, int, int] = (64, 128, 256, 512),
        depths: Tuple[int, int, int, int] = (2, 2, 6, 2),
        mlp_ratio: int = 4,
        drop_path_rate: float = 0.1,
        layer_scale_init: float = 1e-6,
        stem_stride: int = 4,
    ):
        super().__init__()
        assert len(dims) == 4 and len(depths) == 4
        k = stem_stride
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=k, stride=stem_stride, padding=0),
            LayerNorm2d(dims[0]),
        )

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
        self.out_dim = dims[-1]

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
        x = self.pool(x).flatten(1)
        return x


class ResNetEncoder(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


class EfficientNetEncoder(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.features = model.features
        self.avgpool = model.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


class IVFMultiTaskHead(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        feat_dim: int,
        dropout: float,
        enable_stage: bool,
        enable_morph: bool,
        num_stage_classes: int,
        num_morph_classes: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()
        self.quality_head = nn.Linear(feat_dim, 1)
        self.stage_head = nn.Linear(feat_dim, num_stage_classes) if enable_stage else None
        self.morph_head = nn.Linear(feat_dim, num_morph_classes) if enable_morph else None

    def forward(self, x: torch.Tensor) -> dict:
        feats = self.encoder(x)
        feats = self.dropout(feats)
        logits_quality = self.quality_head(feats).squeeze(1)
        logits_stage = self.stage_head(feats) if self.stage_head is not None else None
        logits_morph = self.morph_head(feats) if self.morph_head is not None else None
        return {
            "logits_quality": logits_quality,
            "logits_stage": logits_stage,
            "logits_morph": logits_morph,
        }


class IVFMorphStageHead(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        feat_dim: int,
        dropout: float,
        enable_quality: bool,
        enable_stage: bool,
        enable_morph: bool,
        num_stage_classes: int,
        num_exp_classes: int,
        num_icm_classes: int,
        num_te_classes: int,
        pool_quality: str = "gap",
        pool_aux: str = "gap",
    ):
        super().__init__()
        self.encoder = encoder
        self.encoder_returns_map = bool(getattr(encoder, "returns_feature_map", False))
        if self.encoder_returns_map:
            self.pool_quality = self._build_pool(pool_quality, feat_dim)
            self.pool_aux = self._build_pool(pool_aux, feat_dim)
            self.quality_dropout = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()
        else:
            # Preserve previous behavior for vector encoders.
            self.pool_quality = None
            self.pool_aux = None
            self.dropout = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()
        self.quality_head = nn.Linear(feat_dim, 1) if enable_quality else None
        self.stage_head = nn.Linear(feat_dim, num_stage_classes) if enable_stage else None
        if enable_morph:
            self.exp_head = nn.Linear(feat_dim, num_exp_classes)
            self.icm_head = nn.Linear(feat_dim, num_icm_classes)
            self.te_head = nn.Linear(feat_dim, num_te_classes)
        else:
            self.exp_head = None
            self.icm_head = None
            self.te_head = None

    def _build_pool(self, name: str, feat_dim: int) -> nn.Module:
        name = name.lower()
        if name == "gem":
            return GeM()
        if name == "gap":
            return nn.AdaptiveAvgPool2d(1)
        raise ValueError(f"Unknown pool type: {name}")

    def _pool_features(self, pool: nn.Module, fmap: torch.Tensor) -> torch.Tensor:
        if isinstance(pool, GeM):
            return pool(fmap)
        if isinstance(pool, nn.AdaptiveAvgPool2d):
            return pool(fmap).flatten(1)
        raise ValueError("Unsupported pooling module.")

    def forward(self, x: torch.Tensor) -> dict:
        if self.encoder_returns_map:
            fmap = self.encoder.forward_features(x) if hasattr(self.encoder, "forward_features") else self.encoder(x)
            quality_feats = self._pool_features(self.pool_quality, fmap)
            aux_feats = self._pool_features(self.pool_aux, fmap)
            quality_feats = self.quality_dropout(quality_feats)
        else:
            feats = self.encoder(x)
            feats = self.dropout(feats)
            quality_feats = feats
            aux_feats = feats

        logits_quality = self.quality_head(quality_feats).squeeze(1) if self.quality_head is not None else None
        logits_stage = self.stage_head(aux_feats) if self.stage_head is not None else None
        logits_exp = self.exp_head(aux_feats) if self.exp_head is not None else None
        logits_icm = self.icm_head(aux_feats) if self.icm_head is not None else None
        logits_te = self.te_head(aux_feats) if self.te_head is not None else None
        return {
            "logits_quality": logits_quality,
            "logits_stage": logits_stage,
            "logits_exp": logits_exp,
            "logits_icm": logits_icm,
            "logits_te": logits_te,
        }


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


def build_mobilenet_v3_large_baseline(pretrained: bool, dropout: float) -> nn.Module:
    if MobileNet_V3_Large_Weights is not None:
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_large(weights=weights)
    else:
        model = mobilenet_v3_large(pretrained=pretrained)
    # Keep the 960->1280 expansion in the classifier; only replace final layer.
    model.classifier[2] = nn.Dropout(p=dropout)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 1)
    return BinaryLogitsWrapper(model)


def build_resnet50_encoder(pretrained: bool) -> tuple[nn.Module, int]:
    if ResNet50_Weights is not None:
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
    else:
        model = resnet50(pretrained=pretrained)
    feat_dim = model.fc.in_features
    return ResNetEncoder(model), feat_dim


def build_efficientnet_b0_encoder(pretrained: bool) -> tuple[nn.Module, int]:
    if EfficientNet_B0_Weights is not None:
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
    else:
        model = efficientnet_b0(pretrained=pretrained)
    feat_dim = model.classifier[-1].in_features
    return EfficientNetEncoder(model), feat_dim


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
        - "mobilenet_v3_large_baseline"
        - "ivf_multitask_head"
        - "ivf_effimorphp"
        - "ivf_morph_stage_head"
      cfg.model.dropout: head dropout
      cfg.model.pretrained: false
      cfg.model.encoder_name: optional encoder override for morph-stage head
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

      model:
        name: mobilenet_v3_large_baseline
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
    if name == "mobilenet_v3_large_baseline":
        return build_mobilenet_v3_large_baseline(pretrained=pretrained, dropout=head_dropout)
    if name == "ivf_effimorphp":
        backbone_pretrained = bool(
            _cfg_get(model_cfg, "backbone_pretrained", _cfg_get(model_cfg, "pretrained", True))
        )
        use_simam = bool(_cfg_get(model_cfg, "use_simam", True))
        use_msma = bool(_cfg_get(model_cfg, "use_msma", True))
        use_mcm = bool(_cfg_get(model_cfg, "use_mcm", True))
        use_eca = bool(_cfg_get(model_cfg, "use_eca", True))
        mcm_depth = int(_cfg_get(model_cfg, "mcm_depth", 1))
        mcm_expand = int(_cfg_get(model_cfg, "mcm_expand", 2))
        pooling = str(_cfg_get(model_cfg, "pooling", "gem"))
        gem_p = float(_cfg_get(model_cfg, "gem_p", 3.0))
        reduce_channels = _cfg_get(model_cfg, "reduce_channels", 640)
        if reduce_channels is not None:
            reduce_channels = int(reduce_channels)
        eca_kernel = _cfg_get(model_cfg, "eca_kernel", None)
        if eca_kernel is not None:
            eca_kernel = int(eca_kernel)
        return IVF_EffiMorphPP(
            pretrained=backbone_pretrained,
            use_simam=use_simam,
            use_msma=use_msma,
            use_mcm=use_mcm,
            mcm_depth=mcm_depth,
            mcm_expand=mcm_expand,
            use_eca=use_eca,
            pooling=pooling,
            dropout=head_dropout,
            gem_p=gem_p,
            eca_kernel=eca_kernel,
            reduce_channels=reduce_channels,
        )
    if name == "ivf_multitask_head":
        backbone_name = str(_cfg_get(model_cfg, "backbone", "ivf_convnext_lite")).lower()
        enable_stage = bool(_cfg_get(model_cfg, "enable_stage", False))
        enable_morph = bool(_cfg_get(model_cfg, "enable_morph", False))
        num_stage_classes = int(_cfg_get(model_cfg, "num_stage_classes", 0))
        num_morph_classes = int(_cfg_get(model_cfg, "num_morph_classes", 0))
        if enable_stage and num_stage_classes < 2:
            raise ValueError("enable_stage requires num_stage_classes >= 2")
        if enable_morph and num_morph_classes < 2:
            raise ValueError("enable_morph requires num_morph_classes >= 2")

        if backbone_name in ("ivf_convnext_lite", "convnext_lite", "ivf_cnn_best"):
            encoder = IVFConvNeXtLiteEncoder(
                dims=dims,
                depths=depths,
                mlp_ratio=mlp_ratio,
                drop_path_rate=drop_path_rate,
                layer_scale_init=layer_scale_init,
                stem_stride=stem_stride,
            )
            feat_dim = encoder.out_dim
        elif backbone_name in ("resnet50", "resnet50_baseline"):
            encoder, feat_dim = build_resnet50_encoder(pretrained=pretrained)
        elif backbone_name in ("efficientnet_b0", "efficientnet_b0_baseline"):
            encoder, feat_dim = build_efficientnet_b0_encoder(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown multitask backbone: {backbone_name}")

        return IVFMultiTaskHead(
            encoder=encoder,
            feat_dim=feat_dim,
            dropout=head_dropout,
            enable_stage=enable_stage,
            enable_morph=enable_morph,
            num_stage_classes=num_stage_classes,
            num_morph_classes=num_morph_classes,
        )
    if name == "ivf_morph_stage_head":
        encoder_name = str(
            _cfg_get(model_cfg, "encoder_name", _cfg_get(model_cfg, "backbone", "ivf_convnext_lite"))
        ).lower()
        enable_quality = bool(_cfg_get(model_cfg, "enable_quality", True))
        enable_stage = bool(_cfg_get(model_cfg, "enable_stage", True))
        enable_morph = bool(_cfg_get(model_cfg, "enable_morph", True))
        num_stage_classes = int(_cfg_get(model_cfg, "num_stage_classes", 0))
        num_exp_classes = int(_cfg_get(model_cfg, "num_exp_classes", 6))
        num_icm_classes = int(_cfg_get(model_cfg, "num_icm_classes", 3))
        num_te_classes = int(_cfg_get(model_cfg, "num_te_classes", 3))
        pool_quality = str(_cfg_get(model_cfg, "pooling_quality", "gap"))
        pool_aux = str(_cfg_get(model_cfg, "pooling_aux", "gap"))
        if enable_stage and num_stage_classes < 2:
            raise ValueError("enable_stage requires num_stage_classes >= 2")
        if enable_morph and (num_exp_classes < 2 or num_icm_classes < 2 or num_te_classes < 2):
            raise ValueError("enable_morph requires exp/icm/te classes >= 2")

        if encoder_name in ("effimorphp", "ivf_effimorphp"):
            reduce_channels = _cfg_get(model_cfg, "reduce_channels", 640)
            if reduce_channels is not None:
                reduce_channels = int(reduce_channels)
            eca_kernel = _cfg_get(model_cfg, "eca_kernel", None)
            if eca_kernel is not None:
                eca_kernel = int(eca_kernel)
            encoder = EffiMorphPPEncoder(
                pretrained=pretrained,
                use_simam=bool(_cfg_get(model_cfg, "use_simam", True)),
                use_msma=bool(_cfg_get(model_cfg, "use_msma", True)),
                use_mcm=bool(_cfg_get(model_cfg, "use_mcm", True)),
                mcm_depth=int(_cfg_get(model_cfg, "mcm_depth", 1)),
                mcm_expand=int(_cfg_get(model_cfg, "mcm_expand", 2)),
                use_eca=bool(_cfg_get(model_cfg, "use_eca", True)),
                eca_kernel=eca_kernel,
                reduce_channels=reduce_channels,
            )
            feat_dim = encoder.feature_dim
        elif encoder_name in ("ivf_convnext_lite", "convnext_lite", "ivf_cnn_best"):
            encoder = IVFConvNeXtLiteEncoder(
                dims=dims,
                depths=depths,
                mlp_ratio=mlp_ratio,
                drop_path_rate=drop_path_rate,
                layer_scale_init=layer_scale_init,
                stem_stride=stem_stride,
            )
            feat_dim = encoder.out_dim
        elif encoder_name in ("resnet50", "resnet50_baseline"):
            encoder, feat_dim = build_resnet50_encoder(pretrained=pretrained)
        elif encoder_name in ("efficientnet_b0", "efficientnet_b0_baseline"):
            encoder, feat_dim = build_efficientnet_b0_encoder(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown morph-stage encoder: {encoder_name}")

        return IVFMorphStageHead(
            encoder=encoder,
            feat_dim=feat_dim,
            dropout=head_dropout,
            enable_quality=enable_quality,
            enable_stage=enable_stage,
            enable_morph=enable_morph,
            num_stage_classes=num_stage_classes,
            num_exp_classes=num_exp_classes,
            num_icm_classes=num_icm_classes,
            num_te_classes=num_te_classes,
            pool_quality=pool_quality,
            pool_aux=pool_aux,
        )

    raise ValueError(f"Unknown model name: {name}")
