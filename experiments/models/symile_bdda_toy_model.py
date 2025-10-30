"""Improved Symile model with stronger backbones for BDDA preprocessed tensors."""
from argparse import Namespace
import math
from typing import Iterable, List

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import clip, symile


def init_logit_scale(args) -> float:
    value = float(getattr(args, "logit_scale_init", math.log(1 / 0.07)))
    if getattr(args, "freeze_logit_scale", False):
        return value
    return value


class ConvBlock(nn.Module):
    """Two-layer Conv-BN-ReLU block followed by optional dropout."""
    def __init__(self, in_channels: int, out_channels: int, dropout: float):
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvBackbone(nn.Module):
    """Compact CNN that aggressively downsamples frames before projection."""
    def __init__(
        self,
        args,
        in_channels: int,
        base_channels: int,
        num_blocks: int,
        dropout: float,
    ):
        super().__init__()
        channels = in_channels
        blocks: List[nn.Module] = []
        for block_idx in range(num_blocks):
            out_channels = base_channels * (2 ** block_idx)
            blocks.append(ConvBlock(channels, out_channels, dropout))
            channels = out_channels
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, args.d),
            nn.LayerNorm(args.d),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.head(x)
        return x


class GPSProjection(nn.Module):
    """Multi-layer perceptron for GPS signal alignment."""
    def __init__(self, args, in_features: int):
        super().__init__()
        hidden_dims: Iterable[int] = getattr(args, "gps_hidden_dims", [256, 128])
        dropout = getattr(args, "gps_dropout", 0.0)

        layers: List[nn.Module] = []
        prev_dim = in_features
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden))
            layers.append(nn.LayerNorm(hidden))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden
        layers.append(nn.Linear(prev_dim, args.d))
        layers.append(nn.LayerNorm(args.d))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SymileBDDAModel(pl.LightningModule):
    """
    Three-modality contrastive model featuring convolutional backbones for camera/gaze
    frames and a deeper MLP for GPS signals.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.args = Namespace(**kwargs)

        self.camera_encoder = ConvBackbone(
            self.args,
            in_channels=3,
            base_channels=getattr(self.args, "camera_base_channels", 64),
            num_blocks=getattr(self.args, "camera_num_blocks", 4),
            dropout=getattr(self.args, "camera_dropout", 0.1),
        )
        self.gaze_encoder = ConvBackbone(
            self.args,
            in_channels=3,
            base_channels=getattr(self.args, "gaze_base_channels", 48),
            num_blocks=getattr(self.args, "gaze_num_blocks", 4),
            dropout=getattr(self.args, "gaze_dropout", 0.1),
        )
        self.gps_encoder = GPSProjection(self.args, in_features=getattr(self.args, "gps_dim", 4))

        logit_scale_value = init_logit_scale(self.args)
        self.logit_scale = nn.Parameter(torch.tensor(logit_scale_value))
        if getattr(self.args, "freeze_logit_scale", False):
            self.logit_scale.requires_grad = False

        self.loss_impl = symile if getattr(self.args, "loss_fn", "symile") == "symile" else clip
        self.negative_sampling = getattr(self.args, "negative_sampling", "n")

    def encode_batch(self, batch):
        camera = self.camera_encoder(batch["camera"])
        gaze = self.gaze_encoder(batch["gaze"])
        gps = self.gps_encoder(batch["gps"])
        return camera, gaze, gps

    @staticmethod
    def compute_retrieval_top1(anchor: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        anchor_norm = F.normalize(anchor, dim=1)
        other_norm = F.normalize(other, dim=1)
        sims = anchor_norm @ other_norm.T
        preds = sims.argmax(dim=1)
        labels = torch.arange(anchor.shape[0], device=anchor.device)
        return (preds == labels).float().mean()

    def shared_step(self, batch, stage: str):
        camera, gaze, gps = self.encode_batch(batch)
        logit_scale_exp = self.logit_scale.exp()

        loss = self.loss_impl(camera, gaze, gps, logit_scale_exp, self.negative_sampling)

        self.log(f"{stage}_loss", loss, prog_bar=stage != "test", batch_size=camera.shape[0])
        self.log(f"{stage}_logit_scale", logit_scale_exp, prog_bar=False, batch_size=camera.shape[0])

        with torch.no_grad():
            cam_gaze_acc = self.compute_retrieval_top1(camera, gaze)
            cam_gps_acc = self.compute_retrieval_top1(camera, gps)
            gaze_gps_acc = self.compute_retrieval_top1(gaze, gps)
            self.log(f"{stage}_acc_cam_gaze", cam_gaze_acc, prog_bar=False, batch_size=camera.shape[0])
            self.log(f"{stage}_acc_cam_gps", cam_gps_acc, prog_bar=False, batch_size=camera.shape[0])
            self.log(f"{stage}_acc_gaze_gps", gaze_gps_acc, prog_bar=False, batch_size=camera.shape[0])

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        return optimizer
