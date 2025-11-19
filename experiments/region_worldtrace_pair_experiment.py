#!/usr/bin/env python3
"""
Train a CLIP-style contrastive model on RegionDCL building embeddings and
WorldTrace trajectory tensors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import lightning.pytorch as pl
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import infonce

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGIONDCL_DIR = PROJECT_ROOT / "RegionDCL"
if REGIONDCL_DIR.exists() and str(REGIONDCL_DIR) not in sys.path:
    sys.path.append(str(REGIONDCL_DIR))
try:
    from model.regiondcl import PatternEncoder  # type: ignore
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional dependency
    PatternEncoder = None  # type: ignore

################################################################################
# Argument parsing                                                             #
################################################################################


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Region-level pairwise contrastive training for RegionDCL + WorldTrace."
    )
    default_data_root = Path("Multimodality-preprocess") / "data" / "processed"
    parser.add_argument("--city", type=str, required=True, help="RegionDCL city to train on (e.g., Singapore).")
    parser.add_argument(
        "--region-root",
        type=Path,
        default=default_data_root / "regiondcl",
        help="Directory containing RegionDCL building exports (defaults to Multimodality-preprocess/data/processed/regiondcl).",
    )
    parser.add_argument(
        "--worldtrace-root",
        type=Path,
        default=default_data_root / "worldtrace",
        help="Directory containing WorldTrace trajectory aggregates.",
    )
    parser.add_argument(
        "--split-config",
        type=Path,
        default=None,
        help="Optional JSON describing dataset split assignments. "
        "Supports either {region_id: split} or {split: [region_ids]} formats.",
    )
    parser.add_argument(
        "--split-config-key",
        type=str,
        default="",
        help="If the split JSON stores multiple cities, provide the top-level key for this city.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.8, help="Fraction of samples assigned to training.")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Fraction of samples assigned to validation.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on the number of paired regions.")
    parser.add_argument("--min-buildings", type=int, default=1, help="Skip regions with fewer building vectors.")
    parser.add_argument("--min-trajectories", type=int, default=1, help="Skip regions with fewer trajectories.")

    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per optimizer step.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes.")
    parser.add_argument("--epochs", type=int, default=50, help="Max training epochs.")
    parser.add_argument("--lr", type=float, default=3e-4, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay.")
    parser.add_argument("--drop-last", action="store_true", help="Drop final partial batch during training.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for data splitting.")
    parser.add_argument("--use-seed", action="store_true", help="Enable deterministic seeding.")
    parser.add_argument("--logit-scale-init", type=float, default=math.log(1 / 0.07), help="Initial logit scale.")
    parser.add_argument("--freeze-logit-scale", action="store_true", help="Disable temperature learning.")

    parser.add_argument("--embedding-dim", type=int, default=256, help="Shared embedding dimension.")
    parser.add_argument(
        "--trajectory-point-dim",
        type=int,
        default=128,
        help="Hidden width for per-point trajectory features.",
    )
    parser.add_argument("--trajectory-point-heads", type=int, default=4, help="Attention heads for point-level transformer.")
    parser.add_argument("--trajectory-point-layers", type=int, default=2, help="Number of transformer layers over trajectory points.")
    parser.add_argument("--trajectory-point-ffn", type=int, default=256, help="Feed-forward width for point transformer.")
    parser.add_argument(
        "--trajectory-track-dim",
        type=int,
        default=256,
        help="Hidden width for per-trajectory aggregation.",
    )
    parser.add_argument("--trajectory-track-heads", type=int, default=4, help="Attention heads for trajectory-level transformer.")
    parser.add_argument("--trajectory-track-layers", type=int, default=2, help="Transformer layers aggregating trajectories.")
    parser.add_argument("--trajectory-track-ffn", type=int, default=512, help="Feed-forward width for trajectory transformer.")
    parser.add_argument(
        "--encoder-dropout",
        type=float,
        default=0.1,
        help="Dropout probability applied inside encoder feed-forward layers.",
    )
    parser.add_argument("--max-buildings", type=int, default=512, help="Number of building rows to keep per region.")
    parser.add_argument("--max-trajectories", type=int, default=128, help="Number of trajectories kept per region.")
    parser.add_argument("--max-points", type=int, default=256, help="Number of points kept per trajectory.")
    parser.add_argument(
        "--building-feature-dim",
        type=int,
        default=None,
        help="Override building feature dimension if metadata is missing.",
    )
    parser.add_argument("--regiondcl-hidden-dim", type=int, default=None, help="Hidden dimension inside RegionDCL encoder (defaults to embedding_dim).")
    parser.add_argument("--regiondcl-feedforward", type=int, default=1024, help="Feedforward dimension for RegionDCL transformer.")
    parser.add_argument("--regiondcl-building-head", type=int, default=8, help="Number of attention heads for building encoder.")
    parser.add_argument("--regiondcl-building-layers", type=int, default=2, help="Number of transformer layers for building encoder.")
    parser.add_argument("--regiondcl-building-dropout", type=float, default=0.2, help="Dropout rate inside RegionDCL building encoder.")
    parser.add_argument("--regiondcl-building-activation", type=str, default="relu", choices=["relu", "gelu"], help="Activation function for RegionDCL building encoder.")
    parser.add_argument("--regiondcl-distance-penalty", type=float, default=1.0, help="Initial distance penalty std for RegionDCL encoder.")
    parser.add_argument("--regiondcl-bottleneck-head", type=int, default=8, help="Number of heads in RegionDCL bottleneck encoder.")
    parser.add_argument("--regiondcl-bottleneck-layers", type=int, default=2, help="Transformer layers in RegionDCL bottleneck.")
    parser.add_argument("--regiondcl-bottleneck-dropout", type=float, default=0.2, help="Dropout inside RegionDCL bottleneck.")
    parser.add_argument("--regiondcl-bottleneck-activation", type=str, default="relu", choices=["relu", "gelu"], help="Activation in RegionDCL bottleneck.")

    parser.add_argument("--save-dir", type=Path, default=Path("runs/region_worldtrace_clip"), help="Output directory.")
    parser.add_argument("--ckpt-save-dir", type=Path, default=Path("checkpoints"), help="Checkpoint directory.")
    parser.add_argument("--ckpt-path", type=str, default=None, help="Resume from checkpoint.")
    parser.add_argument("--limit-train-batches", type=float, default=1.0, help="Trainer limit_train_batches.")
    parser.add_argument("--limit-val-batches", type=float, default=1.0, help="Trainer limit_val_batches.")
    parser.add_argument("--limit-test-batches", type=float, default=1.0, help="Trainer limit_test_batches.")
    parser.add_argument("--log-every-n-steps", type=int, default=10, help="Trainer logging frequency.")
    parser.add_argument("--precision", type=str, default="32-true", help="Precision flag passed to Trainer.")
    parser.add_argument("--accelerator", type=str, default="auto", help="Accelerator setting passed to Trainer.")
    parser.add_argument("--devices", type=str, default=None, help="Trainer devices argument (per lightning).")
    parser.add_argument("--strategy", type=str, default=None, help="Optional Lightning strategy string.")

    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-run-id", type=str, default=None, help="Existing W&B run id (resume).")
    parser.add_argument("--wandb-project", type=str, default="symile", help="Weights & Biases project name.")
    return parser.parse_args()


################################################################################
# Data loading                                                                 #
################################################################################


def _resolve_city_dir(base: Path, city: str) -> Path:
    base = base.expanduser()
    candidate = base / city
    if candidate.exists():
        return candidate
    if base.name.lower() == city.lower():
        return base
    raise FileNotFoundError(f"Could not locate city '{city}' under {base}.")


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


@dataclass(frozen=True)
class RegionPair:
    region_id: int
    region_name: str
    building_path: Path
    trajectory_path: Path
    num_buildings: int
    num_trajectories: int


def load_region_pairs(
    region_dir: Path,
    worldtrace_dir: Path,
    min_buildings: int,
    min_trajectories: int,
) -> List[RegionPair]:
    region_manifest = _read_csv_rows(region_dir / "manifest.csv")
    world_manifest = _read_csv_rows(worldtrace_dir / "manifest.csv")
    world_by_region = {
        int(row["region_id"]): row for row in world_manifest if row.get("sample_path", "").strip() != ""
    }

    pairs: List[RegionPair] = []
    for row in region_manifest:
        region_id = int(row["region_id"])
        building_path = region_dir / row["feature_path"]
        world_row = world_by_region.get(region_id)
        if world_row is None:
            continue
        trajectory_path = worldtrace_dir / world_row["sample_path"]
        if not building_path.exists() or not trajectory_path.exists():
            continue
        num_buildings = int(row.get("num_buildings", 0))
        num_trajectories = int(world_row.get("trajectory_count", 0))
        if num_buildings < min_buildings or num_trajectories < min_trajectories:
            continue
        pairs.append(
            RegionPair(
                region_id=region_id,
                region_name=row.get("region_name", f"region_{region_id:05d}"),
                building_path=building_path,
                trajectory_path=trajectory_path,
                num_buildings=num_buildings,
                num_trajectories=num_trajectories,
            )
        )
    if not pairs:
        raise RuntimeError(
            "No overlapping RegionDCL and WorldTrace samples matched. "
            "Check that both manifests exist and the city argument matches."
        )
    return pairs


def _maybe_limit_pairs(pairs: Sequence[RegionPair], limit: Optional[int]) -> List[RegionPair]:
    if limit is None or limit <= 0:
        return list(pairs)
    return list(pairs[:limit])


def _load_split_mapping(path: Path, key: str) -> Dict[int, str]:
    config = json.loads(path.read_text())
    if key:
        config = config[key]
    mapping: Dict[int, str] = {}
    if all(isinstance(v, list) for v in config.values()):
        for split, region_ids in config.items():
            for rid in region_ids:
                mapping[int(rid)] = split
    else:
        mapping = {int(rid): str(split) for rid, split in config.items()}
    return mapping


def _assign_split(
    pairs: Sequence[RegionPair],
    train_fraction: float,
    val_fraction: float,
    seed: int,
    split_config: Optional[Dict[int, str]],
) -> Tuple[List[RegionPair], List[RegionPair], List[RegionPair]]:
    if split_config:
        train, val, test = [], [], []
        for pair in pairs:
            split = split_config.get(pair.region_id, "train")
            if split == "train":
                train.append(pair)
            elif split == "val":
                val.append(pair)
            elif split == "test":
                test.append(pair)
        return train, val, test

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(pairs))
    total = len(pairs)
    n_train = max(1, int(total * train_fraction)) if total > 0 else 0
    remaining = total - n_train
    n_val = min(int(total * val_fraction), remaining)
    remaining -= n_val
    n_test = remaining

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    def gather(idxs: np.ndarray) -> List[RegionPair]:
        return [pairs[i] for i in idxs]

    return gather(train_idx), gather(val_idx), gather(test_idx)


################################################################################
# Dataset + DataModule                                                         #
################################################################################


class RegionTrajectoryDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        pairs: Sequence[RegionPair],
        building_dim: int,
        max_buildings: int,
        max_trajectories: int,
        max_points: int,
    ):
        self.pairs = list(pairs)
        self.building_dim = building_dim
        self.max_buildings = max_buildings
        self.max_trajectories = max_trajectories
        self.max_points = max_points

    def __len__(self) -> int:
        return len(self.pairs)

    @staticmethod
    def _load_npz(path: Path) -> Dict[str, np.ndarray]:
        with np.load(path, allow_pickle=False) as payload:
            return {k: payload[k] for k in payload.files}

    def _load_buildings(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        arrays = self._load_npz(path)
        features = arrays["features"].astype(np.float32, copy=False)
        centroids = arrays["centroids"].astype(np.float32, copy=False)
        num = features.shape[0]
        keep = min(num, self.max_buildings)
        padded = np.zeros((self.max_buildings, self.building_dim), dtype=np.float32)
        xy = np.zeros((self.max_buildings, 2), dtype=np.float32)
        if keep > 0:
            padded[:keep, :] = features[:keep, : self.building_dim]
            xy[:keep, :] = centroids[:keep, :]
        mask = np.ones(self.max_buildings, dtype=bool)
        mask[:keep] = False
        return torch.from_numpy(padded), torch.from_numpy(mask), torch.from_numpy(xy)

    def _load_trajectories(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        arrays = self._load_npz(path)
        lat = arrays["lat"].astype(np.float32, copy=False)
        lon = arrays["lon"].astype(np.float32, copy=False)
        speed = arrays["speed"].astype(np.float32, copy=False)
        time_rel = arrays["time_rel"].astype(np.float32, copy=False)
        mask = arrays["mask"].astype(bool, copy=False)
        lengths = arrays["lengths"].astype(np.int32, copy=False)

        trajs = min(lat.shape[0], self.max_trajectories)
        points = min(lat.shape[1], self.max_points)

        stacked = np.stack(
            [
                np.nan_to_num(lat[:trajs, :points], nan=0.0),
                np.nan_to_num(lon[:trajs, :points], nan=0.0),
                np.nan_to_num(speed[:trajs, :points], nan=0.0),
                np.nan_to_num(time_rel[:trajs, :points], nan=0.0),
            ],
            axis=-1,
        )
        padded = np.zeros((self.max_trajectories, self.max_points, stacked.shape[-1]), dtype=np.float32)
        padded[:trajs, :points] = stacked

        point_mask = np.zeros((self.max_trajectories, self.max_points), dtype=bool)
        point_mask[:trajs, :points] = mask[:trajs, :points]

        traj_mask = np.zeros(self.max_trajectories, dtype=bool)
        traj_mask[:trajs] = (lengths[:trajs] > 0).astype(bool)

        return (
            torch.from_numpy(padded),
            torch.from_numpy(point_mask),
            torch.from_numpy(traj_mask),
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pair = self.pairs[idx]
        buildings, building_mask, building_xy = self._load_buildings(pair.building_path)
        traj, point_mask, traj_mask = self._load_trajectories(pair.trajectory_path)
        return {
            "buildings": buildings,
            "building_mask": building_mask,
            "building_xy": building_xy,
            "trajectories": traj,
            "trajectory_point_mask": point_mask,
            "trajectory_mask": traj_mask,
            "region_id": torch.tensor(pair.region_id, dtype=torch.int32),
        }


class RegionTrajectoryDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_pairs: Sequence[RegionPair],
        val_pairs: Sequence[RegionPair],
        test_pairs: Sequence[RegionPair],
        building_dim: int,
        args: argparse.Namespace,
    ):
        super().__init__()
        self.train_pairs = train_pairs
        self.val_pairs = val_pairs
        self.test_pairs = test_pairs
        self.args = args
        self.building_dim = building_dim
        self.num_workers = max(0, args.num_workers)

    def _build_dataset(self, pairs: Sequence[RegionPair]) -> RegionTrajectoryDataset:
        return RegionTrajectoryDataset(
            pairs,
            building_dim=self.building_dim,
            max_buildings=self.args.max_buildings,
            max_trajectories=self.args.max_trajectories,
            max_points=self.args.max_points,
        )

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.ds_train = self._build_dataset(self.train_pairs)
            self.ds_val = self._build_dataset(self.val_pairs)
        if stage in (None, "test", "validate"):
            self.ds_val = getattr(self, "ds_val", self._build_dataset(self.val_pairs))
            self.ds_test = self._build_dataset(self.test_pairs)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.ds_train,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=self.args.drop_last,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.ds_val,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.ds_test,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
        )


################################################################################
# Model                                                                        #
################################################################################


class BuildingEncoder(nn.Module):
    def __init__(self, input_dim: int, args: argparse.Namespace):
        super().__init__()
        if PatternEncoder is None:
            raise ImportError(
                "RegionDCL PatternEncoder is unavailable. Ensure the RegionDCL repo is present so we can import model.regiondcl."
            )
        hidden_dim = getattr(args, "regiondcl_hidden_dim", None) or args.embedding_dim
        self.encoder = PatternEncoder(
            d_building=input_dim,
            d_poi=1,
            d_hidden=hidden_dim,
            d_feedforward=args.regiondcl_feedforward,
            building_head=args.regiondcl_building_head,
            building_layers=args.regiondcl_building_layers,
            building_dropout=args.regiondcl_building_dropout,
            building_distance_penalty=args.regiondcl_distance_penalty,
            building_activation=args.regiondcl_building_activation,
            bottleneck_head=args.regiondcl_bottleneck_head,
            bottleneck_layers=args.regiondcl_bottleneck_layers,
            bottleneck_dropout=args.regiondcl_bottleneck_dropout,
            bottleneck_activation=args.regiondcl_bottleneck_activation,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        if hidden_dim != args.embedding_dim:
            self.output_proj = nn.Linear(hidden_dim, args.embedding_dim)
        else:
            self.output_proj = nn.Identity()

    def forward(self, features: torch.Tensor, mask: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        # Convert to RegionDCL expected shapes: [seq, batch, dim]
        features_seq = features.permute(1, 0, 2).contiguous()
        xy_seq = xy.permute(1, 0, 2).contiguous()
        encoded = self.encoder(features_seq, mask, xy_seq, poi_feature=None, poi_mask=None)
        encoded = self.output_norm(encoded)
        return self.output_proj(encoded)


class TrajectoryEncoder(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self.point_hidden = args.trajectory_point_dim
        self.track_hidden = args.trajectory_track_dim
        self.embed_dim = args.embedding_dim
        activation = "gelu"

        self.point_embed = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, self.point_hidden),
            nn.GELU(),
        )

        point_layer = nn.TransformerEncoderLayer(
            d_model=self.point_hidden,
            nhead=args.trajectory_point_heads,
            dim_feedforward=args.trajectory_point_ffn,
            dropout=args.encoder_dropout,
            activation=activation,
            batch_first=False,
        )
        self.point_transformer = nn.TransformerEncoder(point_layer, num_layers=args.trajectory_point_layers)

        self.point_to_track = nn.Sequential(
            nn.LayerNorm(self.point_hidden),
            nn.Linear(self.point_hidden, self.track_hidden),
        )

        track_layer = nn.TransformerEncoderLayer(
            d_model=self.track_hidden,
            nhead=args.trajectory_track_heads,
            dim_feedforward=args.trajectory_track_ffn,
            dropout=args.encoder_dropout,
            activation=activation,
            batch_first=False,
        )
        self.track_transformer = nn.TransformerEncoder(track_layer, num_layers=args.trajectory_track_layers)

        self.final_proj = nn.Sequential(
            nn.LayerNorm(self.track_hidden),
            nn.Linear(self.track_hidden, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )

        self.register_parameter(
            "point_pos_embedding",
            nn.Parameter(torch.zeros(args.max_points, self.point_hidden)),
        )
        self.register_parameter(
            "traj_pos_embedding",
            nn.Parameter(torch.zeros(args.max_trajectories, self.track_hidden)),
        )
        nn.init.normal_(self.point_pos_embedding, std=0.02)
        nn.init.normal_(self.traj_pos_embedding, std=0.02)

    def forward(
        self,
        trajectories: torch.Tensor,
        point_mask: torch.Tensor,
        traj_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, num_traj, num_points, feat_dim = trajectories.shape
        device = trajectories.device

        point_mask = point_mask.to(torch.bool)
        traj_mask = traj_mask.to(torch.bool)

        flat = trajectories.view(batch * num_traj, num_points, feat_dim)
        flat_mask = point_mask.view(batch * num_traj, num_points)

        point_repr = self.point_embed(flat)
        pos = self.point_pos_embedding[:num_points].unsqueeze(0)
        point_repr = point_repr + pos.to(device)
        point_repr = point_repr.permute(1, 0, 2).contiguous()

        point_pad_mask = ~flat_mask
        encoded_points = self.point_transformer(point_repr, src_key_padding_mask=point_pad_mask)
        encoded_points = encoded_points.permute(1, 0, 2)

        flat_mask_f = flat_mask.unsqueeze(-1).float()
        denom_point = flat_mask_f.sum(dim=1).clamp(min=1.0)
        traj_repr = (encoded_points * flat_mask_f).sum(dim=1) / denom_point
        traj_repr = self.point_to_track(traj_repr)
        traj_repr = traj_repr.view(batch, num_traj, self.track_hidden)

        traj_pos = self.traj_pos_embedding[:num_traj].unsqueeze(0)
        traj_repr = traj_repr + traj_pos.to(device)
        traj_repr = traj_repr.permute(1, 0, 2).contiguous()

        traj_pad_mask = ~traj_mask
        encoded_traj = self.track_transformer(traj_repr, src_key_padding_mask=traj_pad_mask)
        encoded_traj = encoded_traj.permute(1, 0, 2)

        traj_mask_f = traj_mask.unsqueeze(-1).float()
        denom_traj = traj_mask_f.sum(dim=1).clamp(min=1.0)
        region_repr = (encoded_traj * traj_mask_f).sum(dim=1) / denom_traj
        return self.final_proj(region_repr)


class RegionTrajectoryCLIP(pl.LightningModule):
    def __init__(self, args: argparse.Namespace, building_dim: int):
        super().__init__()
        self.args = args
        self.save_hyperparameters(ignore=["args"])
        self.building_encoder = BuildingEncoder(input_dim=building_dim, args=args)
        self.trajectory_encoder = TrajectoryEncoder(args)
        init_value = torch.tensor(float(args.logit_scale_init))
        self.logit_scale = nn.Parameter(init_value)
        if args.freeze_logit_scale:
            self.logit_scale.requires_grad = False

    @staticmethod
    def compute_retrieval_top1(anchor: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        anchor_norm = F.normalize(anchor, dim=1)
        other_norm = F.normalize(other, dim=1)
        sims = anchor_norm @ other_norm.T
        preds = sims.argmax(dim=1)
        labels = torch.arange(anchor.shape[0], device=anchor.device)
        return (preds == labels).float().mean()

    def encode_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        building_mask = batch["building_mask"].to(torch.bool)
        buildings = self.building_encoder(batch["buildings"], building_mask, batch["building_xy"])
        trajectories = self.trajectory_encoder(
            batch["trajectories"],
            batch["trajectory_point_mask"],
            batch["trajectory_mask"],
        )
        buildings = F.normalize(buildings, dim=1)
        trajectories = F.normalize(trajectories, dim=1)
        return buildings, trajectories

    def shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        building_repr, trajectory_repr = self.encode_batch(batch)
        logit_scale = self.logit_scale.exp()
        loss = infonce(building_repr, trajectory_repr, logit_scale)

        self.log(f"{stage}_loss", loss, prog_bar=stage != "test", batch_size=building_repr.shape[0])
        self.log(f"{stage}_logit_scale", logit_scale, prog_bar=False, batch_size=building_repr.shape[0])

        with torch.no_grad():
            forward_acc = self.compute_retrieval_top1(building_repr, trajectory_repr)
            backward_acc = self.compute_retrieval_top1(trajectory_repr, building_repr)
            self.log(
                f"{stage}_acc_building_to_traj",
                forward_acc,
                prog_bar=False,
                batch_size=building_repr.shape[0],
            )
            self.log(
                f"{stage}_acc_traj_to_building",
                backward_acc,
                prog_bar=False,
                batch_size=building_repr.shape[0],
            )
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


################################################################################
# Entry point                                                                  #
################################################################################


def infer_building_dim(region_dir: Path, override: Optional[int]) -> int:
    if override:
        return override
    metadata_path = region_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Could not infer building feature dimension because {metadata_path} is missing. "
            "Pass --building-feature-dim explicitly."
        )
    metadata = json.loads(metadata_path.read_text())
    feature_dim = metadata.get("feature_dim")
    if feature_dim is None:
        raise KeyError("metadata.json is missing 'feature_dim'. Pass --building-feature-dim to override.")
    return int(feature_dim)


def run():
    args = parse_args()

    region_dir = _resolve_city_dir(args.region_root, args.city)
    worldtrace_dir = _resolve_city_dir(args.worldtrace_root, args.city)
    building_dim = infer_building_dim(region_dir, args.building_feature_dim)

    pairs = load_region_pairs(
        region_dir=region_dir,
        worldtrace_dir=worldtrace_dir,
        min_buildings=args.min_buildings,
        min_trajectories=args.min_trajectories,
    )
    pairs = _maybe_limit_pairs(pairs, args.max_samples)

    split_mapping = None
    if args.split_config:
        split_mapping = _load_split_mapping(args.split_config, args.split_config_key)
    train_pairs, val_pairs, test_pairs = _assign_split(
        pairs,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
        split_config=split_mapping,
    )
    print(
        f"Loaded {len(pairs)} region pairs "
        f"(train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)})."
    )

    if args.use_seed:
        seed_everything(args.seed, workers=True)

    save_dir = args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    args.ckpt_save_dir.mkdir(parents=True, exist_ok=True)

    logger = False
    if args.wandb:
        logger = WandbLogger(
            project=args.wandb_project,
            log_model=False,
            save_dir=args.ckpt_save_dir,
            id=args.wandb_run_id,
        )

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_save_dir,
        filename="{epoch}-{val_loss:.4f}",
        every_n_epochs=1,
        save_top_k=-1,
    )

    trainer_kwargs = dict(
        accelerator=args.accelerator,
        callbacks=[checkpoint_callback],
        default_root_dir=save_dir,
        max_epochs=args.epochs,
        check_val_every_n_epoch=1,
        log_every_n_steps=args.log_every_n_steps,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        limit_test_batches=args.limit_test_batches,
        precision=args.precision,
        enable_progress_bar=True,
        logger=logger,
    )
    if getattr(args, "devices", None) not in (None, "None"):
        trainer_kwargs["devices"] = args.devices
    if getattr(args, "strategy", None) not in (None, "None"):
        trainer_kwargs["strategy"] = args.strategy

    trainer = Trainer(**trainer_kwargs)

    datamodule = RegionTrajectoryDataModule(
        train_pairs, val_pairs, test_pairs, building_dim=building_dim, args=args
    )
    model = RegionTrajectoryCLIP(args=args, building_dim=building_dim)

    trainer.fit(model, datamodule=datamodule, ckpt_path=None if args.ckpt_path in (None, "None") else args.ckpt_path)

    if args.wandb and logger:
        logger.experiment.finish()


if __name__ == "__main__":
    run()
