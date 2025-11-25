"""
Train Symile on the PSI dataset without touching the existing experiment entrypoints.

Expected manifest format (jsonl or json list):
{
    "sample_id": "vid001_000123",
    "frame_path": "frames/vid001_000123.jpg",             # relative to --data_root
    "bbox": [x1, y1, x2, y2],                             # pedestrian box for the anchor frame
    "trajectory": [[x1, y1, x2, y2], ...],                # past K boxes (oldest -> newest) or
    "trajectory_path": "trajectories/vid001_000123.npy",  # np.array of shape (K, 4) or (K, 2)
    "text": "Pedestrian hesitates due to fast traffic",
    "label": 0                                            # optional, used only for logging
}

Preprocessing enforced by this script:
- Visual: crops a square context around the pedestrian box, resizes to 224x224, ImageNet normalize.
- Trajectory: converts bounding boxes to center points, computes relative displacements,
              z-score normalizes using train-set statistics.
- Text: tokenizes to max length (default 32) and uses the CLS embedding.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import lightning.pytorch as pl
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.models import ResNet50_Weights, resnet50
from transformers import AutoModel, AutoTokenizer

from losses import clip, symile, zeroshot_retrieval_logits
from utils import PathToStrEncoder, str_to_bool


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _load_manifest(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with path.open("r", encoding="utf8") as f:
        data = json.load(f)
        if isinstance(data, dict):
            data = data.get("samples", [])
        return data


def _context_crop(img: Image.Image, bbox: List[float], scale: float, image_size: int) -> Image.Image:
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    side = max(w, h) * scale

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    left = max(0.0, cx - side / 2.0)
    top = max(0.0, cy - side / 2.0)
    right = min(img.width, left + side)
    bottom = min(img.height, top + side)

    if right - left < 1.0 or bottom - top < 1.0:
        return img.resize((image_size, image_size))

    cropped = img.crop((left, top, right, bottom))
    return cropped


class PSITriadDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        data_root: Path,
        tokenizer: AutoTokenizer,
        image_size: int,
        context_scale: float,
        traj_steps: int,
        text_max_len: int,
        traj_stats: Optional[Dict[str, torch.Tensor]] = None,
        split: str = "train",
    ):
        self.samples = _load_manifest(manifest_path)
        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {manifest_path}.")

        self.data_root = data_root
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.context_scale = context_scale
        self.traj_steps = traj_steps
        self.text_max_len = text_max_len
        self.split = split

        # Base normalization for everything
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

        # Augmentation transforms (Color only, spatial is handled in __getitem__)
        self.color_aug = transforms.Compose([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
        ])

        self.traj_stats = traj_stats or self._compute_traj_stats()

    def __len__(self) -> int:
        return len(self.samples)

    def _load_trajectory(self, sample: Dict[str, Any]) -> np.ndarray:
        if "trajectory" in sample:
            traj = np.asarray(sample["trajectory"], dtype=np.float32)
        elif "trajectory_path" in sample:
            traj = np.load(self.data_root / sample["trajectory_path"]).astype(np.float32)
        else:
            raise ValueError("Sample missing trajectory information.")

        if traj.ndim != 2 or traj.shape[1] not in (2, 4):
            raise ValueError(f"Trajectory must be (N,2) or (N,4); got {traj.shape}.")
        return traj

    def _trajectory_centers(self, traj: np.ndarray) -> np.ndarray:
        if traj.shape[1] == 4:
            cx = (traj[:, 0] + traj[:, 2]) / 2.0
            cy = (traj[:, 1] + traj[:, 3]) / 2.0
            centers = np.stack([cx, cy], axis=1)
        else:
            centers = traj
        return centers.astype(np.float32)

    def _relative_displacements(self, traj: np.ndarray) -> torch.Tensor:
        centers = self._trajectory_centers(traj)
        centers = centers[-self.traj_steps :]
        if centers.shape[0] < self.traj_steps:
            pad = np.repeat(centers[:1], self.traj_steps - centers.shape[0], axis=0)
            centers = np.concatenate([pad, centers], axis=0)

        deltas = np.diff(centers, axis=0)
        if deltas.shape[0] < self.traj_steps - 1:
            pad = np.zeros((self.traj_steps - 1 - deltas.shape[0], 2), dtype=np.float32)
            deltas = np.concatenate([pad, deltas], axis=0)

        return torch.from_numpy(deltas.astype(np.float32))

    def _compute_traj_stats(self) -> Dict[str, torch.Tensor]:
        deltas = []
        # Sample a subset if dataset is too large to speed up stat computation
        sample_subset = self.samples[:5000] if len(self.samples) > 5000 else self.samples
        for sample in sample_subset:
            traj = self._load_trajectory(sample)
            deltas.append(self._relative_displacements(traj))
        stacked = torch.cat(deltas, dim=0)
        mean = stacked.mean(dim=0)
        std = stacked.std(dim=0)
        std = torch.clamp(std, min=1e-6)
        return {"mean": mean, "std": std}

    def _load_image_pil(self, sample: Dict[str, Any]) -> Image.Image:
        """Returns the PIL image cropped to context, but NOT converted to tensor yet."""
        frame_path = self.data_root / sample["frame_path"]
        img = Image.open(frame_path).convert("RGB")

        if "bbox" in sample:
            bbox = sample["bbox"]
        else:
            traj = self._load_trajectory(sample)
            bbox = traj[-1].tolist() if traj.shape[1] == 4 else [0, 0, img.width, img.height]

        cropped = _context_crop(img, bbox, self.context_scale, self.image_size)
        return cropped

    def _tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.text_max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # 1. Load Data
        image_pil = self._load_image_pil(sample)
        raw_traj = self._load_trajectory(sample)
        traj_tensor = self._relative_displacements(raw_traj)
        
        # 2. Augmentation (Syncing Image and Trajectory)
        if self.split == "train":
            # A. Color Augmentation (does not affect trajectory)
            image_pil = self.color_aug(image_pil)
            
            # B. Horizontal Flip (affects trajectory X coordinates)
            if random.random() < 0.5:
                image_pil = TF.hflip(image_pil)
                # Negate X displacements (dx) -> index 0
                traj_tensor[:, 0] = -traj_tensor[:, 0]

        # 3. Final Tensor Conversion
        # Ensure correct size (ColorJitter keeps size, but crop might have varied if we added it)
        # Here we just ensure it's resized back to target if needed (already handled by _context_crop, but safe to ensure)
        image_pil = image_pil.resize((self.image_size, self.image_size))
        image_tensor = TF.to_tensor(image_pil)
        image_tensor = self.normalize(image_tensor)

        # 4. Trajectory Normalization
        traj_norm = (traj_tensor - self.traj_stats["mean"]) / self.traj_stats["std"]

        # 5. Text
        text = self._tokenize_text(sample["text"])

        label = sample.get("label", 0)
        sample_id = sample.get("sample_id", str(idx))

        return {
            "image": image_tensor,
            "trajectory": traj_norm,
            "text": text,
            "label": torch.tensor(label, dtype=torch.float32),
            "sample_id": sample_id,
        }


class PSITriadDataModule(pl.LightningDataModule):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.text_model_id)
        self.train_ds: Optional[PSITriadDataset] = None
        self.val_ds: Optional[PSITriadDataset] = None
        self.test_ds: Optional[PSITriadDataset] = None

    def setup(self, stage: str) -> None:
        self.train_ds = PSITriadDataset(
            manifest_path=self.args.train_manifest,
            data_root=self.args.data_root,
            tokenizer=self.tokenizer,
            image_size=self.args.image_size,
            context_scale=self.args.context_scale,
            traj_steps=self.args.traj_steps,
            text_max_len=self.args.text_max_len,
            split="train"  # Enable augmentation
        )

        traj_stats = self.train_ds.traj_stats

        self.val_ds = PSITriadDataset(
            manifest_path=self.args.val_manifest,
            data_root=self.args.data_root,
            tokenizer=self.tokenizer,
            image_size=self.args.image_size,
            context_scale=self.args.context_scale,
            traj_steps=self.args.traj_steps,
            text_max_len=self.args.text_max_len,
            traj_stats=traj_stats,
            split="val"  # Disable augmentation
        )

        if self.args.test_manifest and Path(self.args.test_manifest).exists():
            try:
                self.test_ds = PSITriadDataset(
                    manifest_path=self.args.test_manifest,
                    data_root=self.args.data_root,
                    tokenizer=self.tokenizer,
                    image_size=self.args.image_size,
                    context_scale=self.args.context_scale,
                    traj_steps=self.args.traj_steps,
                    text_max_len=self.args.text_max_len,
                    traj_stats=traj_stats,
                    split="test"  # Disable augmentation
                )
            except RuntimeError:
                self.test_ds = None
        else:
            self.test_ds = None

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            persistent_workers=self.args.persistent_workers,
            pin_memory=self.args.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.args.val_batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            persistent_workers=self.args.persistent_workers,
            pin_memory=self.args.pin_memory,
            drop_last=False,
        )

    def test_dataloader(self):
        if self.test_ds is None:
            return None
        return DataLoader(
            self.test_ds,
            batch_size=self.args.test_batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            persistent_workers=self.args.persistent_workers,
            pin_memory=self.args.pin_memory,
            drop_last=False,
        )


class VisionContextEncoder(nn.Module):
    def __init__(self, d: int, freeze_backbone: bool):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2
        backbone = resnet50(weights=weights)
        self.feature_dim = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.backbone = nn.Sequential(*layers)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.project = nn.Linear(self.feature_dim, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x).flatten(1)
        x = self.project(feats)
        return self.norm(x)


class TrajectoryEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float, d: int):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.project = nn.Linear(hidden_dim, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, traj: torch.Tensor) -> torch.Tensor:
        _, h_n = self.rnn(traj)
        h_last = h_n[-1]
        x = self.project(h_last)
        return self.norm(x)


class TextEncoder(nn.Module):
    def __init__(self, model_id: str, freeze_encoder: bool, d: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_id)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.project = nn.Linear(self.encoder.config.hidden_size, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, text: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.encoder(**text)
        cls = outputs.last_hidden_state[:, 0, :]
        x = self.project(cls)
        return self.norm(x)


class SymilePSIModel(pl.LightningModule):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.save_hyperparameters(vars(args))
        self.args = args

        self.loss_fn = symile if args.loss_fn == "symile" else clip

        self.vision_encoder = VisionContextEncoder(args.d, args.freeze_image_backbone)
        self.traj_encoder = TrajectoryEncoder(
            input_dim=2,
            hidden_dim=args.traj_hidden,
            num_layers=args.traj_layers,
            dropout=args.traj_dropout,
            d=args.d,
        )
        self.text_encoder = TextEncoder(args.text_model_id, args.freeze_text_encoder, args.d)

        if args.freeze_logit_scale:
            self.logit_scale = nn.Parameter(torch.ones([]) * args.logit_scale_init).requires_grad_(False)
        else:
            self.logit_scale = nn.Parameter(torch.ones([]) * args.logit_scale_init)

    def forward(self, batch: Dict[str, Any]):
        r_v = self.vision_encoder(batch["image"])
        r_s = self.traj_encoder(batch["trajectory"])
        r_t = self.text_encoder(batch["text"])
        return r_v, r_s, r_t, self.logit_scale.exp()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

    def _compute_one_way_acc(self, target: torch.Tensor, queries: List[torch.Tensor], logit_scale_exp: torch.Tensor):
        """Helper to compute accuracy for a specific target given query modalities."""
        logits = zeroshot_retrieval_logits(target, queries, logit_scale_exp, self.args.loss_fn)
        labels = torch.arange(logits.shape[0], device=logits.device)
        pred = torch.argmax(logits, dim=1)
        return (pred == labels).float().mean()

    def training_step(self, batch, batch_idx):
        r_v, r_s, r_t, logit_scale_exp = self(batch)
        
        # 1. Calculate Training Loss (This optimizes the joint space for ALL tasks)
        loss = self.loss_fn(r_v, r_s, r_t, logit_scale_exp, self.args.negative_sampling)

        # 2. Calculate accuracy for the primary task (Image Retrieval) for progress bar
        acc_image = self._compute_one_way_acc(r_v, [r_s, r_t], logit_scale_exp)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=False)
        self.log("train_acc", acc_image, on_step=True, on_epoch=True, prog_bar=True, sync_dist=False)
        self.log("logit_scale_exp", logit_scale_exp, on_step=True, prog_bar=True, sync_dist=False)
        return loss

    def validation_step(self, batch, batch_idx):
        r_v, r_s, r_t, logit_scale_exp = self(batch)
        loss = self.loss_fn(r_v, r_s, r_t, logit_scale_exp, self.args.negative_sampling)
        
        # --- NEW: Compute accuracies for ALL 3 TASKS ---
        # 1. Given Text + Traj -> Find Image (Original)
        acc_v = self._compute_one_way_acc(target=r_v, queries=[r_s, r_t], logit_scale_exp=logit_scale_exp)
        
        # 2. Given Image + Traj -> Find Text (New Request)
        acc_t = self._compute_one_way_acc(target=r_t, queries=[r_v, r_s], logit_scale_exp=logit_scale_exp)
        
        # 3. Given Image + Text -> Find Trajectory (New Request)
        acc_s = self._compute_one_way_acc(target=r_s, queries=[r_v, r_t], logit_scale_exp=logit_scale_exp)

        # Log everything clearly
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_acc_image", acc_v, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val_acc_text", acc_t, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val_acc_traj", acc_s, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        
        # Keep 'val_acc' as the average of all three for a general health check
        self.log("val_acc", (acc_v + acc_t + acc_s) / 3, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss

    def test_step(self, batch, batch_idx):
        r_v, r_s, r_t, logit_scale_exp = self(batch)
        
        # Compute all 3 accuracies for the test set
        acc_v = self._compute_one_way_acc(target=r_v, queries=[r_s, r_t], logit_scale_exp=logit_scale_exp)
        acc_t = self._compute_one_way_acc(target=r_t, queries=[r_v, r_s], logit_scale_exp=logit_scale_exp)
        acc_s = self._compute_one_way_acc(target=r_s, queries=[r_v, r_t], logit_scale_exp=logit_scale_exp)

        self.log("test_acc_image", acc_v, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_acc_text", acc_t, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_acc_traj", acc_s, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_train_end(self) -> None:
        run_info = {"args": self.args, "final_logit_scale": float(self.logit_scale.exp().item())}
        with open(self.args.run_dir / "run_info.json", "w", encoding="utf8") as f:
            json.dump(run_info, f, indent=4, cls=PathToStrEncoder)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Symile training on PSI tri-modal data.")

    parser.add_argument("--train_manifest", type=Path, required=True, help="json/jsonl manifest for the train split.")
    parser.add_argument("--val_manifest", type=Path, required=True, help="json/jsonl manifest for the val split.")
    parser.add_argument("--test_manifest", type=Path, default=None, help="json/jsonl manifest for the test split.")
    parser.add_argument("--data_root", type=Path, default=Path("."), help="Root directory for frame and trajectory paths.")
    parser.add_argument("--image_size", type=int, default=224, help="Output spatial size for cropped contexts.")
    parser.add_argument("--context_scale", type=float, default=1.6, help="Context square scale around the pedestrian box.")
    parser.add_argument("--traj_steps", type=int, default=15, help="Number of past frames to consume for trajectories.")
    parser.add_argument("--text_max_len", type=int, default=32, help="Max token length for textual explanations.")

    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--val_batch_size", type=int, default=None, help="Validation batch size. Defaults to batch_size.")
    parser.add_argument("--test_batch_size", type=int, default=None, help="Test batch size. Defaults to batch_size.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--persistent_workers", type=str_to_bool, default=False, help="Use persistent workers in DataLoaders.")
    parser.add_argument("--pin_memory", type=str_to_bool, default=False, help="Pin memory for DataLoaders.")
    parser.add_argument("--drop_last", type=str_to_bool, default=True, help="Drop the last incomplete training batch.")

    parser.add_argument("--d", type=int, default=256, help="Projection dimension for all encoders.")
    parser.add_argument("--traj_hidden", type=int, default=256, help="Hidden size for the trajectory GRU.")
    parser.add_argument("--traj_layers", type=int, default=1, help="Number of GRU layers for trajectory encoding.")
    parser.add_argument("--traj_dropout", type=float, default=0.1, help="Dropout inside the trajectory GRU.")
    parser.add_argument("--text_model_id", type=str, default="distilbert-base-uncased", help="HF model id for text encoding.")
    parser.add_argument("--freeze_text_encoder", type=str_to_bool, default=True, help="Freeze DistilBERT/RoBERTa backbone.")
    parser.add_argument("--freeze_image_backbone", type=str_to_bool, default=True, help="Freeze ResNet-50 backbone.")

    parser.add_argument("--loss_fn", type=str, choices=["symile", "clip"], default="symile", help="Training objective.")
    parser.add_argument("--negative_sampling", type=str, choices=["n", "n_squared"], default="n", help="Symile negative sampling.")
    parser.add_argument("--freeze_logit_scale", type=str_to_bool, default=False, help="Freeze learned temperature.")
    parser.add_argument("--logit_scale_init", type=float, default=math.log(1 / 0.07), help="Init value for logit scale.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")

    parser.add_argument("--use_seed", type=str_to_bool, default=True, help="Enable deterministic seeding.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")

    parser.add_argument("--wandb", type=str_to_bool, default=True, help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="symile-psi", help="W&B project name.")

    parser.add_argument("--output_dir", type=Path, default=Path("runs/psi"), help="Root directory for run artifacts.")
    parser.add_argument(
        "--check_val_every_n_epoch", type=int, default=1, help="Validation interval in epochs."
    )
    parser.add_argument("--limit_train_batches", type=float, default=1.0, help="Used for quick debugging.")
    parser.add_argument("--limit_val_batches", type=float, default=1.0, help="Used for quick debugging.")

    args = parser.parse_args()

    if args.val_batch_size is None:
        args.val_batch_size = args.batch_size
    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    return args


def _create_run_directory(base_dir: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rand = random.randint(0, 9999)
    run_dir = base_dir / f"psi_{ts}_{rand:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main():
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir = _create_run_directory(args.output_dir)

    if args.use_seed:
        seed_everything(args.seed, workers=True)

    dm = PSITriadDataModule(args)
    model = SymilePSIModel(args)

    ckpt_dir = args.run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{epoch}-{val_loss:.4f}",
        every_n_epochs=args.check_val_every_n_epoch,
        save_top_k=-1,
    )

    logger = WandbLogger(project=args.wandb_project, save_dir=args.run_dir) if args.wandb else None

    trainer = Trainer(
        callbacks=checkpoint_callback,
        accelerator="auto",
        max_epochs=args.epochs,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        deterministic=args.use_seed,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        logger=logger,
    )

    trainer.fit(model, datamodule=dm)

    if dm.test_ds is not None:
        trainer.test(model, datamodule=dm, ckpt_path="best")
    else:
        print("No test dataset provided or manifest is empty; skipping test phase.")


if __name__ == "__main__":
    main()
    