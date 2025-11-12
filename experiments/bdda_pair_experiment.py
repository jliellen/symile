"""Train pairwise CLIP-style models on BDDA modality pairs without touching the
three-modality Symile pipeline."""

from argparse import ArgumentParser, Namespace
import sys

import lightning.pytorch as pl
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torch
import torch.nn as nn
import torch.nn.functional as F

from args import parse_args_main
import datasets
from losses import infonce
from main import create_save_directory
from models.symile_bdda_toy_model import ConvBackbone, GPSProjection, init_logit_scale


PAIR_CHOICES = ("camera_gaze", "camera_gps", "gaze_gps")


def parse_pairwise_args():
    """
    Parse CLI arguments by first extracting the requested modality pair and then
    delegating the remaining options to the standard Symile arg parser.
    """
    original_argv = sys.argv[:]
    help_requested = any(arg in ("-h", "--help") for arg in original_argv[1:])

    pair_parser = ArgumentParser(add_help=False)
    pair_parser.add_argument(
        "--pair",
        default="camera_gaze",
        choices=PAIR_CHOICES,
        help="Which two BDDA modalities to contrast (default: camera_gaze).",
    )

    if help_requested:
        verbose_pair_parser = ArgumentParser(description="Pairwise BDDA options")
        verbose_pair_parser.add_argument(
            "--pair",
            default="camera_gaze",
            choices=PAIR_CHOICES,
            help="Which two BDDA modalities to contrast (default: camera_gaze).",
        )
        verbose_pair_parser.print_help()
        print("\n--- Additional Symile arguments ---\n")

    pair_args, remaining = pair_parser.parse_known_args()

    try:
        sys.argv = [original_argv[0]] + remaining
        args = parse_args_main()
    finally:
        sys.argv = original_argv

    setattr(args, "pair", pair_args.pair)
    return args


class PairwiseSymileBDDAModel(pl.LightningModule):
    """
    Lightweight two-modality variant that reuses the original BDDA encoders but
    optimizes the CLIP InfoNCE objective on a selected pair.
    """

    def __init__(self, args: Namespace):
        super().__init__()
        self.args = Namespace(**vars(args))
        self.modalities = {
            "camera_gaze": ("camera", "gaze"),
            "camera_gps": ("camera", "gps"),
            "gaze_gps": ("gaze", "gps"),
        }[self.args.pair]

        # Track hyperparameters for checkpoint inspection.
        self.save_hyperparameters(
            {
                "pair": self.args.pair,
                "d": getattr(self.args, "d", 128),
                "lr": getattr(self.args, "lr", 1e-3),
                "weight_decay": getattr(self.args, "weight_decay", 0.0),
            }
        )

        if "camera" in self.modalities:
            self.camera_encoder = ConvBackbone(
                self.args,
                in_channels=3,
                base_channels=getattr(self.args, "camera_base_channels", 64),
                num_blocks=getattr(self.args, "camera_num_blocks", 4),
                dropout=getattr(self.args, "camera_dropout", 0.1),
            )
        if "gaze" in self.modalities:
            self.gaze_encoder = ConvBackbone(
                self.args,
                in_channels=3,
                base_channels=getattr(self.args, "gaze_base_channels", 48),
                num_blocks=getattr(self.args, "gaze_num_blocks", 4),
                dropout=getattr(self.args, "gaze_dropout", 0.1),
            )
        if "gps" in self.modalities:
            self.gps_encoder = GPSProjection(
                self.args, in_features=getattr(self.args, "gps_dim", 4)
            )

        logit_scale_value = init_logit_scale(self.args)
        self.logit_scale = nn.Parameter(torch.tensor(logit_scale_value))
        if getattr(self.args, "freeze_logit_scale", False):
            self.logit_scale.requires_grad = False

    def encode_batch(self, batch):
        reps = {}
        if "camera" in self.modalities:
            reps["camera"] = self.camera_encoder(batch["camera"])
        if "gaze" in self.modalities:
            reps["gaze"] = self.gaze_encoder(batch["gaze"])
        if "gps" in self.modalities:
            reps["gps"] = self.gps_encoder(torch.nan_to_num(batch["gps"], nan=0.0))
        return reps

    @staticmethod
    def compute_retrieval_top1(anchor: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        anchor_norm = F.normalize(anchor, dim=1)
        other_norm = F.normalize(other, dim=1)
        sims = anchor_norm @ other_norm.T
        preds = sims.argmax(dim=1)
        labels = torch.arange(anchor.shape[0], device=anchor.device)
        return (preds == labels).float().mean()

    def shared_step(self, batch, stage: str):
        reps = self.encode_batch(batch)
        left, right = (reps[self.modalities[0]], reps[self.modalities[1]])
        logit_scale_exp = self.logit_scale.exp()

        loss = infonce(left, right, logit_scale_exp)

        self.log(f"{stage}_loss", loss, prog_bar=stage != "test", batch_size=left.shape[0])
        self.log(
            f"{stage}_logit_scale", logit_scale_exp, prog_bar=False, batch_size=left.shape[0]
        )

        with torch.no_grad():
            acc_forward = self.compute_retrieval_top1(left, right)
            acc_backward = self.compute_retrieval_top1(right, left)
            self.log(
                f"{stage}_acc_{self.modalities[0]}_{self.modalities[1]}",
                acc_forward,
                prog_bar=False,
                batch_size=left.shape[0],
            )
            self.log(
                f"{stage}_acc_{self.modalities[1]}_{self.modalities[0]}",
                acc_backward,
                prog_bar=False,
                batch_size=left.shape[0],
            )

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
        )
        return optimizer


def run():
    args = parse_pairwise_args()

    if getattr(args, "loss_fn", "symile") != "clip":
        print("Pairwise training enforces --loss_fn clip. Overriding argument value.")
        setattr(args, "loss_fn", "clip")

    # Prepare filesystem locations.
    save_dir = create_save_directory(args)
    setattr(args, "save_dir", save_dir)
    print(f"Saving run artifacts to: {save_dir}")

    if args.use_seed:
        seed_everything(args.seed, workers=True)

    if args.wandb:
        logger = WandbLogger(
            project="symile",
            log_model=False,
            save_dir=args.ckpt_save_dir,
            id=args.wandb_run_id,
        )
    else:
        logger = False

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_dir,
        filename="{epoch}-{val_loss:.4f}",
        every_n_epochs=args.check_val_every_n_epoch,
        save_top_k=-1,
    )

    trainer = Trainer(
        callbacks=checkpoint_callback,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        deterministic=args.use_seed,
        enable_progress_bar=True,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        log_every_n_steps=1,
        logger=logger,
        max_epochs=args.epochs,
        num_sanity_val_steps=0,
    )

    dm = datasets.SymileBDDADataModule(args)
    model = PairwiseSymileBDDAModel(args)

    if args.ckpt_path in (None, "None"):
        trainer.fit(model, datamodule=dm)
    else:
        trainer.fit(model, datamodule=dm, ckpt_path=args.ckpt_path)

    if args.wandb:
        logger.experiment.finish()


if __name__ == "__main__":
    run()
