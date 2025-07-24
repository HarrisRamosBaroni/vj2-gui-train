import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import copy
from pathlib import Path
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import logging
import wandb
import time
import argparse
import signal

from src.models.utils.modules import ACBlock as Block
from src.models.utils.modules import build_action_block_causal_attention_mask
from src.utils.tensors import trunc_normal_
from src.utils.logging import AverageMeter, CSVLogger, get_logger, gpu_timer
from vj2_dataloader import init_preprocessed_data_loader
from config import ROLLOUT_HORIZON, ACTION_BLOCKS_PER_WINDOW, OBSERVATIONS_PER_WINDOW
from vj2gui_utils import init_opt
from vj2gui_predictor import VJ2GUIPredictor

logger = get_logger()

class VJEPATrainer:
    def __init__(self, args):
        self.device = f"cuda:{args.local_rank}"
        torch.cuda.set_device(self.device)

        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.num_frames = OBSERVATIONS_PER_WINDOW
        self.num_workers = args.num_workers
        self.rollout_horizon = ROLLOUT_HORIZON
        self.save_every_epochs = args.save_every_epochs

        self.predictor = VJ2GUIPredictor(num_frames=self.num_frames).to(self.device)
        self.predictor = DDP(self.predictor, device_ids=[args.local_rank])

        self.scaler = GradScaler()

        self.unsupervised_loader, self.unsupervised_sampler = init_preprocessed_data_loader(
            processed_data_dir=args.processed_data_dir,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
        self.ipe = len(self.unsupervised_loader)

        self.validation_loader = None
        if args.validation_data_dir:
            self.validation_loader, _ = init_preprocessed_data_loader(
                processed_data_dir=args.validation_data_dir,
                batch_size=self.batch_size,
                num_workers=self.num_workers
            )

        self.optimizer, self.scheduler, self.wd_scheduler = init_opt(
            encoder=None,
            predictor=self.predictor.module,
            iterations_per_epoch=self.ipe,
            start_lr=0.000075,
            ref_lr=0.000425,
            warmup=15,
            anneal=15,
            num_epochs=self.num_epochs
        )

        self.crop_size = 256
        self.patch_size = 16
        self.tokens_per_frame = int((self.crop_size // self.patch_size) ** 2)
        self.loss_exp = 1

        self.start_epoch = 0
        self.best_loss = float("inf")
        self.current_epoch = 0
        self.current_loss = float("inf")

        self.run_dir = Path("checkpoints") / time.strftime("%Y%m%d_%H%M%S")
        self.run_dir.mkdir(parents=True, exist_ok=True)

        if dist.get_rank() == 0:
            self.run = wandb.init(
                project="vjepa2-ac",
                name=f"run_{time.strftime('%Y%m%d_%H%M%S')}",
                config=dict(
                    epochs=self.num_epochs,
                    batch_size=self.batch_size,
                    lr_start=0.000075,
                    lr_ref=0.000425,
                    warmup=15,
                    anneal=15,
                    rollout_horizon=self.rollout_horizon,
                    tokens_frame=self.tokens_per_frame,
                ),
            )
            wandb.watch(self.predictor.module, log="gradients", log_freq=100)

        signal.signal(signal.SIGINT, lambda signum, frame: self._save_interrupted_checkpoint())

    def _save_checkpoint(self, tag, epoch, loss):
        if dist.get_rank() != 0:
            return
        ckpt = {
            "epoch": epoch,
            "loss": loss,
            "predictor": self.predictor.module.state_dict(),
            "opt": self.optimizer.state_dict(),
        }
        if hasattr(self.scheduler, "state_dict"):
            ckpt["sched"] = self.scheduler.state_dict()
        if hasattr(self.wd_scheduler, "state_dict"):
            ckpt["wd_sched"] = self.wd_scheduler.state_dict()

        path = self.run_dir / f"vjepa2_{tag}.pt"
        temp_path = self.run_dir / f"vjepa2_{tag}.pt.tmp"
        try:
            torch.save(ckpt, temp_path)
            os.rename(temp_path, path)
            logger.info(f"✅  Saved checkpoint → {path}")
        except Exception as e:
            logger.error(f"❌ Error saving checkpoint: {e}")
            if temp_path.exists():
                os.remove(temp_path)
        finally:
            if temp_path.exists():
                os.remove(temp_path)

    def _save_interrupted_checkpoint(self):
        logger.info("Caught Ctrl+C. Saving interrupted checkpoint...")
        self._save_checkpoint("interrupted", self.current_epoch, self.current_loss)
        logger.info("Interrupted checkpoint saved. Exiting.")
        exit()

    def training_loop(self):
        if self.unsupervised_sampler:
            self.unsupervised_sampler.set_epoch(self.start_epoch)

        for epoch in range(self.start_epoch, self.start_epoch + self.num_epochs):
            self.current_epoch = epoch
            loss_meter = AverageMeter()
            jloss_meter = AverageMeter()
            sloss_meter = AverageMeter()

            loader = iter(self.unsupervised_loader)
            for itr, sample in enumerate(loader):
                loss, jloss, sloss, lr, wd = self.train_step(sample)

                loss_meter.update(loss)
                jloss_meter.update(jloss)
                sloss_meter.update(sloss)

                if dist.get_rank() == 0:
                    wandb.log({
                        "iter": epoch * len(loader) + itr,
                        "train/loss": loss,
                        "train/jloss": jloss,
                        "train/sloss": sloss,
                        "lr": lr,
                        "wd": wd,
                        "epoch": epoch + 1,
                    })

            self.current_loss = loss_meter.avg

            if self.validation_loader:
                val_loss, val_jloss, val_sloss = self.validate_epoch()
                if dist.get_rank() == 0:
                    wandb.log({
                        "val/loss": val_loss,
                        "val/jloss": val_jloss,
                        "val/sloss": val_sloss,
                    })
            self._save_checkpoint("last", epoch, self.current_loss)

            if self.save_every_epochs > 0 and (epoch + 1) % self.save_every_epochs == 0:
                self._save_checkpoint(f"epoch_{epoch+1}", epoch, self.current_loss)

        if dist.get_rank() == 0:
            wandb.finish()

    def validate_epoch(self):
        self.predictor.eval()
        loss_meter, jloss_meter, sloss_meter = AverageMeter(), AverageMeter(), AverageMeter()

        loader = iter(self.validation_loader)
        for sample in loader:
            loss, jloss, sloss = self.validation_step(sample)
            loss_meter.update(loss)
            jloss_meter.update(jloss)
            sloss_meter.update(sloss)

        self.predictor.train()
        return loss_meter.avg, jloss_meter.avg, sloss_meter.avg

    @torch.no_grad()
    def validation_step(self, sample):
        embeddings, actions = self.load_trajectory(sample)
        z_all = F.layer_norm(embeddings, (embeddings.size(-1),))
        h_all = z_all

        with autocast():
            z_tf, z_ar_final = self.forward_predictions(z_all, actions)
            jloss = self.loss_fn(z_tf, h_all[:, 1:])
            sloss = self.loss_fn(z_ar_final.unsqueeze(1), h_all[:, self.rollout_horizon].unsqueeze(1))
            return float(jloss + sloss), float(jloss), float(sloss)

    def train_step(self, sample):
        lr, wd = self.scheduler.step(), self.wd_scheduler.step()
        embeddings, actions = self.load_trajectory(sample)
        z_all = F.layer_norm(embeddings, (embeddings.size(-1),))
        h_all = z_all

        self.optimizer.zero_grad()
        with autocast():
            z_tf, z_ar_final = self.forward_predictions(z_all, actions)
            jloss = self.loss_fn(z_tf, h_all[:, 1:])
            sloss = self.loss_fn(z_ar_final.unsqueeze(1), h_all[:, self.rollout_horizon].unsqueeze(1))
            loss = jloss + sloss

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return float(loss), float(jloss), float(sloss), lr, wd

    def load_trajectory(self, sample):
        embeddings, actions = sample
        return embeddings.to(self.device, non_blocking=True), actions.to(self.device, dtype=torch.float, non_blocking=True)

    def forward_predictions(self, z_all, actions):
        B, T_seq, _, _ = actions.shape
        actions_flat = actions.view(B, T_seq, -1)
        z_tf = self.predictor(z_all[:, :-1], actions_flat)

        z_rollout = z_all[:, 0].unsqueeze(1)
        for i in range(self.rollout_horizon):
            a = actions_flat[:, i].unsqueeze(1)
            z_rollout = self.predictor(z_rollout, a)

        return z_tf, z_rollout.squeeze(1)

    def loss_fn(self, z_pred, h_target):
        return torch.mean(torch.abs(z_pred - h_target) ** self.loss_exp) / self.loss_exp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VJ2-GUI Agent with Multi-GPU")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--processed_data_dir", type=str, required=True)
    parser.add_argument("--validation_data_dir", type=str, required=True)
    parser.add_argument("--save_every_epochs", type=int, default=0)
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    dist.init_process_group("nccl")

    trainer = VJEPATrainer(args)
    trainer.training_loop()
