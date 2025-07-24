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

from src.models.utils.modules import ACBlock as Block
from src.models.utils.modules import build_action_block_causal_attention_mask
from src.utils.tensors import trunc_normal_
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from src.utils.logging import AverageMeter, CSVLogger, get_logger, gpu_timer
from vj2_dataloader import init_preprocessed_data_loader
from config import ROLLOUT_HORIZON, ACTION_BLOCKS_PER_WINDOW, OBSERVATIONS_PER_WINDOW

from vj2gui_utils import init_opt

from vj2gui_predictor import VJ2GUIPredictor
# from vj2gui import VJEPA2Wrapper
import os
import logging
import wandb
import time
import argparse
import signal
 
logger = logging.getLogger()
 
class VJEPATrainer:
    def __init__(self, num_epochs, batch_size, num_workers, processed_data_dir, validation_data_dir, save_every_epochs, unfreeze_encoder, checkpoint_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device
        self.num_frames = OBSERVATIONS_PER_WINDOW
        self.num_workers = num_workers
        self.rollout_horizon = ROLLOUT_HORIZON
        self.save_every_epochs = save_every_epochs
        
        self.encoder = None
        self.target_encoder = None

        self.predictor = VJ2GUIPredictor(num_frames=self.num_frames).to(device)
        self.scaler = GradScaler()
        
        self.unsupervised_loader, self.unsupervised_sampler = init_preprocessed_data_loader(
            processed_data_dir=processed_data_dir,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
        self.ipe = len(self.unsupervised_loader)
        
        self.validation_loader = None
        if validation_data_dir:
            self.validation_loader, _ = init_preprocessed_data_loader(
                processed_data_dir=validation_data_dir,
                batch_size=self.batch_size,
                num_workers=self.num_workers
            )
        
        # Optimizer only handles the predictor. TODO: remove encoder stuff
        self.optimizer, self.scheduler, self.wd_scheduler = init_opt(
            encoder = None,
            predictor = self.predictor,
            iterations_per_epoch = self.ipe,
            start_lr  = 0.000075,
            ref_lr = 0.000425,
            warmup = 15,
            anneal = 15,
            num_epochs = num_epochs
        )
        self.crop_size = 256
        self.patch_size = 16
        self.tokens_per_frame = int((self.crop_size // self.patch_size) ** 2)
        self.loss_exp = 1
        
        self.start_epoch = 0
        self.best_loss = float("inf")
        self.current_epoch = 0
        self.current_loss = float("inf")

        if checkpoint_path:
            self.start_epoch, self.best_loss = self._load_checkpoint(checkpoint_path)
            logger.info(f"Resuming training from epoch {self.start_epoch} with best loss {self.best_loss:.4f}")
        else:
            self.run_dir   = Path("checkpoints") / time.strftime("%Y%m%d_%H%M%S")
            self.run_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Starting new training run. Checkpoints will be saved to {self.run_dir}")

        self.run = wandb.init(
            project="vjepa2-ac",
            name   = f"run_{time.strftime('%Y%m%d_%H%M%S')}",
            config = dict(
                epochs       = num_epochs,
                batch_size   = batch_size,
                lr_start     = 0.000075,
                lr_ref       = 0.000425,
                warmup       = 15,
                anneal       = 15,
                rollout_horizon = self.rollout_horizon,
                tokens_frame = self.tokens_per_frame,
            ),
        )
        wandb.watch(
            self.predictor,
            log="gradients",
            log_freq=100,
        )

        # Register signal handler for graceful shutdown
        signal.signal(signal.SIGINT, lambda signum, frame: self._save_interrupted_checkpoint())


        
    def _save_checkpoint(self, tag, epoch, loss):
        ckpt = {
            "epoch":     epoch,
            "loss":      loss,
            "predictor": self.predictor.state_dict(),
            "opt":       self.optimizer.state_dict(),
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

    def _load_checkpoint(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.predictor.load_state_dict(ckpt["predictor"])
        self.optimizer.load_state_dict(ckpt["opt"])
        if "sched" in ckpt and hasattr(self.scheduler, "load_state_dict"):
            self.scheduler.load_state_dict(ckpt["sched"])
        if "wd_sched" in ckpt and hasattr(self.wd_scheduler, "load_state_dict"):
            self.wd_scheduler.load_state_dict(ckpt["wd_sched"])
        
        self.run_dir = Path(checkpoint_path).parent
        
        return ckpt["epoch"] + 1, ckpt["loss"]

    def training_loop(self):
        print("Initializing loader...")
        if self.unsupervised_sampler is not None:
            self.unsupervised_sampler.set_epoch(self.start_epoch)
        
        for epoch in range(self.start_epoch, self.start_epoch + self.num_epochs):
            self.current_epoch = epoch
            loss_meter = AverageMeter()
            jloss_meter = AverageMeter()
            sloss_meter = AverageMeter()

            # Use a fresh iterator for each epoch
            loader = iter(self.unsupervised_loader)
            
            print("Epoch %d" % (epoch + 1))
            for itr, sample in enumerate(loader):
                (
                    loss,
                    jloss,
                    sloss,
                    _new_lr,
                    _new_wd,
                ) = self.train_step(sample)
                
                loss_meter.update(loss)
                jloss_meter.update(jloss)
                sloss_meter.update(sloss)
                
                print(f"Itr {itr+1}/{len(loader)} --- Loss: {loss_meter.avg:.4f} (TF: {jloss_meter.avg:.4f}, AR: {sloss_meter.avg:.4f}), LR: {_new_lr:.6f}, WD: {_new_wd:.6f}")
                wandb.log(
                    {
                        "iter": epoch * len(loader) + itr,
                        "train/loss": loss,
                        "train/jloss": jloss,
                        "train/sloss": sloss,
                        "lr": _new_lr,
                        "wd": _new_wd,
                        "epoch": epoch + 1,
                    },
                    step=epoch * len(loader) + itr,
                )

            epoch_loss = loss_meter.avg
            self.current_loss = epoch_loss
            
            val_loss, val_jloss, val_sloss = -1, -1, -1
            if self.validation_loader:
                val_loss, val_jloss, val_sloss = self.validate_epoch()
                print(f"--- Epoch {epoch+1} Summary --- Train Loss: {epoch_loss:.4f} --- Val Loss: {val_loss:.4f}")
                wandb.log({
                    "val/loss": val_loss,
                    "val/jloss": val_jloss,
                    "val/sloss": val_sloss,
                }, step=epoch * len(loader) + len(loader) - 1)
            else:
                print(f"--- Epoch {epoch+1} Summary --- Train Loss: {epoch_loss:.4f} ---")

            self._save_checkpoint("last", epoch, epoch_loss)

            if self.validation_loader and val_loss < self.best_loss:
                self.best_loss = val_loss
                self._save_checkpoint("best", epoch, val_loss)
                logger.info(f" New best validation loss {val_loss:.4f} at epoch {epoch+1}!")

            if self.save_every_epochs > 0 and (epoch + 1) % self.save_every_epochs == 0:
                self._save_checkpoint(f"epoch_{epoch+1}", epoch, epoch_loss)

        wandb.finish()

    
    def validate_epoch(self):
        self.predictor.eval()
        loss_meter = AverageMeter()
        jloss_meter = AverageMeter()
        sloss_meter = AverageMeter()

        loader = iter(self.validation_loader)
        for itr, sample in enumerate(loader):
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

            h_tf_targets = h_all[:, 1:]
            h_rollout_target = h_all[:, self.rollout_horizon]

            jloss = self.loss_fn(z_tf, h_tf_targets)
            sloss = self.loss_fn(z_ar_final.unsqueeze(1), h_rollout_target.unsqueeze(1))
            
            loss = jloss + sloss
        
        return float(loss), float(jloss), float(sloss)

    def load_trajectory(self, sample):
        """Loads and prepares a single training trajectory."""
        embeddings, actions = sample
        
        embeddings = embeddings.to(self.device, non_blocking=True)
        actions = actions.to(self.device, dtype=torch.float, non_blocking=True)
        
        return embeddings, actions

    def train_step(self, sample):
        _new_lr = self.scheduler.step()
        _new_wd = self.wd_scheduler.step()
        
        embeddings, actions = self.load_trajectory(sample)
        z_all = F.layer_norm(embeddings, (embeddings.size(-1),))
        h_all = z_all
        
        self.optimizer.zero_grad()
        with autocast():
            z_tf, z_ar_final = self.forward_predictions(z_all, actions)

            h_tf_targets = h_all[:, 1:]  # teacher-forced targets z_1, ..., z_7
            
            h_rollout_target = h_all[:, self.rollout_horizon]  # rollout target is z_{rollout_horizon}

            jloss = self.loss_fn(z_tf, h_tf_targets)
            
            sloss = self.loss_fn(z_ar_final.unsqueeze(1), h_rollout_target.unsqueeze(1))
            
            loss = jloss + sloss
        
        self.scaler.scale(loss).backward() # Use scaler for backward
        self.scaler.step(self.optimizer) # Use scaler for optimizer step
        self.scaler.update() # Update scaler
        
        return (
            float(loss),
            float(jloss),
            float(sloss),
            _new_lr,
            _new_wd,
        )

    
    def forward_predictions(self, z_all, actions):
        """
        Performs both teacher-forced and autoregressive predictions based on
        the full sequence of latent states.
        """
        B, T_seq, _, _ = actions.shape
        actions_flat = actions.view(B, T_seq, -1)  # [B, T_seq, ACTIONS_PER_BATCH, ACTION_DIM] -> [B, T_seq, ACTIONS_PER_BATCH * ACTION_DIM]

        z_tf = self._step_predictor(z_all[:, :-1], actions_flat)

        z_rollout_current = z_all[:, 0].unsqueeze(1)
        for i in range(self.rollout_horizon):
            action_current = actions_flat[:, i].unsqueeze(1)
            z_rollout_next = self._step_predictor(z_rollout_current, action_current)
            z_rollout_current = z_rollout_next
        
        z_ar_final = z_rollout_current.squeeze(1)

        return z_tf, z_ar_final

    def _step_predictor(self, z, a):
        """
        Helper for a single prediction step.
        - z: [B, T_seq, N, D]
        - a: [B, T_seq, A_dim]
        """
        pred = self.predictor(z, a)
        pred = F.layer_norm(pred, (pred.size(-1),))
        return pred
        
    def loss_fn(self, z_pred, h_target):
        """
        Calculates the loss between the predicted visual tokens (z) and the
        target visual tokens (h).
        """
        # Ensure dimensions match for broadcasting if needed
        # z_pred: [B, Seq, Latent, N, D]
        # h_target: [B, Seq, Latent, N, D]
        return torch.mean(torch.abs(z_pred - h_target) ** self.loss_exp) / self.loss_exp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VJ2-GUI Agent")
    parser.add_argument("-e", "--num_epochs", type=int, default=1, help="Number of training epochs to run during the training session. Default: 1")
    parser.add_argument("-b", "--batch_size", type=int, default=4, help="Batch size for training. Default: 4")
    parser.add_argument("-w", "--num_workers", type=int, default=1, help="Number of worker processes for data loading. Default: 1")
    parser.add_argument("-l", '--log', default='info', choices=['debug', 'info', 'warning', 'error', 'critical'])
    parser.add_argument("-t", "--processed_data_dir", type=str, required=True, help="Directory containing the pre-processed training .npz files. Required.")
    parser.add_argument("-c", "--load_checkpoint", type=str, default=None, help="Path to a checkpoint file to resume training from.")
    parser.add_argument("-v", "--validation_data_dir", type=str, required=True, help="Directory containing the pre-processed validation .npz files. Required.")
    parser.add_argument("-s", "--save_every_epochs", type=int, default=0, help="Save a checkpoint every this many epochs. 0 to disable. Default: 0")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper()))

    trainer = VJEPATrainer(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        processed_data_dir=args.processed_data_dir,
        validation_data_dir=args.validation_data_dir,
        save_every_epochs=args.save_every_epochs,
        unfreeze_encoder=False,
        checkpoint_path=args.load_checkpoint
    )
    trainer.training_loop()
