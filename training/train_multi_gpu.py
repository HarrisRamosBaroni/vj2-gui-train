import torch
import torch.nn.functional as F
from pathlib import Path
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import wandb
import time
import argparse
import signal
import datetime

from training.dataloader import init_preprocessed_data_loader
from config import ROLLOUT_HORIZON, OBSERVATIONS_PER_WINDOW
from src.utils.logging import AverageMeter, get_logger
from training.utils import init_opt
from src.models.predictor import VJ2GUIPredictor
from testing.model_info import analyze_my_model

logger = get_logger()

def _ddp_mean(x: float, device) -> float:
    """Average a scalar across all DDP ranks."""
    t = torch.tensor([x], dtype=torch.float32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return float(t.item())

class VJEPATrainer:
    def __init__(self, args):
        if args.device_type == "cuda":
            self.device = torch.device(f"cuda:{args.local_rank}")
        else:
            self.device = torch.device("cpu")

        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.num_frames = OBSERVATIONS_PER_WINDOW
        self.num_workers = args.num_workers
        self.rollout_horizon = ROLLOUT_HORIZON
        self.save_every_epochs = args.save_every_epochs
        self.save_every_iters = args.save_every_iters

        self.predictor = VJ2GUIPredictor(num_frames=self.num_frames,depth=24).to(self.device)
        
        # DDP wrapper - only pass device_ids for CUDA
        if args.device_type == "cuda":
            self.predictor = DDP(self.predictor, device_ids=[args.local_rank])
        else:
            self.predictor = DDP(self.predictor)

        model_stats = analyze_my_model(self.predictor, verbose=True)
        
        print(f"[DDP] Rank {dist.get_rank()}: Model initialized on {self.device}")


        self.scaler = GradScaler()

        self.unsupervised_loader, self.unsupervised_sampler = init_preprocessed_data_loader(
            processed_data_dir=args.processed_data_dir,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
        self.ipe = len(self.unsupervised_loader)

        self.validation_loader = None
        self.validation_sampler = None
        if args.validation_data_dir:
            self.validation_loader, self.validation_sampler = init_preprocessed_data_loader(
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

        self.crop_size = 256 # pixels
        self.patch_size = 16 # pixels
        self.tokens_per_frame = int((self.crop_size // self.patch_size) ** 2) # count
        self.loss_exp = 1 # 1 for L1, 2 for L2

        # Initializing loss tracking
        self.start_epoch = 0
        self.best_val_loss = float("inf")
        self.current_epoch = 0
        self.current_loss = float("inf")
        self.global_step = 0

        self.run_dir = Path("checkpoints") / time.strftime("%Y%m%d_%H%M%S")
        self.run_dir.mkdir(parents=True, exist_ok=True) # Create directory if it doesn't exists

        if dist.get_rank() == 0:
            # Initialize Weights & Biases
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
            wandb.watch(self.predictor.module, log="gradients", log_freq=50) # Log geradients

        signal.signal(signal.SIGINT, lambda signum, frame: self._save_interrupted_checkpoint())

    def _save_checkpoint(self, tag, step, val_loss=None):
        if dist.get_rank() != 0:
            return
        
        # Only save predictor weights as requested
        ckpt = {
            "global_step": step,
            "predictor": self.predictor.module.state_dict(),
        }
        if val_loss is not None:
            ckpt["val_loss"] = val_loss

        path = self.run_dir / f"vjepa2_{tag}.pt"
        temp_path = self.run_dir / f"vjepa2_{tag}.pt.tmp"
        try:
            torch.save(ckpt, temp_path)
            os.rename(temp_path, path)
            if val_loss is not None:
                logger.info(f"✅  Saved checkpoint → {path} (val_loss: {val_loss:.6f})")
            else:
                logger.info(f"✅  Saved checkpoint → {path} (step: {step})")
        except Exception as e:
            logger.error(f"❌ Error saving checkpoint: {e}")
            if temp_path.exists():
                os.remove(temp_path)
        finally:
            if temp_path.exists():
                os.remove(temp_path)

    def _save_interrupted_checkpoint(self):
        logger.info("Caught Ctrl+C. Saving interrupted checkpoint...")
        self._save_checkpoint("interrupted", self.global_step)
        logger.info("Interrupted checkpoint saved. Exiting.")
        exit()

    def training_loop(self):
        print(f"[DDP] Rank {dist.get_rank()}: Starting training loop for {self.num_epochs} epochs")
        
        for epoch in range(self.start_epoch, self.start_epoch + self.num_epochs):
            if isinstance(self.unsupervised_sampler, torch.utils.data.DistributedSampler):
                self.unsupervised_sampler.set_epoch(epoch)
            if isinstance(self.validation_sampler, torch.utils.data.DistributedSampler):
                self.validation_sampler.set_epoch(epoch)
            
            if dist.get_rank() == 0:
                print(f"[DDP] Starting epoch {epoch + 1}/{self.start_epoch + self.num_epochs}")
            self.current_epoch = epoch
            loss_meter = AverageMeter()
            jloss_meter = AverageMeter()
            sloss_meter = AverageMeter()

            loader = iter(self.unsupervised_loader)
            for itr, sample in enumerate(loader):
                self.global_step += 1
                loss, jloss, sloss, lr, wd = self.train_step(sample)
                # jlos: teacher forcing loss, sloss: rollout loss

                loss_meter.update(loss)
                jloss_meter.update(jloss)
                sloss_meter.update(sloss)

                if dist.get_rank() == 0:
                    wandb.log({
                        "iter": self.global_step,
                        "train/loss": loss,
                        "train/jloss": jloss,
                        "train/sloss": sloss,
                        "lr": lr,
                        "wd": wd,
                        "epoch": epoch + 1,
                    })

                # Routine checkpoint saving every N iterations
                if self.global_step % self.save_every_iters == 0:
                    if dist.get_rank() == 0:
                        self._save_checkpoint(f"step_{self.global_step}", self.global_step)

                # Validation every 100 steps
                if itr % 100 == 0:
                    print(f"[DDP] Rank {dist.get_rank()}: Entering validation at epoch {epoch}, iter {itr}")
                    dist.barrier()
                    val_loss = val_jloss = val_sloss = None
                    if self.validation_loader:
                        val_loss, val_jloss, val_sloss = self.validate_epoch()
                        device = next(self.predictor.parameters()).device
                        val_loss = _ddp_mean(val_loss, device)
                        val_jloss = _ddp_mean(val_jloss, device)
                        val_sloss = _ddp_mean(val_sloss, device)
                        
                        # Save best model if validation loss improved
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            if dist.get_rank() == 0:
                                self._save_checkpoint("best", self.global_step, val_loss)
                                logger.info(f"🏆 New best validation loss: {val_loss:.6f}")
                    
                    dist.barrier()
                    print(f"[DDP] Rank {dist.get_rank()}: Exiting validation at epoch {epoch}, iter {itr}")
                    if dist.get_rank() == 0:
                        if self.validation_loader:
                            wandb.log({
                                "val/loss": val_loss,
                                "val/jloss": val_jloss,
                                "val/sloss": val_sloss,
                                "best_val_loss": self.best_val_loss,
                                "step": self.global_step,
                            })
            self.current_loss = loss_meter.avg
            
            # Only save epoch-based checkpoints if explicitly requested (for backwards compatibility)
            if self.save_every_epochs > 0 and (epoch + 1) % self.save_every_epochs == 0:
                if dist.get_rank() == 0:
                    self._save_checkpoint(f"epoch_{epoch+1}", self.global_step)

        print(f"[DDP] Rank {dist.get_rank()}: Training loop completed")
        if dist.get_rank() == 0:
            wandb.finish()

    def validate_epoch(self):
        self.predictor.eval() # remember to convert back to train later
        loss_meter, jloss_meter, sloss_meter = AverageMeter(), AverageMeter(), AverageMeter()

        loader = iter(self.validation_loader)
        for sample in loader:
            loss, jloss, sloss = self.validation_step(sample)
            loss_meter.update(loss)
            jloss_meter.update(jloss)
            sloss_meter.update(sloss)

        self.predictor.train() # convert back to train mode, we remembered :)
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
        
        # Validate sequence length for rollout horizon
        seq_len = embeddings.size(1)
        if seq_len < self.rollout_horizon + 1:
            raise ValueError(f"Sequence length {seq_len} is too short for rollout_horizon {self.rollout_horizon}. Need at least {self.rollout_horizon + 1}")
        
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
    parser.add_argument("--save_every_iters", type=int, default=500, help="Save checkpoint every N iterations")
    parser.add_argument("--local_rank", type=int, default=-1)  # Changed default to -1
    parser.add_argument("--dist-backend", default=None, help="nccl|gloo")
    parser.add_argument("--device", default=None, help="cuda|cpu")
    args = parser.parse_args()

    # Backend-agnostic initialization
    use_cuda = torch.cuda.is_available()
    backend = args.dist_backend or ("nccl" if use_cuda else "gloo")
    device = args.device or ("cuda" if use_cuda else "cpu")
    
    print(f"[DDP] Initializing with backend={backend}, device={device}")
    
    # Initialize process group using environment variables
    dist.init_process_group(backend=backend, init_method="env://",
                           timeout=datetime.timedelta(seconds=120))
    local_rank = int(os.environ["LOCAL_RANK"])
    
    if device == "cuda":
        torch.cuda.set_device(local_rank)
    
    args.local_rank = local_rank
    args.device_type = device
    
    print(f"[DDP] Rank {dist.get_rank()}/{dist.get_world_size()}, local_rank={local_rank}, device={device}")

    trainer = VJEPATrainer(args)
    trainer.training_loop()

    # Cleanup
    dist.destroy_process_group()
