import torch
import torch.nn.functional as F
from pathlib import Path
# from torch.cuda.amp import autocast, GradScaler
from torch.amp import autocast, GradScaler
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
from gui_world_model.predictor import VJ2GUIPredictor
from gui_world_model.predictor_cross_attention import VJ2GUIPredictor as VJ2GUIPredictorCrossAttention
from gui_world_model.predictor_film import VJ2GUIPredictorFiLM
from gui_world_model.predictor_a_encoded import VJ2GUIPredictorActionEncoded
from training.validators.input_sensitivity import InputSensitivityValidator
from training.validators.film_gamma import FilmGammaValidator
from training.validators.loss_distribution import LossDistributionValidator
from gui_world_model.utils.loading import load_model

# from testing.model_info import analyze_my_model

MODEL_REGISTRY = {
    "vanilla": VJ2GUIPredictor,
    "cross_attention": VJ2GUIPredictorCrossAttention,
    "film": VJ2GUIPredictorFiLM,
    "encode_a": VJ2GUIPredictorActionEncoded,
}

VALIDATOR_REGISTRY = {
    "input_sensitivity": InputSensitivityValidator,
    "film_gamma": FilmGammaValidator,
    "loss_distribution": LossDistributionValidator,
}

logger = get_logger()

def _ddp_mean(x: float, device) -> float:
    """Average a scalar across all DDP ranks."""
    t = torch.tensor([x], dtype=torch.float32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return float(t.item())

class VJEPATrainer:
    def __init__(self, args, model):
        self.args = args
        self.model_type = args.model_type
        if args.device_type == "cuda":
            self.device = torch.device(f"cuda:{args.local_rank}")
        else:
            self.device = torch.device("cpu")

        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.num_frames = OBSERVATIONS_PER_WINDOW
        self.tubelet_size = 2
        self.num_workers = args.num_workers
        self.rollout_horizon = args.rollout_horizon
        self.save_every_epochs = args.save_every_epochs
        self.save_every_iters = args.save_every_iters

        self.predictor = model.to(self.device)
        # DDP wrapper - only pass device_ids for CUDA
        if args.device_type == "cuda":
            self.predictor = DDP(self.predictor, device_ids=[args.local_rank])
        else:
            self.predictor = DDP(self.predictor)

        # model_stats = analyze_my_model(self.predictor, verbose=True)
        
        print(f"[DDP] Rank {dist.get_rank()}: Model initialized on {self.device}")

        self.scaler = GradScaler('cuda', init_scale=2.**16, enabled=True)

        # Dataloader initialization
        if args.manifest:
            # Manifest-based loading
            self.unsupervised_loader, self.unsupervised_sampler = init_preprocessed_data_loader(
                processed_data_dir=args.processed_data_dir,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                manifest_path=args.manifest,
                split_name='train'
            )
            self.validation_loader, self.validation_sampler = init_preprocessed_data_loader(
                processed_data_dir=args.processed_data_dir, # Same directory
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                manifest_path=args.manifest,
                split_name='validation'
            )
        else:
            # Directory-based loading (backward compatibility)
            self.unsupervised_loader, self.unsupervised_sampler = init_preprocessed_data_loader(
                processed_data_dir=args.processed_data_dir,
                batch_size=self.batch_size,
                num_workers=self.num_workers
            )
            self.validation_loader = None
            self.validation_sampler = None
            if args.validation_data_dir:
                self.validation_loader, self.validation_sampler = init_preprocessed_data_loader(
                    processed_data_dir=args.validation_data_dir,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers
                )
        
        self.ipe = len(self.unsupervised_loader)

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

        # If a checkpoint was provided via args, try to resume from it.
        if getattr(args, "load_checkpoint", None):
            try:
                self._load_checkpoint(args.load_checkpoint)
            except Exception as e:
                logger.error(f"❌ Error loading checkpoint {args.load_checkpoint}: {e}")
                logger.error("Terminating training due to checkpoint loading failure.")
                exit(1)
        else:
            run_dir_name = f"{time.strftime('%Y%m%d_%H%M%S')}_{self.model_type}"
            self.run_dir = Path("resource/checkpoints") / run_dir_name
            self.run_dir.mkdir(parents=True, exist_ok=True)

        self.validators = []
        if dist.get_rank() == 0 and args.validators:
            for validator_name in args.validators:
                validator_args = {
                    "frequency": args.validator_frequency
                }
                self.validators.append(VALIDATOR_REGISTRY[validator_name](**validator_args))

        self.validation_cycle_count = 0

        if dist.get_rank() == 0:
            # Initialize Weights & Biases
            self.run = wandb.init(
                project=f"vjepa2-ac",
                name=f"run_{self.run_dir.name}",
                config=vars(self.args),
            )
            wandb.watch(self.predictor.module, log="gradients", log_freq=50)

        signal.signal(signal.SIGINT, lambda signum, frame: self._save_interrupted_checkpoint())

    def _save_checkpoint(self, tag, step, val_loss=None):
        if dist.get_rank() != 0:
            return

        # Save predictor + optimizer + scheduler states when available
        config = self.predictor.module.get_config()
        config["model_type"] = self.model_type

        ckpt = {
            "global_step": step,
            "predictor": self.predictor.module.state_dict(),
            "predictor_config": config,
        }
        if hasattr(self, "scaler") and self.scaler is not None:
            ckpt["scaler"] = self.scaler.state_dict()
        try:
            if hasattr(self, "optimizer") and self.optimizer is not None:
                ckpt["opt"] = self.optimizer.state_dict()
        except Exception as e:
            logger.debug(f"Could not include optimizer state in checkpoint: {e}")

        if hasattr(self, "scheduler") and hasattr(self.scheduler, "state_dict"):
            try:
                ckpt["sched"] = self.scheduler.state_dict()
            except Exception as e:
                logger.debug(f"Could not include scheduler state in checkpoint: {e}")
        if hasattr(self, "wd_scheduler") and hasattr(self.wd_scheduler, "state_dict"):
            try:
                ckpt["wd_sched"] = self.wd_scheduler.state_dict()
            except Exception as e:
                logger.debug(f"Could not include wd_scheduler state in checkpoint: {e}")

        if val_loss is not None:
            ckpt["val_loss"] = val_loss

        path = self.run_dir / f"vjepa_{self.model_type}_{tag}.pt"
        temp_path = self.run_dir / f"vjepa_{self.model_type}_{tag}.pt.tmp"
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

    def _load_checkpoint(self, checkpoint_path: str):
        """
        Load a checkpoint to resume training.

        This function uses the canonical `load_model` utility to restore the
        model's architecture and weights, and then loads the optimizer,
        schedulers, and other training-specific states.
        """
        # --- 1. Load Model using Canonical Loader ---
        # The canonical loader handles instantiation and weight loading.
        # We set the model to train() mode after loading, as the loader sets it to eval().
        model = load_model(
            model_path=checkpoint_path,
            device=self.device,
            model_registry=MODEL_REGISTRY,
            model_type=self.model_type,
            prepare_for_inference=False
        ).train()

        # Wrap with DDP
        if self.args.device_type == "cuda":
            self.predictor = DDP(model, device_ids=[self.args.local_rank])
        else:
            self.predictor = DDP(model)

        # --- 2. Load Training-Specific States (Optimizer, Schedulers, etc.) ---
        ckpt = torch.load(checkpoint_path, map_location=self.device)

        # Re-initialize optimizers with the new model's parameters
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

        if "opt" in ckpt and hasattr(self, "optimizer"):
            try:
                self.optimizer.load_state_dict(ckpt["opt"])
            except Exception as e:
                logger.error(f"Error loading optimizer state: {e}")

        if "sched" in ckpt and hasattr(self, "scheduler") and hasattr(self.scheduler, "load_state_dict"):
            try:
                self.scheduler.load_state_dict(ckpt["sched"])
            except Exception as e:
                logger.error(f"Error loading scheduler state: {e}")
        if "wd_sched" in ckpt and hasattr(self, "wd_scheduler") and hasattr(self.wd_scheduler, "load_state_dict"):
            try:
                self.wd_scheduler.load_state_dict(ckpt["wd_sched"])
            except Exception as e:
                logger.error(f"Error loading wd_scheduler state: {e}")

        # Restore bookkeeping fields if present
        if "global_step" in ckpt:
            self.global_step = ckpt["global_step"]
        if "val_loss" in ckpt:
            self.best_val_loss = ckpt["val_loss"]
        if "epoch" in ckpt:
            try:
                self.start_epoch = int(ckpt["epoch"]) + 1
            except Exception:
                pass
        if "scaler" in ckpt and hasattr(self, "scaler"):
            try:
                self.scaler.load_state_dict(ckpt["scaler"])
            except Exception as e:
                logger.error(f"Error loading scaler state: {e}")

        self.run_dir = Path(checkpoint_path).parent

        logger.info(f"Loaded checkpoint from {checkpoint_path} (global_step={self.global_step}, start_epoch={self.start_epoch}, best_val_loss={self.best_val_loss})")

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
                if self.save_every_iters > 0 and self.global_step % self.save_every_iters == 0:
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
                    
                    # --- Run Validators ---
                    self.validation_cycle_count += 1
                    if dist.get_rank() == 0 and self.validators:
                        logger.info("Running custom validators...")
                        for validator in self.validators:
                            if validator.should_run(self.validation_cycle_count):
                                try:
                                    validator.run(
                                        model=self.predictor.module, # unwrap DDP
                                        validation_loader=self.validation_loader,
                                        device=self.device,
                                        global_step=self.global_step
                                    )
                                except Exception as e:
                                    logger.error(f"Validator {type(validator).__name__} failed: {e}")

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

        with autocast(device_type='cuda'):
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
        
        with autocast(device_type='cuda'):
            z_tf, z_ar_final = self.forward_predictions(z_all, actions)
            jloss = self.loss_fn(z_tf, h_all[:, 1:])
            sloss = self.loss_fn(z_ar_final.unsqueeze(1), h_all[:, self.rollout_horizon].unsqueeze(1))
            loss = jloss + sloss
        
        # logger.info(f"Loss value before checking isfinite: {loss.item()}")
        if torch.isfinite(loss):
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            logger.info(f"Skipping step due to non-finite loss: {loss.item()}")

        return float(loss), float(jloss), float(sloss), lr, wd

    def load_trajectory(self, sample):
        embeddings, actions = sample
        
        # Validate sequence length for rollout horizon
        seq_len = embeddings.size(1)
        if seq_len < self.rollout_horizon + 1:
            raise ValueError(f"Sequence length {seq_len} is too short for rollout_horizon {self.rollout_horizon}. Need at least {self.rollout_horizon + 1}")
        
        return embeddings.to(self.device, non_blocking=True), actions.to(self.device, dtype=torch.float, non_blocking=True)

    def forward_predictions(self, z_all, actions):
        if hasattr(self.predictor.module, 'actions_formatter'):
            formatted_actions = self.predictor.module.actions_formatter(actions)
        else:
            B, T_seq, _, _ = actions.shape
            formatted_actions = actions.view(B, T_seq, -1)

        z_tf = self.predictor(z_all[:, :-1], formatted_actions)

        z_rollout = z_all[:, 0].unsqueeze(1)
        for i in range(self.rollout_horizon):
            a = formatted_actions[:, i].unsqueeze(1)
            z_rollout = self.predictor(z_rollout, a)

        return z_tf, z_rollout.squeeze(1)

    def loss_fn(self, z_pred, h_target):
        return torch.mean(torch.abs(z_pred - h_target) ** self.loss_exp) / self.loss_exp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Trainer for V-JEPA Predictor Models")

    # Core arguments
    parser.add_argument("--model_type", type=str, required=True, choices=MODEL_REGISTRY.keys(), help="Type of model to train")
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--processed_data_dir", type=str, required=True)
    parser.add_argument("--validation_data_dir", type=str, required=False)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--save_every_iters", type=int, default=500)
    parser.add_argument("--save_every_epochs", type=int, default=0) # Disabled by default
    parser.add_argument("--load_checkpoint", type=str, default=None)
    
    # Validator arguments
    parser.add_argument("--validators", nargs='*', default=[], choices=VALIDATOR_REGISTRY.keys(),
                        help="List of validators to run during validation.")
    parser.add_argument("--validator_frequency", type=int, default=1,
                        help="Run validators every N validation cycles.")

    # DDP arguments
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--dist-backend", default=None, help="nccl|gloo")
    parser.add_argument("--device", default=None, help="cuda|cpu")

    # Training parameters
    parser.add_argument("--rollout_horizon", type=int, default=ROLLOUT_HORIZON)

    # --- Model-specific arguments ---
    # Transformer block arguments (used by all models)
    parser.add_argument("--depth", type=int, default=12, help="Depth of self-attention predictor model")
    parser.add_argument("--num_heads", type=int, default=16, help="Number of multi attention heads")
    
    # Cross-Attention specific
    parser.add_argument("--num_action_tokens", type=int, default=8, help="Number of action tokens for cross-attention")
    parser.add_argument("--cross_attn_depth", type=int, default=3, help="Number of cross-attention blocks")

    # FiLM specific
    parser.add_argument("--film_d_a", type=int, default=32)
    parser.add_argument("--film_d_c", type=int, default=128)
    parser.add_argument("--film_layers_modulated", type=int, default=6)

    # Action-Encoded specific
    parser.add_argument("--frozen_action_encoder_checkpoint", type=str, default=None, help="Path to frozen CNN action encoder checkpoint for encode_a model")

    args = parser.parse_args()

    # Validate arguments
    if args.model_type == "encode_a" and not args.frozen_action_encoder_checkpoint:
        parser.error("--frozen_action_encoder_checkpoint is required when model_type is 'encode_a'")

    # --- DDP Initialization ---
    use_cuda = torch.cuda.is_available()
    backend = args.dist_backend or ("nccl" if use_cuda else "gloo")
    device = args.device or ("cuda" if use_cuda else "cpu")
    
    print(f"[DDP] Initializing with backend={backend}, device={device}")
    
    dist.init_process_group(backend=backend, init_method="env://", timeout=datetime.timedelta(seconds=120))
    local_rank = int(os.environ["LOCAL_RANK"])
    
    if device == "cuda":
        torch.cuda.set_device(local_rank)
    
    args.local_rank = local_rank
    args.device_type = device
    
    print(f"[DDP] Rank {dist.get_rank()}/{dist.get_world_size()}, local_rank={local_rank}, device={device}")

    # --- Model Instantiation ---
    model_class = MODEL_REGISTRY[args.model_type]
    
    # Filter args to only pass relevant ones to the model constructor
    model_arg_names = model_class.__init__.__code__.co_varnames[1:] # Exclude 'self'
    model_args = {k: v for k, v in vars(args).items() if k in model_arg_names}
    
    if dist.get_rank() == 0:
        print(f"Instantiating model '{args.model_type}' with args: {model_args}")

    model = model_class(**model_args)
    
    # --- Trainer Initialization and Execution ---
    trainer = VJEPATrainer(args, model)
    trainer.training_loop()

    # --- Cleanup ---
    dist.destroy_process_group()