import torch
import torch.nn as nn
import torch.nn.functional as F


from training.action_dataloader import PreprocessedGUIAgentDataset
from torch.utils.data import Dataset, DataLoader, random_split
import seaborn as sns

import torch
import torch.nn as nn
import os

# Config
SEQ_LEN = 250      # shortened from 900
FEAT_DIM = 3       # Input: [x, y, p] - Output will be 4D [x, y, p0, p1]
MODEL_DIM = 32     # transformer hidden size
LATENT_DIM = 3     # compressed vector
BATCH_SIZE = 160


CKPT_DIR = "./checkpoints_autoenc"
RUN_NAME = "latent-sphere-run"      # keep in sync with wandb name if you want
RESUME = False                       # try to resume if a checkpoint exists
SAVE_EVERY_EPOCH = True             # also save "last" each epoch
BEST_FILENAME = "best.pt"
LAST_FILENAME = "last.pt"

os.makedirs(CKPT_DIR, exist_ok=True)

def ckpt_path(name):
    return os.path.join(CKPT_DIR, name)

def save_checkpoint(path, model, optimizer, epoch, best_val_loss):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "config": {
            "SEQ_LEN": SEQ_LEN, "FEAT_DIM": FEAT_DIM, "MODEL_DIM": MODEL_DIM,
            "LATENT_DIM": LATENT_DIM, "BATCH_SIZE": BATCH_SIZE, "RUN_NAME": RUN_NAME
        }
    }, path)

def load_checkpoint(path, model, optimizer, map_location=None):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    start_epoch = int(ckpt.get("epoch", -1)) + 1
    best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
    return start_epoch, best_val_loss




class TinyTransformerAutoencoder(nn.Module):
    def __init__(self, FEAT_DIM=3, MODEL_DIM=MODEL_DIM, SEQ_LEN=SEQ_LEN, LATENT_DIM=LATENT_DIM, ema_alpha=0.0, ema_window_pct=0.1):  # Input is still 3D
        super().__init__()
        
        # Store EMA parameters
        self.ema_alpha = ema_alpha  # EMA smoothing factor (0.0 = no smoothing)
        self.ema_window_pct = ema_window_pct  # EMA window size as percentage of sequence length

        self.input_proj = nn.Linear(3, MODEL_DIM)  # Input: [x, y, p] - 3 dimensions
        self.pos_embed = nn.Parameter(torch.randn(1, SEQ_LEN, MODEL_DIM))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=MODEL_DIM, nhead=2, dropout=0.5, batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.cls_token = nn.Parameter(torch.randn(1, 1, MODEL_DIM))
        self.to_latent = nn.Linear(MODEL_DIM, LATENT_DIM)

        self.latent_to_model = nn.Linear(LATENT_DIM, MODEL_DIM)
        self.decoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.output_proj = nn.Linear(MODEL_DIM, 4)  # [x, y, p0, p1]

    def forward(self, x):
        B = x.shape[0]

        # Project input & add positional embedding
        x = self.input_proj(x) + self.pos_embed  # [B, SEQ_LEN, MODEL_DIM]

        # Add [CLS] token to summarize
        cls = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls, x], dim=1)

        # Encode
        encoded = self.encoder(x)
        latent = self.to_latent(encoded[:, 0, :])  # Grab [CLS]

        # Decode
        decoded_seed = self.latent_to_model(latent).unsqueeze(1).repeat(1, SEQ_LEN, 1)
        decoded = self.decoder(decoded_seed + self.pos_embed)
        output = self.output_proj(decoded)
        
        # Apply EMA filtering to raw output (before probabilistic interpretation)
        output = ema_filter(output, self.ema_alpha, self.ema_window_pct)

        return latent, output


def ema_filter(action_sequence, alpha=0.0, window_size_pct=0.1):
    """
    Applies differentiable exponential moving average filtering to action sequences with windowed memory.
    
    Args:
        action_sequence: [B, SEQ_LEN, ACTION_DIM] tensor of action sequences
        alpha: float, smoothing factor (0.0 = no smoothing, 0 < alpha < 1). Higher alpha = more smoothing
        window_size_pct: float, window size as percentage of sequence length (default 0.1 = 10%)
        
    Returns:
        Filtered action sequence of same shape [B, SEQ_LEN, ACTION_DIM]
    """
    # If alpha is 0, no smoothing needed - return original sequence
    if alpha == 0.0:
        return action_sequence
    
    B, T, D = action_sequence.shape
    
    # Calculate window size (minimum 1, maximum T)
    window_size = max(1, min(T, int(T * window_size_pct)))
    
    # Initialize output tensor
    filtered = torch.zeros_like(action_sequence)
    
    # Initialize EMA state with first timestep
    ema_state = action_sequence[:, 0, :]  # [B, ACTION_DIM]
    filtered[:, 0, :] = ema_state
    
    # Apply EMA filter across time dimension with windowed reset
    for t in range(1, T):
        # Reset EMA state if we've exceeded the window size
        if t % window_size == 0:
            ema_state = action_sequence[:, t, :]
        else:
            # EMA update: state = alpha * state + (1 - alpha) * new_value
            ema_state = alpha * ema_state + (1 - alpha) * action_sequence[:, t, :]
        
        filtered[:, t, :] = ema_state
    
    return filtered

    
def sphere_loss(latent, target_radius=1.0):
    # Compute norm of each latent vector (batch-wise)
    lengths = torch.norm(latent, dim=1)  # [B]
    # Penalize how far from target_radius each vector is
    return torch.mean((lengths - target_radius) ** 2)


def cosine_repulsion_loss(latent):
    """
    Computes cosine repulsion loss to encourage diverse latent representations.
    Penalizes high cosine similarity between different samples in the batch.
    
    Args:
        latent: [B, LATENT_DIM] tensor of latent vectors
        
    Returns:
        Scalar loss value
    """
    # Normalize latent vectors to unit length for cosine similarity
    latent_norm = F.normalize(latent, p=2, dim=1)  # [B, LATENT_DIM]
    
    # Compute cosine similarity matrix: [B, B]
    cosine_sim_matrix = torch.mm(latent_norm, latent_norm.t())  # [B, B]
    
    # Remove diagonal (self-similarity = 1.0) by masking
    batch_size = latent.size(0)
    mask = ~torch.eye(batch_size, dtype=torch.bool, device=latent.device)
    off_diagonal_sims = cosine_sim_matrix[mask]
    
    # Take mean of absolute similarities to penalize both positive and negative correlations
    return torch.mean(torch.abs(off_diagonal_sims))


def probabilistic_loss_with_masking(reconstructed, original, latent, alpha=1.0, beta=0.0, coord_weight=1.0, class_weight=1.0):
    recon_xy_raw = reconstructed[:, :, :2]     # [B, 250, 2]
    recon_probs = reconstructed[:, :, 2:]      # [B, 250, 2]
    
    orig_xy = original[:, :, :2]               # [B, 250, 2]
    orig_p = original[:, :, 2]                 # [B, 250]

    # === Predict touch mask from logits ===
    touch_pred = torch.argmax(F.softmax(recon_probs, dim=-1), dim=-1)  # [B, 250]
    pred_touch_mask = touch_pred.unsqueeze(-1).float()                 # [B, 250, 1]

    # === Mask coords using prediction ===
    recon_xy_masked = recon_xy_raw * pred_touch_mask
    orig_xy_masked = orig_xy

    # === Coord loss: L1 loss only on predicted touch timesteps
    coord_loss = F.l1_loss(recon_xy_masked, orig_xy_masked)

    # === Class loss: predicted logits vs GT p
    class_loss = F.cross_entropy(recon_probs.view(-1, 2), orig_p.view(-1).long())

    # === Regularization losses
    reg_loss = sphere_loss(latent)
    repulsion_loss = cosine_repulsion_loss(latent)

    # === Combine
    total_loss_val = coord_weight * coord_loss + class_weight * class_loss + alpha * reg_loss + beta * repulsion_loss
    return total_loss_val, coord_loss, class_loss, reg_loss, repulsion_loss

def total_loss(reconstructed, original, latent, alpha=1.0, beta=0.0, touch_weight=1.0):
    """Legacy function - redirects to probabilistic loss for backward compatibility"""
    total_loss_val, coord_loss, class_loss, reg_loss, repulsion_loss = probabilistic_loss_with_masking(
        reconstructed, original, latent, alpha=alpha, beta=beta,
        coord_weight=1.0, class_weight=touch_weight
    )
    return total_loss_val, coord_loss + class_loss, reg_loss, repulsion_loss




def chunk_windows(actions: torch.Tensor, win=250, step=250):
    """
    actions: [L, 3] tensor
    returns: [W, win, 3] windows (no overlap if step==win)
    """
    L, C = actions.shape
    if L < win:
        return torch.empty(0, win, C)  # or pad if you prefer
    starts = torch.arange(0, L - win + 1, step)
    return torch.stack([actions[s:s+win] for s in starts], dim=0)  # [W, 250, 3]


def evaluate_validation_loss(model, val_loader, device='cuda', alpha=0.1):
    model.eval()
    total_val_loss = 0.0
    num_batches = 0
    latents = []

    with torch.no_grad():
        for x_batch in val_loader: 
            x_batch = x_batch.to(device)
            B, T, L, C = x_batch.shape
            x_batch = x_batch.view(-1, L, C) 

            for i in range(0, x_batch.size(0), BATCH_SIZE):
                x_input = x_batch[i:i+BATCH_SIZE]
                x_input = torch.stack([normalize_zscore(x) for x in x_input], dim=0)
                latent, recon = model(x_input)
                loss, _, _, _, _ = probabilistic_loss_with_masking(recon, x_input, latent, alpha=alpha, beta=0.0)
                total_val_loss += loss.item()
                
                num_batches += 1
                latents.append(latent.cpu())
                
    latents_concat = torch.cat(latents, dim=0)  
    print(latents_concat[:5])
    print("Norms:", torch.norm(latents_concat, dim=1)[:5])
         
    print(latents_concat.shape)
    plot_latents_on_sphere(latents_concat)
    

    return total_val_loss / num_batches if num_batches > 0 else float('inf')

def collect_test_latents(model, test_loader, device='cuda'):
    model.eval()
    latents = []

    # with torch.no_grad():
    #     for (emb, x_batch) in test_loader:
    #         x_batch = x_batch.to(device)
    #         B, T, L, C = x_batch.shape
    #         x_batch = x_batch.view(-1, L, C)  # [B*T, 250, 3]
            
            

            # x_inp_hmp = x_batch[0:1]
            # x_input = x_batch[i:i+4]
            # x_input = torch.stack([normalize_zscore(x) for x in x_input], dim=0)
            # x_inp_hmp = torch.stack([normalize_zscore(x) for x in x_inp_hmp], dim=0)
            # latent, recon = model(x_input)
            # latent_hmap, recon_hmap = model(x_inp_hmp)
            
            # recon_hmap = recon_hmap.squeeze(0)  # [250, 3]
            # x_inp_hmp = x_inp_hmp.squeeze(0)  # [250, 3]

            # # Plot heatmaps or overlay
            # # plot_input_vs_reconstruction(x_inp_hmp, recon_hmap)
            # latents.append(latent.cpu())
            
    # return torch.cat(latents, dim=0)  # [N, 3]
    with torch.no_grad():
        for x_batch in test_loader:
            x_batch = x_batch.to(device)
            B, T, L, C = x_batch.shape
            x_batch = x_batch.view(-1, L, C)  # [B*T, 250, 3]

            for i in range(0, x_batch.size(0), BATCH_SIZE):
                x_input = x_batch[i:i+BATCH_SIZE]
                x_input = torch.stack([normalize_zscore(x) for x in x_input], dim=0)
                latent, _ = model(x_input)
                latents.append(latent.cpu())

    return torch.cat(latents, dim=0)  # [N, 3] ✅ multiple latents!


def normalize_zscore(seq):
    x = seq[:, 0]
    y = seq[:, 1]
    a = seq[:, 2]

    x = (x - x.mean()) / (x.std() + 1e-8)
    y = (y - y.mean()) / (y.std() + 1e-8)

    return torch.stack([x, y, a], dim=1)



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_latents_on_sphere(latents):
    latents = latents.detach()  # ✅ detach from autograd
    # latents = latents / latents.norm(dim=1, keepdim=True)

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')

    x, y, z = latents[:,0], latents[:,1], latents[:,2]
    x = x.numpy()  # ✅ now safe
    y = y.numpy()
    z = z.numpy()
    
    ax.scatter(x, y, z, c='blue', alpha=0.6, s=15)

    # Plot unit sphere wireframe
    u, v = torch.linspace(0, 2 * torch.pi, 100), torch.linspace(0, torch.pi, 100)
    x_sphere = torch.outer(torch.cos(u), torch.sin(v))
    y_sphere = torch.outer(torch.sin(u), torch.sin(v))
    z_sphere = torch.outer(torch.ones_like(u), torch.cos(v))
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.2, linewidth=0.3)

    ax.set_title("Latent Vectors on Unit Sphere")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    plt.savefig("latents_sphere.png")
    wandb.log({"latents_sphere": wandb.Image("latents_sphere.png")})
    plt.show()



def interpret_probabilistic_output(output):
    """
    Interprets probabilistic model output and applies masking.
    
    Args:
        output: [B, 250, 4] = [x, y, p0, p1] raw decoder output
        
    Returns:
        xy_masked: [B, 250, 2] coordinates masked by touch prediction
        touch_probs: [B, 250, 2] softmax probabilities [p0, p1]
        touch_pred: [B, 250] predicted touch state (0 or 1)
    """
    xy_raw = output[:, :, :2]              # [B, 250, 2]
    probs_logits = output[:, :, 2:]        # [B, 250, 2]
    
    # Get probabilities and predictions
    touch_probs = F.softmax(probs_logits, dim=-1)  # [B, 250, 2]
    touch_pred = torch.argmax(touch_probs, dim=-1)  # [B, 250]
    
    # Apply masking: zero out coordinates when predicted touch = 0
    touch_mask = touch_pred.unsqueeze(-1).float()  # [B, 250, 1]
    xy_masked = xy_raw * touch_mask
    
    return xy_masked, touch_probs, touch_pred


def convert_to_legacy_format(xy_masked, touch_pred):
    """
    Converts new probabilistic output back to legacy [x, y, p] format.
    
    Args:
        xy_masked: [B, 250, 2] masked coordinates
        touch_pred: [B, 250] predicted touch state
        
    Returns:
        legacy_output: [B, 250, 3] in [x, y, p] format
    """
    touch_expanded = touch_pred.unsqueeze(-1).float()  # [B, 250, 1]
    return torch.cat([xy_masked, touch_expanded], dim=-1)


def plot_input_vs_reconstruction(x_input, recon_output):
    """
    x_input: [250, 3] original input
    recon_output: [250, 3] reconstructed output
    """
    x_input = x_input.detach().cpu().numpy()
    recon_output = recon_output.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(x_input.T, ax=axes[0], cmap="YlGnBu", cbar=False)
    axes[0].set_title("Original Input (x, y, actionType)")
    axes[0].set_ylabel("Channels")
    axes[0].set_xlabel("Timestep")

    sns.heatmap(recon_output.T, ax=axes[1], cmap="YlGnBu", cbar=False)
    axes[1].set_title("Reconstructed Output")
    axes[1].set_ylabel("Channels")
    axes[1].set_xlabel("Timestep")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("running")
    import wandb
    from torch.utils.data import Subset, DataLoader
    wandb.init(project="vjepa-autoencoder", name="latent-sphere-run")   
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model = TinyTransformerAutoencoder().to(device)
   # dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, FEAT_DIM)
   # latent, recon = model(dummy_input)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    best_val_loss = float("inf")
    start_epoch = 0
    last_ckpt = ckpt_path(LAST_FILENAME)
    if RESUME and os.path.isfile(last_ckpt):
        try:
            start_epoch, best_val_loss = load_checkpoint(last_ckpt, model, optimizer, map_location=device)
            print(f"[Resume] Loaded '{last_ckpt}' (start at epoch {start_epoch}, best_val_loss={best_val_loss:.6f})")
        except Exception as e:
            print(f"[Resume] Failed to load '{last_ckpt}': {e}")
    
    processed_dir = "./processed_actions/final_trajectories"

    
    dataset = PreprocessedGUIAgentDataset(
        processed_data_dir=processed_dir
    )
    
    # import os
    # import numpy as np
    # from tqdm import tqdm
    # import random
    # save_dir = "./processed_data_half_noops"
    # os.makedirs(save_dir, exist_ok=True)
    # valid_idxs = []
    # for i in range(len(dataset)-1):
    #     emb, seq = dataset[i]
    #     flat = seq.reshape(-1, 3) if seq.ndim == 3 else seq
        
    #     if (flat.abs().sum() == 0):  # all zeros → drop 50% of time
    #         if random.random() < 0.5:  
    #             continue  # skip this file entirely
        
    #     # keep otherwise
    #     valid_idxs.append(i)
    #     print("running")
    #     npz_path = dataset.file_indices[i]['path']  # your dataset must store file paths
    #     data = np.load(npz_path)
    #     filename = os.path.basename(npz_path)
    #     np.savez_compressed(
    #         os.path.join(save_dir, filename),
    #         **{k: data[k] for k in data}
    #     )

    
    total_size = len(dataset)

    train_size = int(0.7 * total_size)
    val_size   = int(0.15 * total_size)
    test_size  = total_size - train_size - val_size  # ensures total = 100%

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True, 
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True, 
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True, 
        num_workers=0,
        pin_memory=True,
    )

    window_size = 250
    loss = 0
    latents = []
    for epoch in range(5):
        model.train()
        for itr, (x_batch) in enumerate(train_loader): 
            all_windows = []
            B, T, L, C = x_batch.shape
            x_batch = x_batch.view(-1, L, C)
            for i in range(0, x_batch.size(0), BATCH_SIZE):

                x_input = x_batch[i:i+BATCH_SIZE].to(device)   # [B, 250, 3]
                print(x_input.shape)
                #x_input = torch.stack([normalize_zscore(x) for x in x_input], dim=0)
                optimizer.zero_grad()
                latent, recon = model(x_input)
                # latents.append(latent.cpu())
                # latents_plot = torch.cat(latents, dim=0) 
                # print(latents_plot.shape)
                
                loss, coord_loss, class_loss, reg_loss, repulsion_loss = probabilistic_loss_with_masking(recon, x_input, latent, alpha=0.5, beta=0.0)
                wandb.log({
                    "recon_loss": coord_loss.item(),
                    "class_loss": class_loss.item(), 
                    "reg_loss": reg_loss.item(),
                    "repulsion_loss": repulsion_loss.item()
                })
                wandb.log({"train_loss": loss.item()})

                print(f"Batch {itr}  Loss: {loss.item():.4f}")
                if i % 100:
                    print("we're doing validation")
                    val_loss = evaluate_validation_loss(model, val_loader, device=device, alpha=0.5)
                    wandb.log({"val_loss": val_loss})
                    
                loss.backward()
                optimizer.step()
            # plot_latents_on_sphere(latents_plot)
        
        if SAVE_EVERY_EPOCH:
            save_checkpoint(ckpt_path(LAST_FILENAME), model, optimizer, epoch, best_val_loss)

        # === NEW: save "best" on improvement ===
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(ckpt_path(BEST_FILENAME), model, optimizer, epoch, best_val_loss)
            print(f"[Checkpoint] New best ({best_val_loss:.6f}) saved to {ckpt_path(BEST_FILENAME)}")
        all_latents = collect_test_latents(model, test_loader, device)
        plot_latents_on_sphere(all_latents)
        print(f"Epoch {epoch} | Validation Loss: {val_loss:.4f}")
        print(f"Epoch {epoch} Loss: {loss.item():.4f}")

    all_latents = collect_test_latents(model, test_loader, device)
    plot_latents_on_sphere(all_latents)
    wandb.finish()

