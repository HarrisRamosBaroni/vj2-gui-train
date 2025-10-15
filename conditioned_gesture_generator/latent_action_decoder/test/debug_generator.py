import torch
from conditioned_gesture_generator.latent_action_decoder.dtw_cvae_decoder import GestureGenerator

def run_debug():
    print("--- Running GestureGenerator Debug Script ---")

    # --- 1. Configuration ---
    # These values are taken from the training script defaults
    z_dim = 128
    s_dim = 1024
    style_dim = 64
    T_steps = 250
    K_ctrl_pts = 32
    nhead = 8
    num_decoder_layers = 6
    num_s_tokens = 256
    batch_size = 4

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 2. Instantiate Model ---
    print("Instantiating GestureGenerator...")
    generator = GestureGenerator(
        z_dim=z_dim,
        s_dim=s_dim,
        style_dim=style_dim,
        T_steps=T_steps,
        K_ctrl_pts=K_ctrl_pts,
        nhead=nhead,
        num_decoder_layers=num_decoder_layers,
        num_s_tokens=num_s_tokens
    ).to(device)
    generator.eval() # Set to evaluation mode

    print("Model Initialized. Bias of output_head:")
    print(generator.output_head.bias.data)

    # --- 3. Create Dummy Inputs ---
    print("\nCreating dummy non-zero inputs...")
    z = torch.randn(batch_size, z_dim, device=device)
    s = torch.randn(batch_size, num_s_tokens, s_dim, device=device)
    z_style = torch.randn(batch_size, style_dim, device=device)

    print(f"Input shapes: z={z.shape}, s={s.shape}, z_style={z_style.shape}")

    # --- 4. Forward Pass ---
    print("\nPerforming forward pass...")
    with torch.no_grad():
        # We manually step through the forward pass to inspect intermediate values.
        
        # a. Create context from z and s
        z_token = generator.z_proj(z).unsqueeze(1)
        s_tokens = generator.s_proj(s) + generator.s_pos_embed
        context = torch.cat([z_token, s_tokens], dim=1)
        
        # b. Create initial control point sequence from style
        ctrl_pts_input = generator.style_proj(z_style).unsqueeze(1).repeat(1, generator.K_ctrl_pts, 1)
        ctrl_pts_input += generator.ctrl_pt_pos_emb
        
        # c. Run through the transformer decoder
        decoded_ctrl_pts_features = generator.transformer_decoder(ctrl_pts_input, memory=context)
        
        # d. Get logits from the final output head
        output_ctrl_pts_logits = generator.output_head(decoded_ctrl_pts_features)
        
        # e. Apply activation functions (clamp/sigmoid)
        clamped_coords = torch.clamp(output_ctrl_pts_logits[..., :2], 0.0, 1.0)
        touch_sigmoid = torch.sigmoid(output_ctrl_pts_logits[..., 2]).unsqueeze(-1)
        final_ctrl_pts = torch.cat([clamped_coords, touch_sigmoid], dim=-1)

        # f. Interpolate to get final trajectory
        trajectory = generator.interpolate_to_trajectory(final_ctrl_pts)

    # --- 5. Analyze Outputs ---
    print("\n--- Analysis of Outputs ---")

    # Analyze control point logits
    x_logits = output_ctrl_pts_logits[..., 0]
    y_logits = output_ctrl_pts_logits[..., 1]
    print("\nControl Point LOGITS (from output_head):")
    print(f"  x_logits | min: {x_logits.min():.4f}, max: {x_logits.max():.4f}, mean: {x_logits.mean():.4f}")
    print(f"  y_logits | min: {y_logits.min():.4f}, max: {y_logits.max():.4f}, mean: {y_logits.mean():.4f}")

    # Analyze final control points (after clamp/sigmoid)
    x_ctrl_pts = final_ctrl_pts[..., 0]
    y_ctrl_pts = final_ctrl_pts[..., 1]
    print("\nFinal Control Points (after clamp/sigmoid):")
    print(f"  x_coords | min: {x_ctrl_pts.min():.4f}, max: {x_ctrl_pts.max():.4f}, mean: {x_ctrl_pts.mean():.4f}")
    print(f"  y_coords | min: {y_ctrl_pts.min():.4f}, max: {y_ctrl_pts.max():.4f}, mean: {y_ctrl_pts.mean():.4f}")

    # Analyze final trajectory
    x_traj = trajectory[..., 0]
    y_traj = trajectory[..., 1]
    print("\nFinal Interpolated Trajectory:")
    print(f"  x_traj | min: {x_traj.min():.4f}, max: {x_traj.max():.4f}, mean: {x_traj.mean():.4f}")
    print(f"  y_traj | min: {y_traj.min():.4f}, max: {y_traj.max():.4f}, mean: {y_traj.mean():.4f}")

    print("\n--- Debug Script Finished ---")


if __name__ == "__main__":
    run_debug()
