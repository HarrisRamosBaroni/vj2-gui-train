To switch this codebase from diffusion (ε-prediction + DDIM) to **rectified flow (velocity prediction + ODE integration)** you only need surgical edits in two places:

* the **model API/targets** (no architecture change), and
* the **trainer/sampler & args**.

Below is a clean checklist with drop-in snippets.

---

# A) Changes in your **model code** (diffusion\_gesture\_model.py)

1. **Interpret outputs as velocity, not noise**
   No structural change. Just treat:

* `model_output['coordinates']` **as velocity** $u_\theta(x_t,t,c)$ instead of noise.
* Rename helper for clarity (optional):

  * `compute_coordinate_loss(...)` → `compute_velocity_loss(...)` (same MSE).

```python
# keep signature; just rename for readability
def compute_velocity_loss(self, model_output: Dict[str, torch.Tensor],
                          target_velocity: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(model_output['coordinates'], target_velocity)
```

(You can keep the old function name if you prefer—just pass velocity targets.)

2. **Allow continuous t ∈ \[0,1]**
   You already can. Ensure every call to `time_embed(t)` passes **float32 in \[0,1]** (not long indices). No code change needed in `SinusoidalPositionalEmbedding` or `ResidualBlock`; just feed float `t`.

3. **Keep FiLM, CFG, pen head unchanged**

* FiLM generation and CFG logic is identical; you’ll apply CFG to **velocity** now.
* `forward_pen_prediction` stays as the post-integration clean pass.

4. **Remove DDIM Sampler & schedule use**

* Delete/stop using `DDIMSampler` and `create_diffusion_schedule` from the model file.
* Add a **FlowSampler** (Euler/Heun ODE). See section C below.

5. **No change** to `GestureDiffusionModelLAM` beyond (1) target=velocity and (2) sampler swap, (3) time dtype.

---

# B) Changes in your **training code** (trainer)

### Replace diffusion noising with rectified-flow batch sampling

6. **Add an RF batch constructor** (replaces `add_noise` during training):

```python
def sample_rectified_flow_batch(x1: torch.Tensor,
                                noisy: bool = False,
                                sigma: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    x1: [B, L, 3] absolute coords+pen (use coords for flow; pen kept for BCE)
    Returns:
      x_t_in: [B, L, 3]  -> coords at time t with zero pen channel
      t:     [B]         -> float32 in [0,1]
      v_tar: [B, L, 2]   -> target velocity field
    """
    device = x1.device
    B, L, _ = x1.shape
    x1c = x1[..., :2]
    x0 = torch.randn_like(x1c)                      # base Gaussian
    t = torch.rand(B, device=device)                # uniform [0,1]
    t_ = t[:, None, None]                           # [B,1,1]

    if not noisy:
        x_t = (1 - t_) * x0 + t_ * x1c
        v_tar = (x1c - x0)                          # constant in t
    else:
        eps = torch.randn_like(x1c)
        g = sigma * t_ * (1 - t_)                   # smoothing
        gp = sigma * (1 - 2 * t_)
        x_t = (1 - t_) * x0 + t_ * x1c + g * eps
        v_tar = (x1c - x0) + gp * eps

    x_t_in = torch.cat([x_t, torch.zeros(B, L, 1, device=device)], dim=-1)
    return x_t_in, t, v_tar
```

7. **Rewrite `train_step` to use RF targets**
   Replace your “PASS 1” block:

```python
# --- PASS 1: Rectified Flow training (velocity prediction) ---
x_t_in, t, v_target = sample_rectified_flow_batch(gesture_sequences,
                                                  noisy=False,  # or True with sigma
                                                  sigma=getattr(self.args, 'rf_sigma', 0.1))

# Note: latent_frames CFG-dropout stays as you had in prepare_batch()
out = self.model(x_t_in, t, latent_frames, sequence_length)  # out['coordinates'] ≡ velocity
coord_loss = self.model.compute_velocity_loss(out, v_target)
```

Delete the old `add_noise(...)` call and all references to `coordinate_noise`.

8. **Keep the pen pass unchanged**
   Still do “PASS 2” using clean coordinates (exactly your current code). `t` can be zeros or the same float `t`; both are fine since pen uses a separate head after generation too.

9. **Validation**
   Mirror the training change: use `sample_rectified_flow_batch` to get `x_t_in, t, v_target`, run the forward, compute the velocity MSE. Pen validation unchanged.

10. **Reconstruction loss block**

* You can keep it, but it must call the **flow sampler** (not DDIM). See (14).

### Args & schedule cleanup

11. **Remove diffusion schedule args**
    Delete/ignore:

* `--num_timesteps`, `--beta_start`, `--beta_end` and any call to `create_diffusion_schedule`.
* The `self.diffusion_schedule` object and any dependent helpers (`create_sparse_inference_schedule` unless you want for viz labels).

Add RF-specific args:

```python
parser.add_argument("--rf_noisy", action="store_true",
                    help="Use noisy rectified flow g(t) smoothing")
parser.add_argument("--rf_sigma", type=float, default=0.1,
                    help="Noise scale for g(t)=sigma*t*(1-t) if --rf_noisy")
parser.add_argument("--flow_steps", type=int, default=20,
                    help="ODE steps for inference (Euler/Heun)")
parser.add_argument("--ode_method", type=str, default="heun", choices=["euler","heun"])
```

12. **Delete DDIMSampler**
    Remove its construction and usage:

```python
# self.sampler = DDIMSampler(...)
```

Replace with FlowSampler (below).

13. **Time dtype**
    Ensure `t` passed into the model is `float32` in `[0,1]` everywhere (training/validation/visualization).

### Generation & visualization

14. **Add a FlowSampler** (ODE). Use it wherever you generated sequences.

```python
class FlowSampler:
    def __init__(self, model, device="cuda", method="heun"):
        self.model = model
        self.device = device
        self.method = method

    @torch.no_grad()
    def sample(self, shape, latent_frames=None, sequence_length=None,
               steps=20, cfg_scale=1.0):
        B, L, C = shape
        device = self.device
        x = torch.randn(B, L, 2, device=device)                # base x(0) ~ N(0,I)
        dt = 1.0 / steps

        for k in range(steps):
            t0 = torch.full((B,), k/steps, device=device)
            xin = torch.cat([x, torch.zeros(B, L, 1, device=device)], dim=-1)

            if cfg_scale > 1.0 and latent_frames is not None:
                u_c = self.model(xin, t0, latent_frames, sequence_length)['coordinates']
                u_u = self.model(xin, t0, None, None)['coordinates']
                u0 = u_u + cfg_scale * (u_c - u_u)
            else:
                u0 = self.model(xin, t0, latent_frames, sequence_length)['coordinates']

            if self.method == "euler" or k == steps - 1:
                x = x + dt * u0
            else:
                # Heun (RK2)
                x_euler = x + dt * u0
                t1 = torch.full((B,), (k+1)/steps, device=device)
                xin1 = torch.cat([x_euler, torch.zeros(B, L, 1, device=device)], dim=-1)
                if cfg_scale > 1.0 and latent_frames is not None:
                    u_c1 = self.model(xin1, t1, latent_frames, sequence_length)['coordinates']
                    u_u1 = self.model(xin1, t1, None, None)['coordinates']
                    u1 = u_u1 + cfg_scale * (u_c1 - u_u1)
                else:
                    u1 = self.model(xin1, t1, latent_frames, sequence_length)['coordinates']
                x = x + 0.5 * dt * (u0 + u1)

        # pen post-pass (unchanged)
        x3 = torch.cat([x, torch.zeros(B, L, 1, device=device)], dim=-1)
        dummy_t = torch.zeros(B, device=device)
        pen_logits = self.model.forward_pen_prediction(x3, dummy_t, latent_frames, sequence_length)
        pen = (torch.sigmoid(pen_logits) > 0.5).float()

        return torch.cat([x, pen], dim=-1)  # [B,L,3]
```

Construct it in `__init__`:

```python
self.sampler = FlowSampler(self.model, device=self.device, method=self.args.ode_method)
```

15. **Swap all DDIM sampling calls**

* In `compute_reconstruction_loss(...)`:

```python
generated_sequences = self.sampler.sample(
    shape=(B, L, 3),
    latent_frames=latent_frames,
    sequence_length=sequence_length,
    steps=self.args.flow_steps,
    cfg_scale=self.args.cfg_scale
)
```

* In `generate_validation_samples(...)`: same replacement (use `steps=50` if you like symmetry with your old 50-step DDIM).

16. **Sparse-steps visualization helper**

* Replace `create_sparse_inference_schedule` with a simple list of `t` values: e.g., `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`.
* If you want to show intermediate states, run the FlowSampler loop manually (copy its body and store `x` after selected k’s). Your current plotting code remains valid—only the source of intermediate `x_t` changes.

---

# C) Small correctness/stability tips

* **Normalization**: Make sure your dataset normalization (mean/std) is applied consistently to both $x_1$ and the base $x_0 \sim \mathcal N(0,I)$. If your data is in $[0,1]$, consider standardizing so base and data scales match; otherwise the velocity magnitudes may be skewed.
* **RF variant**: Start with **deterministic RF** (noisy=False). If training is jittery or overfits, turn on `--rf_noisy --rf_sigma 0.1`.
* **CFG**: Begin with `cfg_scale=1–3`. You can keep your FiLM CFG-dropout in `prepare_batch` unchanged.
* **Pen loss timing**: Keeping the clean pen pass in the same step is fine; it doesn’t backprop through the ODE anyway.

---

## Minimal diff summary

* ❌ Remove: `create_diffusion_schedule`, `DDIMSampler`, `add_noise`, all beta/alpha schedule usage.
* ✅ Add: `sample_rectified_flow_batch`, `FlowSampler`, new args (`--rf_noisy`, `--rf_sigma`, `--flow_steps`, `--ode_method`).
* 🔁 Replace:

  * Training PASS 1 → predict **velocity** on RF batches (MSE to `v_target`).
  * Validation PASS 1 same change.
  * All generation calls → `FlowSampler.sample(...)`.
* ⚙️ Keep: FiLM conditioning, CFG dropout, pen head + clean pass, visualizations (with minor wording change “DDIM” → “Flow/ODE”).

That’s it. Plug these in and you’ve effectively converted your pipeline to rectified flow with zero architectural churn.
