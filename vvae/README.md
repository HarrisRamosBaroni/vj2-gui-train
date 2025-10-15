# VideoVAE+ Standalone Package

A standalone, self-contained package for using VideoVAE+ encoder and decoder for inference. This package can be easily migrated to other projects.

## Directory Structure

```
vvae/
├── src/
│   ├── models/              # VAE model definitions
│   │   ├── autoencoder2plus1d_1dcnn.py
│   │   ├── autoencoder.py
│   │   └── autoencoder_temporal.py
│   ├── modules/             # Model components
│   │   ├── ae_modules.py
│   │   ├── attention_temporal_videoae.py
│   │   ├── t5.py
│   │   ├── utils.py
│   │   └── losses/
│   │       ├── __init__.py
│   │       └── contperceptual.py
│   └── distributions.py
├── utils/
│   └── common_utils.py      # Config and loading utilities
├── configs/
│   ├── config_16z.yaml      # 16-channel latent model config
│   └── config_4z.yaml       # 4-channel latent model config
├── weights/
│   └── sota-4-16z.ckpt      # Model checkpoint (16z)
├── example.py               # Example usage script
└── README.md                # This file
```

## Requirements

```bash
pip install torch torchvision omegaconf
```

Optional (for text-guided models):
```bash
pip install transformers
```

## Quick Start

### Basic Usage

```python
import torch
from omegaconf import OmegaConf
from utils.common_utils import instantiate_from_config

# Load model
config = OmegaConf.load('configs/config_16z.yaml')
model = instantiate_from_config(config.model)
model = model.to('cuda')
model.eval()

# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# Prepare video tensor [B, C, T, H, W], normalized to [-1, 1]
video = torch.randn(1, 3, 16, 256, 256).cuda()

# Encode
with torch.no_grad():
    posterior = model.encode(video)
    latent = posterior.mode()  # Use mean (no sampling)

# Decode
with torch.no_grad():
    reconstructed = model.decode(latent)
```

### Using Encoder and Decoder Separately

```python
# Access encoder and decoder directly
encoder = model.encoder
decoder = model.decoder

# Encode only
with torch.no_grad():
    posterior = encoder(video)
    latent = posterior.mode()

# Decode only
with torch.no_grad():
    output = decoder(latent)
```

### Run Example Script

```bash
python example.py
```

## Model Specifications

### 16-Channel Model (config_16z.yaml)
- **Latent channels**: 16
- **Temporal compression**: 4x (16 frames → 4 latent frames)
- **Spatial compression**: ~8x in each dimension
- **Checkpoint**: `weights/sota-4-16z.ckpt`

### 4-Channel Model (config_4z.yaml)
- **Latent channels**: 4
- **Temporal compression**: 4x (16 frames → 4 latent frames)
- **Spatial compression**: ~8x in each dimension
- **Checkpoint**: `weights/sota-4-4z.ckpt` (download separately)

## Input/Output Format

### Video Input
- **Shape**: `[Batch, Channels, Time, Height, Width]`
- **Channels**: 3 (RGB)
- **Time**: Must be divisible by 4
- **Value range**: [-1, 1] (normalized)

### Latent Output (16z model)
- **Shape**: `[Batch, 16, Time/4, Height/8, Width/8]`
- For example: `[1, 3, 16, 256, 256]` → `[1, 16, 4, 32, 32]`

### Latent Output (4z model)
- **Shape**: `[Batch, 4, Time/4, Height/8, Width/8]`
- For example: `[1, 3, 16, 256, 256]` → `[1, 4, 4, 32, 32]`

## Complete Video Processing Pipeline

### Step-by-Step: Video File → Tensor → Encode → Decode → Video File

#### Step 1: Load Video and Convert to Tensor
```python
from decord import VideoReader, cpu

# Load video file
video_reader = VideoReader(video_path, ctx=cpu(0), width=desired_width, height=desired_height)

# Get FPS (important: preserve this for later!)
original_fps = video_reader.get_avg_fps()  # e.g., 24, 30, 60

# Read all frames
frame_indices = list(range(len(video_reader)))
frames = video_reader.get_batch(frame_indices)  # NumPy array [T, H, W, C], values [0, 255]

# Convert to PyTorch tensor
frames = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float()  # [C, T, H, W]

# Normalize to [-1, 1]
frames = (frames / 255 - 0.5) * 2
```

**At this stage:**
- Video is now a PyTorch tensor: `[C, T, H, W]` = `[3, num_frames, height, width]`
- Values are normalized to `[-1, 1]`
- **FPS is stored separately** - the tensor doesn't have FPS, just frames stacked in time dimension

#### Step 2: Encode to Latent Space
```python
# Add batch dimension
frames = frames.unsqueeze(0)  # [1, C, T, H, W]

# Encode
with torch.no_grad():
    posterior = model.encode(frames)
    latent = posterior.mode()  # [1, 16, T/4, H/8, W/8] for 16z model
```

#### Step 3: Decode from Latent Space
```python
# Decode
with torch.no_grad():
    reconstructed = model.decode(latent)  # [1, C, T, H, W], values in [-1, 1]
```

#### Step 4: Save Video File
```python
import torchvision

# Denormalize from [-1, 1] to [0, 255]
video_tensor = torch.clamp((reconstructed + 1) / 2, 0, 1) * 255
video_tensor = video_tensor.squeeze(0).to(torch.uint8)  # [C, T, H, W]

# Rearrange to [T, H, W, C] for saving
video_tensor = video_tensor.permute(1, 2, 3, 0)

# Save with ORIGINAL FPS preserved
torchvision.io.write_video(
    save_path,
    video_tensor,
    fps=original_fps,  # Use the same FPS from step 1!
    options={'codec': 'libx264', 'crf': '15'}
)
```

### Understanding FPS in the Pipeline

**Important:** The tensor itself doesn't have FPS - it's just frames stacked along the time dimension.

- **FPS is metadata** that you preserve separately from the original video
- Even though temporal dimension is compressed 4x in latent space (e.g., 16 frames → 4 latent frames), the output video maintains the **same FPS as input**
- Example:
  - Input: 30fps video with 120 frames → tensor `[3, 120, H, W]`
  - Latent: `[16, 30, H/8, W/8]` (temporal 4x compression)
  - Output: 30fps video with 120 frames

### Complete Example with Video File

```python
from decord import VideoReader, cpu
import torch
import torchvision
from omegaconf import OmegaConf
from utils.common_utils import instantiate_from_config

# 1. Load model
config = OmegaConf.load('configs/config_16z.yaml')
model = instantiate_from_config(config.model).cuda().eval()

# 2. Load video
video_reader = VideoReader('input.mp4', ctx=cpu(0))
fps = video_reader.get_avg_fps()
frames = video_reader.get_batch(list(range(len(video_reader))))

# 3. Preprocess
video = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float()
video = (video / 255 - 0.5) * 2  # Normalize to [-1, 1]
video = video.unsqueeze(0).cuda()  # Add batch dim

# 4. Encode & Decode
with torch.no_grad():
    latent = model.encode(video).mode()
    reconstructed = model.decode(latent)

# 5. Postprocess & Save
output = torch.clamp((reconstructed + 1) / 2, 0, 1) * 255
output = output.squeeze(0).cpu().to(torch.uint8).permute(1, 2, 3, 0)
torchvision.io.write_video('output.mp4', output, fps=fps)
```

## Important Notes

1. **Frame Count**: Input video must have number of frames divisible by 4
2. **Normalization**: Input must be normalized to [-1, 1], output is also in [-1, 1]
3. **FPS Preservation**: FPS is preserved separately as metadata, not in the tensor
4. **Tensor Shape**: Video tensor is `[C, T, H, W]` or `[B, C, T, H, W]` with batch dimension
5. **Frozen Weights**: All parameters should be frozen for inference
6. **Device**: Supports both CUDA and CPU (CUDA recommended)
7. **Memory**: For large videos, process in chunks to avoid OOM
8. **Dependencies**: Requires `decord` for video loading: `pip install decord`

## Migrating to Another Project

This entire `vvae/` folder is self-contained and can be copied to any project:

```bash
cp -r vvae /path/to/your/project/
```

Then in your project:

```python
import sys
sys.path.append('/path/to/your/project/vvae')

from omegaconf import OmegaConf
from utils.common_utils import instantiate_from_config

config = OmegaConf.load('/path/to/your/project/vvae/configs/config_16z.yaml')
model = instantiate_from_config(config.model)
```

## Citation

```bibtex
@article{xing2024videovae,
  title={VideoVAE+: Large Motion Video Autoencoding with Cross-modal Video VAE},
  author={Xing, Yazhou and Fei, Yang and He, Yingqing and Chen, Jingye and Xie, Jiaxin and Chi, Xiaowei and Chen, Qifeng},
  journal={arXiv preprint arXiv:2412.17805},
  year={2024}
}
```

## License

Please follow [CC-BY-NC-ND](https://github.com/VideoVerses/VideoVAEPlus/blob/main/LICENSE).
