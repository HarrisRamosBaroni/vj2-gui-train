import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import cv2
import numpy as np

import os

"""
VJEPA2:
    - Trained on 4 FPS, 16-frame clips (~4 seconds)
    - Can generalize up to 64 frames, but most robust at 16
"""

class VJEPA2Wrapper(nn.Module):
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu',
                 model_name='vjepa2_vit_large', num_frames=16, image_size=256):
        super().__init__()
        print("THOME = ",os.getenv("TORCH_HOME"))
        self.device = device
        self.num_frames = num_frames  # Number of frames per batch element
        self.image_size = image_size

        # Load model (returns transformer encoder)
        # self.encoder, _ = torch.hub.load('facebookresearch/vjepa2', model_name)
        self.encoder,_ = torch.hub.load(
            'facebookresearch/vjepa2',
            model_name,
            source='github',
            trust_repo=True,
            hub_dir='/cs/student/projects1/rai/2023/kevinxie/vj2-gui'
        )
        self.encoder = self.encoder.to(device).eval()

        # Preprocessing: [H,W,C] -> [C,H,W], normalized to [0,1]
        # ImageNet normalization values
        normalize_mean = [0.485, 0.456, 0.406]
        normalize_std = [0.229, 0.224, 0.225]
        self.transform = T.Compose([
            T.Resize((self.image_size, self.image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=normalize_mean, std=normalize_std)
        ])
        # self.transform = T.Compose([
        #     T.ToPILImage(),
        #     T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        #     T.CenterCrop(image_size), # MIGHT NEED TO CHANGE
        #     T.ToTensor(),
        # ])

    def forward(self, x: torch.Tensor):
        """
        x: [B, T, C, H, W]
        Returns: [B, T_actual, N, D] where T_actual = T // tubelet_size
        """
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, T, H, W]
        z = self.encoder(x.to(self.device))        # [B, T_actual * N, D]

        tubelet_size = 2 
        T_actual = T // tubelet_size
        N = z.shape[1] // T_actual
        assert z.shape[1] == T_actual * N, f"Unexpected token count: {z.shape[1]} != {T_actual} × {N}"

        return z.view(B, T_actual, N, z.shape[2])  # [B, T_actual, N, D]

    @torch.no_grad()
    def from_video(self, video_path: str, fps: int = 4) -> torch.Tensor:
        """
        Preprocess video: uniformly sample at `fps`, chunk into segments of `num_frames`.
        Returns: video batch tensor of shape [B, T, C, H, W] where T = self.num_frames
        """

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open {video_path}")

        # Read video metadata
        vid_fps = cap.get(cv2.CAP_PROP_FPS)  # Original FPS
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / vid_fps

        # Determine indices to sample at desired FPS
        total_target_frames = int(duration_sec * fps)
        frame_idxs = np.linspace(0, total_frames - 1, total_target_frames).astype(int)

        # Read and transform selected frames
        frames = []
        for idx in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = cap.read()
            if not success:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # [H,W,3]
            frame_tensor = self.transform(frame_rgb)  # -> [3, H, W]
            frames.append(frame_tensor)

        cap.release()

        total = len(frames)
        usable = (total // self.num_frames) * self.num_frames
        frames = frames[:usable]  # Drop remainder

        if usable < self.num_frames:
            raise ValueError("Not enough frames to form a single chunk.")

        video_tensor = torch.stack(frames)  # [F, C, H, W] = [usable, 3, H, W]
        video_tensor = video_tensor.view(-1, self.num_frames, 3, self.image_size, self.image_size)  # [B, T, C, H, W] where batch B = usable/T

        return video_tensor.to(self.device)

if __name__ == "__main__":
    # --- Load and encode video ---
    encoder = VJEPA2Wrapper(num_frames=16)
    video_tensor = encoder.from_video("videos/test1.mp4", fps=4)  # [B, T, C, H, W]
    z_all = encoder(video_tensor)  # [B, N, D]

    B, T, N, D = z_all.shape
    assert z_all.ndim == 4, f"Expected [B, T, N, D], got {z_all.shape}"
    print(f"✅ Encoder output shape OK: {z_all.shape}")

    # --- Simulate rollout from last encoded chunk ---
    z_T = z_all[-1:].unsqueeze(1)  # [1, 1, 1568, 1024]
    a_T = torch.randn(1, 1, 12).to(z_T.device)  # dummy action [1, 1, 12]

