"""
Video loading and saving utilities for VideoVAE+.
"""

import torch
import logging
from decord import VideoReader, cpu
import torchvision
import av
import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def data_processing(video_path, resolution):
    """Load and preprocess video data."""
    try:
        video_reader = VideoReader(video_path, ctx=cpu(0))
        video_resolution = video_reader[0].shape

        # Rescale resolution to match specified limits
        resolution = [
            min(video_resolution[0], resolution[0]),
            min(video_resolution[1], resolution[1]),
        ]
        video_reader = VideoReader(
            video_path, ctx=cpu(0), width=resolution[1], height=resolution[0]
        )

        video_length = len(video_reader)
        vid_fps = video_reader.get_avg_fps()
        frame_indices = list(range(0, video_length))
        frames = video_reader.get_batch(frame_indices)
        assert (
            frames.shape[0] == video_length
        ), f"Frame mismatch: {len(frames)} != {video_length}"

        frames = (
            torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float()
        )  # [t, h, w, c] -> [c, t, h, w]
        frames = (frames / 255 - 0.5) * 2  # Normalize to [-1, 1]
        return frames, vid_fps
    except Exception as e:
        logging.error(f"Error processing video {video_path}: {e}")
        return None, None


def save_video(tensor, save_path, fps: float):
    """Save video tensor to a file."""
    try:
        tensor = torch.clamp((tensor + 1) / 2, 0, 1) * 255
        arr = tensor.detach().cpu().squeeze().to(torch.uint8)
        c, t, h, w = arr.shape

        torchvision.io.write_video(save_path, arr.permute(1, 2, 3, 0), fps=fps, options={'codec': 'libx264', 'crf': '15'})
        logging.info(f"Video saved to {save_path}")
    except Exception as e:
        logging.error(f"Error saving video {save_path}: {e}")


def load_video_cfr(path, target_fps=30, max_duration=None, target_size=(512, 512)):
    """
    Load video and resample to constant frame rate (CFR).

    This function ensures constant FPS by:
    1. Computing target frame interval: dt = 1/target_fps (time between frames)
    2. For each decoded frame at timestamp t:
       - Check if enough time has elapsed (t >= last_t)
       - If yes, emit frame(s) at regular intervals (last_t, last_t+dt, last_t+2*dt, ...)
       - Continue until last_t catches up to current timestamp t

    This approach:
    - DUPLICATES frames if video runs slower than target_fps (frame held longer)
    - DROPS frames if video runs faster than target_fps (skip intermediate frames)
    - Guarantees output has exactly target_fps spacing

    Args:
        path: Path to video file
        target_fps: Desired output frame rate (frames per second)
        max_duration: Maximum duration to load in seconds (None = load all)
        target_size: Resize frames to (height, width). None = no resize

    Returns:
        np.ndarray: Video frames with shape (N, H, W, 3), sampled at constant FPS
    """
    container = av.open(path)
    stream = container.streams.video[0]

    # Compute frame interval (seconds per frame at target FPS)
    dt = 1.0 / target_fps
    frames, last_t = [], 0.0

    for packet in container.demux(stream):
        for frame in packet.decode():
            # Get actual timestamp of this decoded frame
            t = float(frame.pts * stream.time_base)

            # Stop loading if we've exceeded max_duration
            if max_duration is not None and t > max_duration:
                container.close()
                logging.info(f"Stopped loading at {t:.2f}s (max_duration={max_duration}s)")
                return np.stack(frames) if frames else np.array([])

            # Emit frames at regular intervals until we catch up to current timestamp
            # This while loop handles both frame duplication (slow source) and dropping (fast source)
            while t >= last_t:
                # Check if the next frame would exceed max_duration
                if max_duration is not None and last_t >= max_duration:
                    container.close()
                    logging.info(f"Reached max_duration={max_duration}s, loaded {len(frames)} frames")
                    return np.stack(frames) if frames else np.array([])

                img = frame.to_ndarray(format="rgb24")

                # Resize if target_size is specified
                if target_size is not None:
                    pil_img = Image.fromarray(img)
                    pil_img = pil_img.resize((target_size[1], target_size[0]), Image.LANCZOS)
                    img = np.array(pil_img)

                frames.append(img)
                last_t += dt  # Advance output time by one frame interval

    container.close()
    return np.stack(frames) if frames else np.array([])
