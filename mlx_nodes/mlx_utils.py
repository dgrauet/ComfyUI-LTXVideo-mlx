"""Conversion utilities between ComfyUI PyTorch tensors and MLX arrays."""

from __future__ import annotations

import gc

import numpy as np
import torch


def aggressive_cleanup() -> None:
    """Free Metal GPU cache and run garbage collection."""
    try:
        import mlx.core as mx

        # mx.metal.clear_cache() is deprecated in MLX >= 0.31
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
    except Exception:
        pass
    gc.collect()


def torch_image_to_pil(tensor: torch.Tensor):
    """Convert ComfyUI IMAGE tensor to PIL Image.

    ComfyUI IMAGE format: (B, H, W, C) float32 [0, 1].
    Returns first batch item as PIL Image.
    """
    from PIL import Image

    img_np = (tensor[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img_np)


def mx_video_frames_to_torch(frames) -> torch.Tensor:
    """Convert MLX video frames to ComfyUI IMAGE tensor.

    Input: mx.array (B, C, F, H, W) in [-1, 1] (bfloat16).
    Output: torch.Tensor (F, H, W, C) float32 [0, 1].
    """
    import mlx.core as mx

    # Convert to float32 in MLX, then to numpy
    frames_np = np.array(frames.astype(mx.float32))
    # (B, C, F, H, W) -> take batch 0 -> (C, F, H, W)
    frames_np = frames_np[0]
    # (C, F, H, W) -> (F, H, W, C)
    frames_np = np.transpose(frames_np, (1, 2, 3, 0))
    # [-1, 1] -> [0, 1]
    frames_np = (frames_np + 1.0) / 2.0
    frames_np = np.clip(frames_np, 0.0, 1.0)
    return torch.from_numpy(frames_np)


def mx_audio_to_torch(waveform, sample_rate: int = 48000) -> dict:
    """Convert MLX waveform to ComfyUI AUDIO dict.

    Input: mx.array (B, C, T) waveform.
    Output: {"waveform": torch.Tensor (B, C, T), "sample_rate": int}
    """
    import mlx.core as mx

    wav_np = np.array(waveform.astype(mx.float32))
    return {
        "waveform": torch.from_numpy(wav_np),
        "sample_rate": sample_rate,
    }
