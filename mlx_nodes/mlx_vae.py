"""MLX VAE encode/decode nodes for LTX-2."""

from __future__ import annotations

import numpy as np
import torch

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False


def _materialize(*arrays):
    """Force MLX to materialize lazy computation graph."""
    mx.eval(*arrays)  # noqa: S307 - MLX graph, not Python eval


class LTXVMLXVAEDecode:
    """Decode video latents to frames using MLX VAE."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("LTXV_MLX_VAE",),
                "latent": ("LTXV_MLX_LATENT",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("video_frames",)
    FUNCTION = "decode"
    CATEGORY = "Lightricks/MLX"

    def decode(self, vae: dict, latent: dict):
        from ltx_core_mlx.utils.memory import aggressive_cleanup

        from .mlx_utils import mx_video_frames_to_torch

        video_latent = latent["video_latent"]
        vae.load_decoders()
        frames = vae.decoder.decode(video_latent)
        _materialize(frames)
        aggressive_cleanup()

        return (mx_video_frames_to_torch(frames),)


class LTXVMLXVAEEncode:
    """Encode video frames to latents using MLX VAE."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("LTXV_MLX_VAE",),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("LTXV_MLX_LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "encode"
    CATEGORY = "Lightricks/MLX"

    def encode(self, vae: dict, image: torch.Tensor):
        from ltx_core_mlx.utils.memory import aggressive_cleanup

        # ComfyUI IMAGE: (F, H, W, C) float32 [0, 1]
        # VAE expects: (B, C, F, H, W) bfloat16 [-1, 1]
        img_np = image.cpu().numpy()
        img_np = np.transpose(img_np, (3, 0, 1, 2))  # (C, F, H, W)
        img_np = img_np[np.newaxis, ...]  # (1, C, F, H, W)
        img_np = img_np * 2.0 - 1.0
        img_mx = mx.array(img_np).astype(mx.bfloat16)

        vae.load_encoder()
        latent = vae.encoder.encode(img_mx)
        _materialize(latent)
        aggressive_cleanup()

        return ({"video_latent": latent},)


class LTXVMLXAudioDecode:
    """Decode audio latents to waveform using MLX audio VAE + vocoder."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("LTXV_MLX_VAE",),
                "latent": ("LTXV_MLX_LATENT",),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "decode"
    CATEGORY = "Lightricks/MLX"

    def decode(self, vae: dict, latent: dict):
        from ltx_core_mlx.utils.memory import aggressive_cleanup

        from .mlx_utils import mx_audio_to_torch

        audio_latent = latent["audio_latent"]
        vae.load_decoders()
        mel = vae.audio_decoder.decode(audio_latent)
        waveform = vae.vocoder(mel)
        _materialize(waveform)
        aggressive_cleanup()

        return (mx_audio_to_torch(waveform, sample_rate=48000),)
