"""MLX latent manipulation nodes for LTX-2."""

from __future__ import annotations

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False


class LTXVMLXSelectLatents:
    """Select a range of frames from MLX video latents."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LTXV_MLX_LATENT",),
                "start_frame": ("INT", {"default": 0, "min": 0}),
                "end_frame": ("INT", {"default": -1}),
            },
        }

    RETURN_TYPES = ("LTXV_MLX_LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "select"
    CATEGORY = "Lightricks/MLX"

    def select(self, latent: dict, start_frame: int = 0, end_frame: int = -1):
        video = latent["video_latent"]
        # video shape: (B, C, F, H, W)
        if end_frame == -1:
            selected = video[:, :, start_frame:]
        else:
            selected = video[:, :, start_frame:end_frame]

        result = dict(latent)
        result["video_latent"] = selected
        return (result,)


class LTXVMLXConcatLatents:
    """Concatenate two MLX video latents along the frame dimension."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent1": ("LTXV_MLX_LATENT",),
                "latent2": ("LTXV_MLX_LATENT",),
            },
        }

    RETURN_TYPES = ("LTXV_MLX_LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "concat"
    CATEGORY = "Lightricks/MLX"

    def concat(self, latent1: dict, latent2: dict):
        v1 = latent1["video_latent"]
        v2 = latent2["video_latent"]
        # Concatenate along frame dimension (dim=2)
        concatenated = mx.concatenate([v1, v2], axis=2)

        result = {"video_latent": concatenated}
        # Concatenate audio if present
        if "audio_latent" in latent1 and "audio_latent" in latent2:
            a1 = latent1["audio_latent"]
            a2 = latent2["audio_latent"]
            result["audio_latent"] = mx.concatenate([a1, a2], axis=2)

        return (result,)
