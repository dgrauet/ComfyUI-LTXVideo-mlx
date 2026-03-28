"""MLX guidance configuration node for LTX-2."""

from __future__ import annotations

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False


class LTXVMLXGuiderConfig:
    """Configure guidance parameters for MLX LTX-2 sampling."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cfg_scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "stg_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "stg_blocks": ("STRING", {"default": "28", "tooltip": "Comma-separated block indices for STG"}),
                "rescale_scale": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "modality_scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("LTXV_MLX_GUIDER_CONFIG",)
    RETURN_NAMES = ("guider_config",)
    FUNCTION = "configure"
    CATEGORY = "Lightricks/MLX"

    def configure(
        self,
        cfg_scale: float = 3.0,
        stg_scale: float = 0.0,
        stg_blocks: str = "28",
        rescale_scale: float = 0.7,
        modality_scale: float = 3.0,
    ):
        blocks = [int(b.strip()) for b in stg_blocks.split(",") if b.strip()]

        from ltx_core_mlx.components.guiders import MultiModalGuiderParams

        video_params = MultiModalGuiderParams(
            cfg_scale=cfg_scale,
            stg_scale=stg_scale,
            stg_blocks=blocks if blocks else None,
            rescale_scale=rescale_scale,
            modality_scale=modality_scale,
        )
        audio_params = MultiModalGuiderParams(
            cfg_scale=7.0,
            stg_scale=stg_scale,
            stg_blocks=blocks if blocks else None,
            rescale_scale=rescale_scale,
            modality_scale=modality_scale,
        )

        config = {
            "video_params": video_params,
            "audio_params": audio_params,
        }
        return (config,)
