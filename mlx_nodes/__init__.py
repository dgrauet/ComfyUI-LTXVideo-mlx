"""MLX nodes for LTX-2 video generation on Apple Silicon.

These nodes are only registered when running on macOS with MLX installed.
On other platforms, this module exports empty mappings.
"""

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    import mlx.core  # noqa: F401

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

if HAS_MLX:
    from .mlx_encode import LTXVMLXTextEncode  # noqa: F401
    from .mlx_guider import LTXVMLXGuiderConfig  # noqa: F401
    from .mlx_latents import LTXVMLXConcatLatents, LTXVMLXSelectLatents  # noqa: F401
    from .mlx_loader import LTXVMLXCheckpointLoader, LTXVMLXTextEncoderLoader  # noqa: F401
    from .mlx_sampler import LTXVMLXBaseSampler, LTXVMLXExtendSampler, LTXVMLXICLoRASampler, LTXVMLXTwoStageSampler  # noqa: F401
    from .mlx_vae import LTXVMLXAudioDecode, LTXVMLXVAEDecode, LTXVMLXVAEEncode  # noqa: F401

    MLX_PREFIX = "\U0001f34e MLX"

    NODE_CLASS_MAPPINGS = {
        "LTXVMLXCheckpointLoader": LTXVMLXCheckpointLoader,
        "LTXVMLXTextEncoderLoader": LTXVMLXTextEncoderLoader,
        "LTXVMLXTextEncode": LTXVMLXTextEncode,
        "LTXVMLXGuiderConfig": LTXVMLXGuiderConfig,
        "LTXVMLXBaseSampler": LTXVMLXBaseSampler,
        "LTXVMLXTwoStageSampler": LTXVMLXTwoStageSampler,
        "LTXVMLXExtendSampler": LTXVMLXExtendSampler,
        "LTXVMLXICLoRASampler": LTXVMLXICLoRASampler,
        "LTXVMLXVAEDecode": LTXVMLXVAEDecode,
        "LTXVMLXVAEEncode": LTXVMLXVAEEncode,
        "LTXVMLXAudioDecode": LTXVMLXAudioDecode,
        "LTXVMLXSelectLatents": LTXVMLXSelectLatents,
        "LTXVMLXConcatLatents": LTXVMLXConcatLatents,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "LTXVMLXCheckpointLoader": f"{MLX_PREFIX} Checkpoint Loader",
        "LTXVMLXTextEncoderLoader": f"{MLX_PREFIX} Text Encoder Loader",
        "LTXVMLXTextEncode": f"{MLX_PREFIX} Text Encode",
        "LTXVMLXGuiderConfig": f"{MLX_PREFIX} Guider Config",
        "LTXVMLXBaseSampler": f"{MLX_PREFIX} Base Sampler",
        "LTXVMLXTwoStageSampler": f"{MLX_PREFIX} Two Stage Sampler",
        "LTXVMLXExtendSampler": f"{MLX_PREFIX} Extend Sampler",
        "LTXVMLXICLoRASampler": f"{MLX_PREFIX} IC-LoRA Sampler",
        "LTXVMLXVAEDecode": f"{MLX_PREFIX} VAE Decode",
        "LTXVMLXVAEEncode": f"{MLX_PREFIX} VAE Encode",
        "LTXVMLXAudioDecode": f"{MLX_PREFIX} Audio Decode",
        "LTXVMLXSelectLatents": f"{MLX_PREFIX} Select Latents",
        "LTXVMLXConcatLatents": f"{MLX_PREFIX} Concat Latents",
    }
