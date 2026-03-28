"""MLX model loading nodes for LTX-2 on Apple Silicon."""

from __future__ import annotations

from pathlib import Path

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False


# Module-level cache for loaded models
_model_cache: dict[str, object] = {}


def release_cache(*prefixes: str) -> None:
    """Release cached models matching any of the given key prefixes.

    Call this to free Metal memory before loading large components.
    If no prefixes given, clears entire cache.
    """
    from .mlx_utils import aggressive_cleanup

    if not prefixes:
        _model_cache.clear()
    else:
        keys_to_remove = [k for k in _model_cache if any(k.startswith(p) for p in prefixes)]
        for k in keys_to_remove:
            del _model_cache[k]
    aggressive_cleanup()


def _resolve_model_dir(model_dir: str) -> Path:
    """Resolve model directory - download from HuggingFace if needed."""
    path = Path(model_dir)
    if path.exists():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_dir))


class LazyMLXModel:
    """Lazy-loading wrapper for the LTX transformer (DiT).

    The transformer is only loaded into Metal memory when .load() is called.
    This enables the text encoder to be used and freed first.
    """

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.transformer = None

    def load(self):
        """Load the transformer into Metal memory."""
        if self.transformer is not None:
            return
        from ltx_core_mlx.model.transformer.model import LTXModel
        from ltx_core_mlx.utils.memory import aggressive_cleanup
        from ltx_core_mlx.utils.weights import apply_quantization, load_split_safetensors

        self.transformer = LTXModel()
        transformer_path = self.model_dir / "transformer.safetensors"
        if not transformer_path.exists():
            transformer_path = self.model_dir / "transformer-distilled.safetensors"
        weights = load_split_safetensors(transformer_path, prefix="transformer.")
        apply_quantization(self.transformer, weights)
        self.transformer.load_weights(list(weights.items()))
        aggressive_cleanup()

    def unload(self):
        """Free the transformer from Metal memory."""
        self.transformer = None
        from .mlx_utils import aggressive_cleanup
        aggressive_cleanup()


class LazyMLXVAE:
    """Lazy-loading wrapper for VAE components.

    Components are loaded on first access and can be freed individually.
    """

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.encoder = None
        self.decoder = None
        self.audio_decoder = None
        self.vocoder = None

    def load_encoder(self):
        if self.encoder is not None:
            return
        from ltx_core_mlx.model.video_vae.video_vae import VideoEncoder
        from ltx_core_mlx.utils.memory import aggressive_cleanup
        from ltx_core_mlx.utils.weights import load_split_safetensors

        self.encoder = VideoEncoder()
        enc_weights = load_split_safetensors(self.model_dir / "vae_encoder.safetensors", prefix="vae_encoder.")
        enc_weights = {
            k.replace("._mean_of_means", ".mean_of_means").replace("._std_of_means", ".std_of_means"): v
            for k, v in enc_weights.items()
        }
        self.encoder.load_weights(list(enc_weights.items()))
        aggressive_cleanup()

    def load_decoders(self):
        from ltx_core_mlx.utils.memory import aggressive_cleanup
        from ltx_core_mlx.utils.weights import load_split_safetensors, remap_audio_vae_keys

        if self.decoder is None:
            from ltx_core_mlx.model.video_vae.video_vae import VideoDecoder

            self.decoder = VideoDecoder()
            vae_weights = load_split_safetensors(self.model_dir / "vae_decoder.safetensors", prefix="vae_decoder.")
            self.decoder.load_weights(list(vae_weights.items()))
            aggressive_cleanup()

        if self.audio_decoder is None:
            from ltx_core_mlx.model.audio_vae.audio_vae import AudioVAEDecoder

            self.audio_decoder = AudioVAEDecoder()
            audio_weights = load_split_safetensors(self.model_dir / "audio_vae.safetensors", prefix="audio_vae.decoder.")
            all_audio = load_split_safetensors(self.model_dir / "audio_vae.safetensors", prefix="audio_vae.")
            for k, v in all_audio.items():
                if k.startswith("per_channel_statistics."):
                    audio_weights[k] = v
            audio_weights = remap_audio_vae_keys(audio_weights)
            self.audio_decoder.load_weights(list(audio_weights.items()))
            aggressive_cleanup()

        if self.vocoder is None:
            from ltx_core_mlx.model.audio_vae.bwe import VocoderWithBWE

            self.vocoder = VocoderWithBWE()
            vocoder_weights = load_split_safetensors(self.model_dir / "vocoder.safetensors", prefix="vocoder.")
            self.vocoder.load_weights(list(vocoder_weights.items()))
            aggressive_cleanup()

    def unload_encoder(self):
        self.encoder = None
        from .mlx_utils import aggressive_cleanup
        aggressive_cleanup()

    def unload_decoders(self):
        self.decoder = None
        self.audio_decoder = None
        self.vocoder = None
        from .mlx_utils import aggressive_cleanup
        aggressive_cleanup()


class LTXVMLXCheckpointLoader:
    """Load LTX-2 model weights in MLX format for Apple Silicon inference."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_dir": ("STRING", {
                    "default": "dgrauet/ltx-2.3-mlx-q8",
                    "tooltip": "Path to MLX model weights directory or HuggingFace repo ID",
                }),
            },
            "optional": {
                "force_reload": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("LTXV_MLX_MODEL", "LTXV_MLX_VAE")
    RETURN_NAMES = ("model", "vae")
    FUNCTION = "load_checkpoint"
    CATEGORY = "Lightricks/MLX"

    def load_checkpoint(self, model_dir: str, force_reload: bool = False):
        """Return a lazy model handle. Actual loading happens on first access.

        This is critical for memory management on Apple Silicon: the text encoder
        (~7GB) must be freed before loading the transformer (~10.5GB). Lazy loading
        ensures models are only loaded when the sampler actually needs them.
        """
        resolved = _resolve_model_dir(model_dir)

        model = LazyMLXModel(resolved)
        vae = LazyMLXVAE(resolved)
        return (model, vae)


class LTXVMLXTextEncoderLoader:
    """Load Gemma-3 text encoder in MLX format."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_dir": ("STRING", {
                    "default": "dgrauet/ltx-2.3-mlx-q8",
                    "tooltip": "Path to MLX model weights directory (for connector weights)",
                }),
                "gemma_model_id": ("STRING", {
                    "default": "mlx-community/gemma-3-12b-it-4bit",
                    "tooltip": "HuggingFace repo ID for Gemma model",
                }),
            },
            "optional": {
                "force_reload": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("LTXV_MLX_TEXT_ENCODER",)
    RETURN_NAMES = ("text_encoder",)
    FUNCTION = "load_encoder"
    CATEGORY = "Lightricks/MLX"

    def load_encoder(self, model_dir: str, gemma_model_id: str = "mlx-community/gemma-3-12b-it-4bit", force_reload: bool = False):
        from ltx_core_mlx.text_encoders.gemma.encoders.base_encoder import GemmaLanguageModel
        from ltx_core_mlx.text_encoders.gemma.feature_extractor import GemmaFeaturesExtractorV2
        from ltx_core_mlx.utils.memory import aggressive_cleanup
        from ltx_core_mlx.utils.weights import load_split_safetensors

        cache_key = f"text_encoder:{model_dir}:{gemma_model_id}"
        if not force_reload and cache_key in _model_cache:
            return (_model_cache[cache_key],)

        resolved = _resolve_model_dir(model_dir)

        # Load Gemma language model
        gemma = GemmaLanguageModel()
        gemma.load(gemma_model_id)
        aggressive_cleanup()

        # Load feature extractor connector
        feature_extractor = GemmaFeaturesExtractorV2()
        connector_weights = load_split_safetensors(resolved / "connector.safetensors", prefix="connector.")
        feature_extractor.connector.load_weights(list(connector_weights.items()))
        aggressive_cleanup()

        encoder = {
            "gemma": gemma,
            "feature_extractor": feature_extractor,
        }

        # NOTE: Text encoder is NOT cached - it's too large (~7GB) to coexist
        # with the transformer (~10.5GB) in Metal memory. The sampler will
        # free the conditioning embeddings' reference to the encoder after use.
        return (encoder,)
