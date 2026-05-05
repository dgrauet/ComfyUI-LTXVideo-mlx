"""MLX text encoding node for LTX-2."""

from __future__ import annotations

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

DEFAULT_NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, washed out, pale, faded, "
    "blotchy, pixelated, grainy, noisy, distorted, warped, weak, bad, poor, ugly, "
    "deformed, disfigured, broken, corrupted, artifacted, glitched, incomplete, "
    "low quality, unfinished, unpolished, amateur, low-res, low-resolution, "
    "watermarked, text overlay, date, time, signature, username, artificial, "
    "synthetic, fake, rendered, 3D, CGI, cartoon, animated, hand-drawn, sketch, "
    "painting, art, illustration, desaturated, oversaturated, color cast, AI artifacts"
)


def _evaluate_arrays(*arrays):
    """Force MLX lazy graph evaluation on the given arrays.

    This calls mx.eval() which is MLX's graph evaluation function,
    NOT Python's eval(). It materializes the lazy computation graph.
    """
    mx.eval(*arrays)


class LTXVMLXTextEncode:
    """Encode text prompts to video and audio embeddings using MLX Gemma encoder."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_encoder": ("LTXV_MLX_TEXT_ENCODER",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "default": DEFAULT_NEGATIVE_PROMPT,
                    "multiline": True,
                }),
                "max_length": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 1024,
                    "step": 64,
                    "tooltip": (
                        "Token sequence length passed to Gemma. The prompt is "
                        "left-padded to this length. Lower values reduce GPU "
                        "command-buffer time per layer; on 32GB Apple Silicon "
                        "(e.g. M2 Pro), values above ~512 can trip the macOS "
                        "GPU watchdog (Impacting Interactivity) for the 12B "
                        "encoder. 256 is a safe default for short prompts."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("LTXV_MLX_CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "Lightricks/MLX"

    def encode(self, text_encoder: dict, prompt: str, negative_prompt: str = "", max_length: int = 1024):
        gemma = text_encoder.get("gemma")
        feature_extractor = text_encoder.get("feature_extractor")
        if gemma is None or feature_extractor is None:
            raise RuntimeError(
                "LTXVMLXTextEncode received an already-freed text_encoder. The "
                "encoder is freed in low_vram mode after each encode call. "
                "Re-run the workflow to reload Gemma, or switch the loader's "
                "memory_profile to 'standard' to keep it cached."
            )

        # Encode positive prompt
        all_hidden_states, attention_mask = gemma.encode_all_layers(prompt, max_length=max_length)
        video_embeds, audio_embeds = feature_extractor(all_hidden_states, attention_mask=attention_mask)

        # Encode negative prompt
        neg_video_embeds = None
        neg_audio_embeds = None
        if negative_prompt:
            neg_hidden, neg_mask = gemma.encode_all_layers(negative_prompt, max_length=max_length)
            neg_video_embeds, neg_audio_embeds = feature_extractor(neg_hidden, attention_mask=neg_mask)

        # Force evaluation of the lazy computation graph so embeddings are
        # materialized BEFORE we (potentially) release Gemma — otherwise lazy
        # ops would follow weight references that are about to be freed.
        arrays_to_eval = [video_embeds, audio_embeds]
        if neg_video_embeds is not None:
            arrays_to_eval.extend([neg_video_embeds, neg_audio_embeds])
        _evaluate_arrays(*arrays_to_eval)

        # Free the encoder ONLY in low_vram mode. In standard mode, leave it
        # intact so subsequent Queue runs reuse the cached loader output and
        # skip the ~10s Gemma reload.
        memory_profile = text_encoder.get("_memory_profile", "standard")
        if memory_profile == "low_vram":
            from ltx_core_mlx.utils.memory import aggressive_cleanup

            text_encoder["gemma"] = None
            text_encoder["feature_extractor"] = None
            del gemma, feature_extractor
            aggressive_cleanup()

        conditioning = {
            "video_embeds": video_embeds,
            "audio_embeds": audio_embeds,
            "neg_video_embeds": neg_video_embeds,
            "neg_audio_embeds": neg_audio_embeds,
        }
        return (conditioning,)
