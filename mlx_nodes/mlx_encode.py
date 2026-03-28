"""MLX text encoding node for LTX-2."""

from __future__ import annotations

from nodes_registry import comfy_node

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


@comfy_node(name="LTXVMLXTextEncode", skip=not HAS_MLX)
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
            },
        }

    RETURN_TYPES = ("LTXV_MLX_CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "Lightricks/MLX"

    def encode(self, text_encoder: dict, prompt: str, negative_prompt: str = ""):
        gemma = text_encoder["gemma"]
        feature_extractor = text_encoder["feature_extractor"]

        # Encode positive prompt
        all_hidden_states, attention_mask = gemma.encode_all_layers(prompt)
        video_embeds, audio_embeds = feature_extractor(all_hidden_states, attention_mask=attention_mask)

        # Encode negative prompt
        neg_video_embeds = None
        neg_audio_embeds = None
        if negative_prompt:
            neg_hidden, neg_mask = gemma.encode_all_layers(negative_prompt)
            neg_video_embeds, neg_audio_embeds = feature_extractor(neg_hidden, attention_mask=neg_mask)

        # Force evaluation of the lazy computation graph
        arrays_to_eval = [video_embeds, audio_embeds]
        if neg_video_embeds is not None:
            arrays_to_eval.extend([neg_video_embeds, neg_audio_embeds])
        _evaluate_arrays(*arrays_to_eval)

        conditioning = {
            "video_embeds": video_embeds,
            "audio_embeds": audio_embeds,
            "neg_video_embeds": neg_video_embeds,
            "neg_audio_embeds": neg_audio_embeds,
        }
        return (conditioning,)
