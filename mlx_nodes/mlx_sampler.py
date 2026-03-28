"""MLX sampling nodes for LTX-2 video generation on Apple Silicon."""

from __future__ import annotations

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False


def _materialize(*arrays):
    """Force MLX to materialize lazy computation graph.

    mx.eval() triggers MLX graph evaluation (similar to tf.Session.run()),
    this is NOT Python's eval() builtin.
    """
    mx.eval(*arrays)  # noqa: S307 - MLX graph evaluation, not Python eval


class LTXVMLXBaseSampler:
    """Text-to-Video and Image-to-Video sampling using MLX on Apple Silicon."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("LTXV_MLX_MODEL",),
                "conditioning": ("LTXV_MLX_CONDITIONING",),
                "vae": ("LTXV_MLX_VAE",),
                "width": ("INT", {"default": 704, "min": 64, "max": 2048, "step": 32}),
                "height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 32}),
                "num_frames": ("INT", {"default": 97, "min": 1, "max": 257, "step": 8}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**31 - 1}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 100}),
            },
            "optional": {
                "image": ("IMAGE",),
                "guider_config": ("LTXV_MLX_GUIDER_CONFIG",),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("video_frames", "audio")
    FUNCTION = "sample"
    CATEGORY = "Lightricks/MLX"

    def sample(
        self,
        model: dict,
        conditioning: dict,
        vae: dict,
        width: int = 704,
        height: int = 480,
        num_frames: int = 97,
        seed: int = 42,
        steps: int = 8,
        image=None,
        guider_config: dict | None = None,
    ):
        from ltx_core_mlx.components.patchifiers import (
            AudioPatchifier,
            VideoLatentPatchifier,
            compute_video_latent_shape,
        )
        from ltx_core_mlx.conditioning.types.latent_cond import (
            VideoConditionByLatentIndex,
            apply_conditioning,
            create_initial_state,
        )
        from ltx_core_mlx.model.transformer.model import X0Model
        from ltx_core_mlx.utils.memory import aggressive_cleanup
        from ltx_core_mlx.utils.positions import (
            compute_audio_positions,
            compute_audio_token_count,
            compute_video_positions,
        )
        from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS

        from .mlx_utils import mx_audio_to_torch, mx_video_frames_to_torch

        video_embeds = conditioning["video_embeds"]
        audio_embeds = conditioning["audio_embeds"]
        neg_video_embeds = conditioning.get("neg_video_embeds")
        neg_audio_embeds = conditioning.get("neg_audio_embeds")

        video_patchifier = VideoLatentPatchifier()
        audio_patchifier = AudioPatchifier()

        # Compute latent shapes
        F, H, W = compute_video_latent_shape(num_frames, height, width)
        video_shape = (1, F * H * W, 128)
        audio_T = compute_audio_token_count(num_frames)
        audio_shape = (1, audio_T, 128)

        # Compute RoPE positions
        video_positions = compute_video_positions(F, H, W)
        audio_positions = compute_audio_positions(audio_T)

        # Create initial noise state
        video_state = create_initial_state(video_shape, seed, positions=video_positions)
        audio_state = create_initial_state(audio_shape, seed + 1, positions=audio_positions)

        # Image-to-Video conditioning (needs VAE encoder)
        if image is not None:
            from ltx_core_mlx.utils.image import prepare_image_for_encoding

            from .mlx_utils import torch_image_to_pil

            vae.load_encoder()
            pil_image = torch_image_to_pil(image)
            img_tensor = prepare_image_for_encoding(pil_image, height, width)
            ref_latent = vae.encoder.encode(img_tensor[:, :, None, :, :])
            _materialize(ref_latent)
            vae.unload_encoder()

            ref_tokens = ref_latent.transpose(0, 2, 3, 4, 1).reshape(1, -1, 128)
            condition = VideoConditionByLatentIndex(frame_indices=[0], clean_latent=ref_tokens)
            video_state = apply_conditioning(video_state, [condition], (F, H, W))

        # --- Load transformer (after text encoder is freed by ComfyUI GC) ---
        model.load()
        dit = model.transformer

        # Sigma schedule
        sigmas = DISTILLED_SIGMAS[: steps + 1] if steps < len(DISTILLED_SIGMAS) else DISTILLED_SIGMAS

        # Build model
        x0_model = X0Model(dit)

        # Choose sampling path based on guidance config
        if guider_config is not None and neg_video_embeds is not None:
            from ltx_core_mlx.components.guiders import create_multimodal_guider_factory
            from ltx_pipelines_mlx.scheduler import ltx2_schedule
            from ltx_pipelines_mlx.utils.samplers import guided_denoise_loop

            # Use dynamic schedule for guided sampling
            num_tokens = F * H * W
            sigmas = ltx2_schedule(steps, num_tokens=num_tokens)

            video_factory = create_multimodal_guider_factory(
                guider_config["video_params"],
                negative_context=neg_video_embeds,
            )
            audio_factory = create_multimodal_guider_factory(
                guider_config["audio_params"],
                negative_context=neg_audio_embeds,
            )

            output = guided_denoise_loop(
                model=x0_model,
                video_state=video_state,
                audio_state=audio_state,
                video_text_embeds=video_embeds,
                audio_text_embeds=audio_embeds,
                video_guider_factory=video_factory,
                audio_guider_factory=audio_factory,
                sigmas=sigmas,
            )
        else:
            from ltx_pipelines_mlx.utils.samplers import denoise_loop

            output = denoise_loop(
                model=x0_model,
                video_state=video_state,
                audio_state=audio_state,
                video_text_embeds=video_embeds,
                audio_text_embeds=audio_embeds,
                sigmas=sigmas,
            )

        # Free transformer before loading VAE decoders
        model.unload()

        # Unpatchify
        video_latent = video_patchifier.unpatchify(output.video_latent, (F, H, W))
        audio_latent = audio_patchifier.unpatchify(output.audio_latent)

        # Load and decode
        vae.load_decoders()

        video_frames = vae.decoder.decode(video_latent)
        _materialize(video_frames)
        aggressive_cleanup()

        mel = vae.audio_decoder.decode(audio_latent)
        waveform = vae.vocoder(mel)
        _materialize(waveform)
        aggressive_cleanup()

        # Convert to ComfyUI formats
        video_torch = mx_video_frames_to_torch(video_frames)
        audio_torch = mx_audio_to_torch(waveform, sample_rate=48000)

        return (video_torch, audio_torch)


class LTXVMLXTwoStageSampler:
    """Two-stage MLX sampling: CFG at half-res, upscale, distilled LoRA refine."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("LTXV_MLX_MODEL",),
                "conditioning": ("LTXV_MLX_CONDITIONING",),
                "vae": ("LTXV_MLX_VAE",),
                "width": ("INT", {"default": 704, "min": 64, "max": 2048, "step": 32}),
                "height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 32}),
                "num_frames": ("INT", {"default": 97, "min": 1, "max": 257, "step": 8}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**31 - 1}),
                "stage1_steps": ("INT", {"default": 30, "min": 1, "max": 100}),
            },
            "optional": {
                "stage2_steps": ("INT", {"default": 3, "min": 1, "max": 100}),
                "image": ("IMAGE",),
                "guider_config": ("LTXV_MLX_GUIDER_CONFIG",),
                "dev_transformer": ("STRING", {"default": "transformer-dev.safetensors"}),
                "distilled_lora": ("STRING", {"default": "ltx-2.3-22b-distilled-lora-384.safetensors"}),
                "distilled_lora_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("video_frames", "audio")
    FUNCTION = "sample"
    CATEGORY = "Lightricks/MLX"

    def sample(
        self,
        model: dict,
        conditioning: dict,
        vae: dict,
        width: int = 704,
        height: int = 480,
        num_frames: int = 97,
        seed: int = 42,
        stage1_steps: int = 30,
        stage2_steps: int = 3,
        image=None,
        guider_config: dict | None = None,
        dev_transformer: str = "transformer-dev.safetensors",
        distilled_lora: str = "ltx-2.3-22b-distilled-lora-384.safetensors",
        distilled_lora_strength: float = 1.0,
    ):
        import json

        from ltx_core_mlx.components.guiders import (
            MultiModalGuiderParams,
            create_multimodal_guider_factory,
        )
        from ltx_core_mlx.components.patchifiers import (
            AudioPatchifier,
            VideoLatentPatchifier,
            compute_video_latent_shape,
        )
        from ltx_core_mlx.conditioning.types.latent_cond import (
            LatentState,
            VideoConditionByLatentIndex,
            apply_conditioning,
            create_initial_state,
            noise_latent_state,
        )
        from ltx_core_mlx.loader.fuse_loras import apply_loras
        from ltx_core_mlx.loader.primitives import LoraStateDictWithStrength, StateDict
        from ltx_core_mlx.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
        from ltx_core_mlx.model.transformer.model import LTXModel, X0Model
        from ltx_core_mlx.model.upsampler import LatentUpsampler
        from ltx_core_mlx.utils.image import prepare_image_for_encoding
        from ltx_core_mlx.utils.memory import aggressive_cleanup
        from ltx_core_mlx.utils.positions import (
            compute_audio_positions,
            compute_audio_token_count,
            compute_video_positions,
        )
        from ltx_core_mlx.utils.weights import apply_quantization, load_split_safetensors
        from ltx_pipelines_mlx.scheduler import STAGE_2_SIGMAS, ltx2_schedule
        from ltx_pipelines_mlx.utils.samplers import denoise_loop, guided_denoise_loop

        from .mlx_utils import mx_audio_to_torch, mx_video_frames_to_torch, torch_image_to_pil

        model_dir = model.model_dir
        video_embeds = conditioning["video_embeds"]
        audio_embeds = conditioning["audio_embeds"]
        neg_video_embeds = conditioning.get("neg_video_embeds")
        neg_audio_embeds = conditioning.get("neg_audio_embeds")

        video_patchifier = VideoLatentPatchifier()
        audio_patchifier = AudioPatchifier()

        # --- Load VAE encoder (needed for I2V + denorm/renorm) ---
        vae.load_encoder()

        # --- Load dev transformer ---
        dev_path = model_dir / dev_transformer
        dit = LTXModel()
        weights = load_split_safetensors(dev_path, prefix="transformer.")
        apply_quantization(dit, weights)
        dit.load_weights(list(weights.items()))
        aggressive_cleanup()

        # --- Load upsampler ---
        upsampler_name = "spatial_upscaler_x2_v1_1"
        config_path = model_dir / f"{upsampler_name}_config.json"
        weights_path = model_dir / f"{upsampler_name}.safetensors"
        if config_path.exists():
            config = json.loads(config_path.read_text()).get("config", {})
            upsampler = LatentUpsampler.from_config(config)
        else:
            upsampler = LatentUpsampler()
        if weights_path.exists():
            up_weights = load_split_safetensors(weights_path, prefix=f"{upsampler_name}.")
            upsampler.load_weights(list(up_weights.items()))
        aggressive_cleanup()

        # --- Stage 1: Half resolution with CFG ---
        half_h, half_w = height // 2, width // 2
        F, H_half, W_half = compute_video_latent_shape(num_frames, half_h, half_w)
        video_shape = (1, F * H_half * W_half, 128)
        audio_T = compute_audio_token_count(num_frames)
        audio_shape = (1, audio_T, 128)

        video_positions_1 = compute_video_positions(F, H_half, W_half)
        audio_positions = compute_audio_positions(audio_T)

        video_state = create_initial_state(video_shape, seed, positions=video_positions_1)
        audio_state = create_initial_state(audio_shape, seed + 1, positions=audio_positions)

        # I2V conditioning at half resolution
        pil_image = None
        if image is not None:
            pil_image = torch_image_to_pil(image)
            enc_h = H_half * 32
            enc_w = W_half * 32
            img_tensor = prepare_image_for_encoding(pil_image, enc_h, enc_w)
            ref_latent = vae.encoder.encode(img_tensor[:, :, None, :, :])
            ref_tokens = ref_latent.transpose(0, 2, 3, 4, 1).reshape(1, -1, 128)
            condition = VideoConditionByLatentIndex(frame_indices=[0], clean_latent=ref_tokens, strength=1.0)
            video_state = apply_conditioning(video_state, [condition], (F, H_half, W_half))
            video_state = noise_latent_state(video_state, sigma=1.0, seed=seed)
            audio_state = noise_latent_state(audio_state, sigma=1.0, seed=seed + 1)

        # Stage 1 sigma schedule
        num_tokens = F * H_half * W_half
        sigmas_1 = ltx2_schedule(stage1_steps, num_tokens=num_tokens)
        x0_model = X0Model(dit)

        # Build guider
        if guider_config is not None:
            video_params = guider_config["video_params"]
            audio_params = guider_config["audio_params"]
        else:
            video_params = MultiModalGuiderParams(
                cfg_scale=3.0, stg_scale=0.0, rescale_scale=0.7,
                modality_scale=3.0, stg_blocks=[28],
            )
            audio_params = MultiModalGuiderParams(
                cfg_scale=7.0, stg_scale=0.0, rescale_scale=0.7,
                modality_scale=3.0, stg_blocks=[28],
            )

        video_factory = create_multimodal_guider_factory(video_params, negative_context=neg_video_embeds)
        audio_factory = create_multimodal_guider_factory(audio_params, negative_context=neg_audio_embeds)

        output_1 = guided_denoise_loop(
            model=x0_model,
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            video_guider_factory=video_factory,
            audio_guider_factory=audio_factory,
            sigmas=sigmas_1,
        )
        aggressive_cleanup()

        # --- Fuse distilled LoRA ---
        def _remap_lora_keys(lora_sd):
            remapped = {}
            for key, value in lora_sd.items():
                new_key = LTXV_LORA_COMFY_RENAMING_MAP.apply_to_key(key)
                new_key = new_key.replace(".linear_1.", ".linear1.").replace(".linear_2.", ".linear2.")
                new_key = new_key.replace("audio_ff.net.0.proj.", "audio_ff.proj_in.")
                new_key = new_key.replace("audio_ff.net.2.", "audio_ff.proj_out.")
                remapped[new_key] = value
            return remapped

        lora_path = model_dir / distilled_lora
        if lora_path.exists():
            import mlx.utils

            lora_raw = dict(mx.load(str(lora_path)))
            lora_remapped = _remap_lora_keys(lora_raw)
            flat_params = mlx.utils.tree_flatten(dit.parameters())
            flat_model = {k: v for k, v in flat_params if isinstance(v, mx.array)}
            model_sd = StateDict(sd=flat_model, size=0, dtype=set())
            lora_sd = StateDict(sd=lora_remapped, size=0, dtype=set())
            lora_with_strength = LoraStateDictWithStrength(lora_sd, distilled_lora_strength)
            fused = apply_loras(model_sd, [lora_with_strength])
            dit.load_weights(list(fused.sd.items()))
            aggressive_cleanup()

        # --- Upscale ---
        video_half = video_patchifier.unpatchify(output_1.video_latent, (F, H_half, W_half))
        video_mlx = video_half.transpose(0, 2, 3, 4, 1)
        video_denorm = vae.encoder.denormalize_latent(video_mlx)
        video_denorm = video_denorm.transpose(0, 4, 1, 2, 3)
        video_upscaled = upsampler(video_denorm)
        video_up_mlx = video_upscaled.transpose(0, 2, 3, 4, 1)
        video_upscaled = vae.encoder.normalize_latent(video_up_mlx)
        video_upscaled = video_upscaled.transpose(0, 4, 1, 2, 3)
        _materialize(video_upscaled)

        H_full = H_half * 2
        W_full = W_half * 2

        # I2V conditioning at full resolution for Stage 2
        conditionings_2 = []
        if pil_image is not None:
            enc_h_full = H_full * 32
            enc_w_full = W_full * 32
            img_tensor = prepare_image_for_encoding(pil_image, enc_h_full, enc_w_full)
            ref_latent = vae.encoder.encode(img_tensor[:, :, None, :, :])
            ref_tokens = ref_latent.transpose(0, 2, 3, 4, 1).reshape(1, -1, 128)
            conditionings_2.append(VideoConditionByLatentIndex(frame_indices=[0], clean_latent=ref_tokens, strength=1.0))

        # Free upsampler
        del upsampler
        aggressive_cleanup()

        # --- Stage 2: Refine at full resolution ---
        video_tokens, _ = video_patchifier.patchify(video_upscaled)
        sigmas_2 = STAGE_2_SIGMAS[: stage2_steps + 1] if stage2_steps < len(STAGE_2_SIGMAS) else STAGE_2_SIGMAS
        start_sigma = sigmas_2[0]

        mx.random.seed(seed + 2)
        noise = mx.random.normal(video_tokens.shape).astype(mx.bfloat16)
        noisy_tokens = noise * start_sigma + video_tokens * (1.0 - start_sigma)

        video_positions_2 = compute_video_positions(F, H_full, W_full)

        video_state_2 = LatentState(
            latent=noisy_tokens,
            clean_latent=video_tokens,
            denoise_mask=mx.ones((1, video_tokens.shape[1], 1), dtype=mx.bfloat16),
            positions=video_positions_2,
        )

        if conditionings_2:
            video_state_2 = apply_conditioning(video_state_2, conditionings_2, (F, H_full, W_full))

        audio_tokens_1 = output_1.audio_latent
        audio_state_2 = LatentState(
            latent=audio_tokens_1,
            clean_latent=audio_tokens_1,
            denoise_mask=mx.ones((1, audio_tokens_1.shape[1], 1), dtype=audio_tokens_1.dtype),
            positions=audio_positions,
        )
        audio_state_2 = noise_latent_state(audio_state_2, sigma=start_sigma, seed=seed + 2)

        output_2 = denoise_loop(
            model=x0_model,
            video_state=video_state_2,
            audio_state=audio_state_2,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            sigmas=sigmas_2,
        )
        aggressive_cleanup()

        # Free transformer + VAE encoder before loading decoders
        del dit
        vae.unload_encoder()
        aggressive_cleanup()

        # Unpatchify
        video_latent = video_patchifier.unpatchify(output_2.video_latent, (F, H_full, W_full))
        audio_latent = audio_patchifier.unpatchify(output_2.audio_latent)

        # Decode video
        vae.load_decoders()
        video_frames = vae.decoder.decode(video_latent)
        _materialize(video_frames)
        aggressive_cleanup()

        # Decode audio
        mel = vae.audio_decoder.decode(audio_latent)
        waveform = vae.vocoder(mel)
        _materialize(waveform)
        aggressive_cleanup()

        # Convert to ComfyUI formats
        video_torch = mx_video_frames_to_torch(video_frames)
        audio_torch = mx_audio_to_torch(waveform, sample_rate=48000)

        return (video_torch, audio_torch)


class LTXVMLXExtendSampler:
    """Extend existing video using MLX on Apple Silicon."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("LTXV_MLX_MODEL",),
                "conditioning": ("LTXV_MLX_CONDITIONING",),
                "vae": ("LTXV_MLX_VAE",),
                "source_video": ("IMAGE",),
                "extend_frames": ("INT", {"default": 48, "min": 8, "max": 257, "step": 8}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**31 - 1}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100}),
            },
            "optional": {
                "direction": (["after", "before"], {"default": "after"}),
                "guider_config": ("LTXV_MLX_GUIDER_CONFIG",),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("video_frames", "audio")
    FUNCTION = "sample"
    CATEGORY = "Lightricks/MLX"

    def sample(
        self,
        model: dict,
        conditioning: dict,
        vae: dict,
        source_video,
        extend_frames: int = 48,
        seed: int = 42,
        steps: int = 30,
        direction: str = "after",
        guider_config: dict | None = None,
    ):
        import numpy as np

        from ltx_core_mlx.components.patchifiers import (
            AudioPatchifier,
            VideoLatentPatchifier,
            compute_video_latent_shape,
        )
        from ltx_core_mlx.conditioning.types.latent_cond import (
            VideoConditionByLatentIndex,
            apply_conditioning,
            create_initial_state,
            noise_latent_state,
        )
        from ltx_core_mlx.model.transformer.model import X0Model
        from ltx_core_mlx.utils.memory import aggressive_cleanup
        from ltx_core_mlx.utils.positions import (
            compute_audio_positions,
            compute_audio_token_count,
            compute_video_positions,
        )
        from ltx_pipelines_mlx.scheduler import ltx2_schedule
        from ltx_pipelines_mlx.utils.samplers import guided_denoise_loop

        from .mlx_utils import mx_audio_to_torch, mx_video_frames_to_torch

        video_embeds = conditioning["video_embeds"]
        audio_embeds = conditioning["audio_embeds"]
        neg_video_embeds = conditioning.get("neg_video_embeds")
        neg_audio_embeds = conditioning.get("neg_audio_embeds")

        video_patchifier = VideoLatentPatchifier()
        audio_patchifier = AudioPatchifier()

        # Encode source video frames
        # source_video is (F_src, H, W, C) float32 [0, 1] -> need (1, C, F, H, W) [-1, 1]
        vae.load_encoder()
        src_np = source_video.cpu().numpy()
        src_np = np.transpose(src_np, (3, 0, 1, 2))  # (C, F, H, W)
        src_np = src_np[np.newaxis, ...]  # (1, C, F, H, W)
        src_np = src_np * 2.0 - 1.0  # [0,1] -> [-1,1]
        src_mx = mx.array(src_np).astype(mx.bfloat16)

        source_latent = vae.encoder.encode(src_mx)
        _materialize(source_latent)
        vae.unload_encoder()

        # Compute dimensions for extended video
        src_frames = source_video.shape[0]
        total_frames = src_frames + extend_frames
        _, H_src, W_src, _ = source_video.shape
        height, width = H_src, W_src

        F, H, W = compute_video_latent_shape(total_frames, height, width)
        video_shape = (1, F * H * W, 128)
        audio_T = compute_audio_token_count(total_frames)
        audio_shape = (1, audio_T, 128)

        video_positions = compute_video_positions(F, H, W)
        audio_positions = compute_audio_positions(audio_T)

        video_state = create_initial_state(video_shape, seed, positions=video_positions)
        audio_state = create_initial_state(audio_shape, seed + 1, positions=audio_positions)

        # Apply conditioning from source frames
        source_tokens = source_latent.transpose(0, 2, 3, 4, 1).reshape(1, -1, 128)
        F_src_latent = source_latent.shape[2]

        if direction == "after":
            frame_indices = list(range(F_src_latent))
        else:
            frame_indices = list(range(F - F_src_latent, F))

        condition = VideoConditionByLatentIndex(
            frame_indices=frame_indices,
            clean_latent=source_tokens,
            strength=1.0,
        )
        video_state = apply_conditioning(video_state, [condition], (F, H, W))
        video_state = noise_latent_state(video_state, sigma=1.0, seed=seed)
        audio_state = noise_latent_state(audio_state, sigma=1.0, seed=seed + 1)

        # Load transformer
        model.load()
        dit = model.transformer

        # Sigma schedule
        num_tokens = F * H * W
        sigmas = ltx2_schedule(steps, num_tokens=num_tokens)
        x0_model = X0Model(dit)

        # Guided denoise
        from ltx_core_mlx.components.guiders import (
            MultiModalGuiderParams,
            create_multimodal_guider_factory,
        )

        if guider_config is not None:
            video_params = guider_config["video_params"]
            audio_params = guider_config["audio_params"]
        else:
            video_params = MultiModalGuiderParams(cfg_scale=3.0, rescale_scale=0.7, modality_scale=3.0, stg_blocks=[28])
            audio_params = MultiModalGuiderParams(cfg_scale=7.0, rescale_scale=0.7, modality_scale=3.0, stg_blocks=[28])

        video_factory = create_multimodal_guider_factory(video_params, negative_context=neg_video_embeds)
        audio_factory = create_multimodal_guider_factory(audio_params, negative_context=neg_audio_embeds)

        output = guided_denoise_loop(
            model=x0_model,
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            video_guider_factory=video_factory,
            audio_guider_factory=audio_factory,
            sigmas=sigmas,
        )
        aggressive_cleanup()

        # Free transformer, load decoders
        model.unload()

        # Unpatchify
        video_latent = video_patchifier.unpatchify(output.video_latent, (F, H, W))
        audio_latent = audio_patchifier.unpatchify(output.audio_latent)

        # Decode
        vae.load_decoders()
        video_frames = vae.decoder.decode(video_latent)
        _materialize(video_frames)
        aggressive_cleanup()

        mel = vae.audio_decoder.decode(audio_latent)
        waveform = vae.vocoder(mel)
        _materialize(waveform)
        aggressive_cleanup()

        video_torch = mx_video_frames_to_torch(video_frames)
        audio_torch = mx_audio_to_torch(waveform, sample_rate=48000)

        return (video_torch, audio_torch)


class LTXVMLXICLoRASampler:
    """IC-LoRA conditioned video generation using MLX on Apple Silicon.

    Two-stage pipeline: Stage 1 at half-res with fused LoRA for control signal
    conditioning (depth, pose, edges), Stage 2 upscale + refine with clean model.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("LTXV_MLX_MODEL",),
                "conditioning": ("LTXV_MLX_CONDITIONING",),
                "vae": ("LTXV_MLX_VAE",),
                "reference_video": ("IMAGE",),
                "lora_path": ("STRING", {
                    "default": "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
                    "tooltip": "Path to IC-LoRA .safetensors or HuggingFace repo ID",
                }),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "reference_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "width": ("INT", {"default": 704, "min": 64, "max": 2048, "step": 32}),
                "height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 32}),
                "num_frames": ("INT", {"default": 97, "min": 1, "max": 257, "step": 8}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**31 - 1}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 100}),
            },
            "optional": {
                "image": ("IMAGE",),
                "stage2_steps": ("INT", {"default": 3, "min": 1, "max": 100}),
                "conditioning_attention_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("video_frames", "audio")
    FUNCTION = "sample"
    CATEGORY = "Lightricks/MLX"

    def sample(
        self, model, conditioning, vae, reference_video,
        lora_path="Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
        lora_strength=1.0, reference_strength=1.0,
        width=704, height=480, num_frames=97, seed=42, steps=8,
        image=None, stage2_steps=3, conditioning_attention_strength=1.0,
    ):
        import json
        import logging
        from pathlib import Path

        import numpy as np
        from PIL import Image as PILImage
        from safetensors import safe_open

        from ltx_core_mlx.components.patchifiers import AudioPatchifier, VideoLatentPatchifier, compute_video_latent_shape
        from ltx_core_mlx.conditioning.types.attention_strength_wrapper import ConditioningItemAttentionStrengthWrapper
        from ltx_core_mlx.conditioning.types.latent_cond import (
            LatentState, VideoConditionByLatentIndex, apply_conditioning, create_initial_state, noise_latent_state,
        )
        from ltx_core_mlx.conditioning.types.reference_video_cond import VideoConditionByReferenceLatent
        from ltx_core_mlx.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraStateDictWithStrength, SafetensorsStateDictLoader, StateDict, apply_loras
        from ltx_core_mlx.model.transformer.model import X0Model
        from ltx_core_mlx.model.upsampler import LatentUpsampler
        from ltx_core_mlx.utils.image import prepare_image_for_encoding
        from ltx_core_mlx.utils.memory import aggressive_cleanup
        from ltx_core_mlx.utils.positions import compute_audio_positions, compute_audio_token_count, compute_video_positions
        from ltx_core_mlx.utils.weights import apply_quantization, load_split_safetensors
        from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS, STAGE_2_SIGMAS
        from ltx_pipelines_mlx.utils.samplers import denoise_loop

        from .mlx_utils import mx_audio_to_torch, mx_video_frames_to_torch, torch_image_to_pil

        logger = logging.getLogger(__name__)
        model_dir = model.model_dir
        video_embeds = conditioning["video_embeds"]
        audio_embeds = conditioning["audio_embeds"]
        video_patchifier = VideoLatentPatchifier()
        audio_patchifier = AudioPatchifier()

        # --- Resolve LoRA path ---
        lora_local = Path(lora_path)
        if not lora_local.exists():
            from huggingface_hub import snapshot_download
            repo_dir = Path(snapshot_download(lora_path))
            safetensors_files = list(repo_dir.glob("*.safetensors"))
            if not safetensors_files:
                raise FileNotFoundError(f"No .safetensors in {repo_dir}")
            lora_local = safetensors_files[0]

        reference_downscale_factor = 1
        try:
            with safe_open(str(lora_local), framework="numpy") as f:
                metadata = f.metadata() or {}
                reference_downscale_factor = int(metadata.get("reference_downscale_factor", 1))
        except Exception as e:
            logger.warning(f"Failed to read LoRA metadata: {e}")

        # --- Load transformer and fuse LoRA ---
        model.load()
        dit = model.transformer
        if lora_strength != 0:
            import mlx.utils
            model_weights = dict(mlx.utils.tree_flatten(dit.parameters()))
            model_sd = StateDict(sd=model_weights, size=0, dtype=set())
            lora_sd = SafetensorsStateDictLoader().load(str(lora_local), sd_ops=LTXV_LORA_COMFY_RENAMING_MAP)
            fused_sd = apply_loras(model_sd, [LoraStateDictWithStrength(lora_sd, lora_strength)])
            apply_quantization(dit, fused_sd.sd)
            dit.load_weights(list(fused_sd.sd.items()))
            aggressive_cleanup()

        # --- Stage 1: Half-res with IC-LoRA ---
        vae.load_encoder()
        half_h, half_w = height // 2, width // 2
        F, H_half, W_half = compute_video_latent_shape(num_frames, half_h, half_w)
        audio_T = compute_audio_token_count(num_frames)
        video_positions_1 = compute_video_positions(F, H_half, W_half)
        audio_positions = compute_audio_positions(audio_T)
        video_state = create_initial_state((1, F * H_half * W_half, 128), seed, positions=video_positions_1)
        audio_state = create_initial_state((1, audio_T, 128), seed + 1, positions=audio_positions)

        # Optional I2V
        pil_image = None
        if image is not None:
            pil_image = torch_image_to_pil(image)
            img_t = prepare_image_for_encoding(pil_image, H_half * 32, W_half * 32)
            ref_lat = vae.encoder.encode(img_t[:, :, None, :, :])
            _materialize(ref_lat)
            video_state = apply_conditioning(
                video_state,
                [VideoConditionByLatentIndex(frame_indices=[0], clean_latent=ref_lat.transpose(0, 2, 3, 4, 1).reshape(1, -1, 128), strength=1.0)],
                (F, H_half, W_half),
            )

        # Encode reference video
        scale = reference_downscale_factor
        _, ref_H_lat, ref_W_lat = compute_video_latent_shape(num_frames, half_h // scale, half_w // scale)
        ref_h, ref_w = ref_H_lat * 32, ref_W_lat * 32
        ref_np = reference_video.cpu().numpy()
        ref_frames = []
        for i in range(min(ref_np.shape[0], num_frames)):
            fr = PILImage.fromarray((ref_np[i] * 255).astype(np.uint8))
            fr = fr.resize((ref_w, ref_h), PILImage.Resampling.LANCZOS)
            ref_frames.append(np.array(fr).astype(np.float32) / 255.0)
        ref_arr = np.transpose(np.stack(ref_frames), (3, 0, 1, 2))[np.newaxis] * 2.0 - 1.0
        encoded_ref = vae.encoder.encode(mx.array(ref_arr).astype(mx.bfloat16))
        _materialize(encoded_ref)

        ref_F, ref_H, ref_W = encoded_ref.shape[2], encoded_ref.shape[3], encoded_ref.shape[4]
        ic_cond = VideoConditionByReferenceLatent(
            reference_latent=encoded_ref.transpose(0, 2, 3, 4, 1).reshape(1, -1, 128),
            reference_positions=compute_video_positions(ref_F, ref_H, ref_W),
            downscale_factor=scale, strength=reference_strength,
        )
        if conditioning_attention_strength < 1.0:
            ic_cond = ConditioningItemAttentionStrengthWrapper(conditioning=ic_cond, attention_mask=conditioning_attention_strength)

        vae.unload_encoder()
        video_state = ic_cond.apply(video_state, (F, H_half, W_half))

        # Denoise Stage 1
        sigmas_1 = DISTILLED_SIGMAS[: steps + 1] if steps < len(DISTILLED_SIGMAS) else DISTILLED_SIGMAS
        output_1 = denoise_loop(
            model=X0Model(dit), video_state=video_state, audio_state=audio_state,
            video_text_embeds=video_embeds, audio_text_embeds=audio_embeds, sigmas=sigmas_1,
        )
        aggressive_cleanup()

        gen_tokens = output_1.video_latent[:, : F * H_half * W_half, :]
        video_half = video_patchifier.unpatchify(gen_tokens, (F, H_half, W_half))

        # --- Stage 2: Upscale + refine (clean model) ---
        vae.load_encoder()
        upsampler_name = "spatial_upscaler_x2_v1_1"
        config_path = model_dir / f"{upsampler_name}_config.json"
        weights_path = model_dir / f"{upsampler_name}.safetensors"
        upsampler = LatentUpsampler.from_config(json.loads(config_path.read_text()).get("config", {})) if config_path.exists() else LatentUpsampler()
        if weights_path.exists():
            upsampler.load_weights(list(load_split_safetensors(weights_path, prefix=f"{upsampler_name}.").items()))
        aggressive_cleanup()

        v = video_half.transpose(0, 2, 3, 4, 1)
        video_upscaled = vae.encoder.normalize_latent(
            upsampler(vae.encoder.denormalize_latent(v).transpose(0, 4, 1, 2, 3)).transpose(0, 2, 3, 4, 1)
        ).transpose(0, 4, 1, 2, 3)
        _materialize(video_upscaled)
        H_full, W_full = H_half * 2, W_half * 2

        conds_2 = []
        if pil_image is not None:
            img_t2 = prepare_image_for_encoding(pil_image, H_full * 32, W_full * 32)
            ref_lat2 = vae.encoder.encode(img_t2[:, :, None, :, :])
            conds_2.append(VideoConditionByLatentIndex(frame_indices=[0], clean_latent=ref_lat2.transpose(0, 2, 3, 4, 1).reshape(1, -1, 128), strength=1.0))

        del upsampler
        vae.unload_encoder()

        # Reload clean transformer (without LoRA)
        model.unload()
        model.load()

        video_tokens_up, _ = video_patchifier.patchify(video_upscaled)
        sigmas_2 = STAGE_2_SIGMAS[: stage2_steps + 1] if stage2_steps < len(STAGE_2_SIGMAS) else STAGE_2_SIGMAS
        s0 = sigmas_2[0]
        mx.random.seed(seed + 2)
        noisy = mx.random.normal(video_tokens_up.shape).astype(mx.bfloat16) * s0 + video_tokens_up * (1.0 - s0)

        vs2 = LatentState(latent=noisy, clean_latent=video_tokens_up,
                          denoise_mask=mx.ones((1, video_tokens_up.shape[1], 1), dtype=mx.bfloat16),
                          positions=compute_video_positions(F, H_full, W_full))
        if conds_2:
            vs2 = apply_conditioning(vs2, conds_2, (F, H_full, W_full))

        at1 = output_1.audio_latent
        as2 = LatentState(latent=at1, clean_latent=at1,
                          denoise_mask=mx.ones((1, at1.shape[1], 1), dtype=at1.dtype), positions=audio_positions)
        as2 = noise_latent_state(as2, sigma=s0, seed=seed + 2)

        output_2 = denoise_loop(
            model=X0Model(model.transformer), video_state=vs2, audio_state=as2,
            video_text_embeds=video_embeds, audio_text_embeds=audio_embeds, sigmas=sigmas_2,
        )
        model.unload()

        video_latent = video_patchifier.unpatchify(output_2.video_latent, (F, H_full, W_full))
        audio_latent = audio_patchifier.unpatchify(output_2.audio_latent)

        vae.load_decoders()
        video_frames = vae.decoder.decode(video_latent)
        _materialize(video_frames)
        aggressive_cleanup()
        mel = vae.audio_decoder.decode(audio_latent)
        waveform = vae.vocoder(mel)
        _materialize(waveform)
        aggressive_cleanup()

        return (mx_video_frames_to_torch(video_frames), mx_audio_to_torch(waveform, sample_rate=48000))
