"""Record paired image-editing latent trajectories without altering model updates."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from diffusers import (
    StableDiffusion3Img2ImgPipeline,
    StableDiffusionInstructPix2PixPipeline,
)

from plot_trajectories import make_plots


class RecordInitialLatents:
    """Capture x₀ at its actual creation, before the scheduler's first update."""

    def prepare_latents(self, *args, **kwargs):
        initial = super().prepare_latents(*args, **kwargs)
        self.initial_latents = initial.detach().float().cpu().clone()
        return initial


class RecordedSD35(RecordInitialLatents, StableDiffusion3Img2ImgPipeline):
    """Image-to-image editing with a flow-matching SD3 / SD3.5 scheduler."""


class RecordedInstructPix2Pix(RecordInitialLatents, StableDiffusionInstructPix2PixPipeline):
    """Diffusion editor with an additional VAE-encoded image condition."""

    def prepare_image_latents(self, *args, **kwargs):
        image_latents = super().prepare_image_latents(*args, **kwargs)
        # The first entry is the actual image condition (even under 3-way CFG).
        self.conditioning_latents = image_latents[:1].detach().float().cpu().clone()
        return image_latents


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(choice)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def timestep_to_float(timestep) -> float:
    """Scheduler callbacks pass a tensor; keep one Python float per update."""
    if isinstance(timestep, torch.Tensor):
        return float(timestep.detach().to(dtype=torch.float32).flatten()[0].cpu().item())
    return float(timestep)


def parameter_nbytes(module) -> int:
    if module is None:
        return 0
    return sum(parameter.numel() * parameter.element_size() for parameter in module.parameters())


def should_cpu_offload(pipe, device: torch.device) -> bool:
    """Offload when weights plus a transformer activation reserve exceed free GPU memory."""
    if device.type != "cuda":
        return False
    free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
    names = ("transformer", "unet", "text_encoder", "text_encoder_2", "text_encoder_3", "vae")
    weight_bytes = sum(parameter_nbytes(getattr(pipe, name, None)) for name in names)
    return weight_bytes + 8 * (1024**3) > free_bytes


def resolve_dtype(choice: str, device: torch.device) -> torch.dtype:
    if choice == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[choice]
    if device.type == "cpu" and dtype != torch.float32:
        raise ValueError("Use --dtype fp32 for CPU inference")
    return dtype


@torch.inference_mode()
def encode_sd35_image(pipe: RecordedSD35, image: Image.Image, device: torch.device) -> torch.Tensor:
    """Use a deterministic VAE posterior mode as the starting image code.

    In contrast to the pipeline's default stochastic VAE posterior sample, this
    mode makes the only image-to-image perturbation the controlled shot noise.
    The returned code is already in the SD3 model's scale/shift convention, which
    is what StableDiffusion3Img2ImgPipeline.prepare_latents expects when the
    image tensor already has ``vae.config.latent_channels`` channels. The VAE
    itself stays in float32; only the returned code is cast to the transformer dtype.
    """
    pixels = pipe.image_processor.preprocess(image)
    pixels = pixels.to(device=device, dtype=torch.float32)
    mode = pipe.vae.encode(pixels).latent_dist.mode()
    shift = pipe.vae.config.shift_factor
    scale = pipe.vae.config.scaling_factor
    return ((mode - shift) * scale).to(dtype=pipe.transformer.dtype)


@torch.inference_mode()
def decode_sd35_result(
    pipe: RecordedSD35, scaled_latents: torch.Tensor, device: torch.device
) -> Image.Image:
    """Decode SD3/SD3.5 latents with the float32 VAE."""
    scaled_latents = scaled_latents.to(device=device, dtype=torch.float32)
    latents = (scaled_latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    rgb = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(rgb, output_type="pil")[0]


@torch.inference_mode()
def record_one(pipe, image: Image.Image, args, device: torch.device):
    pipe.initial_latents = None
    pipe.conditioning_latents = None
    records = []
    timesteps = []

    def after_update(_pipeline, index, timestep, callback_kwargs):
        # This callback occurs AFTER scheduler.step(), never before it.
        records.append(callback_kwargs["latents"][0].detach().float().cpu().clone())
        timesteps.append(timestep_to_float(timestep))
        return callback_kwargs  # preserve the pipeline's model state unchanged

    generator = torch.Generator(device=device).manual_seed(args.seed)
    settings = dict(
        prompt=args.prompt,
        image=image,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        callback_on_step_end=after_update,
        callback_on_step_end_tensor_inputs=["latents"],
        output_type="pil",
    )

    if args.model == "sd35":
        # The SD3 img2img pipeline accepts already encoded image latents.
        # It then mixes them with a seeded Gaussian using scheduler.scale_noise().
        conditioning = encode_sd35_image(pipe, image, device)
        # Copy the clean code before prepare_latents mixes it with noise.
        conditioning_array = conditioning[0].detach().float().cpu().numpy()
        settings["image"] = conditioning
        # The flow scheduler's dynamic shift depends on the TRUE image size.
        # Pipeline defaults to 1024 otherwise, even with a 512x512 latent input.
        settings["height"] = image.height
        settings["width"] = image.width
        settings["strength"] = args.strength
        settings["output_type"] = "latent"  # decoded in fp32 immediately below
    else:
        conditioning = None  # captured from the actual InstructPix2Pix pipeline
        conditioning_array = None
        settings["image_guidance_scale"] = args.image_guidance_scale

    result = pipe(**settings)
    if pipe.initial_latents is None:
        raise RuntimeError("The pipeline did not invoke prepare_latents")
    if len(records) == 0:
        raise RuntimeError("The pipeline did not execute any denoising updates")

    initial = pipe.initial_latents[0]
    trajectory = torch.stack([initial, *records]).numpy()
    if conditioning_array is None:
        if pipe.conditioning_latents is None:
            raise RuntimeError("InstructPix2Pix image-condition latents were not captured")
        conditioning_array = pipe.conditioning_latents[0].numpy()
    edited_latents = result.images
    edited = (
        decode_sd35_result(pipe, edited_latents[:1], device)
        if args.model == "sd35" else edited_latents[0]
    )
    return trajectory, conditioning_array, np.asarray(timesteps), edited


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["ip2p", "sd35"], default="ip2p")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Local Diffusers directory or Hugging Face model ID")
    parser.add_argument("--image-1", type=Path, default=Path("sample_images/original.png"))
    parser.add_argument("--image-2", type=Path, default=Path("sample_images/shot_noise.png"))
    parser.add_argument("--output-dir", type=Path, default=Path("test_outputs"))
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "make the mountains into a desert and add a properller aircraft "
            "flying right to left in the sky leaving a cloud trail"
        ),
    )
    parser.add_argument("--steps", type=int, default=None,
                        help="IP2P default 100; SD3.5 Large default 28")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--strength", type=float, default=0.6,
                        help="SD3.5 img2img default: fraction of the noising schedule")
    parser.add_argument("--guidance-scale", type=float, default=None,
                        help="IP2P default 7.5; SD3.5 Large model-card default 3.5")
    parser.add_argument("--image-guidance-scale", type=float, default=1.5,
                        help="InstructPix2Pix pipeline default")
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    parser.add_argument("--dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    parser.add_argument(
        "--offload",
        choices=["auto", "on", "off"],
        default="auto",
        help="Move one SD3/IP2P module to the GPU at a time when the checkpoint does not fit",
    )
    parser.add_argument("--fps", type=int, default=3)
    args = parser.parse_args()

    if args.model == "ip2p":
        if args.steps is None:
            args.steps = 100
        if args.guidance_scale is None:
            args.guidance_scale = 7.5
    else:
        if args.steps is None:
            args.steps = 28
        if args.guidance_scale is None:
            args.guidance_scale = 3.5
    if args.steps < 2:
        parser.error("--steps must be at least 2")
    if not (0 < args.strength <= 1):
        parser.error("--strength must be in (0, 1]")
    if args.image_guidance_scale < 1:
        parser.error("--image-guidance-scale must be >= 1")
    if args.fps <= 0:
        parser.error("--fps must be positive")

    image_1 = Image.open(args.image_1).convert("RGB")
    image_2 = Image.open(args.image_2).convert("RGB")
    if image_1.size != image_2.size:
        raise ValueError("Both image files must have exactly the same image dimensions")
    if any(v % 16 for v in image_1.size):
        raise ValueError("Image dimensions must be multiples of 16")

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    models = {
        "ip2p": (RecordedInstructPix2Pix, "timbrooks/instruct-pix2pix"),
        "sd35": (RecordedSD35, "stabilityai/stable-diffusion-3.5-medium"),
    }
    pipeline_class, default_model_path = models[args.model]
    model_path = args.model_path or default_model_path

    print(f"Loading {model_path} ({args.model}) on {device}, {dtype} ...")
    from_pretrained_kwargs = {"dtype": dtype}
    if args.model == "sd35":
        # Official SD3.5 guidance: encode and decode with a float32 VAE.
        from diffusers import AutoencoderKL

        from_pretrained_kwargs["vae"] = AutoencoderKL.from_pretrained(
            model_path, subfolder="vae", dtype=torch.float32
        )
    pipe = pipeline_class.from_pretrained(model_path, **from_pretrained_kwargs)
    use_offload = args.offload == "on" or (args.offload == "auto" and should_cpu_offload(pipe, device))
    if use_offload:
        if device.type != "cuda":
            parser.error("--offload on requires a CUDA device")
        gpu_id = 0 if device.index is None else device.index
        print("Enabling model CPU offload so the checkpoint fits in GPU memory.")
        pipe.enable_model_cpu_offload(gpu_id=gpu_id)
    else:
        pipe.to(device)
    pipe.set_progress_bar_config(disable=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Recording original image ...")
    original, cond_original, ts_original, edit_original = record_one(pipe, image_1, args, device)
    print("Recording shot-noise image (same prompt, fresh generator with same seed) ...")
    protected, cond_protected, ts_protected, edit_protected = record_one(
        pipe, image_2, args, device
    )
    if original.shape != protected.shape:
        raise RuntimeError(f"Trajectory shapes differ: {original.shape}, {protected.shape}")
    if not np.array_equal(ts_original, ts_protected):
        raise RuntimeError("Scheduler timestep sequences do not match")

    np.savez_compressed(
        args.output_dir / "latents_full.npz",
        original=original,
        protected=protected,
        conditioning_original=cond_original,
        conditioning_protected=cond_protected,
        timesteps=ts_original,
    )
    edit_original.save(args.output_dir / "edited_original.png")
    edit_protected.save(args.output_dir / "edited_protected.png")

    metadata = {
        "model_type": args.model,
        "model_path": model_path,
        "image_1": str(args.image_1),
        "image_2": str(args.image_2),
        "prompt": args.prompt,
        "num_inference_steps_requested": args.steps,
        "num_actual_denoising_updates": int(original.shape[0] - 1),
        "strength_sd35_only": args.strength if args.model == "sd35" else None,
        "guidance_scale": args.guidance_scale,
        "settings_source": (
            "InstructPix2Pix pipeline defaults: 100 steps, guidance_scale 7.5, "
            "image_guidance_scale 1.5."
            if args.model == "ip2p" else
            "SD3.5 Large model card: 28 steps, guidance_scale 3.5. "
            "SD3 img2img pipeline default strength 0.6."
        ),
        "image_guidance_scale_ip2p_only": (
            args.image_guidance_scale if args.model == "ip2p" else None
        ),
        "seed_shared_between_runs": args.seed,
        "compute_dtype": str(dtype).replace("torch.", ""),
        "model_cpu_offload": use_offload,
        "pca_protocol": (
            "One PCA is fitted on image 1 step-0 spatial latent vectors and reused "
            "for every denoising state and for image 2."
        ),
        "full_latent_tensor_shape_per_image": list(original.shape),
        "conditioning_image_latent_shape_per_image": list(cond_original.shape),
        "timesteps_used_for_updates": ts_original.tolist(),
        "step_zero_definition": "after image/noise preparation, before first denoiser call",
        "sampling_note": (
            "SD3.5 uses the VAE posterior MODE (deterministic), then official "
            "image-to-image flow scheduler noise mixing with the same seeded noise."
            if args.model == "sd35" else
            "InstructPix2Pix uses the official VAE MODE image-conditioning path; "
            "denoising state x0 is independently sampled Gaussian noise."
        ),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Raw array per image: {original.shape} [states, channels, height, width]")
    print(f"Actual denoising updates: {original.shape[0] - 1}")
    if args.model == "ip2p":
        print("The two x0 denoising states should match exactly: both start from seeded noise.")
        print(f"Maximum absolute x0 difference: {np.max(np.abs(original[0] - protected[0])):.5g}")
    make_plots(args.output_dir / "latents_full.npz", args.output_dir, args.fps)


if __name__ == "__main__":
    main()
