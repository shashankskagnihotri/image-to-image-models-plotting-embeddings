"""Download the Diffusers example photograph and add physically motivated shot noise."""

import argparse
from pathlib import Path
from urllib.request import urlopen
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps


SOURCE_URL = (
    "https://huggingface.co/datasets/diffusers/diffusers-images-docs/"
    "resolve/main/mountain.png"
)


def add_shot_noise(image: Image.Image, peak_photons: float, seed: int) -> Image.Image:
    """Apply independent Poisson photon-count noise in linear-light RGB."""
    if peak_photons <= 0:
        raise ValueError("peak_photons must be positive")
    rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    rng = np.random.default_rng(seed)
    counts = rng.poisson(linear * peak_photons)
    noisy_linear = np.clip(counts / peak_photons, 0.0, 1.0)
    srgb = np.where(
        noisy_linear <= 0.0031308,
        12.92 * noisy_linear,
        1.055 * np.power(noisy_linear, 1.0 / 2.4) - 0.055,
    )
    return Image.fromarray(np.rint(srgb * 255).astype(np.uint8), "RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-url", default=SOURCE_URL)
    parser.add_argument("--size", type=int, default=512, help="Square image size, divisible by 16")
    parser.add_argument("--peak-photons", type=float, default=35.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=Path("sample_images"))
    args = parser.parse_args()

    if args.size <= 0 or args.size % 16:
        parser.error("--size must be a positive multiple of 16")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading image from {args.source_url}")
    with urlopen(args.source_url, timeout=60) as response:
        photo = Image.open(BytesIO(response.read()))
        photo.load()
    photo = ImageOps.exif_transpose(photo).convert("RGB")
    photo = ImageOps.fit(photo, (args.size, args.size), Image.Resampling.LANCZOS)
    protected = add_shot_noise(photo, args.peak_photons, args.seed)

    photo.save(args.output_dir / "original.png")
    protected.save(args.output_dir / "shot_noise.png")
    (args.output_dir / "SOURCE.txt").write_text(
        f"Original internet image: {args.source_url}\n"
        f"Preparation: EXIF transpose, center crop, resize to {args.size} x {args.size}.\n"
        f"Variant: independent Poisson photon counts per linear-light RGB channel.\n"
        f"Peak photons per channel: {args.peak_photons:g}\n"
        f"Random seed: {args.seed}\n"
        "The noise is illustrative and is not a validated editing protection.\n",
        encoding="utf-8",
    )
    print(f"Saved two images to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
