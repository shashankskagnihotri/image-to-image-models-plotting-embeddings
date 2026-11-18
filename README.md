# Latent PCA trajectories

Paired image-edit experiment. The same prompt, scheduler settings, and seed are applied to `sample_images/original.png` and `sample_images/shot_noise.png`. One PCA is fitted on image 1's step-0 spatial latent vectors (`x_0[:, y, x]`, one C-dimensional observation per spatial site) and reused for every later denoising state and for image 2.

## Environment

The conda environment lives inside this directory. From this directory:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -p .conda python=3.11 pip -y
conda activate ./.conda
python -m pip install -r requirements.txt
```

`requirements.txt` installs PyTorch 2.14.1 and torchvision 0.29.1 from the CUDA 13.0 wheel index (`cu130`), which matches the NVIDIA driver on the RTX 5000 Ada used for the recorded run. The remaining packages are Diffusers, Transformers, Accelerate, and the plotting stack. Activate the same environment before every command below:

```bash
conda activate ./.conda
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

## Sample images

`sample_images/original.png` and `sample_images/shot_noise.png` are already prepared (512×512, Poisson shot noise, seed 2026). Regenerate them with:

```bash
python prepare_examples.py
```

## Run

The shot-noise image is Poisson photon noise applied to `sample_images/original.png` (peak 35 photons per channel, seed 2026). Both runs use the prompt `make the mountains into a desert and add a properller aircraft flying right to left in the sky leaving a cloud trail`.

InstructPix2Pix uses the pipeline defaults: 100 steps, guidance scale 7.5, image guidance scale 1.5.

```bash
python run_experiment.py --model ip2p \
    --model-path models/instruct-pix2pix \
    --image-1 sample_images/original.png \
    --image-2 sample_images/shot_noise.png \
    --output-dir test_outputs/ip2p
```

Stable Diffusion 3.5 Large uses the model-card settings of 28 steps and guidance scale 3.5, and the image-to-image pipeline's default strength of 0.6. The VAE stays in float32. `--offload auto` moves one module onto the GPU at a time when the weights do not fit.

```bash
python run_experiment.py --model sd35 \
    --model-path /ceph/sagnihot/cache/huggingface/hub/models--stabilityai--stable-diffusion-3.5-large/snapshots/ceddf0a7fdf2064ea28e2213e3b84e4afa170a0f \
    --image-1 sample_images/original.png \
    --image-2 sample_images/shot_noise.png \
    --output-dir test_outputs/sd35
```

Rebuild the figures from a saved trajectory without loading the model:

```bash
python plot_trajectories.py \
    --data test_outputs/ip2p/latents_full.npz \
    --output-dir test_outputs/ip2p
```

## Outputs

Each `test_outputs/<model>/` directory contains:

- `latents_full.npz` — states `[S, C, H, W]` for both images, VAE condition arrays, timesteps
- `pca_basis.npz` — mean `[C]`, components `[2, C]`, explained variance
- `projected_points.npz` — both trajectories projected with that one basis, `[S, H*W, 2]`
- `metadata.json` — run settings and tensor shapes
- `edited_original.png`, `edited_protected.png`
- `pca_animation.gif`, `pca_all_steps.pdf`, `pca_step_0.pdf`, `pca_step_final.pdf`
- `pca_anchor_trajectories.pdf`, `pca_conditioning_image_latents.pdf`
- `latent_distance_metrics.pdf`, `distance_by_step.csv`

State 0 is the latent after noise preparation and before the first scheduler update. It is not diffusion time `t = 0`.
