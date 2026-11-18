"""Fit PCA on all spatial vectors at clean step 0; reuse it everywhere."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA


CLEAN_COLOR = "#1668A8"
PROTECTED_COLOR = "#DC6739"


def flatten_spatial(latents: np.ndarray) -> np.ndarray:
    """Convert [steps, channels, height, width] into [steps, positions, channels]."""
    if latents.ndim != 4:
        raise ValueError(f"Expected [steps, channels, H, W], got {latents.shape}")
    steps, channels, height, width = latents.shape
    return latents.transpose(0, 2, 3, 1).reshape(steps, height * width, channels)


def project_data(clean: np.ndarray, protected: np.ndarray):
    if clean.shape != protected.shape:
        raise ValueError(f"Both trajectories need the same shape: {clean.shape}, {protected.shape}")
    if clean.shape[0] < 2 or clean.shape[1] < 2:
        raise ValueError("Need at least one denoising step and two latent channels")
    x_clean = flatten_spatial(clean).astype(np.float64)
    x_protected = flatten_spatial(protected).astype(np.float64)
    pca = PCA(n_components=2, svd_solver="full")
    pca.fit(x_clean[0])
    p_clean = np.stack([pca.transform(x) for x in x_clean])
    p_protected = np.stack([pca.transform(x) for x in x_protected])
    return pca, x_clean, x_protected, p_clean, p_protected


def spatial_grid(height: int, width: int, count: int = 6) -> np.ndarray:
    """Stable spatial anchors, shared between both runs and every timestep."""
    ys = np.unique(np.linspace(0, height - 1, count).astype(int))
    xs = np.unique(np.linspace(0, width - 1, count).astype(int))
    return np.array([y * width + x for y in ys for x in xs], dtype=int)


def plot_bounds(a: np.ndarray, b: np.ndarray):
    combined = np.concatenate([a.reshape(-1, 2), b.reshape(-1, 2)], axis=0)
    lower = combined.min(axis=0)
    upper = combined.max(axis=0)
    margin = np.maximum((upper - lower) * 0.06, 0.01)
    return lower - margin, upper + margin


def draw_step(ax, clean, protected, step, anchors, bounds, explained, times):
    ax.clear()
    # All spatial locations are plotted. Anchor paths are an additional overlay.
    ax.scatter(clean[step, :, 0], clean[step, :, 1], s=5, alpha=0.22,
               c=CLEAN_COLOR, linewidths=0, rasterized=True, label="Original")
    ax.scatter(protected[step, :, 0], protected[step, :, 1], s=5, alpha=0.22,
               c=PROTECTED_COLOR, linewidths=0, rasterized=True, label="Shot noise")
    if step:
        for points, color in [(clean, CLEAN_COLOR), (protected, PROTECTED_COLOR)]:
            for anchor in anchors:
                trajectory = points[: step + 1, anchor]
                ax.plot(trajectory[:, 0], trajectory[:, 1], c=color, lw=0.75, alpha=0.42)
            ax.scatter(points[step, anchors, 0], points[step, anchors, 1],
                       c=color, s=11, marker="o", alpha=0.9, zorder=4)
    ax.set_xlim(bounds[0][0], bounds[1][0])
    ax.set_ylim(bounds[0][1], bounds[1][1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"PC1 (fit on original x₀; {explained[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% variance)")
    label = "before first denoising update" if step == 0 else f"after update {step}"
    # One five-point star per image: the mean of that image's projected embeddings.
    for points, color, name in (
        (clean, CLEAN_COLOR, "Original mean"),
        (protected, PROTECTED_COLOR, "Shot noise mean"),
    ):
        center = points[step].mean(axis=0)
        ax.scatter(
            center[0], center[1], marker="*", s=220, c=color,
            edgecolors="black", linewidths=0.7, zorder=6, label=name,
        )
    ax.set_title(f"Same PCA basis · {label} / {len(clean) - 1}\n{times[step]}")
    ax.legend(loc="upper right", markerscale=1.4, frameon=True)


def make_plots(data_path: Path, output_dir: Path, fps: int = 3) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(data_path) as archive:
        clean = archive["original"]
        protected = archive["protected"]
        conditioning_original = archive["conditioning_original"]
        conditioning_protected = archive["conditioning_protected"]
        timesteps = archive["timesteps"]

    pca, x_clean, x_protected, p_clean, p_protected = project_data(clean, protected)
    steps, channels, height, width = clean.shape
    anchors = spatial_grid(height, width)
    bounds = plot_bounds(p_clean, p_protected)
    explained = pca.explained_variance_ratio_

    np.savez_compressed(
        output_dir / "pca_basis.npz",
        mean=pca.mean_, components=pca.components_,
        explained_variance_ratio=explained, singular_values=pca.singular_values_,
    )
    np.savez_compressed(
        output_dir / "projected_points.npz", original=p_clean.astype(np.float32),
        protected=p_protected.astype(np.float32), spatial_shape=np.array([height, width]),
    )

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams.update({"pdf.fonttype": 42, "figure.facecolor": "white"})
    labels = ["Start: before first scheduler step"] + [
        f"Scheduler timestep used: {float(t):g}" for t in timesteps
    ]
    if len(labels) != steps:
        raise ValueError(f"Need {steps - 1} update timesteps, received {len(timesteps)}")

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    with PdfPages(output_dir / "pca_all_steps.pdf") as pdf:
        with PillowWriter(fps=fps).saving(fig, str(output_dir / "pca_animation.gif"), dpi=110) as gif:
            for step in range(steps):
                draw_step(ax, p_clean, p_protected, step, anchors, bounds, explained, labels)
                pdf.savefig(fig, dpi=180)
                gif.grab_frame()
                if step in (0, steps - 1):
                    fig.savefig(output_dir / ("pca_step_0.pdf" if step == 0 else "pca_step_final.pdf"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    for data, color, name in [
        (p_clean, CLEAN_COLOR, "Original"), (p_protected, PROTECTED_COLOR, "Shot noise")
    ]:
        for anchor in anchors:
            ax.plot(data[:, anchor, 0], data[:, anchor, 1], color=color, alpha=0.38, lw=0.9)
        ax.scatter(data[0, anchors, 0], data[0, anchors, 1], marker="o",
                   color=color, s=18, alpha=0.65, label=f"{name}: start")
        ax.scatter(data[-1, anchors, 0], data[-1, anchors, 1], marker="x",
                   color=color, s=24, alpha=0.9, label=f"{name}: end")
        center = data[-1].mean(axis=0)
        ax.scatter(
            center[0], center[1], marker="*", s=220, c=color,
            edgecolors="black", linewidths=0.7, zorder=6, label=f"{name} mean",
        )
    ax.set(xlim=(bounds[0][0], bounds[1][0]), ylim=(bounds[0][1], bounds[1][1]),
           xlabel="PC1", ylabel="PC2", title="Trajectories of matched spatial coordinates\n"
           "One PCA fitted only on original step 0")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right")
    fig.savefig(output_dir / "pca_anchor_trajectories.pdf")
    plt.close(fig)

    # Per-position full-channel distances, not distances between image means.
    difference = x_clean - x_protected
    rms_difference = np.sqrt(np.mean(np.sum(difference ** 2, axis=2), axis=1))
    rms_clean_motion = np.sqrt(np.mean(np.sum((x_clean - x_clean[0]) ** 2, axis=2), axis=1))
    rms_protected_motion = np.sqrt(
        np.mean(np.sum((x_protected - x_protected[0]) ** 2, axis=2), axis=1)
    )
    projected_difference = p_clean - p_protected
    rms_projected = np.sqrt(np.mean(np.sum(projected_difference ** 2, axis=2), axis=1))
    np.savetxt(
        output_dir / "distance_by_step.csv",
        np.column_stack([np.arange(steps), rms_difference, rms_projected,
                         rms_clean_motion, rms_protected_motion]),
        delimiter=",", comments="",
        header="state_index,rms_clean_vs_protected_full_channels,"
               "rms_clean_vs_protected_pca2d,rms_original_from_start,"
               "rms_protected_from_start",
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    step_axis = np.arange(steps)
    axes[0].plot(step_axis, rms_difference, color="black", lw=2, label="Full channels")
    axes[0].plot(step_axis, rms_projected, color="#8290A6", lw=2,
                 linestyle="--", label="Only PC1–PC2")
    axes[0].set(xlabel="State index (0 is pre-denoising)", ylabel="RMS Euclidean distance",
                title="Original vs protected, matching spatial positions")
    axes[0].legend()
    axes[1].plot(step_axis, rms_clean_motion, color=CLEAN_COLOR, label="Original", lw=2)
    axes[1].plot(step_axis, rms_protected_motion, color=PROTECTED_COLOR, label="Shot noise", lw=2)
    axes[1].set(xlabel="State index", ylabel="RMS Euclidean distance from own x₀",
                title="Movement in full latent-channel space")
    axes[1].legend()
    fig.savefig(output_dir / "latent_distance_metrics.pdf")
    plt.close(fig)

    if conditioning_original.size and conditioning_protected.size:
        if conditioning_original.shape != (channels, height, width):
            raise ValueError("Conditioning latent has an unexpected shape")
        z0 = conditioning_original.transpose(1, 2, 0).reshape(-1, channels)
        z1 = conditioning_protected.transpose(1, 2, 0).reshape(-1, channels)
        q0, q1 = pca.transform(z0.astype(np.float64)), pca.transform(z1.astype(np.float64))
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        ax.scatter(q0[:, 0], q0[:, 1], c=CLEAN_COLOR, s=5, alpha=0.25,
                   linewidths=0, rasterized=True, label="Original image")
        ax.scatter(q1[:, 0], q1[:, 1], c=PROTECTED_COLOR, s=5, alpha=0.25,
                   linewidths=0, rasterized=True, label="Protected image")
        for points, color, name in (
            (q0, CLEAN_COLOR, "Original mean"),
            (q1, PROTECTED_COLOR, "Shot noise mean"),
        ):
            center = points.mean(axis=0)
            ax.scatter(
                center[0], center[1], marker="*", s=220, c=color,
                edgecolors="black", linewidths=0.7, zorder=6, label=name,
            )
        ax.set(xlabel="PC1 (same original x₀ basis)", ylabel="PC2",
               title="VAE image latents projected into the denoising PCA basis\n"
                     "Not part of the denoising trajectory")
        ax.legend(markerscale=3)
        fig.savefig(output_dir / "pca_conditioning_image_latents.pdf")
        plt.close(fig)

    print(f"PCA fitted on all {height * width} original x₀ positions; {channels} channels per point")
    print(f"Variance explained by PC1 + PC2: {100 * explained.sum():.2f}%")
    print(f"Wrote plots and arrays to {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("test_outputs/latents_full.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("test_outputs"))
    parser.add_argument("--fps", type=int, default=3)
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    make_plots(args.data, args.output_dir, args.fps)


if __name__ == "__main__":
    main()
