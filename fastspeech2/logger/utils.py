import io

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
from torchvision.transforms import ToTensor

plt.switch_backend("agg")


def plot_images(imgs, config):
    """
    Combine several images into one figure.
    """
    names = config.writer.names
    figsize = config.writer.figsize
    fig, axes = plt.subplots(1, len(names), figsize=figsize)
    for i in range(len(names)):
        img = imgs[i].permute(1, 2, 0)
        axes[i].imshow(img)
        axes[i].set_title(names[i])
        axes[i].axis("off")
    buf = io.BytesIO()
    fig.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    image = ToTensor()(PIL.Image.open(buf))
    plt.close()

    return image


def plot_spectrogram(spectrogram, title=None, ylabel="Mel Frequency", figsize=(10, 4)):
    """
    Plot a single spectrogram.
    """
    if isinstance(spectrogram, torch.Tensor):
        spectrogram = spectrogram.detach().cpu().numpy()

    if spectrogram.ndim == 3:
        spectrogram = spectrogram.squeeze(0)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        spectrogram, aspect="auto", origin="lower", interpolation="none", cmap="viridis"
    )
    ax.set_xlabel("Time Frame")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    plt.colorbar(im, ax=ax, format="%+2.0f")
    plt.tight_layout()

    return fig


def plot_spectrogram_comparison(
    pred_spec, target_spec, title_prefix="", figsize=(15, 8)
):
    """
    Plot predicted and target spectrograms side by side with difference.
    """
    if isinstance(pred_spec, torch.Tensor):
        pred_spec = pred_spec.detach().cpu().numpy()
    if isinstance(target_spec, torch.Tensor):
        target_spec = target_spec.detach().cpu().numpy()

    if pred_spec.ndim == 3:
        pred_spec = pred_spec.squeeze(0)
    if target_spec.ndim == 3:
        target_spec = target_spec.squeeze(0)

    diff = pred_spec - target_spec
    fig, axes = plt.subplots(3, 1, figsize=figsize)

    im1 = axes[0].imshow(
        target_spec, aspect="auto", origin="lower", interpolation="none", cmap="viridis"
    )
    axes[0].set_title(f"{title_prefix}Target Mel-Spectrogram")
    axes[0].set_ylabel("Mel Frequency")
    plt.colorbar(im1, ax=axes[0], format="%+2.0f")

    im2 = axes[1].imshow(
        pred_spec, aspect="auto", origin="lower", interpolation="none", cmap="viridis"
    )
    axes[1].set_title(f"{title_prefix}Predicted Mel-Spectrogram")
    axes[1].set_ylabel("Mel Frequency")
    plt.colorbar(im2, ax=axes[1], format="%+2.0f")

    im3 = axes[2].imshow(
        diff,
        aspect="auto",
        origin="lower",
        interpolation="none",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
    )
    axes[2].set_title(f"{title_prefix}Difference (Predicted - Target)")
    axes[2].set_xlabel("Time Frame")
    axes[2].set_ylabel("Mel Frequency")
    plt.colorbar(im3, ax=axes[2], format="%+2.2f")

    plt.tight_layout()

    return fig


def plot_pitch_comparison(pred_pitch, target_pitch, title_prefix="", figsize=(12, 4)):
    """
    Plot predicted and target pitch contours.
    """
    if isinstance(pred_pitch, torch.Tensor):
        pred_pitch = pred_pitch.detach().cpu().numpy()
    if isinstance(target_pitch, torch.Tensor):
        target_pitch = target_pitch.detach().cpu().numpy()

    pred_pitch = pred_pitch.flatten()
    target_pitch = target_pitch.flatten()

    fig, ax = plt.subplots(figsize=figsize)

    time_steps = np.arange(len(target_pitch))
    ax.plot(time_steps, target_pitch, label="Target", linewidth=2, alpha=0.7)
    ax.plot(time_steps, pred_pitch, label="Predicted", linewidth=2, alpha=0.7)

    ax.set_xlabel("Time Frame")
    ax.set_ylabel("Pitch (Hz)")
    ax.set_title(f"{title_prefix}Pitch Contour Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def plot_energy_comparison(
    pred_energy, target_energy, title_prefix="", figsize=(12, 4)
):
    """
    Plot predicted and target energy contours.
    """
    if isinstance(pred_energy, torch.Tensor):
        pred_energy = pred_energy.detach().cpu().numpy()
    if isinstance(target_energy, torch.Tensor):
        target_energy = target_energy.detach().cpu().numpy()

    pred_energy = pred_energy.flatten()
    target_energy = target_energy.flatten()

    fig, ax = plt.subplots(figsize=figsize)

    time_steps = np.arange(len(target_energy))
    ax.plot(time_steps, target_energy, label="Target", linewidth=2, alpha=0.7)
    ax.plot(time_steps, pred_energy, label="Predicted", linewidth=2, alpha=0.7)

    ax.set_xlabel("Time Frame")
    ax.set_ylabel("Energy")
    ax.set_title(f"{title_prefix}Energy Contour Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def plot_duration_comparison(
    pred_duration, target_duration, phonemes=None, title_prefix="", figsize=(12, 6)
):
    """
    Plot predicted and target duration values as bar charts.
    """
    if isinstance(pred_duration, torch.Tensor):
        pred_duration = pred_duration.detach().cpu().numpy()
    if isinstance(target_duration, torch.Tensor):
        target_duration = target_duration.detach().cpu().numpy()

    pred_duration = pred_duration.flatten()
    target_duration = target_duration.flatten()

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(target_duration))
    width = 0.35

    ax.bar(x - width / 2, target_duration, width, label="Target", alpha=0.8)
    ax.bar(x + width / 2, pred_duration, width, label="Predicted", alpha=0.8)

    ax.set_xlabel("Phoneme Index")
    ax.set_ylabel("Duration (frames)")
    ax.set_title(f"{title_prefix}Duration Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    if phonemes is not None and len(phonemes) <= 30:
        ax.set_xticks(x)
        ax.set_xticklabels(phonemes, rotation=45, ha="right")

    plt.tight_layout()

    return fig


def plot_attention_weights(
    attention_weights, title="Attention Weights", figsize=(10, 8)
):
    """
    Plot attention weights as a heatmap.
    """
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()

    if attention_weights.ndim == 3:
        attention_weights = attention_weights.squeeze(0)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        attention_weights,
        aspect="auto",
        origin="lower",
        interpolation="none",
        cmap="viridis",
    )
    ax.set_xlabel("Source Position")
    ax.set_ylabel("Target Position")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    return fig
