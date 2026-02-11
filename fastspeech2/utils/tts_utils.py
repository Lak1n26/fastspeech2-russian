from pathlib import Path

import numpy as np
import torch


def get_mask_from_lengths(lengths, max_len=None):
    """
    Create a boolean mask from sequence lengths.
    """
    if max_len is None:
        max_len = torch.max(lengths).item()

    batch_size = lengths.size(0)
    ids = (
        torch.arange(0, max_len, device=lengths.device)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    mask = ids < lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def pad_1d(sequences, pad_value=0):
    """
    Pad a list of 1D sequences to the same length.
    """
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)

    padded = torch.full((batch_size, max_len), pad_value, dtype=sequences[0].dtype)

    for i, seq in enumerate(sequences):
        padded[i, : len(seq)] = seq

    return padded


def pad_2d(sequences, pad_value=0):
    """
    Pad a list of 2D sequences to the same length.
    """
    max_len = max(seq.size(1) for seq in sequences)
    batch_size = len(sequences)
    dim = sequences[0].size(0)

    padded = torch.full((batch_size, dim, max_len), pad_value, dtype=sequences[0].dtype)

    for i, seq in enumerate(sequences):
        padded[i, :, : seq.size(1)] = seq

    return padded


def get_alignment_2d(durations):
    """
    Create alignment matrix from durations for length regulation.
    """
    batch_size, max_phoneme_len = durations.shape
    max_mel_len = torch.sum(durations, dim=1).max().item()

    alignment = torch.zeros(
        batch_size, max_mel_len, max_phoneme_len, device=durations.device
    )

    for i in range(batch_size):
        mel_pos = 0
        for j in range(max_phoneme_len):
            duration = durations[i, j].item()
            if duration > 0:
                alignment[i, mel_pos : mel_pos + int(duration), j] = 1
                mel_pos += int(duration)

    return alignment


def expand_durations(embeddings, durations):
    """
    Expand embeddings according to durations (length regulation).
    """
    batch_size, hidden_dim, phoneme_len = embeddings.shape
    mel_lens = torch.sum(durations, dim=1).long()
    max_mel_len = torch.max(mel_lens).item()

    expanded = torch.zeros(
        batch_size,
        hidden_dim,
        max_mel_len,
        device=embeddings.device,
        dtype=embeddings.dtype,
    )

    for i in range(batch_size):
        mel_pos = 0
        for j in range(phoneme_len):
            duration = int(durations[i, j].item())
            if duration > 0:
                expanded[i, :, mel_pos : mel_pos + duration] = (
                    embeddings[i, :, j].unsqueeze(-1).expand(-1, duration)
                )
                mel_pos += duration

    return expanded


def normalize_pitch(pitch, pitch_min=80.0, pitch_max=400.0):
    """
    Normalize pitch to [-3, 3] range using z-score normalization.
    """
    pitch = pitch.clone()

    voiced_mask = pitch > 0

    if not voiced_mask.any():
        return pitch

    voiced_pitch = pitch[voiced_mask]
    voiced_pitch = torch.clamp(voiced_pitch, min=pitch_min, max=pitch_max)

    pitch_mean = voiced_pitch.mean()
    pitch_std = voiced_pitch.std() + 1e-8
    pitch[voiced_mask] = (voiced_pitch - pitch_mean) / pitch_std
    pitch[voiced_mask] = torch.clamp(pitch[voiced_mask], min=-3.0, max=3.0)

    return pitch


def denormalize_pitch(normalized_pitch, pitch_mean, pitch_std):
    """
    Denormalize pitch from [-3, 3] range back to Hz.
    """
    pitch = normalized_pitch.clone()

    voiced_mask = pitch != 0

    if not voiced_mask.any():
        return pitch

    pitch[voiced_mask] = pitch[voiced_mask] * pitch_std + pitch_mean
    pitch[voiced_mask] = torch.clamp(pitch[voiced_mask], min=0.0)

    return pitch


def normalize_energy(energy, energy_min, energy_max):
    """
    Normalize energy to [0, 1] range.
    """
    return (energy - energy_min) / (energy_max - energy_min + 1e-8)


def denormalize_energy(normalized_energy, energy_min, energy_max):
    """
    Denormalize energy from [0, 1] range.
    """
    return normalized_energy * (energy_max - energy_min) + energy_min


def save_model_checkpoint(model, optimizer, scheduler, epoch, step, save_path):
    """
    Save model checkpoint.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict()
        if scheduler is not None
        else None,
    }

    torch.save(checkpoint, save_path)


def load_model_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """
    Load model checkpoint.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    step = checkpoint.get("step", 0)

    return epoch, step


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_number(num):
    """
    Format large numbers with K, M, B suffixes.
    """
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    else:
        return str(num)


def get_lr(optimizer):
    """
    Get current learning rate from optimizer.
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def clip_grad_norm(parameters, max_norm, norm_type=2.0):
    """
    Clip gradient norm of parameters.
    """
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=norm_type)


def move_batch_to_device(batch, device):
    """
    Move all tensors in batch to device.
    """
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    return batch
