from pathlib import Path

import numpy as np
import torch
import torchaudio


def load_audio(audio_path, sample_rate=22050):
    """
    Load audio file and resample to target sample rate.
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    waveform, sr = torchaudio.load(audio_path)

    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    return waveform


def save_audio(waveform, audio_path, sample_rate=22050):
    """
    Save audio waveform to file.
    """
    audio_path = Path(audio_path)
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform)

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    torchaudio.save(str(audio_path), waveform, sample_rate)


def mel_spectrogram(
    waveform,
    sample_rate=22050,
    n_fft=1024,
    win_length=1024,
    hop_length=256,
    n_mels=80,
    f_min=0.0,
    f_max=8000.0,
):
    """
    Compute mel-spectrogram from waveform.
    """
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        power=1.0,
    )

    mel_spec = mel_transform(waveform)
    mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
    if mel_spec.shape[0] == 1:
        mel_spec = mel_spec.squeeze(0)

    return mel_spec


def inverse_mel_spectrogram(
    mel_spec,
    sample_rate=22050,
    n_fft=1024,
    win_length=1024,
    hop_length=256,
    n_mels=80,
    f_min=0.0,
    f_max=8000.0,
):
    """
    Compute inverse mel-spectrogram (Griffin-Lim algorithm).
    """
    mel_spec = torch.exp(mel_spec)
    if mel_spec.ndim == 2:
        mel_spec = mel_spec.unsqueeze(0)

    inverse_mel_transform = torchaudio.transforms.InverseMelScale(
        n_stft=n_fft // 2 + 1,
        n_mels=n_mels,
        sample_rate=sample_rate,
        f_min=f_min,
        f_max=f_max,
    )

    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        power=1.0,
    )

    spec = inverse_mel_transform(mel_spec)
    waveform = griffin_lim(spec)

    return waveform


def extract_pitch(
    waveform,
    sample_rate=22050,
    hop_length=256,
    f_min=80.0,
    f_max=400.0,
    normalize=True,
):
    """
    Extract pitch (F0) from waveform using PyWorld or torchaudio.
    """
    try:
        import pyworld as pw

        if isinstance(waveform, torch.Tensor):
            waveform_np = waveform.squeeze(0).cpu().numpy()
        else:
            waveform_np = waveform
        waveform_np = waveform_np.astype(np.float64)
        f0, t = pw.dio(
            waveform_np,
            sample_rate,
            frame_period=hop_length / sample_rate * 1000,
            f0_floor=f_min,
            f0_ceil=f_max,
        )
        f0 = pw.stonemask(waveform_np, f0, t, sample_rate)

        pitch = torch.from_numpy(f0).float()

    except ImportError:
        if waveform.ndim == 2:
            waveform = waveform.squeeze(0)
        frame_length = hop_length * 2
        num_frames = (waveform.shape[0] - frame_length) // hop_length + 1

        pitch = torch.zeros(num_frames)

        for i in range(num_frames):
            start = i * hop_length
            end = start + frame_length
            frame = waveform[start:end]

            autocorr = torch.nn.functional.conv1d(
                frame.unsqueeze(0).unsqueeze(0),
                frame.flip(0).unsqueeze(0).unsqueeze(0),
                padding=frame_length - 1,
            ).squeeze()

            min_lag = int(sample_rate / f_max)
            max_lag = int(sample_rate / f_min)

            if max_lag < len(autocorr):
                peak_lag = torch.argmax(autocorr[min_lag:max_lag]) + min_lag
                pitch[i] = sample_rate / peak_lag.float()

    if normalize:
        pitch = normalize_pitch(pitch, f_min=f_min, f_max=f_max)

    return pitch


def normalize_pitch(pitch, f_min=80.0, f_max=400.0):
    """
    Normalize pitch from Hz to [-3, 3] range.
    """
    pitch = pitch.clone()
    voiced_mask = pitch > 0

    if not voiced_mask.any():
        return pitch

    voiced_pitch = pitch[voiced_mask]

    voiced_pitch = torch.clamp(voiced_pitch, min=f_min, max=f_max)
    pitch_mean = voiced_pitch.mean()
    pitch_std = voiced_pitch.std() + 1e-8
    pitch[voiced_mask] = (voiced_pitch - pitch_mean) / pitch_std
    pitch[voiced_mask] = torch.clamp(pitch[voiced_mask], min=-3.0, max=3.0)

    return pitch


def denormalize_pitch(pitch_norm, pitch_mean, pitch_std):
    """
    Denormalize pitch from [-3, 3] range back to Hz.
    """
    pitch = pitch_norm.clone()
    voiced_mask = pitch != 0

    if not voiced_mask.any():
        return pitch
    pitch[voiced_mask] = pitch[voiced_mask] * pitch_std + pitch_mean
    pitch[voiced_mask] = torch.clamp(pitch[voiced_mask], min=0.0)

    return pitch


def extract_energy(mel_spec):
    """
    Extract energy from mel-spectrogram.
    """
    energy = torch.norm(mel_spec, dim=0, p=2)

    return energy


def normalize_waveform(waveform):
    """
    Normalize waveform to [-1, 1] range.
    """
    max_val = torch.abs(waveform).max()
    if max_val > 0:
        return waveform / max_val
    return waveform


def trim_silence(waveform, threshold=0.01, frame_length=2048, hop_length=512):
    """
    Trim silence from the beginning and end of waveform.
    """
    if waveform.ndim == 2:
        waveform = waveform.squeeze(0)

    num_frames = (len(waveform) - frame_length) // hop_length + 1
    energy = torch.zeros(num_frames)

    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frame = waveform[start:end]
        energy[i] = torch.sqrt(torch.mean(frame**2))

    non_silent = energy > threshold

    if non_silent.any():
        start_frame = non_silent.nonzero()[0].item()
        end_frame = non_silent.nonzero()[-1].item() + 1

        start_sample = start_frame * hop_length
        end_sample = min(end_frame * hop_length + frame_length, len(waveform))

        trimmed = waveform[start_sample:end_sample]
    else:
        trimmed = waveform

    if trimmed.ndim == 1:
        trimmed = trimmed.unsqueeze(0)

    return trimmed
