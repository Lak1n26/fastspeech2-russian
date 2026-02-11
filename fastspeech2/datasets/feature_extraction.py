"""
Feature extraction utilities for RUSLAN dataset.

This module provides functions to extract mel-spectrograms, pitch, and energy
from audio files for FastSpeech2 training.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)


class MelSpectrogramExtractor:
    """
    Extract mel-spectrograms from audio.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: Optional[float] = 11025.0,
        normalize_db: bool = True,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sample_rate // 2
        self.normalize_db = normalize_db

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=self.f_max,
            power=2.0,
        )

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract mel-spectrogram from audio.
        """
        mel = self.mel_transform(audio)

        if self.normalize_db:
            mel = torch.clamp(mel, min=1e-10)
            ref = mel.max()
            if ref > 0:
                mel_db = 10.0 * torch.log10(mel) - 10.0 * torch.log10(ref)
            else:
                mel_db = mel
            mel_db = torch.clamp(mel_db, min=-80.0, max=0.0)
            mel = (mel_db + 80.0) / 80.0
            mel = torch.clamp(mel, min=0.0, max=1.0)

        return mel


class PitchExtractor:
    """
    Extract pitch (F0) from audio using STFT-based method.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        hop_length: int = 256,
        f_min: float = 80.0,
        f_max: float = 400.0,
    ):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract pitch from audio.
        """
        try:
            pitch = torchaudio.functional.detect_pitch_frequency(
                audio.unsqueeze(0),
                sample_rate=self.sample_rate,
                frame_time=self.hop_length / self.sample_rate,
            )
            pitch = pitch.squeeze(0)
        except Exception as e:
            logger.warning(f"Pitch extraction failed: {e}. Using zeros.")
            num_frames = audio.shape[0] // self.hop_length
            pitch = torch.zeros(num_frames)

        voiced_mask = pitch >= self.f_min
        pitch = torch.where(voiced_mask, pitch, torch.zeros_like(pitch))
        pitch = torch.clamp(pitch, min=0.0, max=self.f_max)

        return pitch


class EnergyExtractor:
    """
    Extract energy from audio or mel-spectrogram.
    """

    def __init__(self, hop_length: int = 256, norm_type: str = "frame"):
        self.hop_length = hop_length
        self.norm_type = norm_type

    def __call__(
        self, audio: Optional[torch.Tensor] = None, mel: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract energy from audio or mel-spectrogram.
        """
        if mel is not None:
            energy = torch.norm(mel, dim=0, p=2)
        elif audio is not None:
            num_frames = (audio.shape[0] - self.hop_length) // self.hop_length + 1
            energy = torch.zeros(num_frames)

            for i in range(num_frames):
                start = i * self.hop_length
                end = start + self.hop_length
                frame = audio[start:end]
                energy[i] = torch.sqrt(torch.mean(frame**2))
        else:
            raise ValueError("Either audio or mel must be provided")

        if self.norm_type == "frame":
            energy = energy / (torch.max(energy) + 1e-8)

        return energy


class FeatureExtractor:
    """
    Combined feature extractor for all FastSpeech2 features.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        hop_length: int = 256,
        n_mels: int = 80,
        mel_f_min: float = 0.0,
        mel_f_max: Optional[float] = 11025.0,
        mel_normalize_db: bool = True,
        resample: bool = True,
    ):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.resample = resample

        self.mel_extractor = MelSpectrogramExtractor(
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=mel_f_min,
            f_max=mel_f_max,
            normalize_db=mel_normalize_db,
        )

        self.pitch_extractor = PitchExtractor(
            sample_rate=sample_rate,
            hop_length=hop_length,
        )

        self.energy_extractor = EnergyExtractor(
            hop_length=hop_length,
        )

    def __call__(
        self, audio: torch.Tensor, original_sample_rate: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract all features from audio.
        """
        if self.resample and original_sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=original_sample_rate, new_freq=self.sample_rate
            )
            audio = resampler(audio)

        mel = self.mel_extractor(audio)
        pitch = self.pitch_extractor(audio)
        energy = self.energy_extractor(mel=mel)

        min_len = min(mel.shape[-1], pitch.shape[0], energy.shape[0])
        mel = mel[:, :min_len]
        pitch = pitch[:min_len]
        energy = energy[:min_len]

        return mel, pitch, energy

    def extract_from_file(
        self, audio_path: Path
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract features from audio file.
        """
        audio, sample_rate = torchaudio.load(audio_path)

        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0)
        else:
            audio = audio.squeeze(0)

        return self(audio, sample_rate)
