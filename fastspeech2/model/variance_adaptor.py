"""
Variance Adaptor for FastSpeech2.

Contains:
- Duration Predictor
- Pitch Predictor
- Energy Predictor
- Length Regulator
- Variance Adaptor (combines all components)
"""

import json
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastspeech2.model.blocks import ConvBlock
from fastspeech2.utils.tts_utils import denormalize_energy, denormalize_pitch

logger = logging.getLogger(__name__)


class VariancePredictor(nn.Module):
    """
    Variance Predictor for duration, pitch, or energy.
    """

    def __init__(
        self,
        d_model: int = 256,
        kernel_size: int = 3,
        dropout: float = 0.5,
        num_layers: int = 2,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                ConvBlock(d_model if i == 0 else d_model, d_model, kernel_size, dropout)
                for i in range(num_layers)
            ]
        )

        self.linear = nn.Linear(d_model, 1)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        """
        x = x.transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        x = x.transpose(1, 2)
        x = self.linear(x).squeeze(-1)

        if mask is not None:
            mask = mask.to(x.device)
            x = x.masked_fill(mask == 0, 0.0)

        return x


class LengthRegulator(nn.Module):
    """
    Length Regulator.

    Expands hidden states according to predicted durations.
    Each frame is repeated according to its duration.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, x: torch.Tensor, duration: torch.Tensor, max_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Expand sequence according to duration.
        """
        batch_size, seq_len, d_model = x.size()
        duration = torch.nan_to_num(duration, nan=0.0, posinf=300.0, neginf=0.0)
        duration = torch.clamp(duration, min=0, max=300)

        if max_len is None:
            max_len = int(duration.sum(dim=1).max().item())
        else:
            max_len = int(max_len)

        output = torch.zeros(
            batch_size, max_len, d_model, device=x.device, dtype=x.dtype
        )
        lengths = torch.zeros(batch_size, device=x.device, dtype=torch.long)

        for b in range(batch_size):
            pos = 0
            for i in range(seq_len):
                dur = int(duration[b, i].item())
                if dur > 0 and pos < max_len:
                    actual_dur = min(dur, max_len - pos)
                    output[b, pos : pos + actual_dur, :] = (
                        x[b, i, :].unsqueeze(0).expand(actual_dur, -1)
                    )
                    pos += actual_dur
                if pos >= max_len:
                    break
            lengths[b] = pos

        return output, lengths


class VarianceAdaptor(nn.Module):
    """
    Variance Adaptor module.

    Predicts and applies duration, pitch, energy, and emotion to encoder output.
    """

    def __init__(
        self,
        d_model: int = 256,
        kernel_size: int = 3,
        dropout: float = 0.5,
        n_bins: int = 256,
        pitch_min: float = 0.0,
        pitch_max: float = 800.0,
        energy_min: float = 0.0,
        energy_max: float = 1.0,
        pitch_mean: Optional[float] = None,
        pitch_std: Optional[float] = None,
        pitch_stats_path: Optional[str] = None,
        n_emotions: int = 7,
        use_emotion: bool = True,
    ):
        super().__init__()

        self.duration_predictor = VariancePredictor(d_model, kernel_size, dropout)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(d_model, kernel_size, dropout)
        self.pitch_embedding = nn.Embedding(n_bins, d_model)
        self.energy_predictor = VariancePredictor(d_model, kernel_size, dropout)
        self.energy_embedding = nn.Embedding(n_bins, d_model)

        self.use_emotion = use_emotion
        self.n_emotions = n_emotions
        if self.use_emotion:
            self.emotion_predictor = VariancePredictor(d_model, kernel_size, dropout)
            self.emotion_predictor.linear = nn.Linear(d_model, n_emotions)
            self.emotion_embedding = nn.Embedding(n_emotions, d_model)
            logger.info(
                f"Emotion predictor initialized with {n_emotions} emotion classes"
            )

        self.pitch_mean = pitch_mean
        self.pitch_std = pitch_std
        if pitch_stats_path and (self.pitch_mean is None or self.pitch_std is None):
            try:
                with open(pitch_stats_path, "r", encoding="utf-8") as f:
                    stats = json.load(f)
                self.pitch_mean = float(stats.get("pitch_mean"))
                self.pitch_std = float(stats.get("pitch_std"))
            except Exception as exc:
                logger.warning(
                    "Failed to load pitch stats from %s: %s", pitch_stats_path, exc
                )

        self.n_bins = n_bins
        self.pitch_min = pitch_min
        self.pitch_max = pitch_max
        self.energy_min = energy_min
        self.energy_max = energy_max
        if self.pitch_mean is None or self.pitch_std is None:
            pitch_bins = torch.linspace(-3.0, 3.0, n_bins - 1)
        else:
            pitch_bins = torch.linspace(pitch_min, pitch_max, n_bins - 1)
        self.register_buffer("pitch_bins", pitch_bins)
        self.register_buffer(
            "energy_bins", torch.linspace(energy_min, energy_max, n_bins - 1)
        )

    def get_pitch_embedding(
        self,
        pitch: torch.Tensor,
        target_pitch: Optional[torch.Tensor] = None,
        control: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get pitch embedding.
        """
        pitch_pred = pitch

        if target_pitch is not None:
            pitch_to_embed = target_pitch
        else:
            if self.pitch_mean is not None and self.pitch_std is not None:
                pitch_to_embed = denormalize_pitch(
                    pitch, self.pitch_mean, self.pitch_std
                )
                pitch_to_embed = pitch_to_embed * control
                pitch_to_embed = torch.clamp(
                    pitch_to_embed, min=self.pitch_min, max=self.pitch_max
                )
            else:
                pitch_to_embed = pitch * control
                pitch_to_embed = torch.clamp(pitch_to_embed, min=-3.0, max=3.0)

        if target_pitch is not None:
            if self.pitch_mean is not None and self.pitch_std is not None:
                pitch_to_embed = denormalize_pitch(
                    pitch_to_embed, self.pitch_mean, self.pitch_std
                )
                pitch_to_embed = torch.clamp(
                    pitch_to_embed, min=self.pitch_min, max=self.pitch_max
                )
            else:
                pitch_to_embed = torch.clamp(pitch_to_embed, min=-3.0, max=3.0)

        pitch_bins_device = self.pitch_bins.to(pitch_to_embed.device)
        pitch_bins = torch.bucketize(pitch_to_embed, pitch_bins_device)
        pitch_bins = torch.clamp(pitch_bins, min=0, max=self.n_bins - 1)
        pitch_embedding = self.pitch_embedding(pitch_bins)

        return pitch_embedding, pitch_pred

    def get_energy_embedding(
        self,
        energy: torch.Tensor,
        target_energy: Optional[torch.Tensor] = None,
        control: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get energy embedding.
        """
        energy_pred = energy

        if target_energy is not None:
            energy_to_embed = target_energy
        else:
            energy_to_embed = denormalize_energy(
                energy, self.energy_min, self.energy_max
            )
            energy_to_embed = energy_to_embed * control
            energy_to_embed = torch.clamp(
                energy_to_embed, min=self.energy_min, max=self.energy_max
            )

        if target_energy is not None:
            energy_to_embed = torch.clamp(
                energy_to_embed,
                min=self.energy_bins.min().item(),
                max=self.energy_bins.max().item(),
            )

        energy_bins_device = self.energy_bins.to(energy_to_embed.device)
        energy_bins = torch.bucketize(energy_to_embed, energy_bins_device)
        energy_bins = torch.clamp(energy_bins, min=0, max=self.n_bins - 1)
        energy_embedding = self.energy_embedding(energy_bins)

        return energy_embedding, energy_pred

    def get_emotion_embedding(
        self,
        emotion_logits: torch.Tensor,
        target_emotion: Optional[torch.Tensor] = None,
        control: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get emotion embedding.
        """
        emotion_pred = emotion_logits

        if target_emotion is not None:
            batch_size, seq_len, _ = emotion_logits.shape
            emotion_indices = target_emotion.unsqueeze(1).expand(batch_size, seq_len)
        else:
            emotion_probs = F.softmax(emotion_logits, dim=-1)
            if control != 1.0 and self.n_emotions > 5:
                neutral_probs = torch.zeros_like(emotion_probs)
                neutral_probs[:, :, 5] = 1.0  # All probability to neutral
                emotion_probs = (
                    control * emotion_probs + (1.0 - control) * neutral_probs
                )

            emotion_indices = torch.argmax(emotion_probs, dim=-1)

        emotion_embedding = self.emotion_embedding(emotion_indices)

        return emotion_embedding, emotion_pred

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        target_duration: Optional[torch.Tensor] = None,
        target_pitch: Optional[torch.Tensor] = None,
        target_energy: Optional[torch.Tensor] = None,
        target_emotion: Optional[torch.Tensor] = None,
        max_len: Optional[int] = None,
        duration_control: float = 1.0,
        pitch_control: float = 1.0,
        energy_control: float = 1.0,
        emotion_control: float = 1.0,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        """
        Forward pass of variance adaptor.
        """
        log_duration_pred = self.duration_predictor(x, mask)
        log_duration_pred = torch.clamp(log_duration_pred, min=-5.0, max=5.0)

        duration_pred = torch.exp(log_duration_pred) - 1.0
        duration_pred = torch.clamp(duration_pred, min=0)

        emotion_pred = None
        if self.use_emotion:
            x_emotion = x.transpose(1, 2)
            for layer in self.emotion_predictor.layers:
                x_emotion = layer(x_emotion)
            x_emotion = x_emotion.transpose(1, 2)
            emotion_logits = self.emotion_predictor.linear(x_emotion)

            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).to(emotion_logits.device)
                masked_logits = emotion_logits * mask_expanded
                emotion_pred = masked_logits.sum(dim=1) / mask_expanded.sum(
                    dim=1
                ).clamp(min=1.0)
            else:
                emotion_pred = emotion_logits.mean(dim=1)

        if target_duration is not None:
            duration_rounded = target_duration
        else:
            duration_scaled = duration_pred * duration_control
            duration_rounded = torch.round(duration_scaled)
            duration_rounded = torch.clamp(duration_rounded, min=0)
            duration_rounded = torch.where(
                (duration_pred > 0) & (duration_rounded < 1),
                torch.ones_like(duration_rounded),
                duration_rounded,
            )

        x, mel_lengths = self.length_regulator(x, duration_rounded, max_len)
        pitch_pred = self.pitch_predictor(x)
        energy_pred = self.energy_predictor(x)

        if target_pitch is not None and target_pitch.size(-1) != pitch_pred.size(-1):
            max_len = max(target_pitch.size(-1), pitch_pred.size(-1))
            batch_size = target_pitch.size(0)
            device = pitch_pred.device

            if target_pitch.size(-1) < max_len:
                target_pitch_padded = torch.zeros(
                    batch_size, max_len, device=device, dtype=target_pitch.dtype
                )
                target_pitch_padded[:, : target_pitch.size(-1)] = target_pitch.to(
                    device
                )
                target_pitch = target_pitch_padded
                logger.debug(
                    f"Padded target_pitch from {target_pitch.size(-1)} to {max_len}"
                )

            if pitch_pred.size(-1) < max_len:
                pitch_pred_padded = torch.zeros(
                    batch_size, max_len, device=device, dtype=pitch_pred.dtype
                )
                pitch_pred_padded[:, : pitch_pred.size(-1)] = pitch_pred
                pitch_pred = pitch_pred_padded
                logger.debug(
                    f"Padded pitch_pred from {pitch_pred.size(-1)} to {max_len}"
                )

        if target_energy is not None and target_energy.size(-1) != energy_pred.size(-1):
            max_len = max(target_energy.size(-1), energy_pred.size(-1))
            batch_size = target_energy.size(0)
            device = energy_pred.device

            if target_energy.size(-1) < max_len:
                target_energy_padded = torch.zeros(
                    batch_size, max_len, device=device, dtype=target_energy.dtype
                )
                target_energy_padded[:, : target_energy.size(-1)] = target_energy.to(
                    device
                )
                target_energy = target_energy_padded
                logger.debug(
                    f"Padded target_energy from {target_energy.size(-1)} to {max_len}"
                )

            if energy_pred.size(-1) < max_len:
                energy_pred_padded = torch.zeros(
                    batch_size, max_len, device=device, dtype=energy_pred.dtype
                )
                energy_pred_padded[:, : energy_pred.size(-1)] = energy_pred
                energy_pred = energy_pred_padded
                logger.debug(
                    f"Padded energy_pred from {energy_pred.size(-1)} to {max_len}"
                )

        pitch_embedding, pitch_pred = self.get_pitch_embedding(
            pitch_pred, target_pitch, pitch_control
        )
        energy_embedding, energy_pred = self.get_energy_embedding(
            energy_pred, target_energy, energy_control
        )

        x = x + pitch_embedding + energy_embedding

        if self.use_emotion and emotion_pred is not None:
            batch_size, mel_len, _ = x.shape
            emotion_logits_expanded = emotion_pred.unsqueeze(1).expand(
                batch_size, mel_len, self.n_emotions
            )

            emotion_embedding, _ = self.get_emotion_embedding(
                emotion_logits_expanded, target_emotion, emotion_control
            )
            x = x + emotion_embedding

        return x, duration_pred, pitch_pred, energy_pred, mel_lengths, emotion_pred
