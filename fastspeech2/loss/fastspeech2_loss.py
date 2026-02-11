"""
Loss function for FastSpeech2 training.
"""

import logging
from typing import Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FastSpeech2LossWrapper(nn.Module):
    """
    Wrapper for FastSpeech2 loss to integrate with the training framework.
    """

    def __init__(
        self,
        mel_loss_type: str = "mae",
        lambda_mel: float = 1.0,
        lambda_duration: float = 1.0,
        lambda_pitch: float = 1.0,
        lambda_energy: float = 1.0,
        lambda_emotion: float = 1.0,
    ):
        super().__init__()

        self.mel_loss_type = mel_loss_type
        self.lambda_mel = lambda_mel
        self.lambda_duration = lambda_duration
        self.lambda_pitch = lambda_pitch
        self.lambda_energy = lambda_energy
        self.lambda_emotion = lambda_emotion

        if mel_loss_type == "mse":
            self.mel_loss_fn = nn.MSELoss()
        elif mel_loss_type == "mae":
            self.mel_loss_fn = nn.L1Loss()
        else:
            raise ValueError(
                f"Unknown mel_loss_type: {mel_loss_type}. Use 'mse' or 'mae'."
            )

        self.mse_loss = nn.MSELoss()
        self.emotion_loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        mel_pred: torch.Tensor,
        duration_pred: torch.Tensor,
        pitch_pred: torch.Tensor,
        energy_pred: torch.Tensor,
        mel: torch.Tensor,
        duration: torch.Tensor,
        pitch: torch.Tensor,
        energy: torch.Tensor,
        mel_lengths: torch.Tensor = None,
        text_lengths: torch.Tensor = None,
        emotion_pred: torch.Tensor = None,
        emotion: torch.Tensor = None,
        **batch,
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate loss.
        """
        mel_len = min(mel_pred.size(-1), mel.size(-1))
        mel_pred = mel_pred[..., :mel_len]
        mel = mel[..., :mel_len]

        duration_len = min(duration_pred.size(-1), duration.size(-1))
        duration_pred = duration_pred[..., :duration_len]
        duration = duration[..., :duration_len]

        pitch_len = min(pitch_pred.size(-1), pitch.size(-1))
        pitch_pred = pitch_pred[..., :pitch_len]
        pitch = pitch[..., :pitch_len]

        energy_len = min(energy_pred.size(-1), energy.size(-1))
        energy_pred = energy_pred[..., :energy_len]
        energy = energy[..., :energy_len]

        if mel_lengths is not None and not isinstance(mel_lengths, int):
            mel_lengths = mel_lengths.to(mel_pred.device)
            mel_mask = torch.arange(mel_len, device=mel_pred.device).unsqueeze(0)
            mel_mask = (mel_mask < mel_lengths.unsqueeze(1)).unsqueeze(1).float()

            mel_diff = (
                torch.abs(mel_pred - mel)
                if self.mel_loss_type == "mae"
                else (mel_pred - mel) ** 2
            )
            mel_loss = (mel_diff * mel_mask).sum() / (
                mel_mask.sum() * mel_pred.size(1) + 1e-8
            )
        else:
            mel_loss = self.mel_loss_fn(mel_pred, mel)

        log_duration_pred = torch.log(duration_pred + 1.0)
        log_duration_target = torch.log(duration.float() + 1.0)

        if text_lengths is not None and not isinstance(text_lengths, int):
            text_lengths = text_lengths.to(duration_pred.device)

            dur_mask = torch.arange(
                duration_len, device=duration_pred.device
            ).unsqueeze(0)
            dur_mask = (dur_mask < text_lengths.unsqueeze(1)).float()

            dur_diff = (log_duration_pred - log_duration_target) ** 2
            duration_loss = (dur_diff * dur_mask).sum() / (dur_mask.sum() + 1e-8)
        else:
            duration_loss = self.mse_loss(log_duration_pred, log_duration_target)

        if mel_lengths is not None and not isinstance(mel_lengths, int):
            mel_lengths = mel_lengths.to(pitch_pred.device)

            pitch_mask = torch.arange(pitch_len, device=pitch_pred.device).unsqueeze(0)
            pitch_mask = (pitch_mask < mel_lengths.unsqueeze(1)).float()

            pitch_diff = (pitch_pred - pitch) ** 2
            pitch_loss = (pitch_diff * pitch_mask).sum() / (pitch_mask.sum() + 1e-8)
        else:
            pitch_loss = self.mse_loss(pitch_pred, pitch)

        if mel_lengths is not None and not isinstance(mel_lengths, int):
            mel_lengths = mel_lengths.to(energy_pred.device)

            energy_mask = torch.arange(energy_len, device=energy_pred.device).unsqueeze(
                0
            )
            energy_mask = (energy_mask < mel_lengths.unsqueeze(1)).float()

            energy_diff = (energy_pred - energy) ** 2
            energy_loss = (energy_diff * energy_mask).sum() / (energy_mask.sum() + 1e-8)
        else:
            energy_loss = self.mse_loss(energy_pred, energy)

        emotion_loss = torch.tensor(0.0, device=mel_pred.device)
        if emotion_pred is not None and emotion is not None:
            emotion_loss = self.emotion_loss_fn(emotion_pred, emotion)

        total_loss = (
            self.lambda_mel * mel_loss
            + self.lambda_duration * duration_loss
            + self.lambda_pitch * pitch_loss
            + self.lambda_energy * energy_loss
            + self.lambda_emotion * emotion_loss
        )

        loss_dict = {
            "loss": total_loss,
            "mel_loss": mel_loss.detach(),
            "duration_loss": duration_loss.detach(),
            "pitch_loss": pitch_loss.detach(),
            "energy_loss": energy_loss.detach(),
            "emotion_loss": emotion_loss.detach(),
        }

        return loss_dict
