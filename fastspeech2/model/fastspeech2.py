"""
FastSpeech2 Model.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from fastspeech2.model.encoder_decoder import Decoder, Encoder
from fastspeech2.model.variance_adaptor import VarianceAdaptor


class FastSpeech2(nn.Module):
    """
    FastSpeech2: Fast and High-Quality End-to-End Text to Speech.

    Architecture:
    1. Encoder: Converts text tokens to hidden representations
    2. Variance Adaptor: Predicts and applies duration, pitch, energy
    3. Decoder: Converts adapted representations to mel-spectrogram
    """

    def __init__(
        self,
        vocab_size: int = 100,
        d_model: int = 256,
        encoder_layers: int = 4,
        decoder_layers: int = 4,
        num_heads: int = 2,
        d_ff: int = 1024,
        dropout: float = 0.1,
        n_mels: int = 80,
        max_seq_len: int = 5000,
        n_bins: int = 256,
        pitch_min: float = 80.0,
        pitch_max: float = 400.0,
        energy_min: float = 0.0,
        energy_max: float = 1.0,
        pitch_mean: Optional[float] = None,
        pitch_std: Optional[float] = None,
        pitch_stats_path: Optional[str] = None,
        padding_idx: int = 0,
        **kwargs,
    ):
        super().__init__()

        self.encoder = Encoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=encoder_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_seq_len,
            padding_idx=padding_idx,
        )

        self.variance_adaptor = VarianceAdaptor(
            d_model=d_model,
            dropout=dropout,
            n_bins=n_bins,
            pitch_min=pitch_min,
            pitch_max=pitch_max,
            energy_min=energy_min,
            energy_max=energy_max,
            pitch_mean=pitch_mean,
            pitch_std=pitch_std,
            pitch_stats_path=pitch_stats_path,
            n_emotions=kwargs.get("n_emotions", 7),
            use_emotion=kwargs.get("use_emotion", True),
        )

        self.decoder = Decoder(
            d_model=d_model,
            num_layers=decoder_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_seq_len,
            n_mels=n_mels,
        )

        self.n_mels = n_mels

    def forward(
        self,
        text_tokens: torch.Tensor,
        text_lengths: Optional[torch.Tensor] = None,
        target_duration: Optional[torch.Tensor] = None,
        target_pitch: Optional[torch.Tensor] = None,
        target_energy: Optional[torch.Tensor] = None,
        target_emotion: Optional[torch.Tensor] = None,
        mel_lengths: Optional[torch.Tensor] = None,
        max_mel_len: Optional[int] = None,
        duration_control: float = 1.0,
        pitch_control: float = 1.0,
        energy_control: float = 1.0,
        emotion_control: float = 1.0,
        **batch,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of FastSpeech2.
        """
        if max_mel_len is None and mel_lengths is not None:
            max_mel_len = int(mel_lengths.max().item())

        if text_lengths is not None:
            max_text_len = text_tokens.size(1)
            text_mask = self._get_mask_from_lengths(text_lengths, max_text_len)
        else:
            text_mask = None

        encoder_output = self.encoder(text_tokens, text_mask)

        (
            adapted_output,
            duration_pred,
            pitch_pred,
            energy_pred,
            mel_len,
            emotion_pred,
        ) = self.variance_adaptor(
            encoder_output,
            mask=text_mask,
            target_duration=target_duration,
            target_pitch=target_pitch,
            target_energy=target_energy,
            target_emotion=target_emotion,
            max_len=max_mel_len,
            duration_control=duration_control,
            pitch_control=pitch_control,
            energy_control=energy_control,
            emotion_control=emotion_control,
        )

        mel_pred = self.decoder(adapted_output)
        mel_pred = mel_pred.transpose(1, 2)

        if mel_lengths is not None and isinstance(mel_lengths, torch.Tensor):
            max_mel_len = mel_pred.size(-1)
            mel_lengths = mel_lengths.to(mel_pred.device)
            mel_mask = torch.arange(max_mel_len, device=mel_pred.device).unsqueeze(0)
            mel_mask = (mel_mask < mel_lengths.unsqueeze(1)).unsqueeze(1).float()
            mel_pred = mel_pred * mel_mask

        output = {
            "mel_pred": mel_pred,
            "duration_pred": duration_pred,
            "pitch_pred": pitch_pred,
            "energy_pred": energy_pred,
            "mel_lengths": mel_len,
        }

        if emotion_pred is not None:
            output["emotion_pred"] = emotion_pred

        return output

    def inference(
        self,
        text_tokens: torch.Tensor,
        duration_control: float = 1.0,
        pitch_control: float = 1.0,
        energy_control: float = 1.0,
        emotion_control: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference mode (generate mel-spectrogram from text).
        """
        text_lengths = (text_tokens != self.encoder.embedding.padding_idx).sum(dim=1)

        with torch.no_grad():
            output = self.forward(
                text_tokens=text_tokens,
                text_lengths=text_lengths,
                duration_control=duration_control,
                pitch_control=pitch_control,
                energy_control=energy_control,
                emotion_control=emotion_control,
            )

        return output

    @staticmethod
    def _get_mask_from_lengths(
        lengths: torch.Tensor, max_len: Optional[int] = None
    ) -> torch.Tensor:
        """
        Create mask from lengths.
        """
        if max_len is None:
            max_len = lengths.max().item()

        batch_size = lengths.size(0)
        ids = torch.arange(0, max_len, device=lengths.device).unsqueeze(0)
        ids = ids.expand(batch_size, -1)

        mask = ids < lengths.unsqueeze(1)

        return mask

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters:,}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters:,}"

        return result_info


class FastSpeech2Loss(nn.Module):
    """
    Loss function for FastSpeech2.
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
            raise ValueError(f"Unknown mel_loss_type: {mel_loss_type}")

        self.duration_loss_fn = nn.MSELoss()
        self.pitch_loss_fn = nn.MSELoss()
        self.energy_loss_fn = nn.MSELoss()
        self.emotion_loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        mel_pred: torch.Tensor,
        mel_target: torch.Tensor,
        duration_pred: torch.Tensor,
        duration_target: torch.Tensor,
        pitch_pred: torch.Tensor,
        pitch_target: torch.Tensor,
        energy_pred: torch.Tensor,
        energy_target: torch.Tensor,
        mel_lengths: Optional[torch.Tensor] = None,
        text_lengths: Optional[torch.Tensor] = None,
        emotion_pred: Optional[torch.Tensor] = None,
        target_emotion: Optional[torch.Tensor] = None,
        **batch,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate loss.
        """
        mel_loss = self.mel_loss_fn(mel_pred, mel_target)

        log_duration_pred = torch.log(duration_pred + 1.0)
        log_duration_target = torch.log(duration_target.float() + 1.0)
        duration_loss = self.duration_loss_fn(log_duration_pred, log_duration_target)

        pitch_loss = self.pitch_loss_fn(pitch_pred, pitch_target)
        energy_loss = self.energy_loss_fn(energy_pred, energy_target)
        emotion_loss = torch.tensor(0.0, device=mel_pred.device)
        if emotion_pred is not None and target_emotion is not None:
            emotion_loss = self.emotion_loss_fn(emotion_pred, target_emotion)

        total_loss = (
            self.lambda_mel * mel_loss
            + self.lambda_duration * duration_loss
            + self.lambda_pitch * pitch_loss
            + self.lambda_energy * energy_loss
            + self.lambda_emotion * emotion_loss
        )

        loss_dict = {
            "loss": total_loss,
            "mel_loss": mel_loss,
            "duration_loss": duration_loss,
            "pitch_loss": pitch_loss,
            "energy_loss": energy_loss,
            "emotion_loss": emotion_loss,
        }

        return total_loss, loss_dict
