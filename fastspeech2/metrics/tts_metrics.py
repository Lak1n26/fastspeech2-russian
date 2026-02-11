import torch
import torch.nn.functional as F

from fastspeech2.metrics.base_metric import BaseMetric


class MelSpectrogramMAE(BaseMetric):
    """
    Mean Absolute Error (MAE) for mel-spectrograms.
    """

    def __call__(
        self,
        mel_pred: torch.Tensor,
        mel: torch.Tensor,
        mel_lengths: torch.Tensor = None,
        **kwargs
    ):
        """
        Calculate MAE between predicted and target mel-spectrograms.
        """
        min_len = min(mel_pred.size(-1), mel.size(-1))
        mel_pred = mel_pred[..., :min_len]
        mel = mel[..., :min_len]

        if mel_lengths is not None:
            if isinstance(mel_lengths, int):
                mel_lengths = None
            else:
                mel_lengths = mel_lengths.to(mel_pred.device)
                batch_size, mel_dim, max_len = mel_pred.shape
                mask = torch.arange(max_len, device=mel_pred.device).unsqueeze(
                    0
                ) < mel_lengths.unsqueeze(1)
                mask = mask.unsqueeze(1)
                mae = torch.abs(mel_pred - mel) * mask
                return mae.sum() / (mask.sum() * mel_dim + 1e-8)

        return F.l1_loss(mel_pred, mel)


class MelSpectrogramMSE(BaseMetric):
    """
    Mean Squared Error (MSE) for mel-spectrograms.
    """

    def __call__(
        self,
        mel_pred: torch.Tensor,
        mel: torch.Tensor,
        mel_lengths: torch.Tensor = None,
        **kwargs
    ):
        """
        Calculate MSE between predicted and target mel-spectrograms.
        """
        min_len = min(mel_pred.size(-1), mel.size(-1))
        mel_pred = mel_pred[..., :min_len]
        mel = mel[..., :min_len]

        if mel_lengths is not None:
            if isinstance(mel_lengths, int):
                mel_lengths = None
            else:
                mel_lengths = mel_lengths.to(mel_pred.device)
                batch_size, mel_dim, max_len = mel_pred.shape
                mask = torch.arange(max_len, device=mel_pred.device).unsqueeze(
                    0
                ) < mel_lengths.unsqueeze(1)
                mask = mask.unsqueeze(1)
                mse = ((mel_pred - mel) ** 2) * mask
                return mse.sum() / (mask.sum() * mel_dim + 1e-8)

        return F.mse_loss(mel_pred, mel)


class DurationAccuracy(BaseMetric):
    """
    Accuracy metric for duration prediction (percentage of frames within tolerance).
    """

    def __init__(self, tolerance=5, *args, **kwargs):
        """
        Init accuracy metric
        """
        super().__init__(*args, **kwargs)
        self.tolerance = tolerance

    def __call__(
        self,
        duration_pred: torch.Tensor,
        duration: torch.Tensor,
        text_lengths: torch.Tensor = None,
        **kwargs
    ):
        """
        Calculate accuracy of duration predictions.
        """
        min_len = min(duration_pred.size(-1), duration.size(-1))
        duration_pred = duration_pred[..., :min_len]
        duration = duration[..., :min_len]
        duration_pred_rounded = torch.round(duration_pred)
        diff = torch.abs(duration_pred_rounded - duration)

        if text_lengths is not None and not isinstance(text_lengths, int):
            text_lengths = text_lengths.to(duration_pred.device)
            batch_size, max_len = duration_pred.shape
            mask = torch.arange(max_len, device=duration_pred.device).unsqueeze(
                0
            ) < text_lengths.unsqueeze(1)
            correct = (diff <= self.tolerance) & mask
            accuracy = correct.sum().float() / (mask.sum().float() + 1e-8)
        else:
            correct = diff <= self.tolerance
            accuracy = correct.float().mean()

        return accuracy.item()


class PitchMAE(BaseMetric):
    """
    Mean Absolute Error (MAE) for pitch prediction.
    """

    def __call__(
        self,
        pitch_pred: torch.Tensor,
        pitch: torch.Tensor,
        mel_lengths: torch.Tensor = None,
        **kwargs
    ):
        """
        Calculate MAE between predicted and target pitch values.
        """
        min_len = min(pitch_pred.size(-1), pitch.size(-1))
        pitch_pred = pitch_pred[..., :min_len]
        pitch = pitch[..., :min_len]

        if mel_lengths is not None and not isinstance(mel_lengths, int):
            mel_lengths = mel_lengths.to(pitch_pred.device)
            batch_size, max_len = pitch_pred.shape
            mask = torch.arange(max_len, device=pitch_pred.device).unsqueeze(
                0
            ) < mel_lengths.unsqueeze(1)
            mae = torch.abs(pitch_pred - pitch) * mask
            return (mae.sum() / (mask.sum() + 1e-8)).item()
        else:
            return F.l1_loss(pitch_pred, pitch).item()


class EnergyMAE(BaseMetric):
    """
    Mean Absolute Error (MAE) for energy prediction.
    """

    def __call__(
        self,
        energy_pred: torch.Tensor,
        energy: torch.Tensor,
        mel_lengths: torch.Tensor = None,
        **kwargs
    ):
        """
        Calculate MAE between predicted and target energy values.
        """
        min_len = min(energy_pred.size(-1), energy.size(-1))
        energy_pred = energy_pred[..., :min_len]
        energy = energy[..., :min_len]

        if mel_lengths is not None and not isinstance(mel_lengths, int):
            mel_lengths = mel_lengths.to(energy_pred.device)
            batch_size, max_len = energy_pred.shape
            mask = torch.arange(max_len, device=energy_pred.device).unsqueeze(
                0
            ) < mel_lengths.unsqueeze(1)
            mae = torch.abs(energy_pred - energy) * mask
            return (mae.sum() / (mask.sum() + 1e-8)).item()
        else:
            return F.l1_loss(energy_pred, energy).item()


class DurationMAE(BaseMetric):
    """
    Mean Absolute Error (MAE) for duration prediction in frames.
    """

    def __call__(
        self,
        duration_pred: torch.Tensor,
        duration: torch.Tensor,
        text_lengths: torch.Tensor = None,
        **kwargs
    ):
        """
        Calculate MAE between predicted and target durations.
        """
        min_len = min(duration_pred.size(-1), duration.size(-1))
        duration_pred = duration_pred[..., :min_len]
        duration = duration[..., :min_len]
        duration_pred_clamped = torch.clamp(duration_pred, min=0.0, max=1000.0)

        if (
            torch.isnan(duration_pred_clamped).any()
            or torch.isinf(duration_pred_clamped).any()
        ):
            return float("nan")

        if text_lengths is not None and not isinstance(text_lengths, int):
            text_lengths = text_lengths.to(duration_pred.device)
            batch_size, max_len = duration_pred.shape
            mask = torch.arange(max_len, device=duration_pred.device).unsqueeze(
                0
            ) < text_lengths.unsqueeze(1)

            if mask.sum() == 0:
                return float("nan")

            mae = torch.abs(duration_pred_clamped - duration.float()) * mask
            return (mae.sum() / (mask.sum() + 1e-8)).item()
        else:
            return F.l1_loss(duration_pred_clamped, duration.float()).item()


class MelSpectrogramL2Norm(BaseMetric):
    """
    L2 norm distance between predicted and target mel-spectrograms.
    """

    def __call__(
        self,
        mel_pred: torch.Tensor,
        mel: torch.Tensor,
        mel_lengths: torch.Tensor = None,
        **kwargs
    ):
        """
        Calculate L2 norm distance between mel-spectrograms.
        """
        min_len = min(mel_pred.size(-1), mel.size(-1))
        mel_pred = mel_pred[..., :min_len]
        mel = mel[..., :min_len]

        if mel_lengths is not None and not isinstance(mel_lengths, int):
            mel_lengths = mel_lengths.to(mel_pred.device)
            batch_size, mel_dim, max_len = mel_pred.shape
            mask = torch.arange(max_len, device=mel_pred.device).unsqueeze(
                0
            ) < mel_lengths.unsqueeze(1)
            mask = mask.unsqueeze(1)

            diff = (mel_pred - mel) * mask
            l2_norm = torch.sqrt((diff**2).sum()) / torch.sqrt(
                mask.sum() * mel_dim + 1e-8
            )
        else:
            diff = mel_pred - mel
            l2_norm = torch.sqrt((diff**2).mean())

        return l2_norm.item()
