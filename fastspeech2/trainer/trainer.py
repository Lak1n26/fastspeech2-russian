"""
Trainer class for FastSpeech2.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.cuda.amp import autocast

from fastspeech2.metrics.tracker import MetricTracker
from fastspeech2.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class for FastSpeech2.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        if "duration" in batch and "target_duration" not in batch:
            batch["target_duration"] = batch["duration"]
        if "pitch" in batch and "target_pitch" not in batch:
            batch["target_pitch"] = batch["pitch"]
        if "energy" in batch and "target_energy" not in batch:
            batch["target_energy"] = batch["energy"]
        if "emotion" in batch and "target_emotion" not in batch:
            batch["target_emotion"] = batch["emotion"]

        metric_funcs = self.metrics.get("val", self.metrics.get("inference", []))
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        with autocast(enabled=self.use_amp):
            outputs = self.model(**batch)
            batch.update(outputs)
            all_losses = self.criterion(**batch)
            batch.update(all_losses)

        if self.is_train:
            if self.scaler is not None:
                self.scaler.scale(batch["loss"]).backward()
                self.scaler.unscale_(self.optimizer)
                self._clip_grad_norm()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                batch["loss"].backward()
                self._clip_grad_norm()
                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        for loss_name in self.config.writer.loss_names:
            if loss_name in batch:
                metrics.update(loss_name, batch[loss_name].item())

        if metric_funcs is not None:
            for met in metric_funcs:
                metrics.update(met.name, met(**batch))

        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch for FastSpeech2.
        """
        if "mel_pred" in batch and "mel" in batch:
            self._log_spectrogram(batch, mode)

        if mode == "inference":
            self._log_predictions(batch, mode)

        self._log_audio(batch, mode)

    def _log_spectrogram(self, batch, mode="train"):
        """
        Log mel-spectrograms to experiment tracker.
        """
        if self.writer is None:
            return

        num_samples_to_log = min(4, batch["mel_pred"].size(0))

        for idx in range(num_samples_to_log):
            mel_pred = batch["mel_pred"][idx].detach().cpu()
            mel_target = batch["mel"][idx].detach().cpu()

            min_len = min(mel_pred.shape[-1], mel_target.shape[-1])
            mel_pred = mel_pred[:, :min_len]
            mel_target = mel_target[:, :min_len]

            diff = torch.abs(mel_pred - mel_target)
            fig, axes = plt.subplots(3, 1, figsize=(14, 10))

            im0 = axes[0].imshow(
                mel_target.numpy(),
                aspect="auto",
                origin="lower",
                interpolation="none",
                cmap="viridis",
            )
            axes[0].set_title(
                f"{mode.capitalize()} - Target Mel-Spectrogram (Sample {idx+1})"
            )
            axes[0].set_ylabel("Mel bin")
            axes[0].set_xlabel("Frame")
            plt.colorbar(im0, ax=axes[0])

            im1 = axes[1].imshow(
                mel_pred.numpy(),
                aspect="auto",
                origin="lower",
                interpolation="none",
                cmap="viridis",
            )
            axes[1].set_title(
                f"{mode.capitalize()} - Predicted Mel-Spectrogram (Sample {idx+1})"
            )
            axes[1].set_ylabel("Mel bin")
            axes[1].set_xlabel("Frame")
            plt.colorbar(im1, ax=axes[1])

            im2 = axes[2].imshow(
                diff.numpy(),
                aspect="auto",
                origin="lower",
                interpolation="none",
                cmap="hot",
            )
            axes[2].set_title(
                f"{mode.capitalize()} - Absolute Difference |pred - target| (Sample {idx+1})"
            )
            axes[2].set_ylabel("Mel bin")
            axes[2].set_xlabel("Frame")
            plt.colorbar(im2, ax=axes[2])

            plt.tight_layout()

            self.writer.add_figure(f"{mode}/mel_comparison_sample_{idx+1}", fig)
            plt.close(fig)

    def _log_predictions(self, batch, mode="train"):
        """
        Log predicted vs target duration, pitch, energy.
        """
        if self.writer is None:
            return

        idx = 0

        if "duration_pred" in batch and "duration" in batch:
            dur_pred = batch["duration_pred"][idx].detach().cpu().numpy()
            dur_target = batch["duration"][idx].detach().cpu().numpy()

            min_len = min(len(dur_pred), len(dur_target))

            fig, ax = plt.subplots(1, 1, figsize=(12, 4))
            x = np.arange(min_len)
            ax.plot(x, dur_pred[:min_len], label="Predicted", alpha=0.7)
            ax.plot(x, dur_target[:min_len], label="Target", alpha=0.7)
            ax.set_xlabel("Phoneme index")
            ax.set_ylabel("Duration (frames)")
            ax.set_title(f"{mode.capitalize()} - Duration Prediction")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            self.writer.add_figure(f"{mode}/duration", fig)
            plt.close(fig)

        if "pitch_pred" in batch and "pitch" in batch:
            pitch_pred = batch["pitch_pred"][idx].detach().cpu().numpy()
            pitch_target = batch["pitch"][idx].detach().cpu().numpy()

            min_len = min(len(pitch_pred), len(pitch_target))

            fig, ax = plt.subplots(1, 1, figsize=(12, 4))
            x = np.arange(min_len)
            ax.plot(x, pitch_pred[:min_len], label="Predicted", alpha=0.7)
            ax.plot(x, pitch_target[:min_len], label="Target", alpha=0.7)
            ax.set_xlabel("Frame")
            ax.set_ylabel("Pitch (Hz)")
            ax.set_title(f"{mode.capitalize()} - Pitch Prediction")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            self.writer.add_figure(f"{mode}/pitch", fig)
            plt.close(fig)

        if "energy_pred" in batch and "energy" in batch:
            energy_pred = batch["energy_pred"][idx].detach().cpu().numpy()
            energy_target = batch["energy"][idx].detach().cpu().numpy()

            min_len = min(len(energy_pred), len(energy_target))

            fig, ax = plt.subplots(1, 1, figsize=(12, 4))
            x = np.arange(min_len)
            ax.plot(x, energy_pred[:min_len], label="Predicted", alpha=0.7)
            ax.plot(x, energy_target[:min_len], label="Target", alpha=0.7)
            ax.set_xlabel("Frame")
            ax.set_ylabel("Energy")
            ax.set_title(f"{mode.capitalize()} - Energy Prediction")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            self.writer.add_figure(f"{mode}/energy", fig)
            plt.close(fig)

    def _log_audio(self, batch, mode="train"):
        """
        Log synthesized audio to experiment tracker.
        """
        if self.writer is None:
            return

        if mode == "train":
            return

        vocoder_config = self.config.get("vocoder", None)
        if vocoder_config is None or not vocoder_config.get("enabled", False):
            self.logger.debug(
                "Vocoder not enabled or configured. Skipping audio logging."
            )
            return

        try:
            import json

            from waveglow.model import WaveGlow
            from waveglow.modules import UpsampleNet

            if not hasattr(self, "_vocoder_cache"):
                vocoder_path = vocoder_config.get("checkpoint_path")
                waveglow_params_path = vocoder_config.get(
                    "params_path", "waveglow_params.json"
                )

                if vocoder_path is None:
                    self.logger.debug(
                        "Vocoder checkpoint path not configured. Skipping audio logging."
                    )
                    return

                with open(waveglow_params_path, "r") as f:
                    waveglow_params = json.load(f)

                upsample_net = UpsampleNet(
                    upsample_factor=waveglow_params["upsample_net"]["upsample_factor"],
                    upsample_method=waveglow_params["upsample_net"]["upsample_method"],
                    squeeze_factor=waveglow_params["waveglow"]["squeeze_factor"],
                )

                input_channels = waveglow_params["upsample_net"]["input_channels"]
                local_condition_channels = (
                    input_channels * waveglow_params["waveglow"]["squeeze_factor"]
                )
                vocoder = WaveGlow(
                    squeeze_factor=waveglow_params["waveglow"]["squeeze_factor"],
                    num_layers=waveglow_params["waveglow"]["num_layers"],
                    wn_filter_width=waveglow_params["waveglow"]["wn_filter_width"],
                    wn_dilation_layers=waveglow_params["waveglow"][
                        "wn_dilation_layers"
                    ],
                    wn_residual_channels=waveglow_params["waveglow"][
                        "wn_residual_channels"
                    ],
                    wn_dilation_channels=waveglow_params["waveglow"][
                        "wn_dilation_channels"
                    ],
                    wn_skip_channels=waveglow_params["waveglow"]["wn_skip_channels"],
                    local_condition_channels=local_condition_channels,
                )

                checkpoint = torch.load(vocoder_path, map_location=self.device)
                upsample_net.load_state_dict(checkpoint["upsample_net"])
                vocoder.load_state_dict(checkpoint["waveglow"])
                upsample_net.to(self.device).eval()
                vocoder.to(self.device).eval()
                self._vocoder_cache = {
                    "params": waveglow_params,
                    "upsample_net": upsample_net,
                    "vocoder": vocoder,
                }

                self.logger.info("Vocoder loaded successfully for audio logging.")

            waveglow_params = self._vocoder_cache["params"]
            upsample_net = self._vocoder_cache["upsample_net"]
            vocoder = self._vocoder_cache["vocoder"]

            sample_rate = waveglow_params["waveglow"]["sample_rate"]

            num_samples_to_log = min(4, batch["mel_pred"].size(0))

            self.logger.info(
                f"Logging {num_samples_to_log} audio samples for mode={mode}..."
            )

            for idx in range(num_samples_to_log):
                mel_pred = batch["mel_pred"][idx].detach()

                if "mel_lengths" in batch:
                    mel_len = batch["mel_lengths"][idx].item()
                    mel_pred = mel_pred[:, :mel_len]
                else:
                    mel_len = mel_pred.shape[-1]

                if mel_len == 0 or mel_pred.shape[-1] == 0:
                    self.logger.warning(
                        f"Skipping audio sample {idx+1}: empty mel-spectrogram (length={mel_len})"
                    )
                    continue

                text_content = None
                if "text" in batch:
                    if isinstance(batch["text"], list):
                        text_content = (
                            batch["text"][idx] if idx < len(batch["text"]) else None
                        )
                    elif isinstance(batch["text"], str):
                        text_content = batch["text"]

                audio_id = None
                if "audio_ids" in batch and idx < len(batch["audio_ids"]):
                    audio_id = batch["audio_ids"][idx]

                mel_tensor = mel_pred.unsqueeze(0)

                with torch.no_grad():
                    local_condition = upsample_net(mel_tensor)
                    noise = (
                        torch.FloatTensor(
                            1,
                            waveglow_params["waveglow"]["squeeze_factor"],
                            local_condition.shape[2],
                        )
                        .normal_(0.0, 0.6)
                        .to(self.device)
                    )
                    waveform = vocoder(
                        noise,
                        reverse=True,
                        logdet=None,
                        local_condition=local_condition,
                    )
                    waveform = torch.clamp(waveform[0], min=-1.0, max=1.0)
                    audio = waveform.squeeze(0).cpu().numpy()

                audio_metadata = {
                    "sample_idx": idx + 1,
                    "mode": mode,
                }
                if text_content:
                    audio_metadata["text"] = text_content
                if audio_id:
                    audio_metadata["audio_id"] = audio_id
                if "mel_lengths" in batch:
                    audio_metadata["mel_length"] = mel_len

                self.writer.add_audio(
                    f"{mode}/audio_sample_{idx+1}",
                    audio,
                    sample_rate=sample_rate,
                    metadata=audio_metadata,
                )

                log_msg = f"Logged audio sample {idx+1} to {mode}/audio_sample_{idx+1}"
                if text_content:
                    log_msg += (
                        f" | Text: '{text_content[:50]}...'"
                        if len(text_content) > 50
                        else f" | Text: '{text_content}'"
                    )
                self.logger.info(log_msg)

                if "audio" in batch and idx < len(batch["audio"]):
                    target_audio = batch["audio"][idx].detach().cpu().numpy()
                    target_metadata = audio_metadata.copy()
                    target_metadata["type"] = "target"

                    self.writer.add_audio(
                        f"{mode}/audio_target_sample_{idx+1}",
                        target_audio,
                        sample_rate=sample_rate,
                        metadata=target_metadata,
                    )
                    self.logger.info(
                        f"Logged target audio sample {idx+1} to {mode}/audio_target_sample_{idx+1}"
                    )

        except ImportError as e:
            self.logger.warning(
                f"Vocoder modules not available: {e}. Skipping audio logging."
            )
        except FileNotFoundError as e:
            self.logger.warning(
                f"Vocoder files not found: {e}. Skipping audio logging."
            )
        except Exception as e:
            self.logger.error(f"Error generating audio for logging: {e}")
            import traceback

            self.logger.error(traceback.format_exc())
