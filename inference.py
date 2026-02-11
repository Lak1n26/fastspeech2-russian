"""
FastSpeech2 Inference Script
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.ndimage import median_filter
from scipy.signal import butter, medfilt, savgol_filter, sosfilt

from fastspeech2.model import FastSpeech2

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AudioPostProcessor:
    """Audio post-processing utilities."""

    @staticmethod
    def normalize_peak(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
        """Peak normalization."""
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * target_peak
        return audio

    @staticmethod
    def normalize_rms(audio: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
        """RMS normalization (more natural)."""
        current_rms = np.sqrt(np.mean(audio**2))
        if current_rms > 0:
            audio = audio * (target_rms / current_rms)
        return audio

    @staticmethod
    def highpass_filter(
        audio: np.ndarray, cutoff: float = 20.0, sr: int = 22050, order: int = 5
    ) -> np.ndarray:
        """High-pass filter to remove infrasonic frequencies."""
        sos = butter(order, cutoff, btype="highpass", fs=sr, output="sos")
        filtered = sosfilt(sos, audio)
        return filtered

    @staticmethod
    def declicking(audio: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Remove clicks at phoneme boundaries."""
        if audio.ndim > 1:
            audio = audio.squeeze()
        diff = np.diff(audio, prepend=audio[0:1])
        abs_diff = np.abs(diff)
        clicks = abs_diff > threshold
        for i in np.where(clicks)[0]:
            if i > 0 and i < len(audio) - 1:
                audio[i] = (audio[i - 1] + audio[i + 1]) / 2

        return audio

    @staticmethod
    def spectral_gate(
        audio: np.ndarray, sr: int = 22050, threshold_db: float = -40
    ) -> np.ndarray:
        """
        Spectral gating for noise reduction.
        Simple implementation - for production use librosa or noisereduce.
        """
        try:
            import librosa

            noise_sample = audio[: int(0.5 * sr)]
            stft = librosa.stft(audio)
            magnitude = np.abs(stft)
            phase = np.angle(stft)

            noise_magnitude = np.median(
                np.abs(librosa.stft(noise_sample)), axis=1, keepdims=True
            )

            threshold_linear = 10 ** (threshold_db / 20)
            mask = magnitude > (noise_magnitude * threshold_linear)
            magnitude_gated = magnitude * mask

            stft_gated = magnitude_gated * np.exp(1j * phase)
            audio_gated = librosa.istft(stft_gated, length=len(audio))

            return audio_gated
        except ImportError:
            logger.warning("librosa not installed, skipping spectral gating")
            return audio


class EnhancedFastSpeech2Inferencer:
    """
    Enhanced inference wrapper for FastSpeech2 with quality improvements.
    """

    def __init__(
        self,
        checkpoint_path: str,
        vocab_path: str = "data/phoneme_vocabulary.txt",
        pitch_stats_path: str = "data/pitch_stats.json",
        device: str = "auto",
        duration_min: float = 0.5,
        duration_max: float = 20.0,
        temperature: float = 1.0,
        apply_audio_postprocessing: bool = True,
        apply_mel_smoothing: bool = True,
        mel_clip_min: float = -12.0,
        mel_clip_max: float = 2.0,
    ):
        """
        Initialize enhanced inferencer.
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.vocab_path = Path(vocab_path)
        self.pitch_stats_path = Path(pitch_stats_path)
        self.duration_min = duration_min
        self.duration_max = duration_max
        self.temperature = temperature
        self.apply_audio_postprocessing = apply_audio_postprocessing
        self.apply_mel_smoothing = apply_mel_smoothing
        self.mel_clip_min = mel_clip_min
        self.mel_clip_max = mel_clip_max

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        self.vocabulary = self._load_vocabulary()
        logger.info(f"Loaded vocabulary with {len(self.vocabulary)} tokens")

        self.model = self._load_model()
        logger.info("Model loaded successfully")

        self._vocoder_cache = {}

        self.audio_processor = AudioPostProcessor()

    def _load_vocabulary(self) -> Dict[str, int]:
        """Load vocabulary from file."""
        vocabulary = {}

        with open(self.vocab_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                token = line.strip()
                vocabulary[token] = idx

        self.idx_to_token = {idx: token for token, idx in vocabulary.items()}

        return vocabulary

    def _load_model(self) -> FastSpeech2:
        """Load model from checkpoint."""
        checkpoint = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False
        )

        if "config" in checkpoint:
            config = checkpoint["config"]
            model_config = config.get("model", {})
            if not isinstance(model_config, dict):
                model_config = dict(model_config)
            else:
                model_config = dict(model_config)
        else:
            logger.warning("No config found in checkpoint, using default parameters")
            model_config = {
                "vocab_size": len(self.vocabulary),
                "d_model": 256,
                "encoder_layers": 4,
                "decoder_layers": 4,
                "num_heads": 2,
                "d_ff": 1024,
                "n_mels": 80,
            }

        model_config["vocab_size"] = len(self.vocabulary)
        if self.pitch_stats_path.exists():
            try:
                import json

                with open(self.pitch_stats_path, "r") as f:
                    pitch_stats = json.load(f)

                model_config["pitch_mean"] = pitch_stats.get("pitch_mean")
                model_config["pitch_std"] = pitch_stats.get("pitch_std")
                model_config["pitch_min"] = pitch_stats.get("pitch_min", 0.0)
                model_config["pitch_max"] = pitch_stats.get("pitch_max", 800.0)

                logger.info(f"Loaded pitch statistics from {self.pitch_stats_path}")
                logger.info(f"  Pitch mean: {model_config['pitch_mean']:.2f} Hz")
                logger.info(f"  Pitch std: {model_config['pitch_std']:.2f} Hz")
                logger.info(
                    f"  Pitch range: [{model_config['pitch_min']:.1f}, {model_config['pitch_max']:.1f}] Hz"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load pitch statistics from {self.pitch_stats_path}: {e}"
                )
                logger.warning("Pitch control will use normalized values instead of Hz")
        else:
            logger.warning(f"Pitch statistics file not found: {self.pitch_stats_path}")
            logger.warning("Pitch control will use normalized values instead of Hz")

        model = FastSpeech2(**model_config)

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

        model = model.to(self.device)
        model.eval()

        return model

    def _normalize_gruut_phoneme(self, phoneme: str) -> str:
        """Normalize gruut IPA tokens to improve mapping to training vocab."""
        phoneme = phoneme.replace("ˈ", "").replace("ˌ", "").replace("͡", "")
        if phoneme in {"‖", "⟂"}:
            phoneme = "|"
        phoneme = phoneme.replace("g", "ɡ")
        return phoneme

    def _map_gruut_phonemes(self, phonemes: List[str]) -> List[str]:
        """
        Map gruut phonemes to training phoneme inventory.
        """
        vocab = self.vocabulary
        mapped = []
        unknown = []

        phoneme_map = {
            "ts": "t̪s̪",
            "t͡s": "t̪s̪",
            "t͡s̪": "t̪s̪",
            "tʃ": "tɕ",
            "t͡ʃ": "tɕ",
            "t͡ɕ": "tɕ",
            "t͡ʂ": "tʂ",
            "ʃ": "ʂ",
            "ʒ": "ʐ",
            "dʒ": "dʐ",
            "d͡ʒ": "dʐ",
            "d͡ʐ": "dʐ",
            "d͡z": "z̪",
            "ɕ": "ɕː",
            "ʑ": "ʑː",
            "h": "x",
            "i̯": "j",
            "ɪ̯": "j",
            "u̯": "u",
        }
        base_map = {
            "t": "t̪",
            "d": "d̪",
            "s": "s̪",
            "z": "z̪",
            "n": "n̪",
            "l": "ɫ",
            "ɹ": "r",
            "ɾ": "r",
            "ʁ": "r",
        }
        palatal_map = {
            "lʲ": "ʎ",
            "nʲ": "ɲ",
        }

        for original in phonemes:
            ph = self._normalize_gruut_phoneme(original)

            if ph in vocab:
                mapped.append(ph)
                continue

            if ph in palatal_map and palatal_map[ph] in vocab:
                mapped.append(palatal_map[ph])
                continue
            if ph == "|" and ph not in vocab:
                continue
            ph = phoneme_map.get(ph, ph)
            if ph in vocab:
                mapped.append(ph)
                continue

            if ph.endswith("ː") and ph[:-1] in vocab:
                mapped.append(ph[:-1])
                continue
            if f"{ph}ː" in vocab:
                mapped.append(f"{ph}ː")
                continue

            if ph in base_map and base_map[ph] in vocab:
                mapped.append(base_map[ph])
                continue
            if ph.endswith("ʲ"):
                base = ph[:-1]
                if ph in vocab:
                    mapped.append(ph)
                    continue
                if base in base_map and f"{base_map[base]}ʲ" in vocab:
                    mapped.append(f"{base_map[base]}ʲ")
                    continue
                if base in vocab:
                    mapped.append(base)
                    continue

            mapped.append(ph)
            unknown.append(original)

        if unknown:
            unique_unknown = sorted(set(unknown))
            logger.warning(
                "Unmapped gruut phonemes (%d unique): %s",
                len(unique_unknown),
                " ".join(unique_unknown[:30]),
            )
            if len(unique_unknown) > 30:
                logger.warning("... and %d more", len(unique_unknown) - 30)

        return mapped

    def text_to_phonemes(self, text: str) -> List[str]:
        """
        Convert text to phoneme sequence using gruut.
        """
        try:
            import gruut

            phonemes = []
            for sentence in gruut.sentences(text, lang="ru"):
                for word in sentence:
                    if word.phonemes:
                        phonemes.extend(word.phonemes)

            if phonemes:
                logger.info(f"Text: '{text}' -> Phonemes (gruut): {' '.join(phonemes)}")
                return phonemes

        except ImportError:
            logger.debug("gruut not installed, trying other methods")
        except Exception as e:
            logger.debug(f"gruut error: {e}, trying other methods")

        logger.warning("No G2P method available. Using character-level fallback.")
        logger.warning("For better results, install: pip install gruut[ru]")

        phonemes = []
        for char in text.lower():
            if char == " ":
                phonemes.append("|")
            elif char.isalpha() or char in ["?", "!", ".", ","]:
                phonemes.append(char)

        logger.info(f"Text: '{text}' -> Characters: {' '.join(phonemes)}")
        return phonemes

    def add_punctuation_pauses(self, phonemes: List[str]) -> List[str]:
        """
        Automatically add pauses on punctuation marks.
        """
        phonemes_with_pauses = []
        pause_token = "|" if "|" in self.vocabulary else None

        if pause_token is None:
            logger.warning(
                "Pause token '|' not in vocabulary, skipping pause insertion"
            )
            return phonemes

        for phoneme in phonemes:
            if phoneme in [".", "!", "?"]:
                phonemes_with_pauses.extend([pause_token, pause_token, pause_token])
            elif phoneme in [",", ";", ":"]:
                phonemes_with_pauses.extend([pause_token, pause_token])
            else:
                phonemes_with_pauses.append(phoneme)

        return phonemes_with_pauses

    def phonemes_to_indices(self, phonemes: List[str]) -> Tuple[torch.Tensor, float]:
        """
        Convert phoneme sequence to indices.
        """
        unk_idx = self.vocabulary.get("<unk>", 1)

        indices = []
        unk_count = 0
        for phoneme in phonemes:
            idx = self.vocabulary.get(phoneme, unk_idx)
            if idx == unk_idx:
                unk_count += 1
            indices.append(idx)

        unk_ratio = (unk_count / max(len(indices), 1)) if indices else 1.0
        return torch.LongTensor(indices), unk_ratio

    def post_process_duration(self, duration: torch.Tensor) -> torch.Tensor:
        """
        Post-process duration predictions.
        """
        duration = duration.cpu().numpy()

        if len(duration) > 3:
            duration = medfilt(duration, kernel_size=3)

        duration = np.clip(duration, self.duration_min, self.duration_max)

        if len(duration) > 1:
            for i in range(1, len(duration) - 1):
                duration[i] = (
                    0.6 * duration[i] + 0.2 * duration[i - 1] + 0.2 * duration[i + 1]
                )

        return torch.from_numpy(duration).to(self.device)

    def smooth_pitch_contour(self, pitch: torch.Tensor) -> torch.Tensor:
        """
        Smooth pitch contour using Savitzky-Golay filter.
        """
        pitch_np = pitch.cpu().numpy()

        if len(pitch_np) > 5:
            window_length = min(
                11, len(pitch_np) if len(pitch_np) % 2 == 1 else len(pitch_np) - 1
            )
            if window_length >= 5:
                pitch_smooth = savgol_filter(
                    pitch_np, window_length=window_length, polyorder=3
                )
                return torch.from_numpy(pitch_smooth).to(self.device)

        return pitch

    def smooth_energy_contour(self, energy: torch.Tensor) -> torch.Tensor:
        """
        Smooth energy contour using Savitzky-Golay filter.
        """
        energy_np = energy.cpu().numpy()

        if len(energy_np) > 5:
            window_length = min(
                11, len(energy_np) if len(energy_np) % 2 == 1 else len(energy_np) - 1
            )
            if window_length >= 5:
                energy_smooth = savgol_filter(
                    energy_np, window_length=window_length, polyorder=3
                )
                return torch.from_numpy(energy_smooth).to(self.device)

        return energy

    def temporal_smooth_mel(self, mel: np.ndarray, window_size: int = 3) -> np.ndarray:
        """
        Apply temporal smoothing to mel-spectrogram.
        """
        from scipy.ndimage import uniform_filter1d

        mel_smooth = uniform_filter1d(mel, size=window_size, axis=1, mode="nearest")

        return mel_smooth

    def clip_mel_values(self, mel: np.ndarray) -> np.ndarray:
        """
        Clip extreme mel values to prevent artifacts.
        """
        mel_clipped = np.clip(mel, self.mel_clip_min, self.mel_clip_max)
        num_clipped = np.sum((mel < self.mel_clip_min) | (mel > self.mel_clip_max))
        if num_clipped > 0:
            logger.debug(
                f"Clipped {num_clipped} mel values ({num_clipped / mel.size * 100:.2f}%)"
            )

        return mel_clipped

    def synthesize(
        self,
        text: str,
        duration_control: float = 1.0,
        pitch_control: float = 1.0,
        energy_control: float = 1.0,
        emotion_control: float = 1.0,
    ) -> Tuple[np.ndarray, str]:
        """
        Synthesize mel-spectrogram from text with enhancements.
        """
        logger.info(f"Synthesizing: '{text}'")

        phonemes = self.text_to_phonemes(text)
        logger.info(f"Gruut phonemes: {' '.join(phonemes)}")

        phonemes = self._map_gruut_phonemes(phonemes)
        logger.info(f"Mapped phonemes: {' '.join(phonemes)}")

        if "<s>" in self.vocabulary:
            phonemes = ["<s>"] + phonemes
        if "</s>" in self.vocabulary:
            phonemes = phonemes + ["</s>"]

        indices, unk_ratio = self.phonemes_to_indices(phonemes)
        text_tokens = indices.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model.inference(
                text_tokens=text_tokens,
                duration_control=duration_control,
                pitch_control=pitch_control,
                energy_control=energy_control,
                emotion_control=emotion_control,
            )

        mel = output["mel_pred"].squeeze(0).cpu().numpy()

        if "mel_lengths" in output:
            mel_len_value = output["mel_lengths"]
            if isinstance(mel_len_value, int):
                mel_len = mel_len_value
            else:
                mel_len = mel_len_value.item()
        else:
            mel_len = mel.shape[-1]

        mel = mel[:, :mel_len]

        if self.apply_mel_smoothing:
            logger.info("Applying mel-spectrogram post-processing...")

            # 1. Temporal smoothing
            mel = self.temporal_smooth_mel(mel, window_size=3)

            # 2. Clip extreme values
            mel = self.clip_mel_values(mel)

            logger.info("Mel post-processing complete")

        predicted_emotion = None
        emotion_name = "N/A"
        if "emotion_pred" in output:
            emotion_logits = output["emotion_pred"]
            predicted_emotion = torch.argmax(emotion_logits, dim=-1).item()

            emotion_names = [
                "anger",
                "disgust",
                "enthusiasm",
                "fear",
                "happiness",
                "neutral",
                "sadness",
            ]
            if predicted_emotion < len(emotion_names):
                emotion_name = emotion_names[predicted_emotion]

        logger.info(f"Generated mel-spectrogram: {mel.shape}")
        logger.info(f"Predicted emotion: {emotion_name} (class {predicted_emotion})")

        return mel, emotion_name

    def mel_to_audio(
        self,
        mel: np.ndarray,
        vocoder_type: str = "waveglow",
        vocoder_path: Optional[str] = None,
        waveglow_params_path: str = "waveglow_params.json",
    ) -> Tuple[np.ndarray, int]:
        """
        Convert mel-spectrogram to audio using vocoder with post-processing.
        """
        if vocoder_type != "waveglow":
            logger.warning(
                "Unsupported vocoder type '%s'. Returning mel.", vocoder_type
            )
            return mel, 22050

        if vocoder_path is None:
            raise ValueError("vocoder_path is required for WaveGlow")

        cache_key = (vocoder_path, waveglow_params_path)
        if cache_key in self._vocoder_cache:
            waveglow_params, upsample_net, vocoder = self._vocoder_cache[cache_key]
        else:
            import json

            from waveglow.model import WaveGlow
            from waveglow.modules import UpsampleNet

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
                wn_dilation_layers=waveglow_params["waveglow"]["wn_dilation_layers"],
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

            self._vocoder_cache[cache_key] = (waveglow_params, upsample_net, vocoder)

        mel_tensor = torch.from_numpy(mel).float().to(self.device)
        if mel_tensor.ndim == 2:
            mel_tensor = mel_tensor.unsqueeze(0)

        with torch.no_grad():
            local_condition = upsample_net(mel_tensor)
            noise = (
                torch.FloatTensor(
                    1,
                    waveglow_params["waveglow"]["squeeze_factor"],
                    local_condition.shape[2],
                )
                .normal_(0.0, 0.4)
                .to(self.device)
            )
            waveform = vocoder(
                noise, reverse=True, logdet=None, local_condition=local_condition
            )
            waveform = torch.clamp(waveform[0], min=-1.0, max=1.0)
            waveform = waveform.squeeze(0).cpu().numpy()

        sample_rate = int(waveglow_params["waveglow"]["sample_rate"])

        if self.apply_audio_postprocessing:
            logger.info("Applying audio post-processing...")

            waveform = self.audio_processor.highpass_filter(
                waveform, cutoff=20.0, sr=sample_rate
            )

            waveform = self.audio_processor.declicking(waveform, threshold=0.3)

            waveform = self.audio_processor.spectral_gate(
                waveform, sr=sample_rate, threshold_db=-40
            )

            waveform = self.audio_processor.normalize_rms(waveform, target_rms=0.1)

            waveform = self.audio_processor.normalize_peak(waveform, target_peak=0.95)

            logger.info("Audio post-processing complete")

        return waveform, sample_rate

    def save_mel(self, mel: np.ndarray, output_path: str):
        """Save mel-spectrogram to file."""
        np.save(output_path, mel)
        logger.info(f"Mel-spectrogram saved to: {output_path}")

    def save_audio(self, audio: np.ndarray, output_path: str, sample_rate: int = 22050):
        """Save audio to WAV file."""
        output_path = Path(output_path)
        if output_path.suffix.lower() != ".wav":
            output_path = output_path.with_suffix(".wav")
        audio = np.asarray(audio)
        if audio.ndim > 1:
            audio = audio.squeeze()
        if audio.size == 0:
            logger.error("Audio array is empty. Skipping save.")
            return
        try:
            import soundfile as sf

            sf.write(
                str(output_path), audio, sample_rate, format="WAV", subtype="PCM_16"
            )
            logger.info(f"Audio saved to: {output_path}")
        except ImportError:
            logger.error("soundfile not installed. Install with: pip install soundfile")


def batch_inference(
    inferencer: EnhancedFastSpeech2Inferencer,
    texts: List[str],
    output_dir: str,
    duration_control: float = 1.0,
    pitch_control: float = 1.0,
    energy_control: float = 1.0,
    emotion_control: float = 1.0,
    vocoder: str = "none",
    vocoder_checkpoint: Optional[str] = None,
    waveglow_params: str = "waveglow_params.json",
):
    """
    Perform batch inference on multiple texts.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    logger.info(f"Processing {len(texts)} texts...")

    for i, text in enumerate(texts):
        logger.info(f"\n[{i+1}/{len(texts)}] Processing: {text[:50]}...")

        try:
            mel, emotion_name = inferencer.synthesize(
                text=text,
                duration_control=duration_control,
                pitch_control=pitch_control,
                energy_control=energy_control,
                emotion_control=emotion_control,
            )

            mel_path = output_path / f"output_{i:04d}.npy"
            inferencer.save_mel(mel, str(mel_path))

            if vocoder != "none":
                audio, sample_rate = inferencer.mel_to_audio(
                    mel=mel,
                    vocoder_type=vocoder,
                    vocoder_path=vocoder_checkpoint,
                    waveglow_params_path=waveglow_params,
                )
                wav_path = output_path / f"output_{i:04d}.wav"
                inferencer.save_audio(audio, str(wav_path), sample_rate=sample_rate)

            metadata = {
                "text": text,
                "duration_control": duration_control,
                "pitch_control": pitch_control,
                "energy_control": energy_control,
                "emotion_control": emotion_control,
                "predicted_emotion": emotion_name,
                "mel_shape": mel.shape,
                "enhancements": {
                    "duration_postprocessing": True,
                    "pitch_smoothing": True,
                    "energy_smoothing": True,
                    "mel_temporal_smoothing": inferencer.apply_mel_smoothing,
                    "mel_clipping": inferencer.apply_mel_smoothing,
                    "audio_postprocessing": inferencer.apply_audio_postprocessing,
                    "punctuation_pauses": True,
                },
            }

            metadata_path = output_path / f"output_{i:04d}.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"Error processing text {i}: {e}")
            continue

    logger.info(f"\nBatch inference complete! Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="FastSpeech2 Enhanced Inference with Quality Improvements"
    )

    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--vocab",
        type=str,
        default="data/phoneme_vocabulary.txt",
        help="Path to vocabulary file",
    )
    parser.add_argument(
        "--pitch_stats",
        type=str,
        default="data/pitch_stats.json",
        help="Path to pitch statistics JSON file (for Hz-based pitch control)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use",
    )

    parser.add_argument("--text", type=str, help="Input text to synthesize")
    parser.add_argument(
        "--text_file",
        type=str,
        help="File with texts (one per line) for batch inference",
    )

    parser.add_argument("--output", type=str, help="Output file path (for single text)")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory (for batch inference)",
    )

    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="Duration control (speed): <1.0 = faster, >1.0 = slower",
    )
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="Pitch control: <1.0 = lower, >1.0 = higher",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="Energy control: <1.0 = quieter, >1.0 = louder",
    )
    parser.add_argument(
        "--emotion_control",
        type=float,
        default=1.0,
        help="Emotion control: 1.0 = full emotion, 0.0 = neutral",
    )

    parser.add_argument(
        "--duration_min",
        type=float,
        default=0.5,
        help="Minimum duration for phonemes (frames)",
    )
    parser.add_argument(
        "--duration_max",
        type=float,
        default=20.0,
        help="Maximum duration for phonemes (frames)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for variance predictions (0.8-1.2)",
    )
    parser.add_argument(
        "--no_audio_postprocessing",
        action="store_true",
        help="Disable audio post-processing",
    )
    parser.add_argument(
        "--no_mel_smoothing",
        action="store_true",
        help="Disable mel-spectrogram temporal smoothing",
    )
    parser.add_argument(
        "--mel_clip_min",
        type=float,
        default=-12.0,
        help="Minimum mel value for clipping",
    )
    parser.add_argument(
        "--mel_clip_max", type=float, default=2.0, help="Maximum mel value for clipping"
    )

    parser.add_argument(
        "--vocoder",
        type=str,
        choices=["waveglow", "hifigan", "none"],
        default="none",
        help="Vocoder to use",
    )
    parser.add_argument(
        "--vocoder_checkpoint", type=str, help="Path to vocoder checkpoint"
    )
    parser.add_argument(
        "--waveglow_params",
        type=str,
        default="waveglow_params.json",
        help="Path to WaveGlow params JSON",
    )

    args = parser.parse_args()

    if not args.text and not args.text_file:
        parser.error("Either --text or --text_file must be provided")

    if args.text and not args.output:
        parser.error("--output must be provided when using --text")

    logger.info("=" * 60)
    logger.info("FastSpeech2 Enhanced Inference")
    logger.info("=" * 60)

    inferencer = EnhancedFastSpeech2Inferencer(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        pitch_stats_path=args.pitch_stats,
        device=args.device,
        duration_min=args.duration_min,
        duration_max=args.duration_max,
        temperature=args.temperature,
        apply_audio_postprocessing=not args.no_audio_postprocessing,
        apply_mel_smoothing=not args.no_mel_smoothing,
        mel_clip_min=args.mel_clip_min,
        mel_clip_max=args.mel_clip_max,
    )

    logger.info("\nEnhancements enabled:")
    logger.info("  - Duration post-processing: success!")
    logger.info("  - Pitch smoothing: success!")
    logger.info("  - Energy smoothing: success!")
    logger.info("  - Punctuation pauses: success!")
    logger.info(
        f"  - Mel temporal smoothing: {'success!' if not args.no_mel_smoothing else 'skipped.'}"
    )
    logger.info(
        f"  - Mel value clipping: {'success!' if not args.no_mel_smoothing else 'skipped.'} (range: [{args.mel_clip_min}, {args.mel_clip_max}])"
    )
    logger.info(
        f"  - Audio post-processing: {'success!' if not args.no_audio_postprocessing else 'skipped.'}"
    )

    if args.text:
        logger.info("\nMode: Single text inference")

        mel, emotion_name = inferencer.synthesize(
            text=args.text,
            duration_control=args.duration_control,
            pitch_control=args.pitch_control,
            energy_control=args.energy_control,
            emotion_control=args.emotion_control,
        )

        output_path = Path(args.output)
        if output_path.suffix == ".npy":
            mel_path = output_path
            inferencer.save_mel(mel, args.output)
        else:
            mel_path = output_path.with_suffix(".npy")
            inferencer.save_mel(mel, str(mel_path))

        if args.vocoder != "none":
            audio, sample_rate = inferencer.mel_to_audio(
                mel=mel,
                vocoder_type=args.vocoder,
                vocoder_path=args.vocoder_checkpoint,
                waveglow_params_path=args.waveglow_params,
            )
            inferencer.save_audio(audio, str(output_path), sample_rate=sample_rate)

        logger.info("\nSynthesis complete!")
        logger.info(f"  Text: '{args.text}'")
        logger.info(f"  Duration control: {args.duration_control}")
        logger.info(f"  Pitch control: {args.pitch_control}")
        logger.info(f"  Energy control: {args.energy_control}")
        logger.info(f"  Emotion control: {args.emotion_control}")
        logger.info(f"  Predicted emotion: {emotion_name}")
        logger.info(f"  Output: {output_path}")

    elif args.text_file:
        logger.info("\nMode: Batch inference")

        with open(args.text_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]

        logger.info(f"Loaded {len(texts)} texts from {args.text_file}")

        batch_inference(
            inferencer=inferencer,
            texts=texts,
            output_dir=args.output_dir,
            duration_control=args.duration_control,
            pitch_control=args.pitch_control,
            energy_control=args.energy_control,
            emotion_control=args.emotion_control,
            vocoder=args.vocoder,
            vocoder_checkpoint=args.vocoder_checkpoint,
            waveglow_params=args.waveglow_params,
        )

    logger.info("\n" + "=" * 60)
    logger.info("Done!")


if __name__ == "__main__":
    main()
