import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from fastspeech2.datasets.ruslan_dataset import RUSLANDataset
from fastspeech2.utils.io_utils import ROOT_PATH

logger = logging.getLogger(__name__)


class RUSLANFeatureDataset(RUSLANDataset):
    """
    Extended RUSLAN dataset that loads pre-extracted features.

    This dataset is used for FastSpeech2 training when mel-spectrograms,
    pitch, energy, and duration have been pre-computed and cached.

    Features should be stored in the following structure:
    {features_dir}/
        mel/
            {audio_id}.npy
        pitch/
            {audio_id}.npy
        energy/
            {audio_id}.npy
        duration/
            {audio_id}.npy
        text_tokens/
            {audio_id}.npy

    """

    EMOTION_MAP = {
        "anger": 0,
        "disgust": 1,
        "enthusiasm": 2,
        "fear": 3,
        "happiness": 4,
        "neutral": 5,
        "sadness": 6,
    }

    def __init__(
        self,
        features_dir: str = "data/features",
        load_audio: bool = False,
        vocabulary_path: Optional[str] = None,
        emotion_labels_path: Optional[str] = None,
        max_samples: Optional[int] = None,
        *args,
        **kwargs,
    ):
        """
        Initialize feature dataset.
        """
        super().__init__(*args, **kwargs)

        if max_samples is not None and max_samples > 0:
            self._index = self._index[:max_samples]
            logger.info(
                f"Limited dataset to {max_samples} samples for overfitting test"
            )

        self.features_dir = ROOT_PATH / features_dir
        self.load_audio = load_audio

        self.mel_dir = self.features_dir / "mel"
        self.pitch_dir = self.features_dir / "pitch"
        self.energy_dir = self.features_dir / "energy"
        self.duration_dir = self.features_dir / "duration"
        self.text_tokens_dir = self.features_dir / "text_tokens"

        self.has_mel = self.mel_dir.exists()
        self.has_pitch = self.pitch_dir.exists()
        self.has_energy = self.energy_dir.exists()
        self.has_duration = self.duration_dir.exists()
        self.has_text_tokens = self.text_tokens_dir.exists()

        logger.info("Feature availability:")
        logger.info(f"  Mel: {self.has_mel}")
        logger.info(f"  Pitch: {self.has_pitch}")
        logger.info(f"  Energy: {self.has_energy}")
        logger.info(f"  Duration: {self.has_duration}")
        logger.info(f"  Text tokens: {self.has_text_tokens}")

        self.vocabulary = None
        if vocabulary_path is not None:
            vocab_path = ROOT_PATH / vocabulary_path
            if vocab_path.exists():
                self.vocabulary = self._load_vocabulary(vocab_path)
                logger.info(f"Loaded vocabulary with {len(self.vocabulary)} tokens")

        self.emotion_labels = {}
        self.has_emotions = False
        if emotion_labels_path is not None:
            emotion_path = ROOT_PATH / emotion_labels_path
            if emotion_path.exists():
                self.emotion_labels = self._load_emotion_labels(emotion_path)
                self.has_emotions = True
                logger.info(
                    f"Loaded emotion labels for {len(self.emotion_labels)} samples"
                )
            else:
                logger.warning(f"Emotion labels file not found: {emotion_path}")

    def _load_emotion_labels(self, emotion_path: Path) -> Dict[str, int]:
        """
        Load emotion labels from JSON file.
        """
        import json

        emotion_dict = {}
        try:
            with open(emotion_path, "r", encoding="utf-8") as f:
                emotion_data = json.load(f)

            for item in emotion_data:
                audio_id = item["audio_id"]
                emotion_name = item["emotion"]
                emotion_idx = self.EMOTION_MAP.get(emotion_name, 5)
                emotion_dict[audio_id] = emotion_idx

            logger.info(f"Loaded {len(emotion_dict)} emotion labels")
        except Exception as e:
            logger.error(f"Failed to load emotion labels: {e}")

        return emotion_dict

    def _load_vocabulary(self, vocab_path: Path) -> Dict[str, int]:
        """
        Load vocabulary from file.
        """
        vocabulary = {}
        with open(vocab_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                token = line.strip()
                vocabulary[token] = idx
        return vocabulary

    def __getitem__(self, ind):
        """
        Get a single item with pre-extracted features.
        """
        data_dict = self._index[ind]
        audio_id = data_dict["audio_id"]
        text = data_dict["text"]

        instance_data = {"text": text, "audio_id": audio_id}

        if self.has_mel:
            mel_path = self.mel_dir / f"{audio_id}.npy"
            if mel_path.exists():
                mel = np.load(mel_path)
                instance_data["mel"] = torch.FloatTensor(mel)
            else:
                logger.warning(f"Mel file not found: {mel_path}")

        if self.has_pitch:
            pitch_path = self.pitch_dir / f"{audio_id}.npy"
            if pitch_path.exists():
                pitch = np.load(pitch_path)
                instance_data["pitch"] = torch.FloatTensor(pitch)
            else:
                logger.warning(f"Pitch file not found: {pitch_path}")

        if self.has_energy:
            energy_path = self.energy_dir / f"{audio_id}.npy"
            if energy_path.exists():
                energy = np.load(energy_path)
                instance_data["energy"] = torch.FloatTensor(energy)
            else:
                logger.warning(f"Energy file not found: {energy_path}")

        if self.has_duration:
            duration_path = self.duration_dir / f"{audio_id}.npy"
            if duration_path.exists():
                duration = np.load(duration_path)
                instance_data["duration"] = torch.LongTensor(duration)
            else:
                logger.warning(f"Duration file not found: {duration_path}")

        if self.has_text_tokens:
            tokens_path = self.text_tokens_dir / f"{audio_id}.npy"
            if tokens_path.exists():
                text_tokens = np.load(tokens_path)
                instance_data["text_tokens"] = torch.LongTensor(text_tokens)
            elif self.vocabulary is not None:
                tokens = [
                    self.vocabulary.get(char, self.vocabulary.get("<unk>", 0))
                    for char in text
                ]
                instance_data["text_tokens"] = torch.LongTensor(tokens)
            else:
                logger.debug(f"Text tokens file not found: {tokens_path}")
        elif self.vocabulary is not None:
            tokens = [
                self.vocabulary.get(char, self.vocabulary.get("<unk>", 0))
                for char in text
            ]
            instance_data["text_tokens"] = torch.LongTensor(tokens)

        if self.load_audio:
            audio_path = data_dict["audio_path"]
            import torchaudio

            audio, sample_rate = torchaudio.load(audio_path)
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            instance_data["audio"] = audio.squeeze(0)
            instance_data["sample_rate"] = sample_rate

        if self.has_emotions and audio_id in self.emotion_labels:
            instance_data["emotion"] = torch.LongTensor([self.emotion_labels[audio_id]])

        instance_data = self.preprocess_data(instance_data)

        return instance_data

    def verify_features(self, num_samples: int = 10) -> Dict[str, bool]:
        """
        Verify that features are available for a sample of the dataset.
        """
        results = {
            "mel": [],
            "pitch": [],
            "energy": [],
            "duration": [],
            "text_tokens": [],
        }

        num_to_check = min(num_samples, len(self._index))

        for i in range(num_to_check):
            audio_id = self._index[i]["audio_id"]

            results["mel"].append((self.mel_dir / f"{audio_id}.npy").exists())
            results["pitch"].append((self.pitch_dir / f"{audio_id}.npy").exists())
            results["energy"].append((self.energy_dir / f"{audio_id}.npy").exists())
            results["duration"].append((self.duration_dir / f"{audio_id}.npy").exists())
            results["text_tokens"].append(
                (self.text_tokens_dir / f"{audio_id}.npy").exists()
            )

        summary = {}
        for key, values in results.items():
            success_rate = sum(values) / len(values) if values else 0
            summary[key] = {
                "success_rate": success_rate,
                "available": sum(values),
                "total": len(values),
            }

        return summary
