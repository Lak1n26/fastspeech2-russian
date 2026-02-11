import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torchaudio

from fastspeech2.datasets.base_dataset import BaseDataset
from fastspeech2.utils.io_utils import ROOT_PATH

logger = logging.getLogger(__name__)


class RUSLANDataset(BaseDataset):
    """
    Dataset for RUSLAN Russian TTS corpus.

    Loads audio files and corresponding text transcriptions from the RUSLAN dataset.
    """

    def __init__(
        self,
        metadata_path: str = "data/metadata_RUSLAN_22200.csv",
        data_dir: str = "data/RUSLAN",
        max_audio_length: Optional[int] = None,
        min_audio_length: Optional[int] = None,
        *args,
        **kwargs,
    ):
        """
        Initialize RUSLAN dataset.
        """
        self.metadata_path = ROOT_PATH / metadata_path
        self.data_dir = ROOT_PATH / data_dir
        self.max_audio_length = max_audio_length
        self.min_audio_length = min_audio_length

        index = self._create_index()
        if max_audio_length is not None or min_audio_length is not None:
            index = self._filter_by_length(index)

        super().__init__(index, *args, **kwargs)
        logger.info(f"Loaded RUSLAN dataset with {len(self._index)} samples")

    def _create_index(self):
        """
        Create index from the metadata CSV file.
        """
        df = pd.read_csv(
            self.metadata_path,
            sep="|",
            header=None,
            names=["audio_id", "text"],
            dtype={"audio_id": str, "text": str},
        )

        index = []
        for _, row in df.iterrows():
            audio_id = row["audio_id"]
            text = row["text"]
            audio_path = self.data_dir / f"{audio_id}.wav"

            if not audio_path.exists():
                logger.warning(f"Audio file not found: {audio_path}")
                continue

            index.append(
                {
                    "audio_path": str(audio_path),
                    "text": text,
                    "audio_id": audio_id,
                    "path": str(audio_path),
                    "label": text,
                }
            )

        return index

    def _filter_by_length(self, index):
        """
        Filter samples by audio length.
        """
        filtered_index = []

        for item in index:
            try:
                try:
                    # New torchaudio API (>= 0.10)
                    audio_info = torchaudio.info(item["audio_path"])
                    num_frames = audio_info.num_frames
                except AttributeError:
                    # Old torchaudio API (< 0.10)
                    import soundfile as sf

                    info = sf.info(item["audio_path"])
                    num_frames = info.frames

                if (
                    self.min_audio_length is not None
                    and num_frames < self.min_audio_length
                ):
                    continue
                if (
                    self.max_audio_length is not None
                    and num_frames > self.max_audio_length
                ):
                    continue

                item["audio_length"] = num_frames
                filtered_index.append(item)

            except Exception as e:
                logger.warning(f"Error processing {item['audio_path']}: {e}")
                continue

        logger.info(
            f"Filtered dataset from {len(index)} to {len(filtered_index)} samples"
        )

        return filtered_index

    def __getitem__(self, ind):
        """
        Get a single item from the dataset.
        """
        data_dict = self._index[ind]
        audio_path = data_dict["audio_path"]
        text = data_dict["text"]
        audio_id = data_dict["audio_id"]

        audio, sample_rate = torchaudio.load(audio_path)
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        audio = audio.squeeze(0)

        instance_data = {
            "audio": audio,
            "text": text,
            "audio_id": audio_id,
            "audio_path": audio_path,
            "sample_rate": sample_rate,
            "audio_length": audio.shape[0],
        }

        instance_data = self.preprocess_data(instance_data)

        return instance_data

    def load_object(self, path):
        """
        Override load_object to load audio files.
        """
        audio, _ = torchaudio.load(path)
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        return audio.squeeze(0)
