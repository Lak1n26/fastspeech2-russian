import logging
from typing import Any, Dict, List

import torch

logger = logging.getLogger(__name__)


class CollateFnWrapper:
    """
    Wrapper for collate functions to make them compatible with Hydra instantiation.
    """

    def __init__(self, collate_fn):
        self.collate_fn = collate_fn

    def __call__(self, dataset_items):
        return self.collate_fn(dataset_items)


def collate_fn_tts(dataset_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for TTS dataset.

    Handles variable-length audio sequences and text by padding them to the
    maximum length in the batch.
    """
    dataset_items = sorted(dataset_items, key=lambda x: x["audio_length"], reverse=True)

    audios = [item["audio"] for item in dataset_items]
    texts = [item["text"] for item in dataset_items]
    audio_ids = [item["audio_id"] for item in dataset_items]
    sample_rate = dataset_items[0]["sample_rate"]

    audio_lengths = torch.LongTensor([audio.shape[0] for audio in audios])
    text_lengths = torch.LongTensor([len(text) for text in texts])

    max_audio_len = audio_lengths[0].item()
    audio_padded = torch.zeros(len(audios), max_audio_len)

    for i, audio in enumerate(audios):
        audio_padded[i, : audio.shape[0]] = audio

    batch = {
        "audio": audio_padded,
        "audio_lengths": audio_lengths,
        "text": texts,
        "text_lengths": text_lengths,
        "audio_ids": audio_ids,
        "sample_rate": sample_rate,
    }

    return batch


def collate_fn_tts_with_features(dataset_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for TTS dataset with extracted features.

    Used when mel-spectrograms, pitch, energy, and duration are pre-extracted.
    """
    if "mel" in dataset_items[0]:

        def get_mel_time_len(item):
            mel = item["mel"]
            audio_id = item.get("audio_id", "unknown")
            logger.debug(
                f"[COLLATE] get_mel_time_len for {audio_id}: shape = {mel.shape}"
            )

            if mel.shape[0] == 80 and mel.shape[1] != 80:
                time_len = mel.shape[1]
                logger.debug(f"-> Format: (n_mels, time), time_len = {time_len}")
                return time_len  # (n_mels, time)
            if mel.shape[1] == 80 and mel.shape[0] != 80:
                time_len = mel.shape[0]
                logger.debug(f"-> Format: (time, n_mels), time_len = {time_len}")
                return time_len  # (time, n_mels)
            if mel.shape[0] == 80 and mel.shape[1] == 80:
                time_len = mel.shape[1]
                logger.debug(
                    f"-> Format: (n_mels, time) [square], time_len = {time_len}"
                )
                return time_len

            logger.error(f"[COLLATE] ERROR: Unexpected mel shape for {audio_id}")
            logger.error(f"  Shape: {mel.shape}")
            logger.error(f"  Dtype: {mel.dtype}")
            logger.error(f"  Min/Max values: {mel.min(): .3f} / {mel.max(): .3f}")
            logger.error("  Expected: one dimension should be 80 (n_mels)")
            raise ValueError(
                f"Unexpected mel shape: {mel.shape}. Expected one dim == 80."
            )

        dataset_items = sorted(dataset_items, key=get_mel_time_len, reverse=True)

    batch = {}

    if "text" in dataset_items[0]:
        batch["text"] = [item["text"] for item in dataset_items]
        batch["text_lengths"] = torch.LongTensor(
            [len(item["text"]) for item in dataset_items]
        )

    if "text_tokens" in dataset_items[0]:
        text_tokens = [item["text_tokens"] for item in dataset_items]
        text_lengths = torch.LongTensor([len(tokens) for tokens in text_tokens])
        max_text_len = text_lengths.max().item()

        text_tokens_padded = torch.zeros(
            len(text_tokens), max_text_len, dtype=torch.long
        )
        for i, tokens in enumerate(text_tokens):
            text_tokens_padded[i, : len(tokens)] = torch.LongTensor(tokens)

        batch["text_tokens"] = text_tokens_padded
        batch["text_lengths"] = text_lengths

    if "mel" in dataset_items[0]:
        mels = [item["mel"] for item in dataset_items]
        mel_lengths = torch.LongTensor([mel.shape[-1] for mel in mels])
        max_mel_len = mel_lengths.max().item()
        n_mels = mels[0].shape[0]
        mels_padded = torch.zeros(len(mels), n_mels, max_mel_len)
        for i, mel in enumerate(mels):
            mels_padded[i, :, : mel.shape[-1]] = mel

        batch["mel"] = mels_padded
        batch["mel_lengths"] = mel_lengths

    if "duration" in dataset_items[0]:
        durations = [item["duration"] for item in dataset_items]
        max_dur_len = max(dur.shape[0] for dur in durations)

        durations_padded = torch.zeros(len(durations), max_dur_len, dtype=torch.long)
        for i, dur in enumerate(durations):
            if isinstance(dur, torch.Tensor):
                dur_tensor = dur.long()
            else:
                dur_tensor = torch.LongTensor(dur)
            durations_padded[i, : dur_tensor.shape[0]] = dur_tensor

        batch["duration"] = durations_padded

    if "pitch" in dataset_items[0]:
        pitches = [item["pitch"] for item in dataset_items]
        max_pitch_len = max(pitch.shape[0] for pitch in pitches)

        pitches_padded = torch.zeros(len(pitches), max_pitch_len)
        for i, pitch in enumerate(pitches):
            pitch_tensor = (
                torch.FloatTensor(pitch)
                if not isinstance(pitch, torch.Tensor)
                else pitch
            )
            pitches_padded[i, : pitch_tensor.shape[0]] = pitch_tensor

        batch["pitch"] = pitches_padded

    if "energy" in dataset_items[0]:
        energies = [item["energy"] for item in dataset_items]
        max_energy_len = max(energy.shape[0] for energy in energies)

        energies_padded = torch.zeros(len(energies), max_energy_len)
        for i, energy in enumerate(energies):
            energy_tensor = (
                torch.FloatTensor(energy)
                if not isinstance(energy, torch.Tensor)
                else energy
            )
            energies_padded[i, : energy_tensor.shape[0]] = energy_tensor

        batch["energy"] = energies_padded

    if "audio_id" in dataset_items[0]:
        batch["audio_ids"] = [item["audio_id"] for item in dataset_items]

    if "emotion" in dataset_items[0]:
        emotions = [item["emotion"] for item in dataset_items]
        batch["emotion"] = torch.cat(emotions, dim=0)

    return batch


class TTSCollate(CollateFnWrapper):
    """Collate function wrapper for basic TTS dataset."""

    def __init__(self):
        super().__init__(collate_fn_tts)


class TTSFeaturesCollate(CollateFnWrapper):
    """Collate function wrapper for TTS dataset with pre-extracted features."""

    def __init__(self):
        super().__init__(collate_fn_tts_with_features)
