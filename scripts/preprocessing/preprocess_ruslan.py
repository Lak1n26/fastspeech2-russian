"""
Preprocess RUSLAN dataset and extract features for FastSpeech2.
"""

import argparse
import json
import logging
import multiprocessing as mp
from functools import partial
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from fastspeech2.datasets.feature_extraction import FeatureExtractor
from fastspeech2.datasets.ruslan_dataset import RUSLANDataset

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _str_to_bool(value: str) -> bool:
    """Convert string to bool for argparse."""
    if value.lower() not in ["true", "false"]:
        raise ValueError(f"Argument needs to be a boolean, got {value}")
    return {"true": True, "false": False}[value.lower()]


def extract_features_for_sample(
    sample_data: dict, feature_extractor: FeatureExtractor, output_dir: Path
) -> bool:
    """
    Extract and save features for a single sample.
    """
    try:
        audio_id = sample_data["audio_id"]
        audio = sample_data["audio"]
        sample_rate = sample_data["sample_rate"]

        mel, pitch, energy = feature_extractor(audio, sample_rate)
        mel_np = mel.numpy()
        if mel_np.ndim == 2:
            mel_np = mel_np.T
        pitch_np = pitch.numpy()
        energy_np = energy.numpy()

        np.save(output_dir / "mel" / f"{audio_id}.npy", mel_np)
        np.save(output_dir / "pitch" / f"{audio_id}.npy", pitch_np)
        np.save(output_dir / "energy" / f"{audio_id}.npy", energy_np)

        return True

    except Exception as e:
        logger.error(f"Error processing {sample_data.get('audio_id', 'unknown')}: {e}")
        return False


def process_batch(
    indices: list,
    dataset: RUSLANDataset,
    output_dir: Path,
    sample_rate: int,
    hop_length: int,
    n_mels: int,
    mel_f_min: float,
    mel_f_max: float,
    mel_normalize_db: bool,
):
    """
    Process a batch of samples (for multiprocessing).
    """
    feature_extractor = FeatureExtractor(
        sample_rate=sample_rate,
        hop_length=hop_length,
        n_mels=n_mels,
        mel_f_min=mel_f_min,
        mel_f_max=mel_f_max,
        mel_normalize_db=mel_normalize_db,
    )

    for idx in indices:
        sample = dataset[idx]
        extract_features_for_sample(sample, feature_extractor, output_dir)


def preprocess_dataset(
    metadata_path: str = "data/metadata_RUSLAN_22200.csv",
    audio_dir: str = "data/RUSLAN",
    output_dir: str = "data/features",
    sample_rate: int = 22050,
    hop_length: int = 256,
    n_mels: int = 80,
    mel_f_min: float = 0.0,
    mel_f_max: float = 11025.0,
    mel_normalize_db: bool = True,
    num_workers: int = 4,
    limit: int = None,
    normalize_pitch: bool = True,
    f_min: float = 80.0,
    f_max: float = 400.0,
):
    """
    Preprocess entire dataset.
    """
    logger.info("RUSLAN Dataset Preprocessing")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    (output_path / "mel").mkdir(exist_ok=True)
    (output_path / "pitch").mkdir(exist_ok=True)
    (output_path / "energy").mkdir(exist_ok=True)
    (output_path / "duration").mkdir(exist_ok=True)
    (output_path / "text_tokens").mkdir(exist_ok=True)

    logger.info(f"Output directory: {output_path}")

    logger.info("\nLoading dataset...")
    dataset = RUSLANDataset(
        metadata_path=metadata_path, data_dir=audio_dir, limit=limit
    )
    logger.info(f"Loaded {len(dataset)} samples")

    logger.info("\nInitializing feature extractor...")
    logger.info(f"  Sample rate: {sample_rate} Hz")
    logger.info(f"  Hop length: {hop_length}")
    logger.info(f"  N mels: {n_mels}")
    logger.info(f"  Mel f_min: {mel_f_min}")
    logger.info(f"  Mel f_max: {mel_f_max}")
    logger.info(f"  Mel normalize dB: {mel_normalize_db}")

    feature_extractor = FeatureExtractor(
        sample_rate=sample_rate,
        hop_length=hop_length,
        n_mels=n_mels,
        mel_f_min=mel_f_min,
        mel_f_max=mel_f_max,
        mel_normalize_db=mel_normalize_db,
    )

    logger.info(f"\nProcessing dataset with {num_workers} workers...")

    if num_workers <= 1:
        success_count = 0
        for i in tqdm(range(len(dataset)), desc="Extracting features"):
            sample = dataset[i]
            if extract_features_for_sample(sample, feature_extractor, output_path):
                success_count += 1
    else:
        indices = list(range(len(dataset)))
        batch_size = len(indices) // num_workers
        batches = [
            indices[i : i + batch_size] for i in range(0, len(indices), batch_size)
        ]

        process_func = partial(
            process_batch,
            dataset=dataset,
            output_dir=output_path,
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_mels=n_mels,
            mel_f_min=mel_f_min,
            mel_f_max=mel_f_max,
            mel_normalize_db=mel_normalize_db,
        )

        with mp.Pool(num_workers) as pool:
            list(
                tqdm(
                    pool.imap(process_func, batches),
                    total=len(batches),
                    desc="Processing batches",
                )
            )

        success_count = len(dataset)

    logger.info("\nProcessing complete!")
    logger.info(f"Successfully processed: {success_count}/{len(dataset)} samples")
    logger.info(f"Features saved to: {output_path}")

    if normalize_pitch:
        logger.info("\n" + "=" * 60)
        logger.info("Normalizing Pitch Features")
        logger.info("=" * 60)

        pitch_dir = output_path / "pitch"
        pitch_files = sorted(list(pitch_dir.glob("*.npy")))

        logger.info(f"Found {len(pitch_files)} pitch files")

        logger.info("\nCalculating global pitch statistics...")

        all_voiced_pitch = []

        for pitch_file in tqdm(pitch_files, desc="Collecting voiced pitch values"):
            try:
                pitch = torch.from_numpy(np.load(pitch_file)).float()

                voiced_mask = pitch > 0

                if voiced_mask.any():
                    voiced_pitch = pitch[voiced_mask]
                    voiced_pitch = torch.clamp(voiced_pitch, min=f_min, max=f_max)
                    all_voiced_pitch.append(voiced_pitch)

            except Exception as e:
                logger.error(f"Error reading {pitch_file.name}: {e}")

        if not all_voiced_pitch:
            logger.error("No voiced frames found in any pitch file!")
            return

        all_voiced_pitch = torch.cat(all_voiced_pitch)

        pitch_mean = all_voiced_pitch.mean().item()
        pitch_std = all_voiced_pitch.std().item()

        logger.info("\nPitch statistics:")
        logger.info(f"  Mean: {pitch_mean: .2f} Hz")
        logger.info(f"  Std: {pitch_std: .2f} Hz")
        logger.info(f"  Min: {all_voiced_pitch.min().item(): .2f} Hz")
        logger.info(f"  Max: {all_voiced_pitch.max().item(): .2f} Hz")
        logger.info(f"  Total voiced frames: {len(all_voiced_pitch)}")

        stats = {
            "pitch_mean": pitch_mean,
            "pitch_std": pitch_std,
            "pitch_min": f_min,
            "pitch_max": f_max,
            "num_voiced_frames": len(all_voiced_pitch),
            "num_files": len(pitch_files),
        }

        stats_path = output_path.parent / "pitch_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"\nStatistics saved to: {stats_path}")
        logger.info("\nNormalizing all pitch files...")

        for pitch_file in tqdm(pitch_files, desc="Normalizing pitch files"):
            try:
                pitch = torch.from_numpy(np.load(pitch_file)).float()
                voiced_mask = pitch > 0
                if voiced_mask.any():
                    pitch[voiced_mask] = (pitch[voiced_mask] - pitch_mean) / pitch_std
                    pitch[voiced_mask] = torch.clamp(
                        pitch[voiced_mask], min=-3.0, max=3.0
                    )

                    np.save(pitch_file, pitch.numpy())

            except Exception as e:
                logger.error(f"Error normalizing {pitch_file.name}: {e}")

        logger.info("\nPitch normalization complete!")

    logger.info("Feature Verification")
    mel_files = list((output_path / "mel").glob("*.npy"))
    pitch_files = list((output_path / "pitch").glob("*.npy"))
    energy_files = list((output_path / "energy").glob("*.npy"))

    logger.info(f"  Mel files: {len(mel_files)}")
    logger.info(f"  Pitch files: {len(pitch_files)}")
    logger.info(f"  Energy files: {len(energy_files)}")

    if mel_files:
        sample_mel = np.load(mel_files[0])
        sample_pitch = np.load(pitch_files[0])
        sample_energy = np.load(energy_files[0])

        logger.info("\nSample feature shapes:")
        logger.info(f"  Mel: {sample_mel.shape}")
        logger.info(f"  Pitch: {sample_pitch.shape}")
        logger.info(f"  Energy: {sample_energy.shape}")

        if normalize_pitch:
            all_pitch = []
            for pf in pitch_files[:100]:
                p = np.load(pf)
                voiced = p[p != 0]
                if len(voiced) > 0:
                    all_pitch.append(voiced)

            if all_pitch:
                all_pitch = np.concatenate(all_pitch)
                logger.info("\nNormalized pitch statistics (sample):")
                logger.info(f"  Mean: {all_pitch.mean(): .3f}")
                logger.info(f"  Std: {all_pitch.std(): .3f}")
                logger.info(f"  Min: {all_pitch.min(): .3f}")
                logger.info(f"  Max: {all_pitch.max(): .3f}")

    logger.info("Preprocessing complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess RUSLAN dataset for FastSpeech2"
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default="data/metadata_RUSLAN_22200.csv",
        help="Path to metadata CSV file",
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="data/RUSLAN",
        help="Directory containing audio files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/features",
        help="Output directory for extracted features",
    )
    parser.add_argument(
        "--sample_rate", type=int, default=22050, help="Target sample rate"
    )
    parser.add_argument(
        "--hop_length", type=int, default=256, help="Hop length for STFT"
    )
    parser.add_argument("--n_mels", type=int, default=80, help="Number of mel bands")
    parser.add_argument(
        "--mel_f_min", type=float, default=0.0, help="Min frequency for mel filterbank"
    )
    parser.add_argument(
        "--mel_f_max",
        type=float,
        default=11025.0,
        help="Max frequency for mel filterbank",
    )
    parser.add_argument(
        "--mel_normalize_db",
        type=_str_to_bool,
        default=True,
        help="Normalize mel to [0,1] using dB scale (WaveGlow prep)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of parallel workers"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of samples (for testing)"
    )

    args = parser.parse_args()

    preprocess_dataset(
        metadata_path=args.metadata_path,
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        mel_f_min=args.mel_f_min,
        mel_f_max=args.mel_f_max,
        mel_normalize_db=args.mel_normalize_db,
        num_workers=args.num_workers,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
