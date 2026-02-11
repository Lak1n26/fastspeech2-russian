"""
Extract duration from MFA TextGrid alignments.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import textgrid
from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TextGridParser:
    """
    Parser for TextGrid files produced by MFA.
    """

    def __init__(self, hop_length: int = 256, sample_rate: int = 22050):
        """
        Initialize parser.
        """
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.frame_shift = hop_length / sample_rate  # in seconds

    def parse_textgrid(
        self, textgrid_path: str
    ) -> Tuple[List[str], List[float], List[float]]:
        """
        Parse TextGrid file and extract phoneme alignments.
        """
        try:
            tg = textgrid.TextGrid.fromFile(textgrid_path)
        except Exception as e:
            logger.error(f"Error reading TextGrid file {textgrid_path}: {e}")
            return [], [], []

        phonemes = []
        start_times = []
        end_times = []

        phones_tier = None
        for tier in tg.tiers:
            if tier.name.lower() in ["phones", "segments", "phone"]:
                phones_tier = tier
                break

        if phones_tier is None:
            logger.warning(f"No phones tier found in {textgrid_path}")
            return [], [], []

        for interval in phones_tier.intervals:
            phone = interval.mark.strip()
            if not phone or phone == "":
                continue

            phonemes.append(phone)
            start_times.append(interval.minTime)
            end_times.append(interval.maxTime)

        return phonemes, start_times, end_times

    def time_to_frames(self, time_seconds: float) -> int:
        """
        Convert time in seconds to frame index.
        """
        return int(np.round(time_seconds / self.frame_shift))

    def extract_duration(
        self, phonemes: List[str], start_times: List[float], end_times: List[float]
    ) -> np.ndarray:
        """
        Extract duration in frames for each phoneme.
        """
        durations = []

        for start, end in zip(start_times, end_times):
            start_frame = self.time_to_frames(start)
            end_frame = self.time_to_frames(end)
            duration = max(1, end_frame - start_frame)  # At least 1 frame
            durations.append(duration)

        return np.array(durations, dtype=np.int32)

    def extract_phoneme_sequence(self, phonemes: List[str]) -> List[str]:
        """
        Extract clean phoneme sequence.
        """
        silence_labels = {"sil", "sp", "spn", "", "<eps>"}

        clean_phonemes = []
        for phone in phonemes:
            if phone.lower() not in silence_labels:
                clean_phonemes.append(phone)

        return clean_phonemes


def extract_durations_from_alignments(
    alignments_dir: str,
    output_dir: str,
    hop_length: int = 256,
    sample_rate: int = 22050,
    save_phonemes: bool = True,
):
    """
    Extract durations from all TextGrid files in a directory.
    """
    logger.info("Extracting Duration from MFA Alignments")
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    if save_phonemes:
        phonemes_path = output_path.parent / "phonemes"
        phonemes_path.mkdir(exist_ok=True, parents=True)

    logger.info(f"Alignments directory: {alignments_dir}")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Hop length: {hop_length}")
    logger.info(f"Sample rate: {sample_rate}")
    logger.info(f"Frame shift: {hop_length / sample_rate:.4f} seconds")

    parser = TextGridParser(hop_length, sample_rate)

    alignments_path = Path(alignments_dir)
    textgrid_files = list(alignments_path.rglob("*.TextGrid"))

    if not textgrid_files:
        logger.error(f"No TextGrid files found in {alignments_dir}")
        return

    logger.info(f"\nFound {len(textgrid_files)} TextGrid files")

    success_count = 0
    error_count = 0

    logger.info("\nProcessing TextGrid files...")
    for textgrid_file in tqdm(textgrid_files, desc="Extracting durations"):
        try:
            phonemes, start_times, end_times = parser.parse_textgrid(str(textgrid_file))

            if not phonemes:
                logger.warning(f"No phonemes found in {textgrid_file.name}")
                error_count += 1
                continue

            durations = parser.extract_duration(phonemes, start_times, end_times)
            base_name = textgrid_file.stem
            duration_file = output_path / f"{base_name}.npy"
            np.save(duration_file, durations)

            if save_phonemes:
                clean_phonemes = parser.extract_phoneme_sequence(phonemes)
                phoneme_file = phonemes_path / f"{base_name}.npy"
                np.save(phoneme_file, clean_phonemes)

            success_count += 1

        except Exception as e:
            logger.error(f"Error processing {textgrid_file.name}: {e}")
            error_count += 1

    logger.info("Duration Extraction Complete")
    logger.info(f"Successfully processed: {success_count}")
    logger.info(f"Errors: {error_count}")

    duration_files = list(output_path.glob("*.npy"))
    logger.info(f"\nOutput files: {len(duration_files)}")

    if duration_files:
        sample_duration = np.load(duration_files[0])
        logger.info(f"\nSample duration ({duration_files[0].name}):")
        logger.info(f"  Shape: {sample_duration.shape}")
        logger.info(f"  Min: {sample_duration.min()}")
        logger.info(f"  Max: {sample_duration.max()}")
        logger.info(f"  Mean: {sample_duration.mean():.2f}")
        logger.info(f"  Total frames: {sample_duration.sum()}")
        logger.info(
            f"  Duration (seconds): {sample_duration.sum() * hop_length / sample_rate:.2f}"
        )

        all_durations = []
        for dur_file in duration_files[:1000]:
            dur = np.load(dur_file)
            all_durations.extend(dur.tolist())

        all_durations = np.array(all_durations)
        logger.info(
            f"\nOverall statistics (from {min(1000, len(duration_files))} files):"
        )
        logger.info(f"  Total phonemes: {len(all_durations)}")
        logger.info(f"  Min duration: {all_durations.min()} frames")
        logger.info(f"  Max duration: {all_durations.max()} frames")
        logger.info(f"  Mean duration: {all_durations.mean():.2f} frames")
        logger.info(f"  Median duration: {np.median(all_durations):.2f} frames")


def main():
    parser = argparse.ArgumentParser(
        description="Extract duration from MFA TextGrid alignments"
    )
    parser.add_argument(
        "--alignments_dir",
        type=str,
        default="data/ruslan_alignments",
        help="Directory containing TextGrid files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/features/duration",
        help="Output directory for duration files",
    )
    parser.add_argument(
        "--hop_length", type=int, default=256, help="Hop length in samples"
    )
    parser.add_argument(
        "--sample_rate", type=int, default=22050, help="Audio sample rate"
    )
    parser.add_argument(
        "--save_phonemes",
        action="store_true",
        default=True,
        help="Save phoneme sequences",
    )

    args = parser.parse_args()

    extract_durations_from_alignments(
        alignments_dir=args.alignments_dir,
        output_dir=args.output_dir,
        hop_length=args.hop_length,
        sample_rate=args.sample_rate,
        save_phonemes=args.save_phonemes,
    )


if __name__ == "__main__":
    main()
