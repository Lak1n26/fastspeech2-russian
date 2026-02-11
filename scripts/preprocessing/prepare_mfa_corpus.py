"""
Prepare RUSLAN dataset for Montreal Forced Aligner (MFA).
"""

import argparse
import logging
import re
import shutil
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def clean_text_for_mfa(text: str) -> str:
    """
    Clean and normalize text for MFA.

    MFA expects:
    - Lowercase text (optional, depends on dictionary)
    - No punctuation (or minimal punctuation)
    - Normalized numbers, abbreviations, etc.
    """
    text = text.lower()
    text = re.sub(r'[«»""„]', '"', text)
    text = re.sub(r"[—–]", "-", text)
    text = re.sub(r"[^\w\s\.\,\!\?\-\']", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prepare_mfa_corpus(
    metadata_path: str,
    audio_dir: str,
    output_dir: str,
    clean_text: bool = True,
    copy_audio: bool = True,
    limit: int = None,
):
    """
    Prepare corpus for MFA alignment.
    """
    logger.info("Preparing MFA Corpus")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    logger.info(f"Output directory: {output_path}")

    logger.info(f"\nReading metadata from: {metadata_path}")
    df = pd.read_csv(
        metadata_path,
        sep="|",
        header=None,
        names=["audio_id", "text"],
        dtype={"audio_id": str, "text": str},
    )

    if limit is not None:
        df = df.head(limit)
        logger.info(f"Limited to {limit} samples")

    logger.info(f"Total samples: {len(df)}")

    audio_dir_path = Path(audio_dir)
    success_count = 0
    error_count = 0

    logger.info("\nProcessing files...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preparing corpus"):
        audio_id = row["audio_id"]
        text = row["text"]

        audio_file = audio_dir_path / f"{audio_id}.wav"

        if not audio_file.exists():
            logger.warning(f"Audio file not found: {audio_file}")
            error_count += 1
            continue

        dest_audio = output_path / f"{audio_id}.wav"
        dest_text = output_path / f"{audio_id}.txt"

        try:
            if copy_audio:
                shutil.copy2(audio_file, dest_audio)
            else:
                if dest_audio.exists():
                    dest_audio.unlink()
                dest_audio.symlink_to(audio_file.absolute())

            if clean_text:
                text = clean_text_for_mfa(text)

            with open(dest_text, "w", encoding="utf-8") as f:
                f.write(text)

            success_count += 1

        except Exception as e:
            logger.error(f"Error processing {audio_id}: {e}")
            error_count += 1

    logger.info("Corpus Preparation Complete")
    logger.info(f"Successfully processed: {success_count}")
    logger.info(f"Errors: {error_count}")
    logger.info(f"Output directory: {output_path}")

    audio_files = list(output_path.glob("*.wav"))
    text_files = list(output_path.glob("*.txt"))

    logger.info("\nVerification:")
    logger.info(f"  Audio files: {len(audio_files)}")
    logger.info(f"  Text files: {len(text_files)}")

    if text_files:
        sample_text_file = text_files[0]
        with open(sample_text_file, "r", encoding="utf-8") as f:
            sample_text = f.read()
        logger.info(f"\nSample text file ({sample_text_file.name})")
        logger.info(f"  {sample_text}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare RUSLAN dataset for MFA alignment"
    )
    parser.add_argument(
        "--metadata",
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
        default="data/ruslan_mfa_corpus",
        help="Output directory for MFA corpus",
    )
    parser.add_argument(
        "--clean_text",
        action="store_true",
        default=True,
        help="Clean and normalize text for MFA",
    )
    parser.add_argument(
        "--no_clean_text",
        dest="clean_text",
        action="store_false",
        help="Do not clean text",
    )
    parser.add_argument(
        "--copy_audio",
        action="store_true",
        default=True,
        help="Copy audio files (default)",
    )
    parser.add_argument(
        "--symlink_audio",
        dest="copy_audio",
        action="store_false",
        help="Create symlinks instead of copying",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of files (for testing)"
    )

    args = parser.parse_args()

    prepare_mfa_corpus(
        metadata_path=args.metadata,
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        clean_text=args.clean_text,
        copy_audio=args.copy_audio,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
