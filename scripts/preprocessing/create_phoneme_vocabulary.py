"""
Create phoneme vocabulary from MFA alignments.
"""

import argparse
import logging
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_phoneme_vocabulary(
    phonemes_dir: str,
    output_path: str,
    min_frequency: int = 1,
    add_special_tokens: bool = True,
):
    """
    Create phoneme vocabulary from phoneme files.
    """
    logger.info("=" * 60)
    logger.info("Creating Phoneme Vocabulary")
    logger.info("=" * 60)

    phonemes_path = Path(phonemes_dir)

    if not phonemes_path.exists():
        logger.error(f"Phonemes directory not found: {phonemes_dir}")
        return

    phoneme_files = list(phonemes_path.glob("*.npy"))

    if not phoneme_files:
        logger.error(f"No phoneme files found in {phonemes_dir}")
        return

    logger.info(f"Found {len(phoneme_files)} phoneme files")

    logger.info("\nCounting phonemes...")
    phoneme_counter = Counter()

    for phoneme_file in tqdm(phoneme_files, desc="Reading phoneme files"):
        try:
            phonemes = np.load(phoneme_file, allow_pickle=True)
            phoneme_counter.update(phonemes)
        except Exception as e:
            logger.error(f"Error reading {phoneme_file.name}: {e}")

    logger.info(f"\nTotal unique phonemes: {len(phoneme_counter)}")
    logger.info(f"Total phoneme tokens: {sum(phoneme_counter.values())}")

    filtered_phonemes = {
        phoneme: count
        for phoneme, count in phoneme_counter.items()
        if count >= min_frequency
    }

    logger.info(
        f"Phonemes after filtering (min_freq={min_frequency}): {len(filtered_phonemes)}"
    )

    sorted_phonemes = sorted(
        filtered_phonemes.items(), key=lambda x: x[1], reverse=True
    )

    vocabulary = []

    if add_special_tokens:
        vocabulary.extend(
            [
                "<pad>",
                "<unk>",
                "<s>",
                "</s>",
            ]
        )

    for phoneme, count in sorted_phonemes:
        vocabulary.append(phoneme)

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(exist_ok=True, parents=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for token in vocabulary:
            f.write(f"{token}\n")

    logger.info(f"\nVocabulary saved to: {output_path}")
    logger.info(f"Vocabulary size: {len(vocabulary)}")

    logger.info("\n" + "=" * 60)
    logger.info("Vocabulary Statistics")
    logger.info("=" * 60)

    if add_special_tokens:
        logger.info(f"Special tokens: {vocabulary[:4]}")
        logger.info(f"Regular phonemes: {len(vocabulary) - 4}")

    logger.info("\nMost common phonemes:")
    for i, (phoneme, count) in enumerate(sorted_phonemes[:20]):
        logger.info(f"  {i + 1}. '{phoneme}': {count} occurrences")

    if len(sorted_phonemes) > 20:
        logger.info("  ...")
        logger.info(f"  (showing top 20 of {len(sorted_phonemes)})")

    logger.info("\nLeast common phonemes:")
    for phoneme, count in sorted_phonemes[-10:]:
        logger.info(f"  '{phoneme}': {count} occurrences")

    logger.info("\nCreating phoneme-to-index mapping...")
    phoneme_to_idx = {phoneme: idx for idx, phoneme in enumerate(vocabulary)}

    mapping_path = output_path_obj.parent / "phoneme_to_idx.txt"
    with open(mapping_path, "w", encoding="utf-8") as f:
        for phoneme, idx in phoneme_to_idx.items():
            f.write(f"{phoneme}\t{idx}\n")

    logger.info(f"Mapping saved to: {mapping_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create phoneme vocabulary from MFA alignments"
    )
    parser.add_argument(
        "--phonemes_dir",
        type=str,
        default="data/features/phonemes",
        help="Directory containing phoneme .npy files",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/phoneme_vocabulary.txt",
        help="Output path for vocabulary file",
    )
    parser.add_argument(
        "--min_frequency",
        type=int,
        default=1,
        help="Minimum frequency for phoneme to be included",
    )
    parser.add_argument(
        "--no_special_tokens",
        dest="add_special_tokens",
        action="store_false",
        help="Do not add special tokens",
    )

    args = parser.parse_args()

    create_phoneme_vocabulary(
        phonemes_dir=args.phonemes_dir,
        output_path=args.output_path,
        min_frequency=args.min_frequency,
        add_special_tokens=args.add_special_tokens,
    )


if __name__ == "__main__":
    main()
