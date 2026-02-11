"""
Подготовка транскрипций для разметки эмоций.
"""

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List, Set

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_from_csv(csv_path: Path, audio_dir: Path = None) -> Dict[str, str]:
    """
    Извлекает транскрипции из CSV файла формата: audio_id|text
    """
    transcriptions = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            if line_num == 1 and line.startswith("audio_id"):
                continue

            if "|" not in line:
                logger.warning(
                    f"Line {line_num}: no '|' separator, skipping: {line[:50]}"
                )
                continue

            parts = line.split("|", 1)
            if len(parts) != 2:
                logger.warning(
                    f"Line {line_num}: unexpected format, skipping: {line[:50]}"
                )
                continue

            audio_id = parts[0].strip()
            text = parts[1].strip()

            if not audio_id or not text:
                logger.warning(f"Line {line_num}: empty audio_id or text, skipping")
                continue

            transcriptions[audio_id] = text

    logger.info(f"Extracted {len(transcriptions)} transcriptions from {csv_path}")

    if audio_dir and audio_dir.exists():
        existing_audio_ids = set()
        for audio_file in audio_dir.glob("*.wav"):
            audio_id = audio_file.stem
            existing_audio_ids.add(audio_id)

        filtered_transcriptions = {
            audio_id: text
            for audio_id, text in transcriptions.items()
            if audio_id in existing_audio_ids
        }

        missing = len(transcriptions) - len(filtered_transcriptions)
        if missing > 0:
            logger.warning(
                f"Filtered out {missing} transcriptions without corresponding audio files"
            )

        transcriptions = filtered_transcriptions

    return transcriptions


def extract_from_phonemes(phoneme_dir: Path) -> Dict[str, str]:
    """
    Извлекает "транскрипции" из файлов с фонемами.
    Используется как fallback если нет настоящих транскрипций.
    """
    transcriptions = {}

    if not phoneme_dir.exists():
        logger.error(f"Phoneme directory not found: {phoneme_dir}")
        return transcriptions

    for phoneme_file in phoneme_dir.glob("*.txt"):
        audio_id = phoneme_file.stem

        with open(phoneme_file, "r", encoding="utf-8") as f:
            phonemes = f.read().strip()

        if phonemes:
            transcriptions[audio_id] = f"[phonemes: {phonemes}]"

    logger.info(f"Extracted {len(transcriptions)} phoneme sequences from {phoneme_dir}")

    return transcriptions


def save_transcriptions(transcriptions: Dict[str, str], output_path: Path):
    """Сохраняет транскрипции в формат audio_id|text"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for audio_id, text in sorted(transcriptions.items()):
            f.write(f"{audio_id}|{text}\n")

    logger.info(f"Saved {len(transcriptions)} transcriptions to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare transcriptions for emotion annotation"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/metadata_RUSLAN_22200.csv",
        help="Source CSV file with transcriptions",
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="data/RUSLAN",
        help="Audio directory (to filter only existing files)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/transcriptions_clean_ruslan.txt",
        help="Output transcription file",
    )
    parser.add_argument(
        "--from_phonemes",
        type=str,
        default=None,
        help="Extract from phoneme directory (fallback if no transcriptions)",
    )
    parser.add_argument(
        "--min_length", type=int, default=5, help="Minimum text length (characters)"
    )
    parser.add_argument(
        "--max_length", type=int, default=1000, help="Maximum text length (characters)"
    )

    args = parser.parse_args()

    source_path = Path(args.source)
    audio_dir = Path(args.audio_dir) if args.audio_dir else None
    output_path = Path(args.output)

    transcriptions = {}

    if args.from_phonemes:
        phoneme_dir = Path(args.from_phonemes)
        transcriptions = extract_from_phonemes(phoneme_dir)
    elif source_path.exists():
        transcriptions = extract_from_csv(source_path, audio_dir)
    else:
        logger.error(f"Source file not found: {source_path}")
        return

    if not transcriptions:
        logger.error("No transcriptions extracted!")
        return

    original_count = len(transcriptions)
    transcriptions = {
        audio_id: text
        for audio_id, text in transcriptions.items()
        if args.min_length <= len(text) <= args.max_length
    }

    filtered = original_count - len(transcriptions)
    if filtered > 0:
        logger.info(f"Filtered out {filtered} transcriptions by length")

    save_transcriptions(transcriptions, output_path)

    logger.info(f"Всего транскрипций: {len(transcriptions)}")
    if transcriptions:
        text_lengths = [len(text) for text in transcriptions.values()]
        logger.info("Длина текста (символы):")
        logger.info(f"  Минимум: {min(text_lengths)}")
        logger.info(f"  Максимум: {max(text_lengths)}")
        logger.info(f"  Среднее: {sum(text_lengths) / len(text_lengths):.1f}")

        logger.info("\nПримеры (первые 3):")
        for i, (audio_id, text) in enumerate(list(transcriptions.items())[:3]):
            logger.info(f"  {audio_id}: {text[:80]}{'...' if len(text) > 80 else ''}")


if __name__ == "__main__":
    main()
