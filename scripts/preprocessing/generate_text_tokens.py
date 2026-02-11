#!/usr/bin/env python3
"""
Script to generate text_tokens from phoneme files.
"""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm


def load_vocabulary(vocab_path):
    """
    Load phoneme vocabulary from file.
    """
    vocab = {}
    with open(vocab_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            phoneme = line.strip()
            vocab[phoneme] = idx
    return vocab


def convert_phonemes_to_tokens(phonemes, vocab):
    """
    Convert phoneme sequence to token indices.
    """
    unk_idx = vocab.get("<unk>", 1)
    tokens = []

    for phoneme in phonemes:
        token_idx = vocab.get(phoneme, unk_idx)
        tokens.append(token_idx)

    return np.array(tokens, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description="Generate text_tokens from phonemes")
    parser.add_argument(
        "--phonemes_dir",
        type=str,
        default="data/features/phonemes",
        help="Directory containing phoneme files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/features/text_tokens",
        help="Directory to save text_tokens",
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        default="data/phoneme_vocabulary.txt",
        help="Path to phoneme vocabulary file",
    )
    args = parser.parse_args()

    phonemes_dir = Path(args.phonemes_dir)
    output_dir = Path(args.output_dir)
    vocab_path = Path(args.vocab_path)

    if not phonemes_dir.exists():
        raise FileNotFoundError(f"Phonemes directory not found: {phonemes_dir}")

    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading vocabulary from {vocab_path}...")
    vocab = load_vocabulary(vocab_path)
    print(f"Vocabulary size: {len(vocab)}")

    phoneme_files = sorted(list(phonemes_dir.glob("*.npy")))
    print(f"Found {len(phoneme_files)} phoneme files")

    if len(phoneme_files) == 0:
        print("No phoneme files found. Exiting.")
        return

    print("Converting phonemes to tokens...")
    stats = {
        "total": len(phoneme_files),
        "success": 0,
        "failed": 0,
        "total_phonemes": 0,
        "total_tokens": 0,
        "unk_count": 0,
    }

    for phoneme_file in tqdm(phoneme_files):
        try:
            phonemes = np.load(phoneme_file, allow_pickle=True)
            tokens = convert_phonemes_to_tokens(phonemes, vocab)
            unk_idx = vocab.get("<unk>", 1)
            unk_in_file = np.sum(tokens == unk_idx)
            stats["unk_count"] += unk_in_file

            output_file = output_dir / phoneme_file.name
            np.save(output_file, tokens)

            stats["success"] += 1
            stats["total_phonemes"] += len(phonemes)
            stats["total_tokens"] += len(tokens)

        except Exception as e:
            print(f"\nError processing {phoneme_file.name}: {e}")
            stats["failed"] += 1

    print("Conversion complete!")
    print(f"Total files processed: {stats['total']}")
    print(f"Successfully converted: {stats['success']}")
    print(f"Failed: {stats['failed']}")
    print(f"Total phonemes: {stats['total_phonemes']}")
    print(f"Total tokens: {stats['total_tokens']}")
    print(
        f"Unknown tokens: {stats['unk_count']} ({stats['unk_count'] / max(stats['total_tokens'], 1) * 100: .2f}%)"
    )
    print(f"\nOutput directory: {output_dir}")
    print("=" * 60)

    print("\nVerifying random samples...")
    import random

    sample_files = random.sample(phoneme_files, min(3, len(phoneme_files)))

    for phoneme_file in sample_files:
        phonemes = np.load(phoneme_file, allow_pickle=True)
        token_file = output_dir / phoneme_file.name
        tokens = np.load(token_file)

        print(f"\n{phoneme_file.name}")
        print(f"  Phonemes ({len(phonemes)}): {phonemes[:5]}...")
        print(f"  Tokens ({len(tokens)}): {tokens[:5]}...")


if __name__ == "__main__":
    main()
