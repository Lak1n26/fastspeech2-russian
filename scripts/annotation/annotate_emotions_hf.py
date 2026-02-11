"""
Разметка эмоций через HuggingFace.
"""

import argparse
import json
import logging
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    PreTrainedModel,
    Wav2Vec2FeatureExtractor,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


class RussianEmotionRecognizer:
    """
    Распознавание эмоций для русской речи.
    Модель: Aniemore/wav2vec2-xlsr-53-russian-emotion-recognition
    """

    def __init__(
        self,
        model_name: str = "Aniemore/wav2vec2-xlsr-53-russian-emotion-recognition",
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.mps.is_available()
            else "cpu"
        )

        logger.info(f"Loading model: {model_name}")
        logger.info(f"Device: {self.device}")

        class AllTiedWeightsKeysDescriptor:
            def __get__(self, obj, objtype=None):
                if obj is None:
                    return {}
                return getattr(obj, "_tied_weights_keys", None) or {}

        if not hasattr(PreTrainedModel, "all_tied_weights_keys") or callable(
            getattr(PreTrainedModel, "all_tied_weights_keys", None)
        ):
            PreTrainedModel.all_tied_weights_keys = AllTiedWeightsKeysDescriptor()
            logger.info("Applied monkey patch for all_tied_weights_keys compatibility")

        try:
            self.config = AutoConfig.from_pretrained(
                model_name, trust_remote_code=True, revision="main"
            )
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                revision="main",
                use_safetensors=True,
            )
        except Exception as e:
            logger.warning(f"Failed with safetensors, trying without: {e}")
            self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                model_name, trust_remote_code=True, use_safetensors=False
            )

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

        self.id2label = self.config.id2label
        logger.info(f"Emotions: {list(self.id2label.values())}")

        self.success_count = 0
        self.error_count = 0

    def _load_audio(self, audio_path: Path, target_sr: int = 16000):
        """Загружает аудио и ресэмплирует до нужной частоты."""

        speech_array, sampling_rate = sf.read(str(audio_path), dtype="float32")
        speech_tensor = torch.from_numpy(speech_array)

        if speech_tensor.ndim == 1:
            speech_tensor = speech_tensor.unsqueeze(0)
        elif speech_tensor.ndim == 2:
            speech_tensor = speech_tensor.T

        if sampling_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sampling_rate, target_sr)
            speech_tensor = resampler(speech_tensor)

        if speech_tensor.shape[0] > 1:
            speech_tensor = torch.mean(speech_tensor, dim=0, keepdim=True)
        speech = speech_tensor.squeeze().numpy()

        return speech, target_sr

    def predict(self, audio_path: Path) -> Dict:
        """
        Предсказывает эмоцию для аудио файла.
        """
        try:
            speech, sampling_rate = self._load_audio(audio_path)
            inputs = self.feature_extractor(
                speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True
            )
            inputs = {key: inputs[key].to(self.device) for key in inputs}
            with torch.no_grad():
                logits = self.model(**inputs).logits
            scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
            emotion_scores = {
                self.id2label[i]: float(scores[i]) for i in range(len(scores))
            }
            max_emotion_id = scores.argmax()
            max_emotion = self.id2label[max_emotion_id]
            max_score = float(scores[max_emotion_id])

            self.success_count += 1

            return {
                "emotion": max_emotion,
                "emotion_intensity": max_score,
                "scores": emotion_scores,
            }

        except Exception as e:
            self.error_count += 1
            logger.error(f"Error processing {audio_path}: {e}")
            return {
                "emotion": "neutral",
                "emotion_intensity": 0.0,
                "scores": {},
                "error": str(e),
            }

    def annotate_dataset(
        self,
        audio_dir: Path,
        transcriptions: Dict[str, str],
        output_file: Path,
        save_interval: int = 100,
        num_workers: int = 1,
    ) -> List[Dict]:
        """
        Размечает весь датасет.
        """
        results = []

        logger.info(
            f"Processing {len(transcriptions)} audio files with {num_workers} workers..."
        )

        if num_workers <= 1:
            for idx, (audio_id, text) in enumerate(
                tqdm(transcriptions.items(), desc="Annotating"), 1
            ):
                audio_path = audio_dir / f"{audio_id}.wav"

                if not audio_path.exists():
                    logger.warning(f"Audio file not found: {audio_path}")
                    continue

                result = self.predict(audio_path)
                result["audio_id"] = audio_id
                result["text"] = text

                results.append(result)
                if idx % save_interval == 0:
                    self._save_results(results, output_file)
                    logger.info(f"Saved {len(results)} annotations (checkpoint)")
        else:
            results = self._annotate_parallel(
                audio_dir, transcriptions, output_file, save_interval, num_workers
            )

        self._save_results(results, output_file)
        self._print_statistics(results)

        return results

    def _annotate_parallel(
        self,
        audio_dir: Path,
        transcriptions: Dict[str, str],
        output_file: Path,
        save_interval: int,
        num_workers: int,
    ) -> List[Dict]:
        """Параллельная обработка с несколькими воркерами."""
        tasks = []
        for audio_id, text in transcriptions.items():
            audio_path = audio_dir / f"{audio_id}.wav"
            if audio_path.exists():
                tasks.append((audio_id, text, str(audio_path)))
            else:
                logger.warning(f"Audio file not found: {audio_path}")

        results = []

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            worker_fn = partial(
                process_single_audio, model_name=self.model_name, device=self.device
            )
            futures = {executor.submit(worker_fn, task): task for task in tasks}

            with tqdm(total=len(tasks), desc="Annotating (parallel)") as pbar:
                for idx, future in enumerate(as_completed(futures), 1):
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                            self.success_count += 1
                        else:
                            self.error_count += 1
                    except Exception as e:
                        logger.error(f"Worker failed: {e}")
                        self.error_count += 1

                    pbar.update(1)

                    if idx % save_interval == 0:
                        self._save_results(results, output_file)
                        logger.info(f"Saved {len(results)} annotations (checkpoint)")

        return results

    def _save_results(self, results: List[Dict], output_file: Path):
        """Сохраняет результаты в JSON."""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    def _print_statistics(self, results: List[Dict]):
        """Выводит статистику по эмоциям."""
        logger.info(f"Всего размечено: {len(results)}")
        logger.info(f"Успешно: {self.success_count}")
        logger.info(f"Ошибок: {self.error_count}")

        emotion_counts = {}
        total_intensity = 0

        for result in results:
            emotion = result.get("emotion", "unknown")
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            total_intensity += result.get("emotion_intensity", 0)

        logger.info("\nРаспределение эмоций:")
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
            percentage = count / len(results) * 100
            logger.info(f"  {emotion:15s}: {count:5d} ({percentage:5.1f}%)")

        avg_intensity = total_intensity / len(results) if results else 0
        logger.info(f"\nСредняя уверенность модели: {avg_intensity:.3f}")
        logger.info("=" * 50)


def process_single_audio(task, model_name: str, device: str) -> Optional[Dict]:
    """
    Worker function для параллельной обработки одного аудио файла.
    """
    audio_id, text, audio_path = task

    try:
        from transformers import PreTrainedModel

        class AllTiedWeightsKeysDescriptor:
            def __get__(self, obj, objtype=None):
                if obj is None:
                    return {}
                return getattr(obj, "_tied_weights_keys", None) or {}

        if not hasattr(PreTrainedModel, "all_tied_weights_keys") or callable(
            getattr(PreTrainedModel, "all_tied_weights_keys", None)
        ):
            PreTrainedModel.all_tied_weights_keys = AllTiedWeightsKeysDescriptor()

        recognizer = RussianEmotionRecognizer(model_name=model_name, device=device)

        result = recognizer.predict(Path(audio_path))

        result["audio_id"] = audio_id
        result["text"] = text

        return result

    except Exception as e:
        logger.error(f"Error in worker processing {audio_id}: {e}")
        return {
            "audio_id": audio_id,
            "text": text,
            "emotion": "neutral",
            "emotion_intensity": 0.0,
            "scores": {},
            "error": str(e),
        }


def load_transcriptions(transcription_file: Path) -> Dict[str, str]:
    """Загружает транскрипции из файла audio_id|text."""
    transcriptions = {}

    with open(transcription_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue

            parts = line.split("|", 1)
            if len(parts) == 2:
                audio_id = parts[0].strip()
                text = parts[1].strip()
                transcriptions[audio_id] = text

    logger.info(f"Loaded {len(transcriptions)} transcriptions")
    return transcriptions


def filter_by_audio_duration(
    transcriptions: Dict[str, str],
    audio_dir: Path,
    max_duration: float,
) -> Dict[str, str]:
    """Фильтрует транскрипции по длине аудио."""
    import soundfile as sf

    filtered = {}
    skipped = 0

    logger.info(f"Filtering audio by duration (max: {max_duration}s)...")

    for audio_id, text in tqdm(transcriptions.items(), desc="Checking duration"):
        audio_path = audio_dir / f"{audio_id}.wav"

        if not audio_path.exists():
            skipped += 1
            continue

        try:
            info = sf.info(audio_path)
            duration = info.duration

            if duration <= max_duration:
                filtered[audio_id] = text
            else:
                skipped += 1
        except Exception as e:
            logger.warning(f"Failed to read {audio_path}: {e}")
            skipped += 1

    logger.info(f"Kept {len(filtered)} files (≤ {max_duration}s)")
    logger.info(f"Skipped {skipped} files (> {max_duration}s or missing)")

    return filtered


def main():
    parser = argparse.ArgumentParser(
        description="Emotion annotation using Russian HuggingFace model"
    )
    parser.add_argument(
        "--transcriptions",
        type=str,
        default="data/transcriptions_clean_ruslan.txt",
        help="Path to transcriptions file (audio_id|text format)",
    )
    parser.add_argument(
        "--audio_dir", type=str, default="data/RUSLAN", help="Path to audio directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/emotion_labels_hf.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of samples (for testing)"
    )
    parser.add_argument(
        "--max_audio_duration",
        type=float,
        default=None,
        help="Maximum audio duration in seconds (e.g., 10.0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, default: auto)",
    )
    parser.add_argument(
        "--save_interval", type=int, default=100, help="Save checkpoint every N files"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="Number of parallel workers (default: 1, use 10 for parallel processing)",
    )

    args = parser.parse_args()

    transcription_file = Path(args.transcriptions)
    audio_dir = Path(args.audio_dir)
    output_file = Path(args.output)

    if not transcription_file.exists():
        logger.error(f"Transcription file not found: {transcription_file}")
        return

    if not audio_dir.exists():
        logger.error(f"Audio directory not found: {audio_dir}")
        return

    transcriptions = load_transcriptions(transcription_file)

    if args.max_audio_duration:
        transcriptions = filter_by_audio_duration(
            transcriptions,
            audio_dir,
            args.max_audio_duration,
        )

        if not transcriptions:
            logger.error("No audio files left after duration filtering!")
            return

    if args.limit:
        transcriptions = dict(list(transcriptions.items())[: args.limit])
        logger.info(f"Limited to {args.limit} samples")

    logger.info("Initializing emotion recognizer...")
    recognizer = RussianEmotionRecognizer(device=args.device)

    start_time = time.time()

    results = recognizer.annotate_dataset(
        audio_dir=audio_dir,
        transcriptions=transcriptions,
        output_file=output_file,
        save_interval=args.save_interval,
        num_workers=args.num_workers,
    )

    elapsed = time.time() - start_time

    logger.info(f"\nTotal time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    logger.info(f"Speed: {len(results)/elapsed:.2f} files/second")
    logger.info(f"Results saved to: {output_file}")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)
    main()
