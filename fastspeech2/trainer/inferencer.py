"""
Inferencer class for FastSpeech2.
"""

import numpy as np
import torch
from tqdm.auto import tqdm

from fastspeech2.metrics.tracker import MetricTracker
from fastspeech2.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    """
    Inferencer for FastSpeech2.
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        save_path,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
    ):
        """
        Initialize the Inferencer.
        """
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer
        self.device = device
        self.model = model
        self.batch_transforms = batch_transforms
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}
        self.save_path = save_path

        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = None

        if not skip_model_load:
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        """
        Run inference on each partition.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part):
        """
        Run batch through FastSpeech2 model and save predictions.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        outputs = self.model(**batch)
        batch.update(outputs)

        if metrics is not None:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))

        if self.save_path is not None:
            batch_size = batch["mel_pred"].shape[0]
            current_id = batch_idx * batch_size

            for i in range(batch_size):
                mel_pred = batch["mel_pred"][i].clone().cpu().numpy()

                output_dict = {
                    "mel_pred": mel_pred,
                }

                if "duration_pred" in batch:
                    output_dict["duration_pred"] = (
                        batch["duration_pred"][i].clone().cpu().numpy()
                    )

                if "pitch_pred" in batch:
                    output_dict["pitch_pred"] = (
                        batch["pitch_pred"][i].clone().cpu().numpy()
                    )

                if "energy_pred" in batch:
                    output_dict["energy_pred"] = (
                        batch["energy_pred"][i].clone().cpu().numpy()
                    )

                if "mel" in batch:
                    output_dict["mel_target"] = batch["mel"][i].clone().cpu().numpy()

                if "text" in batch:
                    output_dict["text"] = batch["text"][i]

                if "audio_id" in batch:
                    output_dict["audio_id"] = batch["audio_id"][i]

                output_id = current_id + i
                save_file = self.save_path / part / f"output_{output_id}.npz"
                np.savez(save_file, **output_dict)

        return batch

    def _inference_part(self, part, dataloader):
        """
        Run inference on a given partition and save predictions.
        """

        self.is_train = False
        self.model.eval()

        if self.evaluation_metrics is not None:
            self.evaluation_metrics.reset()

        if self.save_path is not None:
            (self.save_path / part).mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                    metrics=self.evaluation_metrics,
                )

        if self.evaluation_metrics is not None:
            return self.evaluation_metrics.result()
        else:
            return {}
