from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image


class CometMLWriter:
    """
    Class for experiment tracking via Comet ML.

    See https://www.comet.com/docs/.
    """

    def __init__(
        self,
        logger,
        project_config,
        project_name,
        workspace=None,
        api_key=None,
        run_id=None,
        run_name=None,
        mode="online",
        **kwargs,
    ):
        """
        API key can be provided via:
        1. api_key parameter
        2. COMET_API_KEY environment variable
        3. .comet.config file
        """
        try:
            from comet_ml import ExistingExperiment, Experiment, OfflineExperiment

            self.run_id = run_id

            experiment_kwargs = {
                "project_name": project_name,
                "workspace": workspace,
            }

            if api_key:
                experiment_kwargs["api_key"] = api_key

            if run_id:
                try:
                    logger.info(
                        f"Attempting to resume existing CometML experiment: {run_id}"
                    )
                    self.experiment = ExistingExperiment(
                        api_key=api_key, experiment_key=run_id
                    )
                    logger.info(f"Successfully resumed CometML experiment: {run_id}")
                except Exception as e:
                    logger.warning(f"Could not resume experiment {run_id}: {e}")
                    logger.info("Creating new experiment instead...")
                    ExperimentClass = (
                        Experiment if mode == "online" else OfflineExperiment
                    )
                    self.experiment = ExperimentClass(**experiment_kwargs)
            else:
                ExperimentClass = Experiment if mode == "online" else OfflineExperiment
                self.experiment = ExperimentClass(**experiment_kwargs)

            if run_name:
                self.experiment.set_name(run_name)

            self.experiment.log_parameters(project_config)

            if kwargs.get("save_code", False):
                self.experiment.log_code(folder="./")

        except ImportError:
            logger.warning("For use CometML install it via \\n\\t pip install comet_ml")

        self.step = 0
        self.mode = ""
        self.timer = datetime.now()

    def set_step(self, step, mode="train"):
        """Define current step and mode for the tracker."""
        self.mode = mode
        previous_step = self.step
        self.step = step

        self.experiment.set_step(step)

        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar(
                "steps_per_sec", (self.step - previous_step) / duration.total_seconds()
            )
            self.timer = datetime.now()

    def _object_name(self, object_name):
        """Update object_name with the current mode."""
        return f"{object_name}_{self.mode}"

    def add_checkpoint(self, checkpoint_path, save_dir):
        """Log checkpoints to the experiment tracker."""
        self.experiment.log_model(
            name=f"checkpoint_step_{self.step}",
            file_or_folder=checkpoint_path,
            overwrite=True,
        )

    def add_scalar(self, scalar_name, scalar):
        """Log a scalar."""
        self.experiment.log_metric(
            name=self._object_name(scalar_name), value=scalar, step=self.step
        )

    def add_scalars(self, scalars):
        """Log several scalars."""
        for scalar_name, scalar in scalars.items():
            self.add_scalar(scalar_name, scalar)

    def add_image(self, image_name, image):
        """Log an image."""
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
            if image.ndim == 3 and image.shape[0] in [1, 3, 4]:
                image = np.transpose(image, (1, 2, 0))
            if image.ndim == 3 and image.shape[2] == 1:
                image = image.squeeze(2)

        self.experiment.log_image(
            image_data=image, name=self._object_name(image_name), step=self.step
        )

    def add_audio(self, audio_name, audio, sample_rate=None, metadata=None):
        """Log audio with optional metadata (including text)."""
        import os
        import tempfile

        import soundfile as sf

        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()

        if audio.ndim == 1:
            pass
        elif audio.ndim == 2:
            if audio.shape[0] == 1:
                audio = audio.squeeze(0)
            elif audio.shape[1] == 1:
                audio = audio.T.squeeze(0)

        # Format: step_<step>_<audio_name>.wav
        safe_name = self._object_name(audio_name).replace("/", "_").replace("\\", "_")
        temp_filename = f"step_{self.step}_{safe_name}.wav"
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, temp_filename)
        sf.write(temp_path, audio, sample_rate or 22050)

        try:
            full_metadata = {
                "audio_name": self._object_name(audio_name),
                "step": self.step,
            }

            if metadata:
                full_metadata.update(metadata)

            self.experiment.log_audio(
                audio_data=temp_path,
                sample_rate=sample_rate or 22050,
                file_name=temp_filename,
                metadata=full_metadata,
                step=self.step,
            )
        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                pass

    def add_text(self, text_name, text):
        """Log text."""
        self.experiment.log_text(text=text, step=self.step)

    def add_histogram(self, hist_name, values_for_hist, bins=None):
        """Log histogram."""
        if isinstance(values_for_hist, torch.Tensor):
            values_for_hist = values_for_hist.detach().cpu().numpy()

        self.experiment.log_histogram_3d(
            values=values_for_hist, name=self._object_name(hist_name), step=self.step
        )

    def add_table(self, table_name, table: pd.DataFrame):
        """Log table."""
        self.experiment.log_table(
            filename=f"{self._object_name(table_name)}.csv",
            tabular_data=table,
            step=self.step,
        )

    def add_figure(self, figure_name, figure):
        """Log matplotlib figure."""
        self.experiment.log_figure(
            figure_name=self._object_name(figure_name), figure=figure, step=self.step
        )
        plt.close(figure)

    def add_spectrogram(self, spectrogram_name, spectrogram, title=None):
        """Log spectrogram."""
        if isinstance(spectrogram, torch.Tensor):
            spectrogram = spectrogram.detach().cpu().numpy()

        if spectrogram.ndim == 3:
            spectrogram = spectrogram.squeeze(0)

        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(
            spectrogram,
            aspect="auto",
            origin="lower",
            interpolation="none",
            cmap="viridis",
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        if title:
            ax.set_title(title)
        plt.colorbar(im, ax=ax)
        plt.tight_layout()

        self.add_figure(spectrogram_name, fig)

    def add_images(self, image_names, images):
        """Log multiple images."""
        for image_name, image in zip(image_names, images):
            self.add_image(image_name, image)

    def add_pr_curve(self, curve_name, labels, predictions, num_thresholds=127):
        """Log precision-recall curve."""
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()

        thresholds = np.linspace(0, 1, num_thresholds)
        precisions = []
        recalls = []

        for threshold in thresholds:
            pred_labels = (predictions >= threshold).astype(int)
            tp = np.sum((pred_labels == 1) & (labels == 1))
            fp = np.sum((pred_labels == 1) & (labels == 0))
            fn = np.sum((pred_labels == 0) & (labels == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            precisions.append(precision)
            recalls.append(recall)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recalls, precisions)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.grid(True)
        plt.tight_layout()

        self.add_figure(curve_name, fig)

    def add_embedding(self, embedding_name, embeddings, labels=None, images=None):
        """Log embeddings."""
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()

        data = []
        for i, emb in enumerate(embeddings):
            row = {"embedding": emb.tolist()}
            if labels is not None:
                row["label"] = labels[i]
            data.append(row)

        df = pd.DataFrame(data)
        self.add_table(embedding_name, df)

    def finish(self):
        """Finish the CometML experiment."""
        if hasattr(self, "experiment"):
            self.experiment.end()
