from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image


class WandBWriter:
    """
    Class for experiment tracking via WandB.

    See https://docs.wandb.ai/.
    """

    def __init__(
        self,
        logger,
        project_config,
        project_name,
        entity=None,
        run_id=None,
        run_name=None,
        mode="online",
        **kwargs,
    ):
        """
        API key is expected to be provided by the user in the terminal.
        """
        try:
            import wandb

            wandb.login()

            self.run_id = run_id

            wandb.init(
                project=project_name,
                entity=entity,
                config=project_config,
                name=run_name,
                resume="allow",
                id=self.run_id,
                mode=mode,
                save_code=kwargs.get("save_code", False),
            )
            self.wandb = wandb

        except ImportError:
            logger.warning("For use wandb install it via \n\t pip install wandb")

        self.step = 0
        self.mode = ""
        self.timer = datetime.now()

    def set_step(self, step, mode="train"):
        """
        Define current step and mode for the tracker.
        """
        self.mode = mode
        previous_step = self.step
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar(
                "steps_per_sec", (self.step - previous_step) / duration.total_seconds()
            )
            self.timer = datetime.now()

    def _object_name(self, object_name):
        """
        Update object_name (scalar, image, etc.) with the
        current mode (partition name)
        """
        return f"{object_name}_{self.mode}"

    def add_checkpoint(self, checkpoint_path, save_dir):
        """
        Log checkpoints to the experiment tracker.
        """
        self.wandb.save(checkpoint_path, base_path=save_dir)

    def add_scalar(self, scalar_name, scalar):
        """
        Log a scalar to the experiment tracker.
        """
        self.wandb.log(
            {
                self._object_name(scalar_name): scalar,
            },
            step=self.step,
        )

    def add_scalars(self, scalars):
        """
        Log several scalars to the experiment tracker.
        """
        self.wandb.log(
            {
                self._object_name(scalar_name): scalar
                for scalar_name, scalar in scalars.items()
            },
            step=self.step,
        )

    def add_image(self, image_name, image):
        """
        Log an image to the experiment tracker.
        """
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
            if image.ndim == 3 and image.shape[0] in [1, 3, 4]:
                image = np.transpose(image, (1, 2, 0))
            if image.ndim == 3 and image.shape[2] == 1:
                image = image.squeeze(2)

        self.wandb.log(
            {self._object_name(image_name): self.wandb.Image(image)}, step=self.step
        )

    def add_audio(self, audio_name, audio, sample_rate=None, metadata=None):
        """
        Log an audio to the experiment tracker.
        """
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()

        if audio.ndim == 1:
            pass
        elif audio.ndim == 2:
            if audio.shape[0] == 1:
                audio = audio.squeeze(0)
            elif audio.shape[1] == 1:
                audio = audio.T.squeeze(0)

        caption = None
        if metadata and "text" in metadata:
            caption = metadata["text"]

        wandb_audio = self.wandb.Audio(audio, sample_rate=sample_rate, caption=caption)

        log_dict = {self._object_name(audio_name): wandb_audio}

        if metadata:
            for key, value in metadata.items():
                if key != "text":
                    log_dict[f"{self._object_name(audio_name)}_meta_{key}"] = value

        self.wandb.log(log_dict, step=self.step)

    def add_text(self, text_name, text):
        """
        Log text to the experiment tracker.
        """
        self.wandb.log(
            {self._object_name(text_name): self.wandb.Html(text)}, step=self.step
        )

    def add_histogram(self, hist_name, values_for_hist, bins=None):
        """
        Log histogram to the experiment tracker.
        """
        if isinstance(values_for_hist, torch.Tensor):
            values_for_hist = values_for_hist.detach().cpu().numpy()

        np_hist = np.histogram(values_for_hist, bins=bins)
        if np_hist[0].shape[0] > 512:
            np_hist = np.histogram(values_for_hist, bins=512)

        hist = self.wandb.Histogram(np_histogram=np_hist)

        self.wandb.log({self._object_name(hist_name): hist}, step=self.step)

    def add_table(self, table_name, table: pd.DataFrame):
        """
        Log table to the experiment tracker.
        """
        self.wandb.log(
            {self._object_name(table_name): self.wandb.Table(dataframe=table)},
            step=self.step,
        )

    def add_figure(self, figure_name, figure):
        """
        Log a matplotlib figure to the experiment tracker.
        """
        self.wandb.log(
            {self._object_name(figure_name): self.wandb.Image(figure)},
            step=self.step,
        )
        plt.close(figure)

    def add_spectrogram(self, spectrogram_name, spectrogram, title=None):
        """
        Log a spectrogram to the experiment tracker.
        """
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
        """
        Log multiple images to the experiment tracker.
        """
        for image_name, image in zip(image_names, images):
            self.add_image(image_name, image)

    def add_pr_curve(self, curve_name, labels, predictions, num_thresholds=127):
        """
        Log precision-recall curve to the experiment tracker.
        """
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
        """
        Log embeddings to the experiment tracker for visualization.
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()

        data = []
        for i, emb in enumerate(embeddings):
            row = {"embedding": emb.tolist()}
            if labels is not None:
                row["label"] = labels[i]
            if images is not None:
                row["image"] = self.wandb.Image(images[i])
            data.append(row)

        table = self.wandb.Table(
            columns=list(data[0].keys()), data=[list(d.values()) for d in data]
        )
        self.wandb.log({self._object_name(embedding_name): table}, step=self.step)

    def finish(self):
        """
        Finish the WandB run.
        """
        if hasattr(self, "wandb"):
            self.wandb.finish()
