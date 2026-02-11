"""
Training script for FastSpeech2.
"""

import sys
import warnings
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from fastspeech2.datasets.data_utils import get_dataloaders
from fastspeech2.trainer import Trainer
from fastspeech2.utils.init_utils import set_random_seed, setup_saving_and_logging

sys.path.append(str(Path(__file__).parent))

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(
    version_base=None,
    config_path="fastspeech2/configs",
    config_name="fastspeech2_train",
)
def main(config):
    """
    Main training script for FastSpeech2.
    """
    set_random_seed(config.trainer.seed)
    project_config = OmegaConf.to_container(config, resolve=True)

    logger = setup_saving_and_logging(config)
    logger.info("=" * 60)
    logger.info("FastSpeech2 Training")
    logger.info("=" * 60)

    force_new_run = config.trainer.get("force_new_cometml_run", False)

    if config.trainer.get("resume_from") is not None and not force_new_run:
        resume_path = (
            Path(config.trainer.save_dir)
            / config.writer.run_name
            / config.trainer.resume_from
        )
        if resume_path.exists():
            try:
                logger.info(f"Checking for run_id in checkpoint: {resume_path}")
                checkpoint = torch.load(
                    resume_path, map_location="cpu", weights_only=False
                )
                if "config" in checkpoint and "writer" in checkpoint["config"]:
                    saved_run_id = checkpoint["config"]["writer"].get("run_id")
                    if saved_run_id:
                        logger.info(f"Found run_id in checkpoint: {saved_run_id}")
                        config.writer.run_id = saved_run_id
                    else:
                        logger.info(
                            "No run_id found in checkpoint, will create new CometML experiment"
                        )
                        config.writer.run_id = None
            except Exception as e:
                logger.warning(f"Could not read checkpoint for run_id: {e}")
                config.writer.run_id = None
    else:
        if force_new_run:
            logger.info("FORCE_NEW_RUN enabled - creating NEW CometML experiment")
        else:
            logger.info(
                "Starting NEW training run - clearing run_id to create new CometML experiment"
            )
        config.writer.run_id = None

    writer = instantiate(config.writer, logger, project_config)

    if hasattr(writer, "run_id") and writer.run_id:
        config.writer.run_id = writer.run_id
        logger.info(f"Saved run_id to config: {writer.run_id}")

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    logger.info(f"Using device: {device}")

    logger.info("\nSetting up dataloaders...")
    dataloaders, batch_transforms = get_dataloaders(config, device)

    logger.info(f"Train dataloader: {len(dataloaders['train'])} batches")
    if "val" in dataloaders:
        logger.info(f"Val dataloader: {len(dataloaders['val'])} batches")

    logger.info("\nBuilding model...")
    model = instantiate(config.model).to(device)
    logger.info(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"\nTotal parameters: {total_params}")
    logger.info(f"Trainable parameters: {trainable_params}")

    if config.trainer.get("freeze_encoder_decoder", False):
        logger.info("\n" + "=" * 60)
        logger.info("FREEZING ENCODER/DECODER FOR EMOTION FINE-TUNING")
        logger.info("=" * 60)

        from fastspeech2.utils.train_utils import (
            freeze_all_except_emotion,
            freeze_encoder_decoder,
            print_trainable_parameters,
        )

        freeze_only_emotion = config.trainer.get("freeze_all_except_emotion", False)

        if freeze_only_emotion:
            logger.info(
                "Strategy: ULTRA-CONSERVATIVE - freeze everything except emotion"
            )
            freeze_all_except_emotion(model)
        else:
            logger.info(
                "Strategy: CONSERVATIVE - freeze encoder/decoder, train full variance adaptor"
            )
            freeze_encoder_decoder(model)

        print_trainable_parameters(model)

    logger.info("\nSetting up loss function...")
    loss_function = instantiate(config.loss_function).to(device)

    metrics = instantiate(config.metrics) if "metrics" in config else None

    logger.info("\nSetting up optimizer...")
    trainable_params_list = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.optimizer, params=trainable_params_list)
    logger.info(f"Optimizer: {optimizer.__class__.__name__}")
    logger.info(f"Learning rate: {config.optimizer.lr}")

    logger.info("\nSetting up LR scheduler...")

    if config.lr_scheduler._target_.endswith("OneCycleLR"):
        if config.lr_scheduler.get("total_steps") is None:
            epoch_len = config.trainer.get("epoch_len")
            if epoch_len is None:
                epoch_len = len(dataloaders["train"])

            if config.trainer.get("resume_from") is not None:
                resume_path = (
                    Path(config.trainer.save_dir)
                    / config.writer.run_name
                    / config.trainer.resume_from
                )
                if resume_path.exists():
                    try:
                        checkpoint = torch.load(
                            resume_path, map_location="cpu", weights_only=False
                        )
                        start_epoch = checkpoint["epoch"] + 1

                        if config.trainer.get("reset_epoch_counter", False):
                            total_steps = config.trainer.n_epochs * epoch_len
                            logger.info(
                                f"Reset epoch counter enabled: using full total_steps = {total_steps}"
                            )
                        else:
                            remaining_epochs = config.trainer.n_epochs - start_epoch + 1
                            total_steps = remaining_epochs * epoch_len
                            logger.info(
                                f"Resuming from epoch {start_epoch}: calculated remaining total_steps = {total_steps}"
                            )
                    except Exception as e:
                        logger.warning(
                            f"Could not calculate remaining steps from checkpoint: {e}"
                        )
                        total_steps = config.trainer.n_epochs * epoch_len
                        logger.info(f"Falling back to full total_steps: {total_steps}")
                else:
                    total_steps = config.trainer.n_epochs * epoch_len
                    logger.info(
                        f"Checkpoint not found, using full total_steps: {total_steps}"
                    )
            else:
                total_steps = config.trainer.n_epochs * epoch_len
                logger.info(f"New training: calculated total_steps = {total_steps}")

            config.lr_scheduler.total_steps = total_steps

    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)
    logger.info(f"LR Scheduler: {lr_scheduler.__class__.__name__}")

    epoch_len = config.trainer.get("epoch_len")

    logger.info("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    logger.info("\n" + "=" * 60)
    logger.info("Training Configuration")
    logger.info("=" * 60)
    logger.info(f"Epochs: {config.trainer.n_epochs}")
    logger.info(f"Batch size: {config.dataloader.batch_size}")
    logger.info(f"Optimizer: {optimizer.__class__.__name__}")
    logger.info(f"Learning rate: {config.optimizer.lr}")
    logger.info(f"Device: {device}")
    logger.info(f"Save dir: {config.trainer.save_dir}")
    logger.info(f"Monitor metric: {config.trainer.monitor}")
    logger.info(f"Save period: {config.trainer.save_period} epochs")
    logger.info(f"Early stopping: {config.trainer.early_stop} epochs")

    if config.trainer.resume_from:
        logger.info(f"Resume from: {config.trainer.resume_from}")

    logger.info("=" * 60)

    try:
        logger.info("\nStarting training...\n")
        trainer.train()
        logger.info("\nTraining completed successfully!")

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")

    except Exception as e:
        logger.error(f"\nTraining failed with error: {e}")
        import traceback

        traceback.print_exc()
        raise

    finally:
        if writer is not None:
            writer.finish()
        logger.info("Training session ended")


if __name__ == "__main__":
    main()
