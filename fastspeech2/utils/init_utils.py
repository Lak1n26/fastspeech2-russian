import logging
import os
import random
import secrets
import shutil
import string
import subprocess

import numpy as np
import torch
from omegaconf import OmegaConf

from fastspeech2.logger.logger import setup_logging
from fastspeech2.utils.io_utils import ROOT_PATH


def set_worker_seed(worker_id):
    """
    Set seed for each dataloader worker.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_random_seed(seed):
    """
    Set random seed for model training or inference.
    """
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def generate_id(length: int = 8) -> str:
    """
    Generate a random base-36 string of `length` digits.
    """
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def log_git_commit_and_patch(save_dir):
    """
    Log current git commit and patch to save dir.
    """
    commit_path = save_dir / "git_commit.txt"
    patch_path = save_dir / "git_diff.patch"
    with commit_path.open("w") as f:
        subprocess.call(["git", "rev-parse", "HEAD"], stdout=f)
    with patch_path.open("w") as f:
        subprocess.call(["git", "diff", "HEAD"], stdout=f)


def resume_config(save_dir):
    """
    Get run_id from resume config to continue logging
    to the same experiment.
    """
    saved_config = OmegaConf.load(save_dir / "config.yaml")
    run_id = saved_config.writer.run_id
    return run_id


def saving_init(save_dir, config):
    """
    Initialize saving by getting run_id.
    """
    run_id = None

    if save_dir.exists():
        if config.trainer.get("resume_from") is not None:
            run_id = resume_config(save_dir)
        elif config.trainer.override:
            shutil.rmtree(str(save_dir))
        elif not config.trainer.override:
            raise ValueError(
                "Save directory exists. Change the name or set override=True"
            )

    save_dir.mkdir(exist_ok=True, parents=True)

    if run_id is None:
        run_id = generate_id(length=config.writer.id_length)

    OmegaConf.set_struct(config, False)
    config.writer.run_id = run_id
    OmegaConf.set_struct(config, True)

    OmegaConf.save(config, save_dir / "config.yaml")

    log_git_commit_and_patch(save_dir)


def setup_saving_and_logging(config):
    """
    Initialize the logger, writer, and saving directory.
    """
    run_name = config.writer.get("run_name")
    save_dir = ROOT_PATH / config.trainer.save_dir / run_name
    saving_init(save_dir, config)

    if config.trainer.get("resume_from") is not None:
        setup_logging(save_dir, append=True)
    else:
        setup_logging(save_dir, append=False)
    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)

    return logger
