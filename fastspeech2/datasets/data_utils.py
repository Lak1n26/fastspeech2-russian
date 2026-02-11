"""
Data utilities for creating dataloaders.
"""

import torch
from hydra.utils import instantiate


def inf_loop(dataloader):
    """
    Infinite wrapper for DataLoader (for iteration-based training).
    """
    while True:
        for batch in dataloader:
            yield batch


def get_dataloaders(config, device):
    """
    Create dataloaders from config.
    """
    dataloaders = {}

    if "datasets" in config:
        if "train" in config.datasets:
            train_dataset = instantiate(config.datasets.train)
            train_dataloader = instantiate(
                config.dataloader, dataset=train_dataset, shuffle=True, drop_last=True
            )
            dataloaders["train"] = train_dataloader

        if "val" in config.datasets:
            val_dataset = instantiate(config.datasets.val)
            val_dataloader = instantiate(
                config.dataloader, dataset=val_dataset, shuffle=False, drop_last=False
            )
            dataloaders["val"] = val_dataloader

        if "test" in config.datasets:
            test_dataset = instantiate(config.datasets.test)
            test_dataloader = instantiate(
                config.dataloader, dataset=test_dataset, shuffle=False, drop_last=False
            )
            dataloaders["test"] = test_dataloader

    batch_transforms = None
    if "transforms" in config and "batch_transforms" in config.transforms:
        batch_transforms = instantiate(config.transforms.batch_transforms)
        if batch_transforms is not None and hasattr(batch_transforms, "to"):
            batch_transforms = batch_transforms.to(device)

    return dataloaders, batch_transforms
