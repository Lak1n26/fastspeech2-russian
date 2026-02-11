import torch


def collate_fn(dataset_items: list[dict]):
    """
    Converts individual items into a batch.
    """

    result_batch = {}

    result_batch["data_object"] = torch.vstack(
        [elem["data_object"] for elem in dataset_items]
    )
    result_batch["labels"] = torch.tensor([elem["labels"] for elem in dataset_items])

    return result_batch
