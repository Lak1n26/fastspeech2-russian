"""
Training utilities for FastSpeech2.

Contains functions for:
- Freezing/unfreezing model parameters
- Learning rate scheduling
- Model checkpoint management
"""

import logging
from typing import List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def freeze_module(module: nn.Module, module_name: str = "module"):
    """
    Freeze all parameters in a module.
    """
    frozen_count = 0
    for param in module.parameters():
        param.requires_grad = False
        frozen_count += 1

    logger.info(f"Froze {frozen_count} parameters in {module_name}")


def unfreeze_module(module: nn.Module, module_name: str = "module"):
    """
    Unfreeze all parameters in a module.
    """
    unfrozen_count = 0
    for param in module.parameters():
        param.requires_grad = True
        unfrozen_count += 1

    logger.info(f"Unfroze {unfrozen_count} parameters in {module_name}")


def freeze_encoder_decoder(model: nn.Module):
    """
    Freeze encoder and decoder, keep variance adaptor trainable.
    """
    if hasattr(model, "encoder"):
        freeze_module(model.encoder, "encoder")

    if hasattr(model, "decoder"):
        freeze_module(model.decoder, "decoder")

    if hasattr(model, "variance_adaptor"):
        logger.info("Variance adaptor remains trainable")
        trainable_params = sum(
            p.numel() for p in model.variance_adaptor.parameters() if p.requires_grad
        )
        logger.info(f"Trainable parameters in variance adaptor: {trainable_params}")


def freeze_all_except_emotion(model: nn.Module):
    """
    Freeze everything except emotion predictor and embedding.
    """
    if hasattr(model, "encoder"):
        freeze_module(model.encoder, "encoder")

    if hasattr(model, "decoder"):
        freeze_module(model.decoder, "decoder")

    if hasattr(model, "variance_adaptor"):
        va = model.variance_adaptor

        if hasattr(va, "duration_predictor"):
            freeze_module(va.duration_predictor, "variance_adaptor.duration_predictor")

        if hasattr(va, "pitch_predictor"):
            freeze_module(va.pitch_predictor, "variance_adaptor.pitch_predictor")
        if hasattr(va, "pitch_embedding"):
            freeze_module(va.pitch_embedding, "variance_adaptor.pitch_embedding")

        if hasattr(va, "energy_predictor"):
            freeze_module(va.energy_predictor, "variance_adaptor.energy_predictor")
        if hasattr(va, "energy_embedding"):
            freeze_module(va.energy_embedding, "variance_adaptor.energy_embedding")

        emotion_trainable = 0
        if hasattr(va, "emotion_predictor"):
            emotion_trainable += sum(
                p.numel() for p in va.emotion_predictor.parameters() if p.requires_grad
            )
            logger.info("Emotion predictor remains trainable")
        if hasattr(va, "emotion_embedding"):
            emotion_trainable += sum(
                p.numel() for p in va.emotion_embedding.parameters() if p.requires_grad
            )
            logger.info("Emotion embedding remains trainable")

        logger.info(f"Trainable emotion parameters: {emotion_trainable}")


def unfreeze_all(model: nn.Module):
    """
    Unfreeze all model parameters.
    """
    if hasattr(model, "encoder"):
        unfreeze_module(model.encoder, "encoder")

    if hasattr(model, "variance_adaptor"):
        unfreeze_module(model.variance_adaptor, "variance_adaptor")

    if hasattr(model, "decoder"):
        unfreeze_module(model.decoder, "decoder")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {total_params}")


def freeze_specific_modules(model: nn.Module, modules_to_freeze: List[str]):
    """
    Freeze specific modules by name.
    """
    for module_name in modules_to_freeze:
        if hasattr(model, module_name):
            module = getattr(model, module_name)
            freeze_module(module, module_name)
        else:
            logger.warning(f"Module '{module_name}' not in model")


def print_trainable_parameters(model: nn.Module):
    """
    Print detailed information about trainable parameters.
    """
    trainable_params = 0
    frozen_params = 0

    logger.info("=" * 60)
    logger.info("Model Parameters Summary")
    logger.info("=" * 60)

    for name, param in model.named_parameters():
        num_params = param.numel()
        if param.requires_grad:
            trainable_params += num_params
        else:
            frozen_params += num_params

    total_params = trainable_params + frozen_params

    logger.info(
        f"Trainable parameters: {trainable_params} ({100 * trainable_params / total_params: .2f}%)"
    )
    logger.info(
        f"Frozen parameters: {frozen_params} ({100 * frozen_params / total_params: .2f}%)"
    )
    logger.info(f"Total parameters: {total_params}")
    logger.info("=" * 60)


def get_emotion_only_params(model: nn.Module) -> List[torch.nn.Parameter]:
    """
    Get parameters related to emotion prediction only.
    """
    emotion_params = []

    if hasattr(model, "variance_adaptor"):
        va = model.variance_adaptor
        if hasattr(va, "emotion_predictor"):
            emotion_params.extend(list(va.emotion_predictor.parameters()))
        if hasattr(va, "emotion_embedding"):
            emotion_params.extend(list(va.emotion_embedding.parameters()))

    logger.info(f"Found {len(emotion_params)} emotion-specific parameters")
    return emotion_params


def setup_fine_tuning_optimizer(
    model: nn.Module,
    base_lr: float = 1e-4,
    emotion_lr: float = 1e-3,
    weight_decay: float = 1e-6,
) -> torch.optim.Optimizer:
    """
    Setup optimizer with different learning rates for emotion components.
    """
    emotion_param_ids = set(id(p) for p in get_emotion_only_params(model))

    emotion_params = []
    base_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if id(param) in emotion_param_ids:
            emotion_params.append(param)
        else:
            base_params.append(param)

    param_groups = [
        {"params": base_params, "lr": base_lr},
        {"params": emotion_params, "lr": emotion_lr},
    ]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    logger.info(
        f"Setup optimizer with {len(base_params)} base params (lr={base_lr}) "
        f"and {len(emotion_params)} emotion params (lr={emotion_lr})"
    )

    return optimizer
