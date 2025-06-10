import torch
import torch.nn as nn
from typing import Optional
from monai.losses import DiceLoss, FocalLoss
from code.training.losses_custom import WeightedBinaryDiceLoss,FocalLossWithSigmoid


def get_loss_function(
    name: str,
    reduction: str = "mean",
    softmax: bool = False,
    sigmoid: bool = True,
    include_background: bool = True,
    to_onehot_y: bool = False,
    weight: Optional[torch.Tensor] = None,
    pos_weight: Optional[torch.Tensor] = None,
    **kwargs
) -> nn.Module:
    """
    Returns a configured loss function.

    Args:
        name (str): One of ["dice", "bce", "focal", "weighted_dice"]
        reduction (str): Loss reduction method: "mean", "sum", or "none".
        softmax (bool): Apply softmax to predictions (Dice/Focal).
        sigmoid (bool): Apply sigmoid to predictions (Dice/Focal).
        include_background (bool): Include background class (Dice/Focal).
        to_onehot_y (bool): Convert target to one-hot (Dice/Focal).
        weight (Tensor, optional): Class weights (Dice/Focal), or per-element weights (BCE).
        pos_weight (Tensor, optional): Positive class weight (BCE only).
        **kwargs: Extra arguments passed to the loss constructor.

    Returns:
        nn.Module: Configured loss function.
    """
    name = name.lower()

    if name == "dice":
        return DiceLoss(
            reduction=reduction,
            sigmoid=sigmoid,
            softmax=softmax,
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            weight=weight,
            **kwargs
        )

    elif name == "bce":
        return nn.BCEWithLogitsLoss(
            reduction=reduction,
            weight=weight,        # per-element weighting (rarely used)
            pos_weight=pos_weight # positive class up-weighting
        )

    elif name == "focal":
        return FocalLossWithSigmoid(
            gamma=kwargs.get("gamma", 2.0),
            weight=weight,
            reduction=reduction)

    elif name == "weighted_dice":
        return WeightedBinaryDiceLoss(
            foreground_weight=kwargs.get("foreground_weight", 1.0),
            background_weight=kwargs.get("background_weight", 0.1),
            smooth=kwargs.get("smooth", 1e-5)
        )

    else:
        raise ValueError(f"Unknown loss function: {name}")
