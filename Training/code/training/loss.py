import torch
import torch.nn as nn
from typing import Optional
from monai.losses import DiceLoss, FocalLoss
from code.training.losses_custom import WeightedBinaryDiceLoss,FocalLossWithSigmoid

# Custom loss wrapper to apply a mask to the loss computation
class MaskLossWrapper(nn.Module):
    def __init__(self, base_loss: nn.Module):
        """
        Wrap a loss to compute only over valid (mask == 1) regions.
        """
        super().__init__()
        self.base_loss = base_loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape or pred.shape != mask.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape}, target {target.shape}, mask {mask.shape}")

        pred_masked = pred * mask
        target_masked = target * mask
        return self.base_loss(pred_masked, target_masked)

# Returns a configured loss function based on the specified name and parameters. 
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

    name = name.lower()

    if name == "dice":
        base = WeightedBinaryDiceLoss(
            foreground_weight=kwargs.get("foreground_weight", 1),
            background_weight=kwargs.get("background_weight", 1),
            smooth=kwargs.get("smooth", 1e-5)
        )
        use_mask = kwargs.get("use_mask", False)

        if use_mask is True:
            return MaskLossWrapper(base)
        else:
            return base

    elif name == "bce":
        return nn.BCEWithLogitsLoss(
            reduction=reduction,
            weight=weight,        
            pos_weight=pos_weight 
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
