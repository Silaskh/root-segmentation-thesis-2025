import torch
import torch.nn as nn
from monai.losses import FocalLoss

#Module containing custom loss functions



#Adapted from MONAI's FocalLoss to apply sigmoid internally
class FocalLossWithSigmoid(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor = None,
        reduction: str = "mean"
    ):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.focal_loss = FocalLoss(
            gamma=gamma,
            weight=weight,
            reduction=reduction
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(input)
        probs = torch.clamp(probs, 1e-4, 1 - 1e-4)
        loss = self.focal_loss(probs, target)
        return loss
                
import torch
import torch.nn as nn

# Custom loss function that computes a weighted binary Dice loss
class WeightedBinaryDiceLoss(nn.Module):
    def __init__(self, foreground_weight=1.0, background_weight=0.1, smooth=1e-5):
        super().__init__()
        self.foreground_weight = foreground_weight
        self.background_weight = background_weight
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape}, target {target.shape}")

        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)

        if torch.sum(target) == 0:
            agreement = 1.0 - torch.abs(pred - target)
            weights = torch.full_like(target, self.background_weight)
            weighted_agreement = torch.sum(weights * agreement)
            weighted_total = torch.sum(weights)
            dice_score = (2.0 * weighted_agreement + self.smooth) / (2.0 * weighted_total + self.smooth)
        else:
            weights = torch.where(target == 1, self.foreground_weight, self.background_weight).to(pred.device)
            intersection = torch.sum(weights * pred * target)
            union = torch.sum(weights * pred) + torch.sum(weights * target)
            dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice_score

