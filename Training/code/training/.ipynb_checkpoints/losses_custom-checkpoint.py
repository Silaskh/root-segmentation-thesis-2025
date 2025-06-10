import torch
import torch.nn as nn
from monai.losses import FocalLoss

class FocalLossWithSigmoid(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor = None,
        reduction: str = "mean"
    ):
        """
        A wrapper around MONAI's FocalLoss that applies sigmoid internally.

        Args:
            gamma (float): Focusing parameter.
            weight (Tensor): Class weights, same as FocalLoss.
            reduction (str): Reduction method.
        """
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.focal_loss = FocalLoss(
            gamma=gamma,
            weight=weight,
            reduction=reduction
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: Raw logits (B, 1, D, H, W)
            target: Ground truth labels, same shape
        Returns:
            Focal loss after applying sigmoid
        """
        probs = self.sigmoid(input)
        return self.focal_loss(probs, target)
class WeightedBinaryDiceLoss(nn.Module):
    def __init__(self, foreground_weight=1.0, background_weight=0.1, smooth=1e-5):
        """
        Label-based voxel-wise weighted Dice loss for binary segmentation.

        Args:
            foreground_weight (float): Weight for voxels where target == 1.
            background_weight (float): Weight for voxels where target == 0.
            smooth (float): Smoothing term to prevent division by zero.
        """
        super().__init__()
        self.foreground_weight = foreground_weight
        self.background_weight = background_weight
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Raw logits (B, 1, D, H, W)
            target: Binary ground truth, same shape

        Returns:
            Dice loss (float tensor)
        """
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape}, target {target.shape}")

        pred = torch.sigmoid(pred)  # <--- Now handles sigmoid internally

        pred = pred.view(-1)
        target = target.view(-1)

        weights = torch.where(target == 1, self.foreground_weight, self.background_weight).to(pred.device)

        intersection = torch.sum(weights * pred * target)
        union = torch.sum(weights * pred) + torch.sum(weights * target)

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice_score
