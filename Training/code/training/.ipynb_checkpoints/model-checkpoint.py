from monai.networks.nets import UNet
import torch

from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, RandSpatialCropd, 
    NormalizeIntensityd, ToTensord, Compose,
    RandFlipd,RandCropByPosNegLabeld,SpatialPadd )

from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    LinearLR
)

unet_base = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16,32, 64,128,256,512),
    strides=(2, 2, 2,2,2),
    num_res_units=2,
    dropout=0.0,
)

def get_scheduler(
    name: str,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    **kwargs
):
    """
    Creates a learning rate scheduler based on a name.

    Args:
        name (str): One of ["cosine", "reduce_on_plateau", "linear", "none"]
        optimizer: torch optimizer
        num_epochs: total number of epochs (used for cosine)

    Returns:
        torch.optim.lr_scheduler._LRScheduler or None
    """
    name = name.lower()

    if name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=kwargs.get("eta_min", 0.0)
        )

    elif name == "reduce_on_plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=kwargs.get("factor", 0.1),
            patience=kwargs.get("patience", 10),
            verbose=kwargs.get("verbose", True)
        )

    elif name == "linear":
        return LinearLR(
            optimizer,
            start_factor=kwargs.get("start_factor", 1.0),
            end_factor=kwargs.get("end_factor", 0.01),
            total_iters=num_epochs
        )

    elif name == "none" or name is None:
        return None

    else:
        raise ValueError(f"Unknown scheduler type: {name}")
        
def get_transforms(split,name):
    if name == "no patch":
        if split == "train":
            return Compose([
                    LoadImaged(keys=["image", "label"]),
                    EnsureChannelFirstd(keys=["image", "label"]),
                    NormalizeIntensityd(keys=["image"]),
                    RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.5),
                    RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.5),
                    RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.5),
                    SpatialPadd(keys=["image", "label"], spatial_size=(416, 416, 416)),
                    ToTensord(keys=["image", "label"]),
            ])
        else:
            return Compose([
                    LoadImaged(keys=["image", "label"]),
                    EnsureChannelFirstd(keys=["image", "label"]),
                    NormalizeIntensityd(keys=["image"]),
                    SpatialPadd(keys=["image", "label"], spatial_size=(416, 416, 416)),
                    ToTensord(keys=["image", "label"])
                ])
        