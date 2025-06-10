from monai.networks.nets import UNet
import torch
from monai.transforms import MapTransform


from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, RandSpatialCropd, 
    NormalizeIntensityd, ToTensord, Compose,
    RandFlipd,RandCropByPosNegLabeld,SpatialPadd )
    
    

from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    LinearLR
)
#Module containing custom transforms and model definitions
# Custom transform to add a binary mask from the label

class AddMaskFromLabel(MapTransform):
    def __call__(self, data):
        d = dict(data)
        d["mask"] = (d["label"] != 0).float()
        return d

# Base UNet model for 3D segmentation tasks
unet_base = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(8,16,32, 64,128,256),
    strides=(2, 2, 2,2,2),
    num_res_units=2,
    dropout=0.0,
)

# Function to get a learning rate scheduler based on the name
def get_scheduler(
    name: str,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    **kwargs
):

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
        
from monai.transforms import RandCropByPosNegLabeld, CenterSpatialCropd

# Function to get the appropriate transforms based on the split and name
def get_transforms(split, name):
    if name == "no_patch":
        if split == "train":
            return Compose([
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                NormalizeIntensityd(keys=["image"]),
                SpatialPadd(keys=["image", "label"], spatial_size=(416, 416, 416)),
                ToTensord(keys=["image", "label"]),
            ])
        else:
            return Compose([
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                NormalizeIntensityd(keys=["image"]),
                SpatialPadd(keys=["image", "label"], spatial_size=(416, 416, 416)),
                ToTensord(keys=["image", "label"]),
            ])

    elif name == "overfit":
        if split == "train":
            return Compose([
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                NormalizeIntensityd(keys=["image"]),
                RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.5),
                RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.5),
                RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.5),
                SpatialPadd(keys=["image", "label"], spatial_size=(480, 480, 480)),
                ToTensord(keys=["image", "label"]),
            ])
        else:
            return Compose([
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                NormalizeIntensityd(keys=["image"]),
                SpatialPadd(keys=["image", "label"], spatial_size=(480, 480, 480)),
                ToTensord(keys=["image", "label"]),
            ])

    elif name == "patch_based":
        patch_size = (416, 416, 416)  
        if split == "train":
            return Compose([
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                NormalizeIntensityd(keys=["image"]),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=patch_size,
                    pos=1,      
                    neg=0,      
                    num_samples=1, 
                    image_key="image",
                    image_threshold=0,  
                ),
                ToTensord(keys=["image", "label"]),
            ])
        else:
            return Compose([
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                NormalizeIntensityd(keys=["image"]),
                CenterSpatialCropd(keys=["image", "label"], roi_size=patch_size),
                ToTensord(keys=["image", "label"]),
            ])

    else:
        raise ValueError(f"Unknown transform pipeline name: {name}")

