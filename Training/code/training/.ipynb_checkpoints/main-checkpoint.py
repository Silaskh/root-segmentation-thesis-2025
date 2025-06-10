print("File opened")

#Library imports
import torch
from torch.optim import Adam
import os
import warnings
import monai
import numpy as np
from torch.cuda.amp import GradScaler



torch.serialization.add_safe_globals([
    monai.data.meta_tensor.MetaTensor,
    monai.utils.enums.MetaKeys,
    monai.utils.enums.SpaceKeys,
    np.core.multiarray._reconstruct,
    np.ndarray,
    np.dtype,
    np.dtypes.Int32DType,
    np.dtypes.Int16DType,
    np.dtypes.UInt8DType,
    np.dtypes.Float32DType,
    np.dtypes.Float64DType,
])


print("Libraies imported") 
#File imports
from code.data.data_sets import og_res_week3_30_2_400_only_root as data_set
from code.data.data_sets import og_res_week3_30_2_400_all as create_val_data_set

from code.data.data_loading import expand_user_paths,get_data_loader

from code.training.model import unet_base as model
from code.training.model import get_transforms,get_scheduler

from code.training.loss import get_loss_function

from code.training.training import train_model


warnings.filterwarnings(
    "ignore",
    message="You are using 'torch.load' with `weights_only=False`.*",
    category=UserWarning
)
print("imports done")

#Variables
LR = 1e-3
NUM_EPOCHS = 25
NUM_LOOPS = 1
ACCUMULATION = 1
NAME = "Test"


NAME += "_" + str(NUM_EPOCHS)

os.makedirs(NAME, exist_ok=True)

#Dataset and data loader
data_set = expand_user_paths(data_set)
val_data_set = expand_user_paths(create_val_data_set())

train_transforms = get_transforms("train","no patch")

val_transforms = get_transforms("validation","no patch")

train_loader = get_data_loader(
    data_set,
    train_transforms,
    cache_dir =  "./persistent_cache/train",
    batch_size = 1,
    shuffle = True,
    num_workers = 4,
    pin_memory = True,)

val_loader = get_data_loader(
    val_data_set,
    val_transforms,
    cache_dir = "./persistent_cache/val",
    batch_size = 1,
    shuffle = False,
    num_workers = 4,
    pin_memory = True,)
#Loss functions

# MONAI DiceLoss with class weights
#loss_fn = get_loss_function("dice", weight=torch.tensor([0.3, 0.7]), include_background=True)

# BCE with positive class emphasis
#loss_fn = get_loss_function("bce",pos_weight=torch.tensor([2.0]))

# Your voxel-weighted Dice
#loss_fn = get_loss_function("weighted_dice",foreground_weight=1.0,background_weight=0.05)

loss_fn = get_loss_function("focal",gamma=1.5,weight=torch.tensor([0.3, 0.7]))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

optimizer = Adam(model.parameters(), lr=LR)

scheduler = get_scheduler(
    name="reduce_on_plateau",
    optimizer=optimizer,
    num_epochs=NUM_EPOCHS,  # not used for this scheduler but required by interface
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=True
)

scaler = GradScaler()
print("Variables declared")

    
#Training

results = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    num_epochs=NUM_EPOCHS,
    loops_per_epoch=NUM_LOOPS,
    accumulation=ACCUMULATION,
    save_dir=f"./{NAME}/models",
    base_name=NAME,
    scaler = scaler
)

print("Training Finished")
#Evaluation
