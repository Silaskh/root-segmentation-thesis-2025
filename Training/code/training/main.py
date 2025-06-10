print("File opened")

#Library imports
import torch
from torch.optim import Adam
import os
from monai.data.meta_tensor import MetaTensor
from monai.utils.enums import MetaKeys, SpaceKeys
import numpy as np
from torch.cuda.amp import GradScaler
import warnings
import sys

# Set the environment variable to allow expandable segments in PyTorch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()
# Add safe globals for serialization
torch.serialization.add_safe_globals([
    MetaTensor,
    MetaKeys,
    SpaceKeys,
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
from code.data.data_sets import get_data_set
from monai.transforms import Compose
from code.data.data_loading import expand_user_paths,get_data_loader

from code.training.model import unet_base as model
from code.training.model import get_transforms,get_scheduler,AddMaskFromLabel
from code.training.utils import plot_loss_curves

from code.training.loss import get_loss_function

from code.training.training import train_model
from code.training.config import load_yaml_config

from code.post.inference import run_sliding_window_inference,generate_projection_heatmaps


warnings.filterwarnings(
    "ignore",
    message="You are using 'torch.load' with `weights_only=False`.*",
    category=UserWarning
)
warnings.filterwarnings("ignore", message=".*weights_only=False.*")

if len(sys.argv) < 2:
    raise ValueError("Usage: python main.py path/to/config.yaml")

config = load_yaml_config(sys.argv[1])

print("imports done")

#Configuration parameters
LR = config["learning_rate"]
NUM_EPOCHS = config["num_epochs"]
NAME = config["name"]
ACCUMULATION = config["accumulation"]
NUM_LOOPS = config["num_loops"]
CURRICULUM_EPOCH = config.get("curriculum_epoch", None)
VAL_FREQ = config.get("val_freq", 1)

NAME += "_" + str(NUM_EPOCHS)

print(NAME)
os.makedirs(NAME, exist_ok=True)
os.makedirs(f"{NAME}/plots", exist_ok=True)
os.makedirs(f"{NAME}/persistent_cache", exist_ok=True)
os.makedirs(f"{NAME}/persistent_cache/train", exist_ok=True)
os.makedirs(f"{NAME}/persistent_cache/val", exist_ok=True)

#Dataset and data loader
data_set = expand_user_paths(get_data_set(config["train_dataset"]))
val_data_set = expand_user_paths(get_data_set(config["val_dataset"]))

if config["curriculum_dataset"] != None:
    curriculum_trainset = expand_user_paths(get_data_set(config["curriculum_dataset"]))
else:
    curriculum_trainset = None

train_transforms = get_transforms("train", config["train_transforms"])
val_transforms = get_transforms("validation", config["val_transforms"])
train_transforms = Compose([
    *train_transforms.transforms,
    AddMaskFromLabel(keys=["label"])
])
val_transforms = Compose([
    *val_transforms.transforms,
    AddMaskFromLabel(keys=["label"])
])

print("Train transform pipeline:")
for t in train_transforms.transforms:
    print(" -", t)

print("\nVal transform pipeline:")
for t in val_transforms.transforms:
    print(" -", t)

train_loader = get_data_loader(
    data_set,
    train_transforms,
    cache_dir =  f"./{NAME}/persistent_cache/train",
    batch_size = 1,
    shuffle = True,
    num_workers = 4,
    pin_memory = True,)

val_loader = get_data_loader(
    val_data_set,
    val_transforms,
    cache_dir = f"./{NAME}/persistent_cache/val",
    batch_size = 1,
    shuffle = False,
    num_workers = 4,
    pin_memory = True,)
  
loss_cfg = config["loss"]
loss_fn = get_loss_function(loss_cfg["type"], **loss_cfg.get("params", {}))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device) 

optimizer = Adam(model.parameters(), lr=LR)

sched_cfg = config["scheduler"]
scheduler = get_scheduler(
    name=sched_cfg["type"],
    optimizer=optimizer,
    num_epochs=NUM_EPOCHS,
    **sched_cfg.get("params", {})
)

scaler = GradScaler()


    
#Training
print("Starting training")
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
    scaler = scaler,
    val_freq = VAL_FREQ,
    curriculum_epoch = CURRICULUM_EPOCH,
    curriculum_trainset = curriculum_trainset,
)

print("Training Finished")
plot_loss_curves(
    results["train_losses"],
    results["val_losses"],
    save_path=f"./{NAME}/plots/loss_curve.png",
    title=f"Loss Curve: {NAME}"
)

#Evaluation
print("Doing inference")

torch.cuda.empty_cache() 
params = config["inference"]

run_sliding_window_inference(
    model=model,
    image_path=params["image_path"],
    output_path=params["output_path"],
    window_size=tuple(params["window_size"]),
    overlap=params.get("overlap", 0.25),
    progress=params.get("progress", True),
)
generate_projection_heatmaps(
    nifti_path=params["image_path"],
    output_dir=f"./{NAME}/plots/heatmaps",
    name="target"
)

generate_projection_heatmaps(
    nifti_path = params["output_path"],
    output_dir=f"./{NAME}/plots/heatmaps",
    name="final_prediction"
)
