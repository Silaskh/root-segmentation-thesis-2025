import os
import torch
import warnings
from torch.utils.data import DataLoader
from typing import Callable, List
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    LinearLR
)
import time
from torch.cuda.amp import autocast, GradScaler
from code.training.loss import MaskLossWrapper
import math
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings(
    "ignore",
    message="You are using 'torch.load' with `weights_only=False`.*",
    category=UserWarning
)
warnings.filterwarnings("ignore", message=".*weights_only=False.*")

#Module containing training functions and utilities

# Function to perform a sanity check on predictions and targets
def debug_sanity_check(pred: torch.Tensor, target: torch.Tensor, epoch: int, max_epochs: int = 10):
    with torch.no_grad():
       
        pred = torch.sigmoid(pred[0])  
        target = target[0]            
        if pred.shape[0] == 1:
            pred = pred[0]
            target = target[0]

        pred_bin = (pred > 0.5).float()

        print(f"[Epoch {epoch}]")
        print(f"  → Prediction mean (after sigmoid): {pred.mean().item():.4f}")
        print(f"  → Prediction > 0.5 count: {pred_bin.sum().item():.0f}")
        print(f"  → Target == 1 count: {(target == 1).sum().item():.0f}")
        print(f"  → Target == 0 count: {(target == 0).sum().item():.0f}")
        print(f"  → Target sum (float): {target.sum().item():.2f}")
        print(f"  → Prediction sum (float): {pred.sum().item():.2f}")
        print("")

# Function to train the model for a specified number of epochs 
def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    loops_per_epoch: int,
    accumulation: int,
    save_dir: str,
    base_name: str,
    scheduler=None,
    scheduler_name=None,
    scaler = None,
    curriculum_epoch = None,
    curriculum_trainset = None,
    val_freq = 1,
) -> dict:
    os.makedirs(save_dir, exist_ok=True)
    train_losses = []
    val_losses = []

    for epoch in range(1, num_epochs + 1):
        start = time.time()
        
        if epoch < 11:
            def debug_hook(pred, target, batch_idx):
                if batch_idx == 0:
                    debug_sanity_check(pred, target, epoch)
        else:
            debug_hook = None  
            
        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            accumulation=accumulation,
            loops_per_epoch=loops_per_epoch,
            scaler = scaler,
            debug_hook = debug_hook
        )
        if math.isnan(train_loss):
            print("Train loss is nan skipping epoch")
            continue
            
        train_losses.append(train_loss)  
        print(f"[Epoch {epoch}] Train: {train_loss:.4f} it took {time.time() - start:.2f} seconds")
        
        if epoch % val_freq == 0:
            print("validating")
            start = time.time()
            val_loss = validate(
                model=model,
                data_loader=val_loader,
                loss_fn=loss_fn,
                device=device
            )
            print(f"Validation finished it took {time.time() - start:.2f} seconds")
            print(f"Validation loss {val_loss}")
            val_losses.append((epoch,val_loss))
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                if epoch % val_freq == 0:
                    scheduler.step(val_loss)
            else:
                scheduler.step()
        
        torch.cuda.empty_cache()
        if epoch % 10 == 0:
            save_path = os.path.join(save_dir, f"{base_name}_epoch{epoch}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint to {save_path}")
            
        if curriculum_epoch is not None and epoch == curriculum_epoch:
            print(f"[Curriculum] Expanding training dataset at epoch {epoch}")
            train_loader.dataset.data = curriculum_trainset

    final_path = os.path.join(save_dir, f"{base_name}_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Saved final model to {final_path}")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses
    }

#Function to train the model for one epoch called by train_model
def train_one_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    accumulation: int = 1,
    loops_per_epoch: int = 1,
    scaler = None,
    debug_hook=None
):
    model.train()
    total_loss = 0.0
    running_loss = 0.0

    for loop in range(loops_per_epoch):
        for step, batch in enumerate(data_loader):
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)
            print(inputs.shape)
           
            outputs = model(inputs)
            if isinstance(loss_fn, MaskLossWrapper):
                mask = batch["mask"].to(device)
                loss = loss_fn(outputs, labels, mask) / accumulation
            else:
                loss = loss_fn(outputs, labels) / accumulation

            if scaler is not None:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

                running_loss += loss.item()

                if (step + 1) % accumulation == 0 or (step + 1) == len(data_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    total_loss += running_loss
                    running_loss = 0.0
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                running_loss += loss.item()

                if (step + 1) % accumulation == 0 or (step + 1) == len(data_loader):
                    optimizer.step()
                    optimizer.zero_grad()
                    total_loss += running_loss
                    running_loss = 0.0
            if debug_hook is not None:
                debug_hook(outputs, labels, step)
    avg_loss = total_loss / (loops_per_epoch * len(data_loader))
    return avg_loss



# Function to validate the model on a validation dataset called by train_model
def validate(
    model: torch.nn.Module,
    data_loader: DataLoader,
    loss_fn: Callable,
    device: torch.device
):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in data_loader:
            inputs, targets = batch["image"].to(device), batch["label"].to(device)
            mask = batch["mask"].to(device)
            outputs = model(inputs)
            if isinstance(loss_fn, MaskLossWrapper):
                mask = batch["mask"].to(device)
                loss = loss_fn(outputs, targets, mask)
            else:
                loss = loss_fn(outputs, targets)
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss
