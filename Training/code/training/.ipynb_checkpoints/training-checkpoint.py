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
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings("ignore", category=UserWarning)
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
    scaler = None
) -> dict:
    os.makedirs(save_dir, exist_ok=True)
    train_losses = []
    val_losses = []

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            accumulation=accumulation,
            loops_per_epoch=loops_per_epoch,
            scaler = scaler
        )
        val_loss = validate(
            model=model,
            data_loader=val_loader,
            loss_fn=loss_fn,
            device=device
        )
        torch.cuda.empty_cache()
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
                
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"[Epoch {epoch}] Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if epoch % 10 == 0:
            save_path = os.path.join(save_dir, f"{base_name}_epoch{epoch}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint to {save_path}")

    final_path = os.path.join(save_dir, f"{base_name}_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Saved final model to {final_path}")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses
    }

def train_one_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    accumulation: int = 1,
    loops_per_epoch: int = 1,
    scaler = None
):
    model.train()
    total_loss = 0.0

    for loop in range(loops_per_epoch):
        for step, batch in enumerate(data_loader):
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)
        
            with autocast():
                outputs = model(inputs)
                loss = loss_fn(outputs, labels) / accumulation_steps  # scale down loss
        
            scaler.scale(loss).backward()
        
            if (step + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item()

    avg_loss = total_loss / (loops_per_epoch * len(data_loader))
    return avg_loss


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
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss
