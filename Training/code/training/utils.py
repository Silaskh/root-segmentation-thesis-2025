import matplotlib.pyplot as plt
import os


import matplotlib.pyplot as plt

# Utility function to plot training and validation loss curves

def plot_loss_curves(train_losses, val_loss_points, save_path="loss_curve.png", title="Training vs Validation Loss"):
    train_epochs = range(1, len(train_losses) + 1)
    val_epochs = [ep for ep, _ in val_loss_points]
    val_losses = [vl for _, vl in val_loss_points]

    plt.figure(figsize=(8, 6))
    plt.plot(train_epochs, train_losses, label="Train Loss", marker='o')
    plt.plot(val_epochs, val_losses, label="Val Loss", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved loss curve to {save_path}")
