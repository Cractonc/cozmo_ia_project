import os
import argparse
import time
import json
from datetime import datetime

try:
    import torch
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader
except ImportError:
    print("Error: PyTorch is required. Please install it with 'pip install torch'")
    exit(1)

try:
    import numpy as np
except ImportError:
    print("Error: numpy is required. Please install it with 'pip install numpy'")
    exit(1)

from model import CozmoNN, CozmoNNDiscrete
from dataset import CozmoDataset, CozmoDiscreteDataset


def build_output_paths(args):
    if args.mode == "discrete":
        return {
            "stats": os.path.join(args.output_dir, "norm_stats_discrete_{}.json".format(args.name)),
            "model": os.path.join(args.output_dir, "cozmo_discrete_nn_{}.pt".format(args.name)),
            "history": os.path.join(args.output_dir, "history_discrete_{}.json".format(args.name)),
            "metadata": os.path.join(args.output_dir, "cozmo_discrete_nn_{}.json".format(args.name)),
        }

    return {
        "stats": os.path.join(args.output_dir, "norm_stats_{}.json".format(args.name)),
        "model": os.path.join(args.output_dir, "cozmo_nn_{}.pt".format(args.name)),
        "history": os.path.join(args.output_dir, "history_{}.json".format(args.name)),
        "metadata": os.path.join(args.output_dir, "cozmo_nn_{}.json".format(args.name)),
    }


def train_continuous_epoch(model, loader, optimizer, loss_weights):
    model.train()
    loss_sum = 0.0
    for imgs, sensors, actions in loader:
        optimizer.zero_grad()
        preds = model(imgs, sensors)
        loss = (F.mse_loss(preds, actions, reduction='none') * loss_weights).mean()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * imgs.size(0)
    return loss_sum / max(1, len(loader.dataset)), None


def eval_continuous(model, loader, loss_weights):
    model.eval()
    loss_sum = 0.0
    if len(loader.dataset) == 0:
        return 0.0, None
    with torch.no_grad():
        for imgs, sensors, actions in loader:
            preds = model(imgs, sensors)
            loss = (F.mse_loss(preds, actions, reduction='none') * loss_weights).mean()
            loss_sum += loss.item() * imgs.size(0)
    return loss_sum / len(loader.dataset), None


def train_discrete_epoch(model, loader, optimizer, class_weights):
    model.train()
    loss_sum = 0.0
    correct = 0
    for imgs, sensors, classes, speeds, heads in loader:
        optimizer.zero_grad()
        preds = model(imgs, sensors)
        loss = F.cross_entropy(preds, classes, weight=class_weights)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * imgs.size(0)
        correct += (torch.argmax(preds, dim=1) == classes).sum().item()
    n = max(1, len(loader.dataset))
    return loss_sum / n, correct / n


def eval_discrete(model, loader, class_weights):
    model.eval()
    loss_sum = 0.0
    correct = 0
    if len(loader.dataset) == 0:
        return 0.0, 0.0
    with torch.no_grad():
        for imgs, sensors, classes, speeds, heads in loader:
            preds = model(imgs, sensors)
            loss = F.cross_entropy(preds, classes, weight=class_weights)
            loss_sum += loss.item() * imgs.size(0)
            correct += (torch.argmax(preds, dim=1) == classes).sum().item()
    return loss_sum / len(loader.dataset), correct / len(loader.dataset)


def save_metadata(path, model, args):
    param_count = model.count_parameters()
    if param_count >= 1e6:
        param_str = "{:.1f}M params".format(param_count / 1e6)
    elif param_count >= 1e3:
        param_str = "{:.1f}k params".format(param_count / 1e3)
    else:
        param_str = "{} params".format(param_count)

    metadata = {
        "displayName": "CozmoPilotDiscrete" if args.mode == "discrete" else "CozmoPilot",
        "version": "2.0-experimental" if args.mode == "discrete" else "1.0",
        "architecture": args.mode,
        "parameters": param_str,
    }
    with open(path, 'w') as mf:
        json.dump(metadata, mf)


def main():
    parser = argparse.ArgumentParser(description="Train Cozmo imitation models (CPU only)")
    parser.add_argument("--data_dir", type=str, default="training_data/", help="Path to training data directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--output_dir", type=str, default="models/", help="Output directory")
    parser.add_argument("--name", type=str, default=None, help="Run name (default: auto-generated timestamp)")
    parser.add_argument("--mode", choices=["continuous", "discrete"], default="continuous", help="Model family to train")

    args = parser.parse_args()

    if args.name is None:
        args.name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    os.makedirs(args.output_dir, exist_ok=True)
    paths = build_output_paths(args)

    print("=== Starting Training Run: {} ({}) ===".format(args.name, args.mode))

    if args.mode == "discrete":
        train_dataset = CozmoDiscreteDataset(data_dir=args.data_dir, split='train', norm_stats_path=paths["stats"])
        val_dataset = CozmoDiscreteDataset(data_dir=args.data_dir, split='val', norm_stats_path=paths["stats"])
        model = CozmoNNDiscrete()
    else:
        train_dataset = CozmoDataset(data_dir=args.data_dir, split='train', norm_stats_path=paths["stats"])
        val_dataset = CozmoDataset(data_dir=args.data_dir, split='val', norm_stats_path=paths["stats"])
        model = CozmoNN()

    if len(train_dataset) == 0:
        print("Error: No training data loaded.")
        return

    if args.mode == "discrete":
        counts = np.bincount(train_dataset.action_classes, minlength=4).astype(np.float64)
        counts = np.maximum(counts, 1)
        total_samples = len(train_dataset)
        raw_weights = total_samples / (4.0 * counts)
        
        # Clipping [0.5, 3.0]
        clipped_weights = np.clip(raw_weights, 0.5, 3.0)
        
        # Forcer mathématiquement la symétrie spatiale (classes 1 et 2)
        sym_weight = (clipped_weights[1] + clipped_weights[2]) / 2.0
        clipped_weights[1] = sym_weight
        clipped_weights[2] = sym_weight
        
        device = next(model.parameters()).device
        class_weights = torch.tensor(clipped_weights, dtype=torch.float32, device=device)
        print("Calculated class weights (4 classes):", clipped_weights)
    else:
        class_weights = None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print("Model instantiated with {:,} parameters.".format(model.count_parameters()))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, verbose=True
    )
    continuous_weights = torch.tensor([1.0, 1.0, 0.3])

    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience_early_stopping = 10

    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': [],
        'train_accuracy': [],
        'val_accuracy': [],
    }

    print("Starting training loop (CPU)...")

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()

        if args.mode == "discrete":
            train_loss, train_acc = train_discrete_epoch(model, train_loader, optimizer, class_weights)
            if len(val_dataset) > 0:
                val_loss, val_acc = eval_discrete(model, val_loader, class_weights)
            else:
                val_loss, val_acc = train_loss, train_acc
        else:
            train_loss, train_acc = train_continuous_epoch(model, train_loader, optimizer, continuous_weights)
            if len(val_dataset) > 0:
                val_loss, val_acc = eval_continuous(model, val_loader, continuous_weights)
            else:
                val_loss, val_acc = train_loss, train_acc

        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)
        history['train_accuracy'].append(train_acc)
        history['val_accuracy'].append(val_acc)

        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start_time
        remaining_time = epoch_time * (args.epochs - epoch)
        acc_text = ""
        if train_acc is not None:
            acc_text = " | Train Acc: {:.1f}% | Val Acc: {:.1f}%".format(train_acc * 100.0, val_acc * 100.0)

        print(
            "Epoch {:02d}/{} | Train Loss: {:.4f} | Val Loss: {:.4f}{} | LR: {:.6f} | Time: {:.1f}s (ETA: {:.1f}m)".format(
                epoch,
                args.epochs,
                train_loss,
                val_loss,
                acc_text,
                current_lr,
                epoch_time,
                remaining_time / 60.0,
            )
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), paths["model"])
            save_metadata(paths["metadata"], model, args)
            print("  -> Best model saved to {} (Val loss: {:.4f})".format(paths["model"], best_val_loss))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience_early_stopping:
                print("Early stopping at epoch {}. No improvement for {} epochs.".format(
                    epoch, patience_early_stopping
                ))
                break

    with open(paths["history"], 'w') as f:
        json.dump(history, f)

    print("Training complete!")


if __name__ == '__main__':
    main()
