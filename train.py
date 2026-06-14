import os
import argparse
import time
import json
from datetime import datetime

# Verification des prérequis
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

from model import CozmoNN
from dataset import CozmoDataset

def main():
    parser = argparse.ArgumentParser(description="Train Cozmo Imitation Learning Model (CPU Only)")
    parser.add_argument("--data_dir", type=str, default="training_data/", help="Path to training data directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--output_dir", type=str, default="models/", help="Output directory")
    parser.add_argument("--name", type=str, default=None, help="Run name (default: auto-generated timestamp)")
    
    args = parser.parse_args()
    
    if args.name is None:
        args.name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
    os.makedirs(args.output_dir, exist_ok=True)
    norm_stats_path = os.path.join(args.output_dir, f"norm_stats_{args.name}.json")
    model_save_path = os.path.join(args.output_dir, f"cozmo_nn_{args.name}.pt")
    history_save_path = os.path.join(args.output_dir, f"history_{args.name}.json")
    
    print(f"=== Starting Training Run: {args.name} ===")
    
    # Datasets & Loaders
    train_dataset = CozmoDataset(data_dir=args.data_dir, split='train', norm_stats_path=norm_stats_path)
    val_dataset = CozmoDataset(data_dir=args.data_dir, split='val', norm_stats_path=norm_stats_path)
    
    if len(train_dataset) == 0:
        print("Error: No training data loaded.")
        return
        
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Model Setup
    model = CozmoNN()
    print(f"Model instantiated with {model.count_parameters():,} parameters.")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # factor=0.5 au lieu de patience=5, ou patience=5
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)
    
    # Loss ponderée: roues comptent plus que la tête
    loss_weights = torch.tensor([1.0, 1.0, 0.3])
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience_early_stopping = 10
    
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    
    print("Starting training loop (CPU)...")
    
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        
        # --- TRAINING PHASE ---
        model.train()
        train_loss_sum = 0.0
        for imgs, sensors, actions in train_loader:
            optimizer.zero_grad()
            preds = model(imgs, sensors)
            
            # Weighted MSE Loss
            loss = (F.mse_loss(preds, actions, reduction='none') * loss_weights).mean()
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item() * imgs.size(0)
            
        train_loss = train_loss_sum / len(train_dataset)
        
        # --- VALIDATION PHASE ---
        model.eval()
        val_loss_sum = 0.0
        if len(val_dataset) > 0:
            with torch.no_grad():
                for imgs, sensors, actions in val_loader:
                    preds = model(imgs, sensors)
                    loss = (F.mse_loss(preds, actions, reduction='none') * loss_weights).mean()
                    val_loss_sum += loss.item() * imgs.size(0)
            val_loss = val_loss_sum / len(val_dataset)
        else:
            val_loss = train_loss # fallback if no val set
            
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)
        
        scheduler.step(val_loss)
        
        # Estimations temps
        epoch_time = time.time() - epoch_start_time
        remaining_time = epoch_time * (args.epochs - epoch)
        
        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"LR: {current_lr:.6f} | "
              f"Time: {epoch_time:.1f}s (ETA: {remaining_time/60:.1f}m)")
              
        # Early Stopping & Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> Best model saved to {model_save_path} (Val loss: {best_val_loss:.4f})")
            
            # Save metadata
            metadata_path = os.path.join(args.output_dir, f"cozmo_nn_{args.name}.json")
            param_count = model.count_parameters()
            param_str = f"{param_count/1e6:.1f}M params" if param_count >= 1e6 else f"{param_count/1e3:.1f}k params" if param_count >= 1e3 else f"{param_count} params"
            
            metadata = {
                "displayName": "CozmoPilot",
                "version": "1.0",
                "parameters": param_str
            }
            with open(metadata_path, 'w') as mf:
                json.dump(metadata, mf)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience_early_stopping:
                print(f"Early stopping at epoch {epoch}. No improvement for {patience_early_stopping} epochs.")
                break
                
    # Sauvegarde de l'historique final
    with open(history_save_path, 'w') as f:
        json.dump(history, f)
        
    print("Training complete!")

if __name__ == '__main__':
    main()
