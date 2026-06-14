import os
import glob
import json
import torch
import numpy as np
from torch.utils.data import Dataset
import random

class CozmoDataset(Dataset):
    def __init__(self, data_dir='training_data/', split='train', val_ratio=0.2, norm_stats_path=None):
        self.data_dir = data_dir
        self.split = split
        self.val_ratio = val_ratio
        
        self.frames = np.empty((0, 60, 80), dtype=np.uint8)
        self.sensors = np.empty((0, 12), dtype=np.float32)
        self.actions = np.empty((0, 3), dtype=np.float32)
        
        self.load_data()
        
        if self.split == 'train':
            self.compute_and_save_norm_stats(norm_stats_path)
        else:
            self.load_norm_stats(norm_stats_path)

    def load_data(self):
        npz_files = sorted(glob.glob(os.path.join(self.data_dir, "*.npz")))
        if not npz_files:
            print(f"[{self.split}] No .npz files found in {self.data_dir}")
            return
            
        # Split par session
        num_sessions = len(npz_files)
        num_val = max(1, int(num_sessions * self.val_ratio)) if num_sessions > 1 else 0
        
        if self.split == 'train':
            selected_files = npz_files[:-num_val] if num_val > 0 else npz_files
        elif self.split == 'val':
            selected_files = npz_files[-num_val:] if num_val > 0 else []
        else:
            selected_files = npz_files
            
        print(f"[{self.split}] Loading {len(selected_files)} sessions...")
        
        frames_list = []
        sensors_list = []
        actions_list = []
        
        for f in selected_files:
            data = np.load(f)
            frames_list.append(data['frames'])
            sensors_list.append(data['sensors'])
            actions_list.append(data['actions'])
            
        if frames_list:
            self.frames = np.concatenate(frames_list, axis=0)
            self.sensors = np.concatenate(sensors_list, axis=0)
            self.actions = np.concatenate(actions_list, axis=0)
            
        duration = len(self.frames) / 20.0 # Assuming 20Hz
        print(f"[{self.split}] Loaded {len(self.frames)} samples (approx {duration/60:.1f} minutes).")

    def compute_and_save_norm_stats(self, save_path):
        if len(self.sensors) == 0:
            return
        
        self.sensors_mean = np.mean(self.sensors, axis=0, keepdims=True)
        self.sensors_std = np.std(self.sensors, axis=0, keepdims=True)
        self.sensors_std[self.sensors_std < 1e-6] = 1e-6 # Eviter division par zéro
        
        if save_path:
            stats = {
                'mean': self.sensors_mean.tolist(),
                'std': self.sensors_std.tolist()
            }
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(stats, f)
            print(f"Saved normalization stats to {save_path}")

    def load_norm_stats(self, load_path):
        if load_path and os.path.exists(load_path):
            with open(load_path, 'r') as f:
                stats = json.load(f)
            self.sensors_mean = np.array(stats['mean'], dtype=np.float32)
            self.sensors_std = np.array(stats['std'], dtype=np.float32)
        else:
            print(f"Warning: Normalization stats not found at {load_path}. Computing from current data.")
            self.compute_and_save_norm_stats(None)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        # Images : Normalisation [0, 1] + Channel dimension
        img = self.frames[idx].astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        
        sensor = self.sensors[idx].astype(np.float32)
        action = self.actions[idx].astype(np.float32)
        
        # Capteurs : Standardisation
        sensor = (sensor - self.sensors_mean[0]) / self.sensors_std[0]
        
        # Data Augmentation (50% de probabilité, uniquement sur train)
        if self.split == 'train':
            # Flip horizontal
            if random.random() > 0.5:
                # Miroir de l'image
                img = np.flip(img, axis=2).copy()
                
                # Inversion roues (actions)
                action[0], action[1] = action[1], action[0]
                
                # Inversion capteurs liés
                sensor[4], sensor[5] = sensor[5], sensor[4] # left_wheel_speed <-> right_wheel_speed
                sensor[2] = -sensor[2]                      # heading_deg (inversé)
                
            # Bruit Gaussien
            if random.random() > 0.5:
                noise = np.random.normal(0, 0.02, img.shape).astype(np.float32)
                img = np.clip(img + noise, 0.0, 1.0)
                
            # Jitter luminosité
            if random.random() > 0.5:
                factor = random.uniform(0.8, 1.2)
                img = np.clip(img * factor, 0.0, 1.0)
                
        return torch.from_numpy(img), torch.from_numpy(sensor), torch.from_numpy(action)
