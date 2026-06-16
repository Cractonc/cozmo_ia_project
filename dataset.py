import os
import glob
import json
import torch
import numpy as np
from torch.utils.data import Dataset
import random

from model import ACTION_CLASS_SPECS, SAFE_SENSOR_INDICES


ACTION_CLASS_ARRAY = np.array([spec[1] for spec in ACTION_CLASS_SPECS], dtype=np.float32)
ACTION_FLIP_MAP = {
    0: 0,  # Avant
    1: 2,  # Gauche -> Droite
    2: 1,  # Droite -> Gauche
    3: 3,  # Arrière/Stop
}

class CozmoDataset(Dataset):
    def __init__(self, data_dir='training_data/', split='train', val_ratio=0.2, norm_stats_path=None, mode='continuous'):
        self.data_dir = data_dir
        self.split = split
        self.val_ratio = val_ratio
        self.mode = mode
        
        if self.mode == 'discrete':
            self.action_templates = np.array([
                [80.0, 80.0],   # 0: avant
                [-60.0, -60.0], # 1: arrière
                [30.0, 80.0],   # 2: gauche
                [80.0, 30.0],   # 3: droite
                [-50.0, 50.0],  # 4: pivot_gauche
                [50.0, -50.0],  # 5: pivot_droite
                [0.0, 0.0]      # 6: stop
            ], dtype=np.float32)

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
                # Miroir de l'image (analyse dynamique de la shape HWC vs CHW)
                if len(img.shape) == 3 and img.shape[2] in (1, 3) and img.shape[0] not in (1, 3):
                    img = np.flip(img, axis=1).copy() # HWC
                else:
                    img = np.flip(img, axis=2).copy() # CHW
                
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
                
        if self.mode == 'discrete':
            diff = action[:2] - self.action_templates
            dist = np.sum(diff * diff, axis=1)
            discrete_label = int(np.argmin(dist))
            action_out = torch.tensor(discrete_label, dtype=torch.long)
        else:
            action_out = torch.from_numpy(action)
            
        return torch.from_numpy(img), torch.from_numpy(sensor), action_out


class CozmoDiscreteDataset(Dataset):
    def __init__(self, data_dir='training_data/', split='train', val_ratio=0.2, norm_stats_path=None):
        self.data_dir = data_dir
        self.split = split
        self.val_ratio = val_ratio

        self.frames = np.empty((0, 60, 80), dtype=np.uint8)
        self.sensors = np.empty((0, len(SAFE_SENSOR_INDICES)), dtype=np.float32)
        self.action_classes = np.empty((0,), dtype=np.int64)
        self.speed_targets = np.empty((0,), dtype=np.float32)
        self.head_targets = np.empty((0,), dtype=np.float32)

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

        num_sessions = len(npz_files)
        num_val = max(1, int(num_sessions * self.val_ratio)) if num_sessions > 1 else 0

        if self.split == 'train':
            selected_files = npz_files[:-num_val] if num_val > 0 else npz_files
        elif self.split == 'val':
            selected_files = npz_files[-num_val:] if num_val > 0 else []
        else:
            selected_files = npz_files

        print(f"[{self.split}] Loading {len(selected_files)} sessions for discrete policy...")

        frames_list = []
        sensors_list = []
        class_list = []
        speed_list = []
        head_list = []

        for f in selected_files:
            data = np.load(f)
            frames = data['frames']
            sensors = data['sensors'][:, SAFE_SENSOR_INDICES].astype(np.float32)
            actions = data['actions'].astype(np.float32)

            classes = self.actions_to_classes(actions[:, :2])
            speeds = self.actions_to_speed_scale(actions[:, :2], classes)

            frames_list.append(frames)
            sensors_list.append(sensors)
            class_list.append(classes)
            speed_list.append(speeds)
            head_list.append(actions[:, 2])

        if frames_list:
            self.frames = np.concatenate(frames_list, axis=0)
            self.sensors = np.concatenate(sensors_list, axis=0)
            
            raw_action_classes = np.concatenate(class_list, axis=0)
            # Remap 7 classes -> 4 classes
            remap = np.array([0, 3, 1, 2, 1, 2, 3], dtype=np.int64)
            self.action_classes = remap[raw_action_classes]
            
            self.speed_targets = np.concatenate(speed_list, axis=0)
            self.head_targets = np.concatenate(head_list, axis=0).astype(np.float32)

        duration = len(self.frames) / 20.0
        print(f"[{self.split}] Loaded {len(self.frames)} samples (approx {duration/60:.1f} minutes).")

    @staticmethod
    def actions_to_classes(wheels):
        diff = wheels[:, None, :] - ACTION_CLASS_ARRAY[None, :, :]
        dist = np.sum(diff * diff, axis=2)
        return np.argmin(dist, axis=1).astype(np.int64)

    @staticmethod
    def actions_to_speed_scale(wheels, classes):
        base = ACTION_CLASS_ARRAY[classes]
        base_mag = np.max(np.abs(base), axis=1)
        action_mag = np.max(np.abs(wheels), axis=1)
        speed = np.zeros_like(action_mag, dtype=np.float32)
        moving = base_mag > 1e-6
        speed[moving] = action_mag[moving] / base_mag[moving]
        return np.clip(speed, 0.0, 1.2).astype(np.float32)

    def compute_and_save_norm_stats(self, save_path):
        if len(self.sensors) == 0:
            return

        self.sensors_mean = np.mean(self.sensors, axis=0, keepdims=True)
        self.sensors_std = np.std(self.sensors, axis=0, keepdims=True)
        self.sensors_std[self.sensors_std < 1e-6] = 1e-6

        if save_path:
            stats = {
                'mean': self.sensors_mean.tolist(),
                'std': self.sensors_std.tolist(),
                'safe_sensor_indices': list(SAFE_SENSOR_INDICES),
                'action_classes': [
                    {"index": i, "name": name, "wheels": list(wheels)}
                    for i, (name, wheels) in enumerate(ACTION_CLASS_SPECS)
                ],
            }
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(stats, f)
            print(f"Saved discrete normalization stats to {save_path}")

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
        img = self.frames[idx].astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        sensor = self.sensors[idx].astype(np.float32)
        sensor = (sensor - self.sensors_mean[0]) / self.sensors_std[0]

        action_class = int(self.action_classes[idx])
        speed_target = np.float32(self.speed_targets[idx])
        head_target = np.float32(self.head_targets[idx])

        if self.split == 'train':
            if random.random() > 0.5:
                if len(img.shape) == 3 and img.shape[2] in (1, 3) and img.shape[0] not in (1, 3):
                    img = np.flip(img, axis=1).copy()
                else:
                    img = np.flip(img, axis=2).copy()
                action_class = ACTION_FLIP_MAP[action_class]
                # SAFE_SENSOR_INDICES[0] is heading_deg.
                sensor[0] = -sensor[0]

            if random.random() > 0.5:
                noise = np.random.normal(0, 0.02, img.shape).astype(np.float32)
                img = np.clip(img + noise, 0.0, 1.0)

            if random.random() > 0.5:
                factor = random.uniform(0.8, 1.2)
                img = np.clip(img * factor, 0.0, 1.0)

        return (
            torch.from_numpy(img),
            torch.from_numpy(sensor),
            torch.tensor(action_class, dtype=torch.long),
            torch.tensor(speed_target, dtype=torch.float32),
            torch.tensor(head_target, dtype=torch.float32),
        )
