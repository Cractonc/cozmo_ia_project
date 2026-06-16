import torch
import torch.nn as nn


ACTION_CLASS_SPECS = [
    ("forward", (80.0, 80.0)),
    ("stop", (0.0, 0.0)),
    ("curve_left", (30.0, 80.0)),
    ("curve_right", (80.0, 30.0)),
    ("rotate_left", (-50.0, 50.0)),
    ("rotate_right", (50.0, -50.0)),
    ("backward", (-60.0, -60.0)),
]

# Remove absolute x/y odometry and wheel-speed feedback from the discrete policy.
SAFE_SENSOR_INDICES = [2, 3, 6, 7, 8, 9, 10, 11]


def action_class_table(device=None, dtype=None):
    return torch.tensor(
        [spec[1] for spec in ACTION_CLASS_SPECS],
        device=device,
        dtype=dtype or torch.float32,
    )


def decode_discrete_outputs(logits, speed_scale, head_angle):
    class_idx = torch.argmax(logits, dim=1)
    table = action_class_table(device=logits.device, dtype=logits.dtype)
    wheels = table[class_idx] * speed_scale.clamp(0.0, 1.2)
    return torch.cat((wheels, head_angle), dim=1)


class CozmoNN(nn.Module):
    def __init__(self):
        super(CozmoNN, self).__init__()
        
        # Branche Vision (CNN léger)
        # Input : (batch, 1, 60, 80)
        self.vision = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),  # -> (batch, 16, 30, 40)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # -> (batch, 32, 15, 20)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> (batch, 64, 8, 10)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d((4, 4)),                          # -> (batch, 64, 4, 4)
            nn.Flatten(),                                          # -> (batch, 1024)
            
            nn.Linear(1024, 128),                                  # -> (batch, 128)
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Branche Capteurs (MLP)
        # Input : (batch, 12)
        self.sensors = nn.Sequential(
            nn.Linear(12, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        # Fusion + Décision
        # Concat(vision_128, sensors_32) -> 160
        self.fusion = nn.Sequential(
            nn.Linear(160, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Tête de sortie pour les actions continues
        self.output = nn.Linear(32, 3)

    def forward(self, image, sensors):
        v = self.vision(image)
        s = self.sensors(sensors)
        
        fused = torch.cat((v, s), dim=1)
        x = self.fusion(fused)
        
        out = self.output(x)
        
        # Activations Tanh avec mise à l'échelle (scaling)
        # Sortie : (left_wheel_speed, right_wheel_speed, head_angle)
        left_wheel = torch.tanh(out[:, 0:1]) * 150.0       # [-150, 150]
        right_wheel = torch.tanh(out[:, 1:2]) * 150.0      # [-150, 150]
        head_angle = torch.tanh(out[:, 2:3]) * 35.0 + 10.0 # [-25, 45]
        
        return torch.cat((left_wheel, right_wheel, head_angle), dim=1)
        
    def count_parameters(self):
        """Retourne le nombre total de paramètres du modèle."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CozmoDiscreteNN(nn.Module):
    """
    Experimental policy for discrete keyboard-style actions.

    Outputs:
      - action logits over ACTION_CLASS_SPECS
      - speed_scale for the selected action
      - head_angle in degrees
    """

    def __init__(self, sensor_dim=len(SAFE_SENSOR_INDICES), num_actions=len(ACTION_CLASS_SPECS)):
        super(CozmoDiscreteNN, self).__init__()

        self.vision = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

        self.sensors = nn.Sequential(
            nn.Linear(sensor_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(160, 96),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(96, 48),
            nn.ReLU(),
        )

        self.action_logits = nn.Linear(48, num_actions)
        self.speed_head = nn.Linear(48, 1)
        self.head_angle = nn.Linear(48, 1)

    def forward(self, image, sensors):
        v = self.vision(image)
        s = self.sensors(sensors)
        fused = torch.cat((v, s), dim=1)
        x = self.fusion(fused)

        logits = self.action_logits(x)
        speed_scale = torch.sigmoid(self.speed_head(x)) * 1.2
        head_angle = torch.tanh(self.head_angle(x)) * 35.0 + 10.0

        return logits, speed_scale, head_angle

    def predict_continuous_action(self, image, sensors):
        logits, speed_scale, head_angle = self.forward(image, sensors)
        return decode_discrete_outputs(logits, speed_scale, head_angle)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CozmoNNv2(nn.Module):
    """
    CozmoNNv2 — Variante avec 13 entrées capteurs.
    Identique à CozmoNN mais la branche capteurs accepte 13 inputs
    (12 capteurs standard + delta_heading normalisé [-1, 1]).
    """
    def __init__(self):
        super(CozmoNNv2, self).__init__()
        
        # Branche Vision (CNN léger)
        # Input : (batch, 1, 60, 80)
        self.vision = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),  # -> (batch, 16, 30, 40)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # -> (batch, 32, 15, 20)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> (batch, 64, 8, 10)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d((4, 4)),                          # -> (batch, 64, 4, 4)
            nn.Flatten(),                                          # -> (batch, 1024)
            
            nn.Linear(1024, 128),                                  # -> (batch, 128)
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Branche Capteurs (MLP)
        # Input : (batch, 13) — 12 capteurs + delta_heading
        self.sensors = nn.Sequential(
            nn.Linear(13, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        # Fusion + Décision
        # Concat(vision_128, sensors_32) -> 160
        self.fusion = nn.Sequential(
            nn.Linear(160, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Tête de sortie pour les actions continues
        self.output = nn.Linear(32, 3)

    def forward(self, image, sensors):
        v = self.vision(image)
        s = self.sensors(sensors)
        
        fused = torch.cat((v, s), dim=1)
        x = self.fusion(fused)
        
        out = self.output(x)
        
        # Activations Tanh avec mise à l'échelle (scaling)
        # Sortie : (left_wheel_speed, right_wheel_speed, head_angle)
        left_wheel = torch.tanh(out[:, 0:1]) * 150.0       # [-150, 150]
        right_wheel = torch.tanh(out[:, 1:2]) * 150.0      # [-150, 150]
        head_angle = torch.tanh(out[:, 2:3]) * 35.0 + 10.0 # [-25, 45]
        
        return torch.cat((left_wheel, right_wheel, head_angle), dim=1)
        
    def count_parameters(self):
        """Retourne le nombre total de paramètres du modèle."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CozmoNNDiscrete(nn.Module):
    """
    Tête de classification pour 7 actions discrètes.
    """
    def __init__(self, sensor_dim=len(SAFE_SENSOR_INDICES), num_actions=4):
        super(CozmoNNDiscrete, self).__init__()
        
        # Branche Vision (CNN léger)
        self.vision = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),  # -> (batch, 16, 30, 40)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # -> (batch, 32, 15, 20)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> (batch, 64, 8, 10)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d((4, 4)),                          # -> (batch, 64, 4, 4)
            nn.Flatten(),                                          # -> (batch, 1024)
            
            nn.Linear(1024, 128),                                  # -> (batch, 128)
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Branche Capteurs (MLP)
        self.sensors = nn.Sequential(
            nn.Linear(sensor_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        # Fusion + Décision
        self.fusion = nn.Sequential(
            nn.Linear(160, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Tête de sortie pour les actions discrètes (7 classes)
        self.output = nn.Linear(32, num_actions)

    def forward(self, image, sensors):
        v = self.vision(image)
        s = self.sensors(sensors)
        
        fused = torch.cat((v, s), dim=1)
        x = self.fusion(fused)
        
        out = self.output(x)
        return out
        
    def count_parameters(self):
        """Retourne le nombre total de paramètres du modèle."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
