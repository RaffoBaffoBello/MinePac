import os
import numpy as np
import torch
import torch.nn as nn


class AZNet(nn.Module):
    def __init__(self, input_shape, action_size):
        super().__init__()
        channels, height, width = input_shape
        self.trunk = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * height * width, action_size),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(1 * height * width, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.trunk(x)
        policy = self.policy_head(x)
        value = self.value_head(x).squeeze(1)
        return policy, value


def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def obs_to_tensor(planes, device):
    arr = np.asarray(planes, dtype=np.float32)
    if arr.ndim == 3:
        arr = np.expand_dims(arr, axis=0)
    tensor = torch.from_numpy(arr).to(device)
    return tensor


def load_model(model_path, input_shape, action_size, device):
    model = AZNet(input_shape, action_size).to(device)
    if model_path and os.path.exists(model_path):
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
    return model
