import os
import time
from collections import deque

import mss
import numpy as np
from PIL import Image
from pynput.keyboard import Controller, Key
import Quartz

import torch
from torch import nn


WINDOW_TITLE = os.getenv("PAC_WINDOW_TITLE", "VideoLeo Pac-Maze")
WINDOW_OWNER = os.getenv("PAC_WINDOW_OWNER", "Python")

CAPTURE_FPS = 8
STEP_INTERVAL = 0.15
KEY_HOLD_TIME = 0.08

FRAME_SIZE = (96, 96)
STACK_SIZE = 64

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "rl_checkpoint.pt")

MOVE_LABELS = ["none", "up", "down", "left", "right"]
ACTION_LABELS = ["none", "c", "v", "space"]


KEY_MAP = {
    "up": Key.up,
    "down": Key.down,
    "left": Key.left,
    "right": Key.right,
    "c": "c",
    "v": "v",
    "space": Key.space,
    "n": "n",
}


def find_window_bounds(title_substring, owner_substring=None):
    options = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
    window_list = Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID)
    matches = []
    for win in window_list:
        name = win.get("kCGWindowName", "") or ""
        owner = win.get("kCGWindowOwnerName", "") or ""
        name_match = title_substring.lower() in name.lower() if title_substring else False
        owner_match = owner_substring and owner_substring.lower() in owner.lower()
        if name_match or owner_match:
            bounds = win.get("kCGWindowBounds", {})
            try:
                x = int(bounds.get("X", 0))
                y = int(bounds.get("Y", 0))
                w = int(bounds.get("Width", 0))
                h = int(bounds.get("Height", 0))
                area = w * h
                if w > 0 and h > 0:
                    label = f"{owner} - {name}".strip(" -")
                    matches.append((area, (x, y, w, h), label))
            except (TypeError, ValueError):
                continue
    if not matches:
        return None, None
    matches.sort(reverse=True, key=lambda item: item[0])
    return matches[0][1], matches[0][2]


def preprocess(frame_bgra):
    frame = np.array(frame_bgra)[:, :, :3]
    img = Image.fromarray(frame)
    img = img.convert("L")
    img = img.resize(FRAME_SIZE, Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


def build_stack(frames, stack_size):
    if not frames:
        return None
    if len(frames) < stack_size:
        pad = [frames[0]] * (stack_size - len(frames))
        stack = pad + list(frames)
    else:
        stack = list(frames)[-stack_size:]
    return np.stack(stack, axis=0)


def action_space():
    actions = []
    for move in MOVE_LABELS:
        for act in ACTION_LABELS:
            actions.append((move, act))
    return actions


class DQN(nn.Module):
    def __init__(self, in_channels, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (FRAME_SIZE[0] // 8) * (FRAME_SIZE[1] // 8), 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        return self.net(x)


def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    actions = action_space()
    num_actions = len(actions)
    device = select_device()

    if not os.path.exists(CHECKPOINT_PATH):
        raise SystemExit("rl_checkpoint.pt not found. Run rl_train.py first.")

    policy = DQN(STACK_SIZE, num_actions).to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    policy.load_state_dict(ckpt.get("policy", policy.state_dict()))
    policy.eval()

    frame_history = deque(maxlen=STACK_SIZE)
    keyboard = Controller()
    current_move = None

    print("RL agent running. Focus the game window.")

    last_window_label = None
    with mss.mss() as sct:
        while True:
            bounds, label = find_window_bounds(WINDOW_TITLE, WINDOW_OWNER)
            if bounds is None:
                print("Window not found. Waiting...")
                time.sleep(1.0)
                continue
            if label and label != last_window_label:
                print(f"Capturing window: {label}")
                last_window_label = label

            x, y, w, h = bounds
            frame = sct.grab({"left": x, "top": y, "width": w, "height": h})
            gray = preprocess(frame)
            frame_history.append(gray)
            state = build_stack(frame_history, STACK_SIZE)
            if state is None:
                time.sleep(1.0 / CAPTURE_FPS)
                continue

            with torch.no_grad():
                inp = torch.from_numpy(state).unsqueeze(0).to(device)
                q = policy(inp)
                action_idx = int(q.argmax(dim=1).item())

            move_label, action_label = actions[action_idx]

            if move_label != current_move:
                if current_move and current_move in KEY_MAP:
                    keyboard.release(KEY_MAP[current_move])
                if move_label != "none":
                    keyboard.press(KEY_MAP[move_label])
                    current_move = move_label
                else:
                    current_move = None

            if action_label in ("c", "space"):
                if move_label != "none" or current_move is not None:
                    keyboard.press(KEY_MAP[action_label])
                    time.sleep(KEY_HOLD_TIME)
                    keyboard.release(KEY_MAP[action_label])
            elif action_label != "none":
                keyboard.press(KEY_MAP[action_label])
                time.sleep(KEY_HOLD_TIME)
                keyboard.release(KEY_MAP[action_label])

            time.sleep(STEP_INTERVAL)


if __name__ == "__main__":
    main()
