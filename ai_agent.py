import os
import time
import random
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
COMMAND_INTERVAL = 0.25
KEY_HOLD_TIME = 0.08
RANDOM_SEED = None

FRAME_SIZE = (96, 96)
STACK_SIZE = 4
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pth")

MOVE_LABELS = ["none", "up", "down", "left", "right"]
ACTION_LABELS = ["none", "c", "v", "space"]

MOVE_CONF_THRESHOLD = 0.2
ACTION_CONF_THRESHOLD = 0.8

DEBUG = True
DEBUG_INTERVAL_SEC = 1.0

KEY_MAP = {
    "up": Key.up,
    "down": Key.down,
    "left": Key.left,
    "right": Key.right,
    "c": "c",
    "v": "v",
    "space": Key.space,
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


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class LightPolicyNet(nn.Module):
    def __init__(self, in_channels, move_classes, action_classes):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.features = nn.Sequential(
            DepthwiseSeparableConv(16, 32, stride=2),
            DepthwiseSeparableConv(32, 64, stride=2),
            DepthwiseSeparableConv(64, 96, stride=2),
            DepthwiseSeparableConv(96, 128, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.move_head = nn.Linear(128, move_classes)
        self.action_head = nn.Linear(128, action_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x).flatten(1)
        move = self.move_head(x)
        action = self.action_head(x)
        return move, action


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


def decide_action(stack, rng, model=None, device="cpu"):
    if model is None:
        move_label = rng.choice(MOVE_LABELS)
        action_label = rng.choice(ACTION_LABELS + ["none", "none", "none"])
        return move_label, action_label, 1.0, 1.0

    with torch.no_grad():
        inp = torch.from_numpy(stack).unsqueeze(0).to(device)
        move_logits, action_logits = model(inp)
        move_probs = torch.softmax(move_logits, dim=1).squeeze(0)
        action_probs = torch.softmax(action_logits, dim=1).squeeze(0)
        move_idx = int(move_probs.argmax().item())
        action_idx = int(action_probs.argmax().item())
        move_conf = float(move_probs[move_idx].item())
        action_conf = float(action_probs[action_idx].item())

    move_label = MOVE_LABELS[move_idx] if move_conf >= MOVE_CONF_THRESHOLD else "none"
    action_label = ACTION_LABELS[action_idx] if action_conf >= ACTION_CONF_THRESHOLD else "none"
    return move_label, action_label, move_conf, action_conf


def tap_key(controller, key, hold_time):
    controller.press(key)
    time.sleep(hold_time)
    controller.release(key)


def main():
    rng = random.Random(RANDOM_SEED)
    keyboard = Controller()
    last_cmd_time = 0.0
    frame_interval = 1.0 / max(1, CAPTURE_FPS)
    model = None
    device = "cpu"
    current_move = None
    last_debug = 0.0
    frame_history = deque(maxlen=STACK_SIZE)

    if os.path.exists(MODEL_PATH):
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        model = LightPolicyNet(STACK_SIZE, len(MOVE_LABELS), len(ACTION_LABELS)).to(device)
        state = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state)
        model.eval()
        print(f"Loaded model from {MODEL_PATH}")
    else:
        print("Model not found, using random actions.")

    print("AI agent starting.")
    print("Click the game window to focus it, then keep it in front.")

    last_window_label = None
    with mss.mss() as sct:
        while True:
            start = time.time()
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
            frame_history.append(preprocess(frame))

            now = time.time()
            if now - last_cmd_time >= COMMAND_INTERVAL:
                stack = build_stack(frame_history, STACK_SIZE)
                if stack is None:
                    continue
                move_label, action_label, move_conf, action_conf = decide_action(stack, rng, model, device)

                if DEBUG and now - last_debug >= DEBUG_INTERVAL_SEC:
                    print(f"move={move_label}({move_conf:.2f}) action={action_label}({action_conf:.2f})")
                    last_debug = now

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
                        tap_key(keyboard, KEY_MAP[action_label], KEY_HOLD_TIME)
                elif action_label != "none":
                    tap_key(keyboard, KEY_MAP[action_label], KEY_HOLD_TIME)

                last_cmd_time = now

            elapsed = time.time() - start
            sleep_for = frame_interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)


if __name__ == "__main__":
    main()
