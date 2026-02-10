import os
import time
import json
import random
from collections import deque

import mss
import numpy as np
from PIL import Image
from pynput.keyboard import Controller, Key
import Quartz

import torch
from torch import nn
import torch.nn.functional as F


WINDOW_TITLE = os.getenv("PAC_WINDOW_TITLE", "VideoLeo Pac-Maze")
WINDOW_OWNER = os.getenv("PAC_WINDOW_OWNER", "Python")

CAPTURE_FPS = 8
STEP_INTERVAL = 0.15
KEY_HOLD_TIME = 0.08
GAME_OVER_HOLD_SEC = 0.5
GAME_OVER_COOLDOWN_SEC = 2.0

FRAME_SIZE = (96, 96)
STACK_SIZE = 4

EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY_STEPS = 10000

GAMMA = 0.99
LR = 2e-4
BATCH_SIZE = 32
REPLAY_SIZE = 50000
WARMUP_STEPS = 1000
TRAIN_EVERY = 4
TARGET_UPDATE_EVERY = 1000

DEATH_PENALTY = -150.0

SCORE_PATH = os.path.join(os.path.dirname(__file__), "score.json")
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "rl_checkpoint.pt")
LOG_PATH = os.path.join(os.path.dirname(__file__), "rl_log.jsonl")
SAVE_EVERY_SEC = 300

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


def detect_game_over(gray_frame):
    h, w = gray_frame.shape
    x0, x1 = int(w * 0.2), int(w * 0.8)
    y0, y1 = int(h * 0.38), int(h * 0.62)
    region = gray_frame[y0:y1, x0:x1]
    if region.size == 0:
        return False
    bright_ratio = (region > 0.9).mean()
    mean_val = float(region.mean())
    return bright_ratio > 0.2 and mean_val > 0.35


def read_score_state(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return (
                int(data.get("score", 0)),
                int(data.get("lives", 0)),
                int(data.get("level", 0)),
            )
    except (OSError, json.JSONDecodeError, ValueError):
        return None


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
    policy = DQN(STACK_SIZE, num_actions).to(device)
    target = DQN(STACK_SIZE, num_actions).to(device)
    target.load_state_dict(policy.state_dict())
    target.eval()
    opt = torch.optim.Adam(policy.parameters(), lr=LR)

    replay = deque(maxlen=REPLAY_SIZE)
    frame_history = deque(maxlen=STACK_SIZE)

    keyboard = Controller()
    current_move = None

    steps = 0
    eps = EPS_START
    last_score = None
    last_lives = None
    last_save = time.time()
    reward_history = deque(maxlen=200)
    game_over_accum = 0.0
    last_game_over_check = time.time()
    last_game_over_press = 0.0

    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        policy.load_state_dict(ckpt.get("policy", policy.state_dict()))
        target.load_state_dict(ckpt.get("target", target.state_dict()))
        opt.load_state_dict(ckpt.get("optimizer", opt.state_dict()))
        steps = int(ckpt.get("steps", 0))
        eps = float(ckpt.get("eps", EPS_START))
        last_score = ckpt.get("last_score", None)
        last_lives = ckpt.get("last_lives", None)
        print(f"Resumed from checkpoint at step {steps}, eps={eps:.3f}")

    print("RL training started. Focus the game window.")

    last_window_label = None
    with mss.mss() as sct:
        try:
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

                now = time.time()
                delta = now - last_game_over_check
                last_game_over_check = now
                if detect_game_over(gray):
                    game_over_accum += delta
                else:
                    game_over_accum = 0.0
                if game_over_accum >= GAME_OVER_HOLD_SEC:
                    if now - last_game_over_press >= GAME_OVER_COOLDOWN_SEC:
                        if current_move and current_move in KEY_MAP:
                            keyboard.release(KEY_MAP[current_move])
                            current_move = None
                        keyboard.press(KEY_MAP["n"])
                        time.sleep(KEY_HOLD_TIME)
                        keyboard.release(KEY_MAP["n"])
                        last_game_over_press = now
                        last_score = None
                        last_lives = None
                        frame_history.clear()
                    time.sleep(1.0)
                    continue

                # Select action
                if random.random() < eps:
                    action_idx = random.randrange(num_actions)
                else:
                    with torch.no_grad():
                        inp = torch.from_numpy(state).unsqueeze(0).to(device)
                        q = policy(inp)
                        action_idx = int(q.argmax(dim=1).item())

                move_label, action_label = actions[action_idx]

                # Apply action
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

                # Next state + reward
                next_frame = sct.grab({"left": x, "top": y, "width": w, "height": h})
                next_gray = preprocess(next_frame)
                frame_history.append(next_gray)
                next_state = build_stack(frame_history, STACK_SIZE)

                reward = 0.0
                done = False
                state_info = read_score_state(SCORE_PATH)
                if state_info:
                    score, lives, _level = state_info
                    if last_score is None:
                        last_score = score
                    if last_lives is None:
                        last_lives = lives
                    reward += score - last_score
                    if lives < last_lives:
                        reward += DEATH_PENALTY
                    last_score = score
                    last_lives = lives
                    if lives <= 0:
                        done = True

                replay.append((state, action_idx, reward, next_state, done))
                reward_history.append(reward)
                steps += 1

                # Epsilon decay
                if steps < EPS_DECAY_STEPS:
                    eps = EPS_START - (EPS_START - EPS_END) * (steps / EPS_DECAY_STEPS)
                else:
                    eps = EPS_END

                # Train
                if len(replay) >= WARMUP_STEPS and steps % TRAIN_EVERY == 0:
                    batch = random.sample(replay, BATCH_SIZE)
                    states, actions_b, rewards, next_states, dones = zip(*batch)
                    states = torch.from_numpy(np.stack(states)).to(device)
                    next_states = torch.from_numpy(np.stack(next_states)).to(device)
                    actions_b = torch.tensor(actions_b, dtype=torch.long, device=device)
                    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
                    dones = torch.tensor(dones, dtype=torch.float32, device=device)

                    q_values = policy(states).gather(1, actions_b.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        next_q = target(next_states).max(dim=1).values
                        target_q = rewards + (1.0 - dones) * GAMMA * next_q
                    loss = F.mse_loss(q_values, target_q)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                if steps % TARGET_UPDATE_EVERY == 0:
                    target.load_state_dict(policy.state_dict())

                if done:
                    if current_move and current_move in KEY_MAP:
                        keyboard.release(KEY_MAP[current_move])
                        current_move = None
                    keyboard.press(KEY_MAP["n"])
                    time.sleep(KEY_HOLD_TIME)
                    keyboard.release(KEY_MAP["n"])
                    time.sleep(1.0)
                    last_score = None
                    last_lives = None

                if time.time() - last_save >= SAVE_EVERY_SEC:
                    avg_reward = float(np.mean(reward_history)) if reward_history else 0.0
                    log_entry = {
                        "time": time.time(),
                        "steps": steps,
                        "eps": eps,
                        "avg_reward_200": avg_reward,
                        "last_score": last_score if last_score is not None else 0,
                        "last_lives": last_lives if last_lives is not None else 0,
                    }
                    try:
                        with open(LOG_PATH, "a", encoding="utf-8") as f:
                            f.write(json.dumps(log_entry) + "\n")
                    except OSError:
                        pass
                    ckpt = {
                        "policy": policy.state_dict(),
                        "target": target.state_dict(),
                        "optimizer": opt.state_dict(),
                        "steps": steps,
                        "eps": eps,
                        "last_score": last_score,
                        "last_lives": last_lives,
                    }
                    torch.save(ckpt, CHECKPOINT_PATH)
                    print(f"Saved checkpoint at step {steps}, avg_reward={avg_reward:.2f}")
                    last_save = time.time()
        except KeyboardInterrupt:
            pass

    ckpt = {
        "policy": policy.state_dict(),
        "target": target.state_dict(),
        "optimizer": opt.state_dict(),
        "steps": steps,
        "eps": eps,
        "last_score": last_score,
        "last_lives": last_lives,
    }
    torch.save(ckpt, CHECKPOINT_PATH)
    print("Saved checkpoint on exit.")


if __name__ == "__main__":
    main()
