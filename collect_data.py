import os
import re
import time
import json
from collections import deque

import mss
import numpy as np
from PIL import Image
from pynput import keyboard
import Quartz


WINDOW_TITLE = "VideoLeo Pac-Maze"
CAPTURE_FPS = 8
FRAME_SIZE = (96, 96)
ACTION_WINDOW_SEC = 0.2
CHUNK_SIZE = 500

MOVE_LABELS = ["none", "up", "down", "left", "right"]
ACTION_LABELS = ["none", "c", "v", "space"]

KEY_TO_MOVE = {
    keyboard.Key.up: "up",
    keyboard.Key.down: "down",
    keyboard.Key.left: "left",
    keyboard.Key.right: "right",
}
KEY_TO_ACTION = {
    keyboard.Key.space: "space",
    keyboard.KeyCode.from_char("c"): "c",
    keyboard.KeyCode.from_char("v"): "v",
}


def find_window_bounds(title_substring):
    options = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
    window_list = Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID)
    matches = []
    for win in window_list:
        name = win.get("kCGWindowName", "") or ""
        if title_substring.lower() in name.lower():
            bounds = win.get("kCGWindowBounds", {})
            try:
                x = int(bounds.get("X", 0))
                y = int(bounds.get("Y", 0))
                w = int(bounds.get("Width", 0))
                h = int(bounds.get("Height", 0))
                area = w * h
                if w > 0 and h > 0:
                    matches.append((area, (x, y, w, h), name))
            except (TypeError, ValueError):
                continue
    if not matches:
        return None
    matches.sort(reverse=True, key=lambda item: item[0])
    return matches[0][1]


def preprocess(frame_bgra):
    frame = np.array(frame_bgra)[:, :, :3]
    img = Image.fromarray(frame)
    img = img.convert("L")
    img = img.resize(FRAME_SIZE, Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(out_dir, exist_ok=True)
    session = os.getenv("PAC_SESSION_NAME") or time.strftime("%Y%m%d_%H%M%S")
    meta_path = os.path.join(out_dir, f"session_{session}_meta.json")
    meta = {
        "window_title": WINDOW_TITLE,
        "capture_fps": CAPTURE_FPS,
        "frame_size": list(FRAME_SIZE),
        "move_labels": MOVE_LABELS,
        "action_labels": ACTION_LABELS,
    }
    if not os.path.exists(meta_path):
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    else:
        print(f"Using existing session metadata: {meta_path}")

    move_state = {k: False for k in KEY_TO_MOVE}
    move_history = deque(maxlen=20)
    last_action = "none"
    last_action_time = 0.0

    def on_press(key):
        nonlocal last_action, last_action_time
        if key in move_state and not move_state[key]:
            move_state[key] = True
            move_history.append(key)
        if key in KEY_TO_ACTION:
            last_action = KEY_TO_ACTION[key]
            last_action_time = time.time()

    def on_release(key):
        if key in move_state:
            move_state[key] = False

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    frames = []
    moves = []
    actions = []
    chunk_idx = 0
    pattern = re.compile(rf"^session_{re.escape(session)}_(\\d+)\\.npz$")
    for name in os.listdir(out_dir):
        match = pattern.match(name)
        if match:
            chunk_idx = max(chunk_idx, int(match.group(1)) + 1)
    frame_interval = 1.0 / max(1, CAPTURE_FPS)

    print("Data collection started.")
    print(f"Session: {session} (starting chunk {chunk_idx:03d})")
    print("Click the game window to focus it. Press Ctrl+C to stop.")

    with mss.mss() as sct:
        try:
            while True:
                start = time.time()
                bounds = find_window_bounds(WINDOW_TITLE)
                if bounds is None:
                    print("Window not found. Waiting...")
                    time.sleep(1.0)
                    continue

                x, y, w, h = bounds
                frame = sct.grab({"left": x, "top": y, "width": w, "height": h})
                frame_small = preprocess(frame)

                now = time.time()
                action_label = "none"
                if now - last_action_time <= ACTION_WINDOW_SEC:
                    action_label = last_action

                move_label = "none"
                for key in reversed(move_history):
                    if move_state.get(key):
                        move_label = KEY_TO_MOVE[key]
                        break
                if move_label == "none":
                    for key, down in move_state.items():
                        if down:
                            move_label = KEY_TO_MOVE[key]
                            break

                frames.append(frame_small)
                moves.append(MOVE_LABELS.index(move_label))
                actions.append(ACTION_LABELS.index(action_label))

                if len(frames) >= CHUNK_SIZE:
                    out_path = os.path.join(out_dir, f"session_{session}_{chunk_idx:03d}.npz")
                    np.savez_compressed(
                        out_path,
                        frames=np.stack(frames),
                        move=np.array(moves, dtype=np.int64),
                        action=np.array(actions, dtype=np.int64),
                    )
                    print(f"Saved {out_path}")
                    frames.clear()
                    moves.clear()
                    actions.clear()
                    chunk_idx += 1

                elapsed = time.time() - start
                sleep_for = frame_interval - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)
        except KeyboardInterrupt:
            pass

    if frames:
        out_path = os.path.join(out_dir, f"session_{session}_{chunk_idx:03d}.npz")
        np.savez_compressed(
            out_path,
            frames=np.stack(frames),
            move=np.array(moves, dtype=np.int64),
            action=np.array(actions, dtype=np.int64),
        )
        print(f"Saved {out_path}")

    print("Data collection finished.")


if __name__ == "__main__":
    main()
