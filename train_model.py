import os
import json
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pth")
META_PATH = os.path.join(os.path.dirname(__file__), "model_meta.json")

FRAME_SIZE = (96, 96)
STACK_SIZE = 4
MOVE_LABELS = ["none", "up", "down", "left", "right"]
ACTION_LABELS = ["none", "c", "v", "space"]

BATCH_SIZE = 128
EPOCHS = 25
LR = 1e-3
VAL_SPLIT = 0.1
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class FrameDataset(Dataset):
    def __init__(self, frames, moves, actions, stack_size):
        self.frames = frames
        self.moves = moves
        self.actions = actions
        self.stack_size = stack_size

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        start = idx - self.stack_size + 1
        if start < 0:
            pad = [self.frames[0]] * (-start)
            stack = pad + [self.frames[i] for i in range(0, idx + 1)]
        else:
            stack = [self.frames[i] for i in range(start, idx + 1)]
        stack = np.stack(stack, axis=0).astype(np.float32) / 255.0
        frame = torch.from_numpy(stack)
        move = int(self.moves[idx])
        action = int(self.actions[idx])
        return frame, move, action


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


def load_npz_files(data_dir):
    frames = []
    moves = []
    actions = []
    for name in sorted(os.listdir(data_dir)):
        if not name.endswith(".npz"):
            continue
        path = os.path.join(data_dir, name)
        data = np.load(path)
        frames.append(data["frames"])
        moves.append(data["move"])
        actions.append(data["action"])
    if not frames:
        return None
    frames = np.concatenate(frames, axis=0)
    moves = np.concatenate(moves, axis=0)
    actions = np.concatenate(actions, axis=0)
    return frames, moves, actions


def main():
    set_seed(SEED)
    if not os.path.isdir(DATA_DIR):
        raise SystemExit("No data directory found. Run collect_data.py first.")

    loaded = load_npz_files(DATA_DIR)
    if loaded is None:
        raise SystemExit("No .npz files found in data directory.")
    frames, moves, actions = loaded

    if frames.shape[1:3] != FRAME_SIZE:
        print(f"Warning: frame size mismatch. Expected {FRAME_SIZE}, got {frames.shape[1:3]}.")

    move_counts = np.bincount(moves, minlength=len(MOVE_LABELS)).astype(np.float32)
    action_counts = np.bincount(actions, minlength=len(ACTION_LABELS)).astype(np.float32)
    print("Move counts:", {MOVE_LABELS[i]: int(c) for i, c in enumerate(move_counts)})
    print("Action counts:", {ACTION_LABELS[i]: int(c) for i, c in enumerate(action_counts)})
    move_weights = np.zeros_like(move_counts)
    action_weights = np.zeros_like(action_counts)
    total_moves = move_counts.sum()
    total_actions = action_counts.sum()
    for i, c in enumerate(move_counts):
        if c > 0:
            move_weights[i] = total_moves / (len(MOVE_LABELS) * c)
    for i, c in enumerate(action_counts):
        if c > 0:
            action_weights[i] = total_actions / (len(ACTION_LABELS) * c)
    print("Move weights:", {MOVE_LABELS[i]: float(f"{w:.3f}") for i, w in enumerate(move_weights)})
    print("Action weights:", {ACTION_LABELS[i]: float(f'{w:.3f}') for i, w in enumerate(action_weights)})

    dataset = FrameDataset(frames, moves, actions, STACK_SIZE)
    val_len = int(len(dataset) * VAL_SPLIT)
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    train_indices = train_set.indices
    train_action_counts = np.bincount(actions[train_indices], minlength=len(ACTION_LABELS)).astype(np.float32)
    action_sample_weights = np.zeros(len(train_indices), dtype=np.float32)
    max_multiplier = 3.0
    base_min = train_action_counts.min() if train_action_counts.min() > 0 else 1.0
    raw = base_min / np.maximum(train_action_counts, 1.0)
    raw = np.clip(raw, 1.0 / max_multiplier, 1.0)
    for i, idx in enumerate(train_indices):
        label = actions[idx]
        action_sample_weights[i] = raw[label]
    sampler = WeightedRandomSampler(action_sample_weights, num_samples=len(train_indices), replacement=True)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = LightPolicyNet(STACK_SIZE, len(MOVE_LABELS), len(ACTION_LABELS)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn_move = nn.CrossEntropyLoss(weight=torch.tensor(move_weights, dtype=torch.float32, device=device))
    loss_fn_action = nn.CrossEntropyLoss(weight=torch.tensor(action_weights, dtype=torch.float32, device=device))

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for frames_batch, move_batch, action_batch in train_loader:
            frames_batch = frames_batch.to(device)
            move_batch = move_batch.to(device)
            action_batch = action_batch.to(device)
            opt.zero_grad()
            move_logits, action_logits = model(frames_batch)
            loss = loss_fn_move(move_logits, move_batch) + loss_fn_action(action_logits, action_batch)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        model.eval()
        correct_move = 0
        correct_action = 0
        total = 0
        move_correct_per_class = np.zeros(len(MOVE_LABELS), dtype=np.int64)
        move_total_per_class = np.zeros(len(MOVE_LABELS), dtype=np.int64)
        action_correct_per_class = np.zeros(len(ACTION_LABELS), dtype=np.int64)
        action_total_per_class = np.zeros(len(ACTION_LABELS), dtype=np.int64)
        with torch.no_grad():
            for frames_batch, move_batch, action_batch in val_loader:
                frames_batch = frames_batch.to(device)
                move_batch = move_batch.to(device)
                action_batch = action_batch.to(device)
                move_logits, action_logits = model(frames_batch)
                move_pred = move_logits.argmax(dim=1)
                action_pred = action_logits.argmax(dim=1)
                correct_move += (move_pred == move_batch).sum().item()
                correct_action += (action_pred == action_batch).sum().item()
                total += move_batch.size(0)
                move_np = move_batch.cpu().numpy()
                move_pred_np = move_pred.cpu().numpy()
                action_np = action_batch.cpu().numpy()
                action_pred_np = action_pred.cpu().numpy()
                for cls in range(len(MOVE_LABELS)):
                    mask = move_np == cls
                    move_total_per_class[cls] += mask.sum()
                    if mask.any():
                        move_correct_per_class[cls] += (move_pred_np[mask] == cls).sum()
                for cls in range(len(ACTION_LABELS)):
                    mask = action_np == cls
                    action_total_per_class[cls] += mask.sum()
                    if mask.any():
                        action_correct_per_class[cls] += (action_pred_np[mask] == cls).sum()

        avg_loss = total_loss / max(1, len(train_loader))
        move_acc = correct_move / max(1, total)
        action_acc = correct_action / max(1, total)
        print(f"Epoch {epoch:02d} | loss {avg_loss:.4f} | move acc {move_acc:.3f} | action acc {action_acc:.3f}")
        move_acc_per_class = {}
        action_acc_per_class = {}
        for i, label in enumerate(MOVE_LABELS):
            denom = move_total_per_class[i]
            move_acc_per_class[label] = float(move_correct_per_class[i] / denom) if denom > 0 else 0.0
        for i, label in enumerate(ACTION_LABELS):
            denom = action_total_per_class[i]
            action_acc_per_class[label] = float(action_correct_per_class[i] / denom) if denom > 0 else 0.0
        print("  move per-class:", {k: round(v, 3) for k, v in move_acc_per_class.items()})
        print("  action per-class:", {k: round(v, 3) for k, v in action_acc_per_class.items()})

    torch.save(model.state_dict(), MODEL_PATH)
    meta = {
        "frame_size": list(FRAME_SIZE),
        "stack_size": STACK_SIZE,
        "move_labels": MOVE_LABELS,
        "action_labels": ACTION_LABELS,
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
