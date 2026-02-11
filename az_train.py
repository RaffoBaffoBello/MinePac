import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from pac_env import PacEnv
from az_model import AZNet, select_device
from az_eval import run_episode

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "az")
BEST_MODEL_PATH = os.path.join(os.path.dirname(__file__), "az_model.pt")
CANDIDATE_PATH = os.path.join(os.path.dirname(__file__), "az_candidate.pt")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

DEFAULT_CONFIG = {
    "tile_size": 24,
    "cyan_effect_ms": 40000,
    "bomb_timer_ms": 4000,
    "bomb_size": 6,
    "reveal_block": 10,
    "ghosts_first_level": 3,
    "ghosts_per_level": 1,
    "ghost_speed_multiplier": 1.02,
    "max_bombs_per_life": 10,
    "points_per_level": 1000,
}


def load_config():
    cfg = DEFAULT_CONFIG.copy()
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                for k in cfg.keys():
                    if k in data:
                        cfg[k] = data[k]
    except FileNotFoundError:
        pass
    except json.JSONDecodeError:
        pass
    return cfg


def load_data(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "az_*.npz")))
    if not files:
        raise FileNotFoundError(f"No az_*.npz files found in {data_dir}")

    print(f"Loading {len(files)} self-play files...")
    sys.stdout.flush()

    obs_list = []
    policy_list = []
    value_list = []

    for idx, path in enumerate(files, start=1):
        data = np.load(path)
        obs_list.append(data["obs"])
        policy_list.append(data["policy"])
        value_list.append(data["value"])
        if idx % 100 == 0 or idx == len(files):
            print(f"  loaded {idx}/{len(files)}")
            sys.stdout.flush()

    obs = np.concatenate(obs_list, axis=0).astype(np.float32)
    policy = np.concatenate(policy_list, axis=0).astype(np.float32)
    value = np.concatenate(value_list, axis=0).astype(np.float32)
    print(f"Loaded samples: {len(obs)} | obs shape {obs.shape} | policy shape {policy.shape}")
    sys.stdout.flush()
    return obs, policy, value


def evaluate_models(best_path, cand_path, games, sims, c_puct, dt_ms, max_steps, seed):
    cfg = load_config()
    env = PacEnv(cfg)
    input_shape = (len(env.get_observation()[0]), env.grid_h, env.grid_w)
    action_size = env.action_count()

    device = select_device()
    best_model = AZNet(input_shape, action_size).to(device)
    cand_model = AZNet(input_shape, action_size).to(device)

    if os.path.exists(best_path):
        best_model.load_state_dict(torch.load(best_path, map_location=device))
    cand_model.load_state_dict(torch.load(cand_path, map_location=device))
    best_model.eval()
    cand_model.eval()

    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 1_000_000, size=games)

    best_scores = []
    cand_scores = []
    wins = 0

    for s in seeds:
        best_result = run_episode(best_model, device, cfg, sims, c_puct, dt_ms, max_steps, seed=int(s))
        cand_result = run_episode(cand_model, device, cfg, sims, c_puct, dt_ms, max_steps, seed=int(s))
        best_scores.append(best_result["score"])
        cand_scores.append(cand_result["score"])
        if cand_result["score"] > best_result["score"]:
            wins += 1

    best_mean = float(np.mean(best_scores)) if best_scores else 0.0
    cand_mean = float(np.mean(cand_scores)) if cand_scores else 0.0
    win_rate = wins / max(1, games)
    return {
        "best_mean": best_mean,
        "cand_mean": cand_mean,
        "win_rate": win_rate,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--no-gating", action="store_true")
    parser.add_argument("--eval-games", type=int, default=6)
    parser.add_argument("--eval-sims", type=int, default=64)
    parser.add_argument("--eval-c-puct", type=float, default=1.5)
    parser.add_argument("--eval-dt-ms", type=int, default=32)
    parser.add_argument("--eval-max-steps", type=int, default=2000)
    parser.add_argument("--min-win-rate", type=float, default=0.55)
    parser.add_argument("--min-score-delta", type=float, default=0.0)
    args = parser.parse_args()

    obs, policy, value = load_data(args.data_dir)
    n = len(obs)
    if n == 0:
        raise RuntimeError("No samples to train on.")

    indices = np.arange(n)
    np.random.shuffle(indices)
    split = int(n * (1 - args.val_split))
    train_idx = indices[:split]
    val_idx = indices[split:] if split < n else indices[:0]

    train_obs = torch.from_numpy(obs[train_idx])
    train_policy = torch.from_numpy(policy[train_idx])
    train_value = torch.from_numpy(value[train_idx])

    val_obs = torch.from_numpy(obs[val_idx])
    val_policy = torch.from_numpy(policy[val_idx])
    val_value = torch.from_numpy(value[val_idx])

    train_ds = TensorDataset(train_obs, train_policy, train_value)
    val_ds = TensorDataset(val_obs, val_policy, val_value)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, drop_last=False)

    input_shape = train_obs.shape[1:]
    action_size = train_policy.shape[1]

    device = select_device()
    model = AZNet(input_shape, action_size).to(device)
    if os.path.exists(BEST_MODEL_PATH):
        state = torch.load(BEST_MODEL_PATH, map_location=device)
        model.load_state_dict(state)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_policy = 0.0
        total_value = 0.0
        steps = 0

        for batch_obs, batch_policy, batch_value in train_loader:
            batch_obs = batch_obs.to(device)
            batch_policy = batch_policy.to(device)
            batch_value = batch_value.to(device)

            optimizer.zero_grad()
            logits, values = model(batch_obs)
            log_probs = F.log_softmax(logits, dim=1)
            policy_loss = -(batch_policy * log_probs).sum(dim=1).mean()
            value_loss = F.mse_loss(values, batch_value)
            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_policy += policy_loss.item()
            total_value += value_loss.item()
            steps += 1

        avg_loss = total_loss / max(1, steps)
        avg_policy = total_policy / max(1, steps)
        avg_value = total_value / max(1, steps)

        model.eval()
        val_policy_loss = 0.0
        val_value_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch_obs, batch_policy, batch_value in val_loader:
                batch_obs = batch_obs.to(device)
                batch_policy = batch_policy.to(device)
                batch_value = batch_value.to(device)
                logits, values = model(batch_obs)
                log_probs = F.log_softmax(logits, dim=1)
                policy_loss = -(batch_policy * log_probs).sum(dim=1).mean()
                value_loss = F.mse_loss(values, batch_value)
                val_policy_loss += policy_loss.item()
                val_value_loss += value_loss.item()
                val_steps += 1
        if val_steps:
            val_policy_loss /= val_steps
            val_value_loss /= val_steps

        print(
            f"Epoch {epoch:02d} | loss {avg_loss:.4f} | policy {avg_policy:.4f} | value {avg_value:.4f}"
            + (f" | val_policy {val_policy_loss:.4f} | val_value {val_value_loss:.4f}" if val_steps else "")
        )

    torch.save(model.state_dict(), CANDIDATE_PATH)
    print(f"Saved candidate to {CANDIDATE_PATH}")

    if args.no_gating:
        shutil.copy(CANDIDATE_PATH, BEST_MODEL_PATH)
        print(f"Gating disabled. Promoted to {BEST_MODEL_PATH}")
        return

    if not os.path.exists(BEST_MODEL_PATH):
        shutil.copy(CANDIDATE_PATH, BEST_MODEL_PATH)
        print(f"No best model found. Promoted to {BEST_MODEL_PATH}")
        return

    eval_result = evaluate_models(
        BEST_MODEL_PATH,
        CANDIDATE_PATH,
        games=args.eval_games,
        sims=args.eval_sims,
        c_puct=args.eval_c_puct,
        dt_ms=args.eval_dt_ms,
        max_steps=args.eval_max_steps,
        seed=123,
    )
    print(
        f"Eval: best_mean={eval_result['best_mean']:.1f} | cand_mean={eval_result['cand_mean']:.1f} | win_rate={eval_result['win_rate']:.2f}"
    )

    if eval_result["win_rate"] >= args.min_win_rate or eval_result["cand_mean"] >= eval_result["best_mean"] + args.min_score_delta:
        shutil.copy(CANDIDATE_PATH, BEST_MODEL_PATH)
        print("Candidate promoted to best.")
    else:
        print("Candidate rejected. Best model kept.")


if __name__ == "__main__":
    main()
