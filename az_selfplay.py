import argparse
import json
import os
import random
import time
from datetime import datetime

import numpy as np
import torch

from pac_env import PacEnv
from az_model import AZNet, select_device
from az_mcts import MCTS

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

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "az")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "az_model.pt")
LOG_PATH = os.path.join(os.path.dirname(__file__), "az_log.jsonl")


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


def select_action(policy, temperature):
    if temperature <= 0:
        return int(np.argmax(policy))
    scaled = np.power(policy, 1.0 / temperature)
    total = scaled.sum()
    if total <= 0:
        return int(np.argmax(policy))
    scaled = scaled / total
    return int(np.random.choice(len(policy), p=scaled))


def play_episode(model, device, config, sims, c_puct, temp, dt_ms, max_steps, seed=None):
    env = PacEnv(config, seed=seed)
    action_size = env.action_count()
    mcts = MCTS(
        model,
        action_size,
        sims=sims,
        c_puct=c_puct,
        device=device,
        dt_ms=dt_ms,
    )

    observations = []
    policies = []
    steps = 0

    channels = None
    while not env.is_terminal() and steps < max_steps:
        policy = mcts.run(env)
        planes, _, _ = env.get_observation()
        if channels is None:
            channels = len(planes)
        observations.append(np.asarray(planes, dtype=np.float32))
        policies.append(policy.astype(np.float32))

        action = select_action(policy, temp)
        env.step_action(action, dt_ms=dt_ms)
        steps += 1

    outcome = env.terminal_value()
    if channels is None:
        channels = len(env.get_observation()[0])
    return {
        "obs": np.stack(observations, axis=0)
        if observations
        else np.zeros((0, channels, env.grid_h, env.grid_w), dtype=np.float32),
        "policy": np.stack(policies, axis=0) if policies else np.zeros((0, action_size), dtype=np.float32),
        "value": np.full((len(observations),), outcome, dtype=np.float32),
        "steps": steps,
        "score": env.score,
        "level": env.level,
        "outcome": outcome,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--sims", type=int, default=64)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--dt-ms", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    cfg = load_config()
    env = PacEnv(cfg)
    input_shape = (len(env.get_observation()[0]), env.grid_h, env.grid_w)
    action_size = env.action_count()

    device = select_device()
    model = AZNet(input_shape, action_size).to(device)
    if os.path.exists(MODEL_PATH):
        state = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state)
    model.eval()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    for episode in range(args.episodes):
        seed = rng.randint(0, 1_000_000) if args.seed is not None else None
        result = play_episode(
            model,
            device,
            cfg,
            sims=args.sims,
            c_puct=args.c_puct,
            temp=args.temp,
            dt_ms=args.dt_ms,
            max_steps=args.max_steps,
            seed=seed,
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(DATA_DIR, f"az_{timestamp}_{episode:03d}.npz")
        np.savez_compressed(out_path, obs=result["obs"], policy=result["policy"], value=result["value"])

        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "ts": time.time(),
                        "episode": episode,
                        "steps": result["steps"],
                        "score": result["score"],
                        "level": result["level"],
                        "outcome": result["outcome"],
                        "file": out_path,
                    }
                )
                + "\n"
            )
        print(f"Saved {out_path} | steps={result['steps']} score={result['score']} level={result['level']} outcome={result['outcome']}")


if __name__ == "__main__":
    main()
