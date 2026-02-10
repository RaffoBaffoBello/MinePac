import argparse
import json
import os
import random

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


def run_episode(model, device, config, sims, c_puct, dt_ms, max_steps, seed=None):
    env = PacEnv(config, seed=seed)
    mcts = MCTS(
        model,
        env.action_count(),
        sims=sims,
        c_puct=c_puct,
        device=device,
        dt_ms=dt_ms,
        dirichlet_eps=0.0,
    )

    steps = 0
    while not env.is_terminal() and steps < max_steps:
        policy = mcts.run(env)
        action = int(np.argmax(policy))
        env.step_action(action, dt_ms=dt_ms)
        steps += 1

    return {
        "score": env.score,
        "level": env.level,
        "steps": steps,
        "outcome": env.terminal_value(),
    }


def load_model(path, input_shape, action_size, device):
    model = AZNet(input_shape, action_size).to(device)
    if path and os.path.exists(path):
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
    else:
        raise FileNotFoundError(f"Model not found: {path}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--best", default="az_model.pt")
    parser.add_argument("--candidate", default="az_candidate.pt")
    parser.add_argument("--games", type=int, default=6)
    parser.add_argument("--sims", type=int, default=64)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--dt-ms", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    cfg = load_config()
    env = PacEnv(cfg)
    input_shape = (len(env.get_observation()[0]), env.grid_h, env.grid_w)
    action_size = env.action_count()

    device = select_device()
    best_model = load_model(args.best, input_shape, action_size, device)
    cand_model = load_model(args.candidate, input_shape, action_size, device)
    best_model.eval()
    cand_model.eval()

    rng = random.Random(args.seed)
    seeds = [rng.randint(0, 1_000_000) for _ in range(args.games)]

    best_scores = []
    cand_scores = []
    wins = 0
    ties = 0

    for s in seeds:
        best_result = run_episode(best_model, device, cfg, args.sims, args.c_puct, args.dt_ms, args.max_steps, seed=s)
        cand_result = run_episode(cand_model, device, cfg, args.sims, args.c_puct, args.dt_ms, args.max_steps, seed=s)
        best_scores.append(best_result["score"])
        cand_scores.append(cand_result["score"])
        if cand_result["score"] > best_result["score"]:
            wins += 1
        elif cand_result["score"] == best_result["score"]:
            ties += 1

    best_mean = float(np.mean(best_scores)) if best_scores else 0.0
    cand_mean = float(np.mean(cand_scores)) if cand_scores else 0.0
    win_rate = wins / max(1, args.games)

    print(
        f"Best avg score: {best_mean:.1f} | Candidate avg score: {cand_mean:.1f} | win_rate={win_rate:.2f} | ties={ties}"
    )


if __name__ == "__main__":
    main()
