import argparse
import json
import os
import time

import numpy as np
import pygame
import torch

from pac_env import PacEnv
from az_model import AZNet, select_device, obs_to_tensor
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
MODEL_PATH = os.path.join(os.path.dirname(__file__), "az_model.pt")
ASSET_PATH = os.path.join(os.path.dirname(__file__), "assets", "reveal.jpg")

BLACK = (10, 10, 15)
WHITE = (240, 240, 240)
YELLOW = (250, 215, 70)
CYAN = (80, 220, 220)
BOMB_COLOR = (255, 200, 60)
GHOST_COLORS = [(220, 60, 60), (255, 105, 180), (80, 220, 220), (255, 165, 60)]
WALL_COLORS = [
    (40, 60, 200),
    (60, 160, 120),
    (200, 110, 60),
    (160, 80, 180),
    (70, 180, 220),
    (200, 200, 70),
    (180, 70, 70),
    (120, 140, 200),
]


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


def draw_env(screen, env, reveal_image, status_line=None):
    tile = env.tile
    width = env.grid_w * tile
    height = env.grid_h * tile

    if reveal_image:
        screen.blit(reveal_image, (0, 0))
        for by in range((env.grid_h + env.reveal_block - 1) // env.reveal_block):
            for bx in range((env.grid_w + env.reveal_block - 1) // env.reveal_block):
                if (bx, by) in env.revealed:
                    continue
                x0 = bx * env.reveal_block * tile
                y0 = by * env.reveal_block * tile
                w = min(env.reveal_block, env.grid_w - bx * env.reveal_block) * tile
                h = min(env.reveal_block, env.grid_h - by * env.reveal_block) * tile
                pygame.draw.rect(screen, BLACK, pygame.Rect(x0, y0, w, h))
    else:
        screen.fill(BLACK)

    wall_color = WALL_COLORS[(env.level - 1) % len(WALL_COLORS)]
    for x, y in env.walls:
        pygame.draw.rect(screen, wall_color, pygame.Rect(x * tile, y * tile, tile, tile))

    for x, y in env.dots:
        pygame.draw.circle(screen, WHITE, (x * tile + tile // 2, y * tile + tile // 2), 2)

    for x, y in env.power:
        pygame.draw.circle(screen, WHITE, (x * tile + tile // 2, y * tile + tile // 2), 6)

    for x, y in env.cyan:
        pygame.draw.circle(screen, CYAN, (x * tile + tile // 2, y * tile + tile // 2), 6)

    for (x, y), remaining in env.dig_timers.items():
        alpha = max(0.2, min(1.0, remaining / 500.0))
        color = (int(200 * alpha), int(200 * alpha), int(200 * alpha))
        pygame.draw.rect(screen, color, pygame.Rect(x * tile, y * tile, tile, tile))

    for b in env.bombs:
        pygame.draw.circle(
            screen,
            BOMB_COLOR,
            (b["x"] * tile + tile // 2, b["y"] * tile + tile // 2),
            tile // 3,
        )

    pac_color = CYAN if env.break_blocks else YELLOW
    pygame.draw.rect(screen, pac_color, pygame.Rect(env.player.x, env.player.y, tile, tile))
    eye = tile // 4
    pygame.draw.circle(screen, BLACK, (int(env.player.x + eye), int(env.player.y + eye)), max(2, tile // 8))
    pygame.draw.circle(screen, BLACK, (int(env.player.x + tile - eye), int(env.player.y + eye)), max(2, tile // 8))

    for idx, g in enumerate(env.ghosts):
        if not g.alive:
            continue
        color = GHOST_COLORS[idx % len(GHOST_COLORS)]
        pygame.draw.rect(screen, color, pygame.Rect(g.x, g.y, tile, tile))
        eye = tile // 4
        pygame.draw.circle(screen, WHITE, (int(g.x + eye), int(g.y + eye)), max(2, tile // 8))
        pygame.draw.circle(screen, WHITE, (int(g.x + tile - eye), int(g.y + eye)), max(2, tile // 8))
        pygame.draw.circle(screen, BLACK, (int(g.x + eye), int(g.y + eye)), max(1, tile // 12))
        pygame.draw.circle(screen, BLACK, (int(g.x + tile - eye), int(g.y + eye)), max(1, tile // 12))

    font = pygame.font.SysFont("Arial", 18)
    hud = font.render(f"Score: {env.score:06d}  Lives: {env.lives}  Level: {env.level}", True, WHITE)
    screen.blit(hud, (10, height - 22))

    if status_line:
        font = pygame.font.SysFont("Arial", 16)
        status = font.render(status_line, True, WHITE)
        screen.blit(status, (10, 4))

    if env.game_over or env.game_won:
        overlay = pygame.Surface((width, height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        screen.blit(overlay, (0, 0))
        msg = "YOU WON" if env.game_won else "GAME OVER"
        big = pygame.font.SysFont("Arial", 64)
        text = big.render(msg, True, WHITE)
        screen.blit(text, (width // 2 - text.get_width() // 2, height // 2 - text.get_height() // 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--dt-ms", type=int, default=32)
    parser.add_argument("--mcts-sims", type=int, default=0)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--force-move", action="store_true")
    parser.add_argument("--episodes", type=int, default=0, help="0 = keep restarting forever")
    args = parser.parse_args()

    os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "60,60")
    cfg = load_config()
    env = PacEnv(cfg)
    input_shape = (len(env.get_observation()[0]), env.grid_h, env.grid_w)

    device = select_device()
    model = AZNet(input_shape, env.action_count()).to(device)
    if os.path.exists(args.model):
        state = torch.load(args.model, map_location=device)
        model.load_state_dict(state)
    model.eval()

    mcts = None
    if args.mcts_sims and args.mcts_sims > 0:
        mcts = MCTS(
            model,
            env.action_count(),
            sims=args.mcts_sims,
            c_puct=args.c_puct,
            device=device,
            dt_ms=args.dt_ms,
        )

    pygame.init()
    screen = pygame.display.set_mode((env.grid_w * env.tile, env.grid_h * env.tile))
    pygame.display.set_caption("AlphaZero Pac-Maze")
    clock = pygame.time.Clock()

    reveal_image = None
    if os.path.exists(ASSET_PATH):
        try:
            reveal_image = pygame.image.load(ASSET_PATH).convert()
            reveal_image = pygame.transform.smoothscale(reveal_image, (env.grid_w * env.tile, env.grid_h * env.tile))
        except pygame.error:
            reveal_image = None

    step_count = 0
    last_action = None
    last_policy = None
    episode = 1

    running = True
    def sample_action_from_logits(logits, mask, temperature):
        logits = logits.astype(np.float32)
        mask = mask.astype(np.float32)
        if args.force_move:
            for i in range(len(mask)):
                move_dir, _ = env.index_to_action(i)
                if move_dir == (0, 0):
                    mask[i] = 0
        if mask.sum() <= 0:
            return int(np.argmax(logits))
        masked = np.where(mask > 0, logits, -1e9)
        if temperature <= 0:
            return int(np.argmax(masked))
        scaled = masked / max(1e-6, temperature)
        scaled = scaled - np.max(scaled)
        probs = np.exp(scaled) * mask
        total = probs.sum()
        if total <= 0:
            return int(np.argmax(masked))
        probs = probs / total
        return int(np.random.choice(len(probs), p=probs))

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if env.is_terminal():
            if args.episodes > 0 and episode >= args.episodes:
                break
            env.reset_all()
            episode += 1
            step_count = 0

        if not env.is_terminal():
            if mcts:
                policy = mcts.run(env)
                action = int(np.argmax(policy))
                last_policy = policy
            else:
                planes, _, _ = env.get_observation()
                obs = obs_to_tensor(planes, device)
                with torch.no_grad():
                    logits, _ = model(obs)
                logits_np = logits.squeeze(0).detach().cpu().numpy()
                mask = np.asarray(env.legal_action_mask(), dtype=np.float32)
                action = sample_action_from_logits(logits_np, mask, args.temperature)
                last_policy = None
            last_action = action
            env.step_action(action, dt_ms=args.dt_ms)
            step_count += 1

        status = f"ep={episode} steps={step_count} action={last_action}"
        if last_policy is not None:
            status += f" pmax={float(np.max(last_policy)):.2f}"
        draw_env(screen, env, reveal_image, status_line=status)
        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()


if __name__ == "__main__":
    main()
