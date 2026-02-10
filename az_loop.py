import argparse
import json
import os
import random
import subprocess
import sys
import threading
import time
from datetime import datetime

import numpy as np
import pygame
import torch

from pac_env import PacEnv
from az_model import AZNet, select_device
from az_mcts import MCTS
from az_selfplay import play_episode, load_config as load_game_config

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "az")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "az_model.pt")
LOG_PATH = os.path.join(os.path.dirname(__file__), "az_log.jsonl")
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


def run_selfplay(model, device, cfg, episodes, sims, c_puct, temp, dt_ms, max_steps, seed):
    os.makedirs(DATA_DIR, exist_ok=True)
    rng = random.Random(seed)

    for ep in range(episodes):
        ep_seed = rng.randint(0, 1_000_000) if seed is not None else None
        result = play_episode(
            model,
            device,
            cfg,
            sims=sims,
            c_puct=c_puct,
            temp=temp,
            dt_ms=dt_ms,
            max_steps=max_steps,
            seed=ep_seed,
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(DATA_DIR, f"az_{timestamp}_{ep:03d}.npz")
        np.savez_compressed(out_path, obs=result["obs"], policy=result["policy"], value=result["value"])

        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "ts": time.time(),
                        "episode": ep,
                        "steps": result["steps"],
                        "score": result["score"],
                        "level": result["level"],
                        "outcome": result["outcome"],
                        "file": out_path,
                    }
                )
                + "\n"
            )
        print(
            f"Self-play saved {out_path} | steps={result['steps']} score={result['score']} level={result['level']} outcome={result['outcome']}"
        )


def train_model(epochs, batch, no_gating):
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "az_train.py"), "--epochs", str(epochs), "--batch", str(batch)]
    if no_gating:
        cmd.append("--no-gating")
    subprocess.run(cmd, check=True)


def play_visual(
    model,
    device,
    cfg,
    fps,
    dt_ms,
    mcts_sims,
    c_puct,
    max_steps,
    hold_sec,
    temperature,
    force_move,
    episodes,
    model_lock,
    reload_secs,
):
    env = PacEnv(cfg)
    mcts = None
    if mcts_sims and mcts_sims > 0:
        mcts = MCTS(
            model,
            env.action_count(),
            sims=mcts_sims,
            c_puct=c_puct,
            device=device,
            dt_ms=dt_ms,
        )

    reveal_image = None
    if os.path.exists(ASSET_PATH):
        try:
            reveal_image = pygame.image.load(ASSET_PATH).convert()
            reveal_image = pygame.transform.smoothscale(reveal_image, (env.grid_w * env.tile, env.grid_h * env.tile))
        except pygame.error:
            reveal_image = None

    screen = pygame.display.set_mode((env.grid_w * env.tile, env.grid_h * env.tile))
    pygame.display.set_caption("AlphaZero Pac-Maze Loop")
    clock = pygame.time.Clock()

    steps = 0
    episode = 1
    running = True
    def sample_action_from_logits(logits, mask, temperature):
        logits = logits.astype(np.float32)
        mask = mask.astype(np.float32)
        if force_move:
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

    last_reload = time.time()
    model_mtime = os.path.getmtime(MODEL_PATH) if os.path.exists(MODEL_PATH) else 0.0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not running:
            break

        if env.is_terminal() or steps >= max_steps:
            if episodes > 0 and episode >= episodes:
                break
            env.reset_all()
            episode += 1
            steps = 0
            continue

        if reload_secs > 0 and time.time() - last_reload >= reload_secs and os.path.exists(MODEL_PATH):
            new_mtime = os.path.getmtime(MODEL_PATH)
            if new_mtime > model_mtime:
                try:
                    state = torch.load(MODEL_PATH, map_location=device)
                    with model_lock:
                        model.load_state_dict(state)
                        model.eval()
                    model_mtime = new_mtime
                    print("Reloaded model for watch.")
                    sys.stdout.flush()
                except OSError:
                    pass
            last_reload = time.time()

        if mcts:
            with model_lock:
                policy = mcts.run(env)
            action = int(np.argmax(policy))
        else:
            planes, _, _ = env.get_observation()
            obs = torch.from_numpy(np.asarray(planes, dtype=np.float32)).unsqueeze(0).to(device)
            with model_lock:
                with torch.no_grad():
                    logits, _ = model(obs)
            logits_np = logits.squeeze(0).detach().cpu().numpy()
            mask = np.asarray(env.legal_action_mask(), dtype=np.float32)
            action = sample_action_from_logits(logits_np, mask, temperature)

        env.step_action(action, dt_ms=dt_ms)
        draw_env(screen, env, reveal_image, status_line=f"ep={episode} steps={steps}")
        pygame.display.flip()
        clock.tick(fps)
        steps += 1

    # Keep the final frame visible for a bit (or until user closes the window).
    if running and hold_sec > 0:
        end_time = time.time() + hold_sec
        while running and time.time() < end_time:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            draw_env(screen, env, reveal_image, status_line="Watching (finished)...")
            pygame.display.flip()
            clock.tick(min(30, fps))

    return running


def train_worker(stop_event, args, cfg, input_shape, action_size, device, model_lock):
    cycle = 0
    while not stop_event.is_set():
        cycle += 1
        print(f"\n=== Train Cycle {cycle} ===")
        sys.stdout.flush()

        # load the latest best model for self-play
        model = AZNet(input_shape, action_size).to(device)
        if os.path.exists(MODEL_PATH):
            try:
                state = torch.load(MODEL_PATH, map_location=device)
                model.load_state_dict(state)
            except OSError:
                pass
        model.eval()

        print("Self-play starting...")
        sys.stdout.flush()
        run_selfplay(
            model,
            device,
            cfg,
            episodes=args.selfplay_episodes,
            sims=args.selfplay_sims,
            c_puct=args.c_puct,
            temp=args.selfplay_temp,
            dt_ms=args.selfplay_dt_ms,
            max_steps=args.selfplay_max_steps,
            seed=args.seed,
        )

        print("Training...")
        sys.stdout.flush()
        train_model(args.train_epochs, args.train_batch, args.no_gating)

        if stop_event.is_set():
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=0, help="0 = infinite")
    parser.add_argument("--selfplay-episodes", type=int, default=2)
    parser.add_argument("--selfplay-sims", type=int, default=64)
    parser.add_argument("--selfplay-temp", type=float, default=1.0)
    parser.add_argument("--selfplay-dt-ms", type=int, default=32)
    parser.add_argument("--selfplay-max-steps", type=int, default=1200)
    parser.add_argument("--train-epochs", type=int, default=5)
    parser.add_argument("--train-batch", type=int, default=64)
    parser.add_argument("--no-gating", action="store_true")
    parser.add_argument("--watch-fps", type=int, default=60)
    parser.add_argument("--watch-dt-ms", type=int, default=32)
    parser.add_argument("--watch-mcts-sims", type=int, default=0)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--watch-max-steps", type=int, default=1200)
    parser.add_argument("--watch-hold-sec", type=float, default=5.0)
    parser.add_argument("--watch-temperature", type=float, default=1.0)
    parser.add_argument("--watch-force-move", action="store_true")
    parser.add_argument("--watch-episodes", type=int, default=0, help="0 = keep restarting during watch")
    parser.add_argument("--watch-first", action="store_true", help="Show a watch window before self-play/training")
    parser.add_argument("--train-while-watching", action="store_true", help="Run self-play/training in background")
    parser.add_argument("--reload-secs", type=float, default=3.0, help="How often to reload model during watching")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "60,60")
    cfg = load_game_config()
    env = PacEnv(cfg)
    input_shape = (len(env.get_observation()[0]), env.grid_h, env.grid_w)

    device = select_device()
    model = AZNet(input_shape, env.action_count()).to(device)
    if os.path.exists(MODEL_PATH):
        state = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state)
    model.eval()

    pygame.init()

    model_lock = threading.Lock()
    stop_event = threading.Event()

    if args.train_while_watching:
        train_thread = threading.Thread(
            target=train_worker,
            args=(stop_event, args, cfg, input_shape, env.action_count(), device, model_lock),
            daemon=True,
        )
        train_thread.start()

        print("Watching current best model (training in background)...")
        sys.stdout.flush()
        running = play_visual(
            model,
            device,
            cfg,
            fps=args.watch_fps,
            dt_ms=args.watch_dt_ms,
            mcts_sims=args.watch_mcts_sims,
            c_puct=args.c_puct,
            max_steps=args.watch_max_steps,
            hold_sec=args.watch_hold_sec,
            temperature=args.watch_temperature,
            force_move=args.watch_force_move,
            episodes=args.watch_episodes,
            model_lock=model_lock,
            reload_secs=args.reload_secs,
        )

        stop_event.set()
        train_thread.join(timeout=5.0)
    else:
        cycle = 0
        running = True
        while running:
            cycle += 1
            print(f"\n=== Cycle {cycle} ===")
            sys.stdout.flush()

            if args.watch_first:
                print("Opening watch window (before training)...")
                sys.stdout.flush()
                running = play_visual(
                    model,
                    device,
                    cfg,
                    fps=args.watch_fps,
                    dt_ms=args.watch_dt_ms,
                    mcts_sims=args.watch_mcts_sims,
                    c_puct=args.c_puct,
                    max_steps=args.watch_max_steps,
                    hold_sec=args.watch_hold_sec,
                    temperature=args.watch_temperature,
                    force_move=args.watch_force_move,
                    episodes=args.watch_episodes,
                    model_lock=model_lock,
                    reload_secs=args.reload_secs,
                )
                if not running:
                    break

            print("Self-play starting...")
            sys.stdout.flush()
            run_selfplay(
                model,
                device,
                cfg,
                episodes=args.selfplay_episodes,
                sims=args.selfplay_sims,
                c_puct=args.c_puct,
                temp=args.selfplay_temp,
                dt_ms=args.selfplay_dt_ms,
                max_steps=args.selfplay_max_steps,
                seed=args.seed,
            )

            print("Training...")
            sys.stdout.flush()
            train_model(args.train_epochs, args.train_batch, args.no_gating)

            # reload best model after training
            model = AZNet(input_shape, env.action_count()).to(device)
            if os.path.exists(MODEL_PATH):
                state = torch.load(MODEL_PATH, map_location=device)
                model.load_state_dict(state)
            model.eval()

            print("Watching current best model...")
            sys.stdout.flush()
            running = play_visual(
                model,
                device,
                cfg,
                fps=args.watch_fps,
                dt_ms=args.watch_dt_ms,
                mcts_sims=args.watch_mcts_sims,
                c_puct=args.c_puct,
                max_steps=args.watch_max_steps,
                hold_sec=args.watch_hold_sec,
                temperature=args.watch_temperature,
                force_move=args.watch_force_move,
                episodes=args.watch_episodes,
                model_lock=model_lock,
                reload_secs=args.reload_secs,
            )

            if args.cycles > 0 and cycle >= args.cycles:
                break

    pygame.quit()


if __name__ == "__main__":
    main()
