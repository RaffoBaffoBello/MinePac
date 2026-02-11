# MinePac

MinePac is a Pac-Man-style maze game built with Python + Pygame, plus two learning pipelines:
- a screen-based RL agent (`rl_train.py`, `rl_agent.py`)
- a headless AlphaZero-style pipeline (`az_*` scripts)

## What Is In This Repo
- `main.py`: playable game
- `config.json`: runtime game configuration
- `assets/reveal.jpg`: optional reveal image
- `pac_env.py`: headless environment used by AlphaZero scripts
- `rl_train.py`, `rl_agent.py`, `plot_rl_progress.py`: RL pipeline
- `az_selfplay.py`, `az_train.py`, `az_eval.py`, `az_play.py`, `az_loop.py`, `az_plot.py`: AlphaZero-style pipeline

## Requirements
- Python 3.10+
- macOS is recommended for the screen-control scripts (`Quartz`, global key input)

Install base game:
```bash
pip install pygame
```

Install AI dependencies:
```bash
pip install torch numpy pillow mss pynput pyobjc-framework-Quartz matplotlib
```

## Run The Game
```bash
python main.py
```

## Controls
- `Arrow keys` (hold): move
- `C`: place a block in facing direction
- `Space`: erase a block in facing direction (with dig timer)
- `V`: drop normal bomb
- `B`: drop nuke bomb (limited per life)
- `P`: pause/resume
- `Esc`: quit
- `Y` / `N` at `GAME OVER`: continue / restart from level 1

## Current Gameplay Rules
- Start with 3 lives.
- `V` bombs use the configured bomb count per life.
- `B` nukes use a separate configured count per life.
- Cyan power temporarily enables block-breaking by contact.
- Level-up requires both:
  - reaching the score threshold for next level
  - trapping yourself (blocked on all four sides)
- `You Win!` appears when all reveal blocks are cleared, then auto-restarts to level 1 after 10 seconds.
- High score and level are persisted.

## Configuration (`config.json`)
Current keys used by the game:
- `tile_size`
- `cyan_effect_ms`
- `bomb_timer_ms`
- `bomb_size`
- `reveal_block`
- `ghosts_first_level`
- `ghosts_per_level`
- `ghost_speed_multiplier`
- `max_bombs_per_life`
- `max_nukes_per_life`
- `points_per_level`

## Persistent Files
- `record.json`: best score + best level
- `score.json`: live state written by `main.py` (used by RL scripts)

## RL Pipeline (Screen-Based)
This pipeline reads the game window, sends global key events, and learns a DQN from score/life changes.

### Train
1. Start the game first:
```bash
python main.py
```
2. In another terminal:
```bash
python rl_train.py
```

Behavior:
- auto-resumes from `rl_checkpoint.pt` if present
- autosaves checkpoint and metrics every 5 minutes
- writes metrics to `rl_log.jsonl`

### Run trained RL policy
```bash
python rl_agent.py
```

### Plot RL metrics
```bash
python plot_rl_progress.py
```
Output: `rl_progress.png`

### RL window matching (optional)
You can override target window matching:
- `PAC_WINDOW_TITLE`
- `PAC_WINDOW_OWNER`

Example:
```bash
PAC_WINDOW_TITLE="VideoLeo Pac-Maze" PAC_WINDOW_OWNER="Python" python rl_train.py
```

## AlphaZero-Style Pipeline (Headless Self-Play + MCTS)
This pipeline does not use screen capture for training. It uses `pac_env.py` directly.

### 1) Generate self-play data
```bash
python az_selfplay.py --episodes 100 --sims 64
```
Output files: `data/az/az_*.npz`
Log file: `az_log.jsonl`

### 2) Train policy/value model
```bash
python az_train.py --epochs 10 --batch 64
```
Outputs:
- candidate: `az_candidate.pt`
- best model: `az_model.pt` (promoted by gating)

### 3) Evaluate best vs candidate (optional)
```bash
python az_eval.py --best az_model.pt --candidate az_candidate.pt --games 12 --sims 64
```

### 4) Watch model play
```bash
python az_play.py
```
Useful options:
```bash
python az_play.py --mcts-sims 32
python az_play.py --temperature 1.2 --force-move
```

### Plot AlphaZero progress
```bash
python az_plot.py
```
Output: `az_progress.png`

## Continuous Loop (`az_loop.py`)
`az_loop.py` can combine self-play, training, and watching.

Sequential loop:
```bash
python az_loop.py --cycles 0
```
(`--cycles 0` means infinite)

Train while watching (background training + hot reload model in viewer):
```bash
python az_loop.py --train-while-watching --watch-episodes 0
```

Useful loop options:
- self-play: `--selfplay-episodes`, `--selfplay-sims`, `--selfplay-temp`, `--selfplay-max-steps`
- score-shaped targets: `--selfplay-score-scale`, `--selfplay-death-penalty`, `--selfplay-temp-final`, `--selfplay-temp-decay-steps`
- training: `--train-epochs`, `--train-batch`, `--no-gating`
- watch: `--watch-mcts-sims`, `--watch-temperature`, `--watch-force-move`, `--watch-fps`, `--reload-secs`

## Overnight Training Example (AlphaZero)
```bash
python az_selfplay.py --episodes 2000 --sims 32 --temp 1.0 --temp-final 0.3 --temp-decay-steps 80 --score-scale 200 --death-penalty 50
python az_train.py --epochs 20 --batch 64
python az_play.py --mcts-sims 32
```

## Troubleshooting
- If RL scripts cannot control keys on macOS, enable Accessibility permissions for your terminal/IDE.
- If RL scripts cannot find the game window, keep the game visible and set `PAC_WINDOW_TITLE` / `PAC_WINDOW_OWNER`.
- On Apple Silicon, PyTorch uses MPS when available.

## License
MIT. See `LICENSE`.
