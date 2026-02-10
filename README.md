# MinePac

A Pac‑Man‑inspired maze game with building, bombs, destructible walls, and an image‑reveal mechanic. Built with Python + Pygame.

**Highlights**
- Place and erase blocks in the maze.
- Bombs that clear a 6×6 area after a delay.
- Cyan power ball temporarily lets you break walls by pushing into them.
- Multi‑level progression by reaching a score threshold and trapping yourself.
- Image reveal: clearing tiles uncovers a hidden picture.

**Controls**
| Action | Key |
| --- | --- |
| Move | Arrow keys (hold to move) |
| Place block | `C` |
| Erase block | `Space` |
| Drop bomb | `V` |
| Pause | `P` |
| Quit | `Esc` |

**Requirements**
- Python 3.10+
- `pygame`

**Install**
```bash
pip install pygame
```

**Run**
```bash
python main.py
```

**Reveal Image**
Place your image at:
```
assets/reveal.jpg
```
As you clear tiles, the image will appear in 10×10 tile blocks (configurable).

**Configuration**
Edit `config.json` to tune gameplay:
- `tile_size`: size of each grid tile in pixels
- `cyan_effect_ms`: duration of cyan power in ms
- `bomb_timer_ms`: bomb fuse time in ms
- `bomb_size`: bomb blast size (tiles)
- `reveal_block`: image reveal block size (tiles)
- `ghosts_first_level`: number of ghosts at level 1
- `ghosts_per_level`: additional ghosts per level
- `ghost_speed_multiplier`: speed multiplier per level
- `max_bombs_per_life`: bombs available per life
- `points_per_level`: score required per level

**Records**
The highest score and level are stored in:
```
record.json
```

**Imitation Learning (AI Agent)**
The project includes a simple imitation learning pipeline that learns from screen input and your keyboard actions.

**AI Requirements**
- `mss` (screen capture)
- `pynput` (keyboard input/output)
- `pyobjc-framework-Quartz` (macOS window capture)
- `pillow` (image preprocessing)
- `torch` (training/inference)

**Install AI Dependencies**
```bash
pip install mss pynput pyobjc-framework-Quartz pillow torch
```

**1) Collect Data**
```bash
python collect_data.py
```
This records frames and your key presses into `.npz` files under `data/`.
You can run it multiple times; training will merge all `.npz` files automatically.

**2) Train**
```bash
python train_model.py
```
This produces `model.pth` and `model_meta.json`.

**3) Run the Agent**
```bash
python ai_agent.py
```
The agent reads the game window image and presses keys globally.
Click the game window to focus it before running.

**4) Self-Training Loop (Optional)**
This project includes a self‑improving loop:
- the agent plays the game
- when the score increases, it saves the recent frames + actions
- it periodically fine‑tunes the model and resumes play

Run it like this (default: retrain after ~2000 new samples):
```bash
python self_train_agent.py
```
This uses `score.json` written by the game to detect score changes.

**Self-Training Log**
Each retrain cycle writes a line to:
```
self_train_log.jsonl
```
Each line records the version and the max score reached by that version, so you can plot improvement later.

**Plot Progress**
```bash
pip install matplotlib
python plot_progress.py
```
This produces `self_train_progress.png`.

**Start from Scratch**
To start fresh, delete:
- `model.pth`
- `model_meta.json`
- `data/*.npz`
- `self_train_log.jsonl`

You can also force a random-start session:
```bash
PAC_START_RANDOM=1 python self_train_agent.py
```

**Notes**
- If the agent cannot control the game on macOS, enable Accessibility for your terminal or IDE.
- On Apple Silicon, PyTorch can use the `mps` backend for faster training/inference.
- The agent only triggers `C` and `Space` when a movement key is active (to match game rules).
- The AI is only as good as the data. Record more varied play for better results.

**Build Windows .exe**
Build on Windows using PyInstaller:
```bash
pip install pyinstaller pygame
pyinstaller --onefile --windowed --add-data "assets;assets" main.py
```
The executable will be in `dist/main.exe`.

**License**
MIT. See `LICENSE`.
