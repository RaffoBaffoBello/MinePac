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

**Build Windows .exe**
Build on Windows using PyInstaller:
```bash
pip install pyinstaller pygame
pyinstaller --onefile --windowed --add-data "assets;assets" main.py
```
The executable will be in `dist/main.exe`.

**License**
Add your preferred license (MIT/Apache/etc.).
