import sys
import os
import json
import random
import math
from collections import deque
import pygame


TILE = 24
BOMB_COLOR = (255, 200, 60)
FPS = 60

BLACK = (10, 10, 15)
BLUE = (40, 60, 200)
YELLOW = (250, 215, 70)
WHITE = (240, 240, 240)
RED = (220, 60, 60)
PINK = (255, 105, 180)
CYAN = (80, 220, 220)
ORANGE = (255, 165, 60)

GHOST_COLORS = [RED, PINK, CYAN, ORANGE]
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

BASE_GHOST_SPEED = 2
MAX_GHOST_SPEED = 6

DEFAULT_CONFIG = {
    "cyan_effect_ms": 40000,
    "bomb_timer_ms": 4000,
    "bomb_size": 6,
    "reveal_block": 10,
    "ghosts_first_level": 3,
    "ghosts_per_level": 1,
    "ghost_speed_multiplier": 1.02,
    "max_bombs_per_life": 10,
    "points_per_level": 1000
}

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
RECORD_PATH = os.path.join(os.path.dirname(__file__), "record.json")


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


CONFIG = load_config()
BOMB_TIMER_MS = int(CONFIG["bomb_timer_ms"])
REVEAL_BLOCK = int(CONFIG["reveal_block"])
BOMB_SIZE = int(CONFIG["bomb_size"])
CYAN_EFFECT_MS = int(CONFIG["cyan_effect_ms"])
GHOSTS_FIRST_LEVEL = int(CONFIG["ghosts_first_level"])
GHOSTS_PER_LEVEL = int(CONFIG["ghosts_per_level"])
GHOST_SPEED_MULTIPLIER = float(CONFIG["ghost_speed_multiplier"])
MAX_BOMBS_PER_LIFE = int(CONFIG["max_bombs_per_life"])
POINTS_PER_LEVEL = int(CONFIG["points_per_level"])


def load_record():
    try:
        with open(RECORD_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                score = int(data.get("score", 0))
                level = int(data.get("level", 0))
                return max(0, score), max(0, level)
    except FileNotFoundError:
        pass
    except json.JSONDecodeError:
        pass
    return 0, 0


def save_record(score, level):
    with open(RECORD_PATH, "w", encoding="utf-8") as f:
        json.dump({"score": int(score), "level": int(level)}, f)

BASE_MAZE = [
    "###########################################",
    "#.#.#....G..........#...........G.....#.#.#",
    "##........####.#####.#.#####.####........##",
    "#....c...o...........#...........o...c....#",
    "###.......###.#####.###.#####.###.......###",
    "#.#.#.......#................#........#.#.#",
    "##.......##.#.###.#####.###.#.####.......##",
    "#..........G..#...#...#...#..G..#.........#",
    "###.......###.###.#.###.###.###.#.......###",
    "#.#.#.....#.....#.#...#.#.....#.#.....#.#.#",
    "##........#.###.#.###.#.#.###.#.#........##",
    "#.........#...#.#.....#.#...#.#...........#",
    "###.......###.#.#####.#.###.#.###.......###",
    "#.#.#.........#...#...#.....#.........#.#.#",
    "##.......####.###.#.###.###.###.##.......##",
    "#.................#.....#.................#",
    "###.......###.###.#.###.#.###.###.......###",
    "#.#.#....o..#.....#.....#.....#..o....#.#.#",
    "##.......##.#.#.#####.#####.#.#.##.......##",
    "#.............#...#...#...#.....#.........#",
    "###.......#########.#.#.#########.......###",
    "#.#.#...............#P#...............#.#.#",
    "##........###.#####.#.#.#####.###........##",
    "#...........#.................#...........#",
    "###......##.#.###.#####.###.#.####......###",
    "#.#.#.........#...#...#...#.....#.....#.#.#",
    "##........###.###.#.###.###.###.#........##",
    "#....c....#.....#.#...#.#.....#.#....c....#",
    "###.......#####.#.###.#.#####.#.#.......###",
    "#.#.#...............#.................#.#.#",
    "###########################################",
]

GRID_W, GRID_H = len(BASE_MAZE[0]), len(BASE_MAZE)
WIDTH, HEIGHT = GRID_W * TILE, GRID_H * TILE


class Player:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.dir = (0, 0)
        self.speed = 4

    def update(self, walls, desired_dir, break_blocks, dig_timers, revealed, reveal_counts, block_required):
        # Turn buffering and generous snap at intersections.
        cx = (self.x // TILE) * TILE + TILE // 2
        cy = (self.y // TILE) * TILE + TILE // 2
        px = self.x + TILE // 2
        py = self.y + TILE // 2
        snap = 14

        # If we're stopped, snap fully to the current tile center to avoid off-grid stuck.
        if self.dir == (0, 0):
            tx = (self.x + TILE // 2) // TILE
            ty = (self.y + TILE // 2) // TILE
            self.x = tx * TILE
            self.y = ty * TILE
            cx = (self.x // TILE) * TILE + TILE // 2
            cy = (self.y // TILE) * TILE + TILE // 2
            px = self.x + TILE // 2
            py = self.y + TILE // 2

        def can_move(d):
            nx = self.x + d[0] * self.speed
            ny = self.y + d[1] * self.speed
            return not hits_wall(nx, ny, walls)

        def record_reveal(tx, ty):
            key = (tx // REVEAL_BLOCK, ty // REVEAL_BLOCK)
            if key not in revealed:
                reveal_counts[key] = reveal_counts.get(key, 0) + 1
                if reveal_counts[key] >= block_required.get(key, REVEAL_BLOCK * REVEAL_BLOCK):
                    revealed.add(key)

        if desired_dir != (0, 0) and desired_dir != self.dir:
            # If close enough to centerline for the turn, snap then turn.
            if desired_dir[0] != 0 and abs(py - cy) <= snap:
                self.y = cy - TILE // 2
                if can_move(desired_dir):
                    self.dir = desired_dir
            elif desired_dir[1] != 0 and abs(px - cx) <= snap:
                self.x = cx - TILE // 2
                if can_move(desired_dir):
                    self.dir = desired_dir

        # If we're stopped and want to go into a wall, allow block breaking.
        if self.dir == (0, 0) and break_blocks and desired_dir != (0, 0):
            tx = (self.x + TILE // 2) // TILE + desired_dir[0]
            ty = (self.y + TILE // 2) // TILE + desired_dir[1]
            if (tx, ty) in walls:
                walls.remove((tx, ty))
                record_reveal(tx, ty)
                if (tx, ty) in dig_timers:
                    del dig_timers[(tx, ty)]
                if can_move(desired_dir):
                    self.dir = desired_dir

        # Keep movement aligned while going straight.
        if self.dir[0] != 0 and abs(py - cy) <= snap:
            self.y = cy - TILE // 2
        if self.dir[1] != 0 and abs(px - cx) <= snap:
            self.x = cx - TILE // 2

        nx, ny = self.x + self.dir[0] * self.speed, self.y + self.dir[1] * self.speed
        if not hits_wall(nx, ny, walls):
            self.x, self.y = nx, ny
        else:
            if break_blocks and self.dir != (0, 0):
                tx = (self.x + TILE // 2) // TILE + self.dir[0]
                ty = (self.y + TILE // 2) // TILE + self.dir[1]
                if (tx, ty) in walls:
                    walls.remove((tx, ty))
                    record_reveal(tx, ty)
                    if (tx, ty) in dig_timers:
                        del dig_timers[(tx, ty)]
                    if not hits_wall(nx, ny, walls):
                        self.x, self.y = nx, ny
                        return
            self.dir = (0, 0)
            # Snap to tile center after a blocked move.
            tx = (self.x + TILE // 2) // TILE
            ty = (self.y + TILE // 2) // TILE
            self.x = tx * TILE
            self.y = ty * TILE

    def rect(self):
        return pygame.Rect(self.x, self.y, TILE, TILE)


def hits_wall(px, py, walls):
    left = px // TILE
    right = (px + TILE - 1) // TILE
    top = py // TILE
    bottom = (py + TILE - 1) // TILE
    for gx in (left, right):
        for gy in (top, bottom):
            if gx < 0 or gy < 0 or gx >= GRID_W or gy >= GRID_H:
                return True
            if (gx, gy) in walls:
                return True
    return False


def make_level_maze(level):
    maze = [list(row) for row in BASE_MAZE]
    h = len(maze)
    w = len(maze[0])
    p = None
    ghosts = set()
    for y in range(h):
        for x in range(w):
            if maze[y][x] == "P":
                p = (x, y)
            elif maze[y][x] == "G":
                ghosts.add((x, y))

    if p:
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = p[0] + dx, p[1] + dy
            if 0 <= nx < w and 0 <= ny < h and maze[ny][nx] == "#":
                maze[ny][nx] = "."

    if level <= 1:
        return ["".join(r) for r in maze]

    rng = random.Random()
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if (x, y) == p or (x, y) in ghosts:
                continue
            if p and abs(x - p[0]) <= 2 and abs(y - p[1]) <= 2:
                continue
            if maze[y][x] == ".":
                if rng.random() < 0.08:
                    maze[y][x] = "#"
            elif maze[y][x] == "#":
                if rng.random() < 0.04:
                    maze[y][x] = "."

    return ["".join(r) for r in maze]


def ghost_speed_for_level(level):
    return min(MAX_GHOST_SPEED, BASE_GHOST_SPEED * (GHOST_SPEED_MULTIPLIER ** (level - 1)))


def wall_color_for_level(level):
    return WALL_COLORS[(level - 1) % len(WALL_COLORS)]


def ghost_count_for_level(level):
    return max(1, GHOSTS_FIRST_LEVEL + (level - 1) * GHOSTS_PER_LEVEL)


def required_score_for_next_level(level):
    return level * POINTS_PER_LEVEL


def build_level(maze):
    walls = set()
    dots = set()
    power = set()
    cyan = set()
    player_start = None
    ghost_starts = []
    rows = len(maze)
    for y in range(rows):
        row = maze[y]
        for x in range(len(row)):
            ch = row[x]
            wx, wy = x * TILE, y * TILE
            if ch == "#":
                walls.add((x, y))
            elif ch == ".":
                dots.add((x, y))
            elif ch == "o":
                power.add((x, y))
            elif ch == "c":
                cyan.add((x, y))
            elif ch == "P":
                player_start = (wx, wy)
            elif ch == "G":
                ghost_starts.append((wx, wy))
    if player_start is None:
        player_start = (TILE, TILE)
    # Ensure player spawn is clear (no walls on or immediately adjacent).
    px = player_start[0] // TILE
    py = player_start[1] // TILE
    for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
        walls.discard((px + dx, py + dy))
    if not ghost_starts:
        ghost_starts = [
            (WIDTH // 2 - TILE, HEIGHT // 2 - TILE),
            (WIDTH // 2 + TILE, HEIGHT // 2 - TILE),
            (WIDTH // 2 - TILE, HEIGHT // 2 + TILE),
            (WIDTH // 2 + TILE, HEIGHT // 2 + TILE),
        ]
    return walls, dots, power, cyan, player_start, ghost_starts


class Ghost:
    def __init__(self, x, y, color, speed):
        self.x = x
        self.y = y
        self.color = color
        self.dir = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        self.speed = speed
        self.recent_tiles = deque(maxlen=6)
        self.alive = True

    def update(self, walls, player_pos, player_rect=None):
        if not self.alive:
            return False
        # Keep aligned to corridor centerlines for smoother turns.
        cx = (self.x // TILE) * TILE + TILE // 2
        cy = (self.y // TILE) * TILE + TILE // 2
        px = self.x + TILE // 2
        py = self.y + TILE // 2
        snap = 6
        if self.dir[0] != 0 and abs(py - cy) <= snap:
            self.y = cy - TILE // 2
        if self.dir[1] != 0 and abs(px - cx) <= snap:
            self.x = cx - TILE // 2

        # Choose a direction near tile centers to chase the player.
        if abs(px - cx) <= snap and abs(py - cy) <= snap:
            tx = (self.x + TILE // 2) // TILE
            ty = (self.y + TILE // 2) // TILE
            if not self.recent_tiles or self.recent_tiles[-1] != (tx, ty):
                self.recent_tiles.append((tx, ty))
            possible = []
            for d in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                if (tx + d[0], ty + d[1]) not in walls:
                    possible.append(d)

            if possible:
                ppx, ppy = player_pos
                def manhattan(d):
                    return abs((tx + d[0]) * TILE - ppx) + abs((ty + d[1]) * TILE - ppy)

                # Prefer non-reverse direction unless it's the only way.
                reverse = (-self.dir[0], -self.dir[1])
                candidates = [d for d in possible if d != reverse] or possible

                # Avoid immediately revisiting recent tiles when possible.
                def target_tile(d):
                    return (tx + d[0], ty + d[1])
                filtered = [d for d in candidates if target_tile(d) not in self.recent_tiles]
                choices = filtered or candidates

                # Add a bit of exploration to avoid oscillating in short paths.
                if len(choices) > 1 and random.random() < 0.25:
                    self.dir = random.choice(choices)
                else:
                    best = min(manhattan(d) for d in choices)
                    near_best = [d for d in choices if manhattan(d) <= best + TILE]
                    self.dir = random.choice(near_best) if near_best else min(choices, key=manhattan)
        else:
            # If not near a center and we're blocked, try to pick a new direction.
            nx, ny = self.x + self.dir[0] * self.speed, self.y + self.dir[1] * self.speed
            if hits_wall(nx, ny, walls):
                tx = (self.x + TILE // 2) // TILE
                ty = (self.y + TILE // 2) // TILE
                possible = []
                for d in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    if (tx + d[0], ty + d[1]) not in walls:
                        possible.append(d)
                if possible:
                    self.dir = random.choice(possible)

        # Move in small steps to avoid tunneling through the player.
        steps = max(1, int(math.ceil(self.speed)))
        step = self.speed / steps
        for _ in range(steps):
            nx, ny = self.x + self.dir[0] * step, self.y + self.dir[1] * step
            if hits_wall(nx, ny, walls):
                break
            self.x, self.y = nx, ny
            if player_rect and self.rect().colliderect(player_rect):
                return True
        return False

    def rect(self):
        return pygame.Rect(self.x, self.y, TILE, TILE)


def draw_ghost(screen, ghost):
    if not ghost.alive:
        return
    x = ghost.x + TILE // 2
    y = ghost.y + TILE // 2

    # Body (rectangle)
    body_rect = pygame.Rect(ghost.x + 2, ghost.y + 2, TILE - 4, TILE - 4)
    pygame.draw.rect(screen, ghost.color, body_rect, border_radius=3)

    # Eyes
    eye_offset_x = 5
    eye_offset_y = 3
    eye_radius = 4
    pupil_radius = 2
    pygame.draw.circle(screen, WHITE, (x - eye_offset_x, y - eye_offset_y), eye_radius)
    pygame.draw.circle(screen, WHITE, (x + eye_offset_x, y - eye_offset_y), eye_radius)
    pygame.draw.circle(screen, BLUE, (x - eye_offset_x + 1, y - eye_offset_y), pupil_radius)
    pygame.draw.circle(screen, BLUE, (x + eye_offset_x + 1, y - eye_offset_y), pupil_radius)


def draw(screen, walls, dots, power, cyan, player, ghosts, score, lives, bombs_left, record_score, record_level, dig_timers, wall_color, level, reveal_image, revealed, reveal_error, bombs, break_blocks):
    screen.fill(BLACK)
    if reveal_image:
        for (bx, by) in revealed:
            x0 = bx * REVEAL_BLOCK
            y0 = by * REVEAL_BLOCK
            w = min(REVEAL_BLOCK, GRID_W - x0)
            h = min(REVEAL_BLOCK, GRID_H - y0)
            screen.blit(
                reveal_image,
                (x0 * TILE, y0 * TILE),
                pygame.Rect(x0 * TILE, y0 * TILE, w * TILE, h * TILE),
            )
    for (x, y) in walls:
        if (x, y) in dig_timers:
            continue
        pygame.draw.rect(screen, wall_color, pygame.Rect(x * TILE, y * TILE, TILE, TILE))

    for (x, y), remaining in dig_timers.items():
        alpha = max(60, int(255 * (remaining / 500)))
        wall_surface = pygame.Surface((TILE, TILE), pygame.SRCALPHA)
        wall_surface.fill((wall_color[0], wall_color[1], wall_color[2], alpha))
        screen.blit(wall_surface, (x * TILE, y * TILE))

    for (x, y) in dots:
        pygame.draw.circle(screen, WHITE, (x * TILE + TILE // 2, y * TILE + TILE // 2), 3)

    for (x, y) in power:
        pygame.draw.circle(screen, WHITE, (x * TILE + TILE // 2, y * TILE + TILE // 2), 6)

    for (x, y) in cyan:
        pygame.draw.circle(screen, CYAN, (x * TILE + TILE // 2, y * TILE + TILE // 2), 6)

    for b in bombs:
        pygame.draw.circle(screen, BOMB_COLOR, (b["x"] * TILE + TILE // 2, b["y"] * TILE + TILE // 2), 5)

    # Pac-Man body (rectangle)
    body_rect = pygame.Rect(player.x + 2, player.y + 2, TILE - 4, TILE - 4)
    pac_color = CYAN if break_blocks else YELLOW
    pygame.draw.rect(screen, pac_color, body_rect, border_radius=4)

    # Big eyes
    eye_offset_x = 5
    eye_offset_y = 6
    eye_radius = 5
    pupil_radius = 2
    center = (player.x + TILE // 2, player.y + TILE // 2)
    pygame.draw.circle(screen, WHITE, (center[0] - eye_offset_x, center[1] - eye_offset_y), eye_radius)
    pygame.draw.circle(screen, WHITE, (center[0] + eye_offset_x, center[1] - eye_offset_y), eye_radius)
    pygame.draw.circle(screen, BLACK, (center[0] - eye_offset_x, center[1] - eye_offset_y), pupil_radius)
    pygame.draw.circle(screen, BLACK, (center[0] + eye_offset_x, center[1] - eye_offset_y), pupil_radius)

    # Smile
    mouth_rect = pygame.Rect(center[0] - 6, center[1] - 1, 12, 8)
    pygame.draw.arc(screen, BLACK, mouth_rect, 0, 3.14159, 3)
    for ghost in ghosts:
        draw_ghost(screen, ghost)

    font = pygame.font.SysFont("Arial", 20)
    hud = font.render(
        f"Score: {score:06d}  Lives: {lives}  Bombs: {bombs_left}  Level: {level}  Next: {required_score_for_next_level(level)}  |  Reach the points, then trap yourself to advance",
        True,
        WHITE,
    )
    screen.blit(hud, (10, HEIGHT - 24))
    if reveal_error:
        warn = font.render(reveal_error, True, WHITE)
        screen.blit(warn, (10, 2))
    else:
        help_text = font.render("C = place blocks  |  Space = erase blocks  |  V = drop bombs", True, WHITE)
        screen.blit(help_text, (10, 2))
        record_text = font.render(f"Record: {record_score:06d} - Level: {record_level}", True, WHITE)
        screen.blit(record_text, (10 + help_text.get_width() + 16, 2))


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("VideoLeo Pac-Maze")
    clock = pygame.time.Clock()

    arrow_state = {
        pygame.K_UP: False,
        pygame.K_DOWN: False,
        pygame.K_LEFT: False,
        pygame.K_RIGHT: False,
    }

    def get_input_dir():
        if arrow_state[pygame.K_UP]:
            return (0, -1)
        if arrow_state[pygame.K_DOWN]:
            return (0, 1)
        if arrow_state[pygame.K_LEFT]:
            return (-1, 0)
        if arrow_state[pygame.K_RIGHT]:
            return (1, 0)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            return (0, -1)
        if keys[pygame.K_DOWN]:
            return (0, 1)
        if keys[pygame.K_LEFT]:
            return (-1, 0)
        if keys[pygame.K_RIGHT]:
            return (1, 0)
        return (0, 0)

    def get_player_tile():
        return player.x // TILE, player.y // TILE

    def get_player_tile_center():
        return (player.x + TILE // 2) // TILE, (player.y + TILE // 2) // TILE

    def can_place_wall(tx, ty):
        tile_rect = pygame.Rect(tx * TILE, ty * TILE, TILE, TILE)
        if player.rect().colliderect(tile_rect):
            return False
        for g in ghosts:
            if g.alive and g.rect().colliderect(tile_rect):
                return False
        return True

    def snap_player_to_tile():
        tx, ty = get_player_tile_center()
        if (tx, ty) in walls:
            return False
        player.x = tx * TILE
        player.y = ty * TILE
        return True

    def attempt_dig(d):
        if d == (0, 0):
            return
        if not snap_player_to_tile():
            return
        px, py = get_player_tile()
        tx = px + d[0]
        ty = py + d[1]
        if (tx, ty) in walls and (tx, ty) not in dig_timers:
            dig_timers[(tx, ty)] = 500

    def attempt_place(d):
        if d == (0, 0):
            return False
        if not snap_player_to_tile():
            return False
        px, py = get_player_tile()
        tx = px + d[0]
        ty = py + d[1]
        if not (0 <= tx < GRID_W and 0 <= ty < GRID_H):
            return False
        if (tx, ty) in walls or (tx, ty) in dig_timers:
            return False
        if not can_place_wall(tx, ty):
            return False
        dots.discard((tx, ty))
        power.discard((tx, ty))
        cyan.discard((tx, ty))
        walls.add((tx, ty))
        tile_rect = pygame.Rect(tx * TILE, ty * TILE, TILE, TILE)
        if hits_wall(player.x, player.y, walls):
            walls.remove((tx, ty))
            return False
        for g in ghosts:
            if g.rect().colliderect(tile_rect):
                walls.remove((tx, ty))
                return False
        return True

    def add_extra_ghosts(maze, base_starts, extra_count):
        if extra_count <= 0:
            return list(base_starts)
        h = len(maze)
        w = len(maze[0])
        p = None
        base_tiles = {(x // TILE, y // TILE) for (x, y) in base_starts}
        open_cells = []
        for y in range(h):
            for x in range(w):
                ch = maze[y][x]
                if ch == "P":
                    p = (x, y)
                if ch in ".oc":
                    if (x, y) not in base_tiles:
                        open_cells.append((x, y))
        if p:
            open_cells = [c for c in open_cells if abs(c[0] - p[0]) + abs(c[1] - p[1]) > 6]
        random.shuffle(open_cells)
        extras = open_cells[:extra_count]
        return list(base_starts) + [(x * TILE, y * TILE) for (x, y) in extras]

    def is_surrounded():
        px = (player.x + TILE // 2) // TILE
        py = (player.y + TILE // 2) // TILE
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = px + dx, py + dy
            if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                if (nx, ny) not in walls:
                    return False
        return True

    def build_block_requirements():
        req = {}
        bx_max = (GRID_W + REVEAL_BLOCK - 1) // REVEAL_BLOCK
        by_max = (GRID_H + REVEAL_BLOCK - 1) // REVEAL_BLOCK
        for by in range(by_max):
            for bx in range(bx_max):
                x0 = bx * REVEAL_BLOCK
                y0 = by * REVEAL_BLOCK
                w = min(REVEAL_BLOCK, GRID_W - x0)
                h = min(REVEAL_BLOCK, GRID_H - y0)
                req[(bx, by)] = w * h
        return req

    def register_clear(x, y):
        key = (x // REVEAL_BLOCK, y // REVEAL_BLOCK)
        if key in revealed:
            return
        reveal_counts[key] = reveal_counts.get(key, 0) + 1
        if reveal_counts[key] >= block_required.get(key, REVEAL_BLOCK * REVEAL_BLOCK):
            revealed.add(key)

    def reset_ghosts():
        for g, start in zip(ghosts, ghost_starts):
            if g.alive:
                g.x, g.y = start
                g.dir = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])

    def apply_player_hit():
        nonlocal lives, game_over, invuln_timer, break_blocks, break_timer, bombs_left
        lives -= 1
        player.x, player.y = player_start
        player.dir = (0, 0)
        break_blocks = False
        break_timer = 0
        bombs_left = MAX_BOMBS_PER_LIFE
        reset_ghosts()
        invuln_timer = 2000
        if lives <= 0:
            game_over = True

    def init_reveal_state():
        rc = {}
        rev = set()
        for y in range(GRID_H):
            for x in range(GRID_W):
                if (x, y) in walls:
                    continue
                if (x, y) in dots or (x, y) in power or (x, y) in cyan:
                    continue
                key = (x // REVEAL_BLOCK, y // REVEAL_BLOCK)
                rc[key] = rc.get(key, 0) + 1
        for key, count in rc.items():
            if count >= block_required.get(key, REVEAL_BLOCK * REVEAL_BLOCK):
                rev.add(key)
        return rev, rc

    block_required = build_block_requirements()

    level = 1
    wall_color = wall_color_for_level(level)
    current_maze = make_level_maze(level)
    walls, dots, power, cyan, player_start, ghost_starts_all = build_level(current_maze)
    base_starts = ghost_starts_all[:1]
    ghost_starts = add_extra_ghosts(current_maze, base_starts, ghost_count_for_level(level) - len(base_starts))
    player = Player(*player_start)
    ghosts = [
        Ghost(x, y, GHOST_COLORS[i % len(GHOST_COLORS)], ghost_speed_for_level(level))
        for i, (x, y) in enumerate(ghost_starts)
    ]
    score = 0
    lives = 3
    power_timer = 0
    invuln_timer = 0
    dig_timers = {}
    place_cooldown = 0
    dig_requested = False
    dig_requested_time = 0
    place_requested_time = 0
    break_blocks = False
    break_timer = 0
    bombs = []
    bomb_positions = set()
    bombs_left = MAX_BOMBS_PER_LIFE
    revealed, reveal_counts = init_reveal_state()
    record_score, record_level = load_record()
    reveal_image = None
    reveal_error = ""
    assets_dir = os.path.join(os.path.dirname(__file__), "assets")
    reveal_path = os.path.join(assets_dir, "reveal.jpg")
    if os.path.exists(reveal_path):
        try:
            reveal_image = pygame.image.load(reveal_path).convert()
            reveal_image = pygame.transform.smoothscale(reveal_image, (WIDTH, HEIGHT))
        except pygame.error:
            reveal_image = None
            reveal_error = "Reveal image failed to load (assets/reveal.jpg)"
    else:
        reveal_error = "Reveal image missing (assets/reveal.jpg)"
    game_over = False
    game_over_choice = None

    running = True
    while running:
        dt = clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in arrow_state:
                    arrow_state[event.key] = True
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif game_over:
                    if event.key == pygame.K_y:
                        game_over_choice = "yes"
                    elif event.key == pygame.K_n:
                        game_over_choice = "no"
                else:
                    if event.key == pygame.K_SPACE:
                        dig_requested = True
                        dig_requested_time = 200
                    elif event.key == pygame.K_c:
                        place_requested_time = 200
                    elif event.key == pygame.K_v:
                        bx = (player.x + TILE // 2) // TILE
                        by = (player.y + TILE // 2) // TILE
                        if bombs_left > 0 and (bx, by) not in bomb_positions:
                            bombs.append({"x": bx, "y": by, "timer": BOMB_TIMER_MS})
                            bomb_positions.add((bx, by))
                            bombs_left -= 1
            elif event.type == pygame.KEYUP:
                if event.key in arrow_state:
                    arrow_state[event.key] = False

        if game_over:
            draw(screen, walls, dots, power, cyan, player, ghosts, score, lives, bombs_left, record_score, record_level, dig_timers, wall_color, level, reveal_image, revealed, reveal_error, bombs, break_blocks)
            font = pygame.font.SysFont("Arial", 28)
            msg = font.render("GAME OVER - Continue? (Y/N)", True, WHITE)
            screen.blit(msg, (WIDTH // 2 - msg.get_width() // 2, HEIGHT // 2 - 20))
            pygame.display.flip()
            if game_over_choice == "yes":
                lives = 3
                invuln_timer = 2000
                break_blocks = False
                break_timer = 0
                bombs = []
                bomb_positions = set()
                bombs_left = MAX_BOMBS_PER_LIFE
                game_over = False
                game_over_choice = None
            elif game_over_choice == "no":
                level = 1
                wall_color = wall_color_for_level(level)
                current_maze = make_level_maze(level)
                walls, dots, power, cyan, player_start, ghost_starts_all = build_level(current_maze)
                base_starts = ghost_starts_all[:1]
                ghost_starts = add_extra_ghosts(current_maze, base_starts, ghost_count_for_level(level) - len(base_starts))
                player = Player(*player_start)
                ghosts = [
                    Ghost(x, y, GHOST_COLORS[i % len(GHOST_COLORS)], ghost_speed_for_level(level))
                    for i, (x, y) in enumerate(ghost_starts)
                ]
                score = 0
                lives = 3
                power_timer = 0
                invuln_timer = 0
                dig_timers = {}
                place_cooldown = 0
                dig_requested = False
                dig_requested_time = 0
                place_requested_time = 0
                break_blocks = False
                break_timer = 0
                revealed, reveal_counts = init_reveal_state()
                bombs = []
                bomb_positions = set()
                bombs_left = MAX_BOMBS_PER_LIFE
                game_over = False
                game_over_choice = None
            continue

        desired_dir = get_input_dir()
        if dig_requested_time > 0:
            dig_requested_time -= dt
            if dig_requested_time <= 0:
                dig_requested = False
        if place_requested_time > 0:
            place_requested_time -= dt

        if dig_requested and desired_dir != (0, 0):
            attempt_dig(desired_dir)
            dig_requested = False

        # Allow slight timing mismatch: if key pressed first, accept direction shortly after.
        if dig_requested_time > 0 and desired_dir != (0, 0):
            attempt_dig(desired_dir)
            dig_requested_time = 0
            dig_requested = False

        if place_requested_time > 0 and desired_dir != (0, 0) and place_cooldown <= 0:
            if attempt_place(desired_dir):
                place_cooldown = 200
            place_requested_time = 0

        if desired_dir == (0, 0):
            player.dir = (0, 0)

        player.update(walls, desired_dir, break_blocks, dig_timers, revealed, reveal_counts, block_required)
        hit_ghost = None
        player_rect = player.rect()
        for ghost in ghosts:
            if ghost.update(walls, (player.x, player.y), player_rect):
                hit_ghost = ghost
                break

        # Update bombs and explode when timers end.
        if bombs:
            exploded = []
            for b in bombs:
                b["timer"] -= dt
                if b["timer"] <= 0:
                    exploded.append(b)
            player_hit_by_bomb = False
            for b in exploded:
                bombs.remove(b)
                bomb_positions.discard((b["x"], b["y"]))
                start_x = b["x"] - (BOMB_SIZE // 2)
                start_y = b["y"] - (BOMB_SIZE // 2)
                end_x = start_x + BOMB_SIZE
                end_y = start_y + BOMB_SIZE
                for y in range(start_y, end_y):
                    for x in range(start_x, end_x):
                        if x < 0 or y < 0 or x >= GRID_W or y >= GRID_H:
                            continue
                        if (x, y) in walls:
                            walls.remove((x, y))
                            register_clear(x, y)
                        if (x, y) in dots:
                            dots.remove((x, y))
                            register_clear(x, y)
                        if (x, y) in power:
                            power.remove((x, y))
                            register_clear(x, y)
                        if (x, y) in cyan:
                            cyan.remove((x, y))
                            register_clear(x, y)
                        if (x, y) in dig_timers:
                            del dig_timers[(x, y)]

                # Kill ghosts in the blast.
                for g, start in zip(ghosts, ghost_starts):
                    if not g.alive:
                        continue
                    gx = (g.x + TILE // 2) // TILE
                    gy = (g.y + TILE // 2) // TILE
                    if start_x <= gx < end_x and start_y <= gy < end_y:
                        g.alive = False

                # Kill Pac-Man if in blast.
                if not player_hit_by_bomb and invuln_timer <= 0:
                    px = (player.x + TILE // 2) // TILE
                    py = (player.y + TILE // 2) // TILE
                    if start_x <= px < end_x and start_y <= py < end_y:
                        apply_player_hit()
                        player_hit_by_bomb = True

        grid_x, grid_y = player.x // TILE, player.y // TILE
        if (grid_x, grid_y) in dots:
            dots.remove((grid_x, grid_y))
            score += 10
            register_clear(grid_x, grid_y)
        if (grid_x, grid_y) in power:
            power.remove((grid_x, grid_y))
            score += 50
            power_timer = 3000
            register_clear(grid_x, grid_y)
        if (grid_x, grid_y) in cyan:
            cyan.remove((grid_x, grid_y))
            score += 50
            break_blocks = True
            break_timer = CYAN_EFFECT_MS
            register_clear(grid_x, grid_y)

        if score > record_score:
            record_score = score
            save_record(record_score, record_level)

        if power_timer > 0:
            power_timer -= dt
        if invuln_timer > 0:
            invuln_timer -= dt
        if break_timer > 0:
            break_timer -= dt
            if break_timer <= 0:
                break_blocks = False
        if place_cooldown > 0:
            place_cooldown -= dt
        if dig_timers:
            done = []
            for (x, y), remaining in list(dig_timers.items()):
                remaining -= dt
                if remaining <= 0:
                    done.append((x, y))
                else:
                    dig_timers[(x, y)] = remaining
            for pos in done:
                if pos in walls:
                    walls.remove(pos)
                    register_clear(pos[0], pos[1])
                if pos in dig_timers:
                    del dig_timers[pos]

        if hit_ghost is None:
            for ghost in ghosts:
                if ghost.alive and player.rect().colliderect(ghost.rect()):
                    hit_ghost = ghost
                    break

        if hit_ghost:
            if power_timer > 0:
                score += 200
                hit_ghost.x, hit_ghost.y = random.choice(ghost_starts)
            elif invuln_timer <= 0:
                apply_player_hit()

        if score >= required_score_for_next_level(level) and is_surrounded():
            level += 1
            wall_color = wall_color_for_level(level)
            current_maze = make_level_maze(level)
            walls, dots, power, cyan, player_start, ghost_starts_all = build_level(current_maze)
            base_starts = ghost_starts_all[:1]
            ghost_starts = add_extra_ghosts(current_maze, base_starts, ghost_count_for_level(level) - len(base_starts))
            player = Player(*player_start)
            ghosts = [
                Ghost(x, y, GHOST_COLORS[i % len(GHOST_COLORS)], ghost_speed_for_level(level))
                for i, (x, y) in enumerate(ghost_starts)
            ]
            power_timer = 0
            invuln_timer = 0
            dig_timers = {}
            place_cooldown = 0
            dig_requested = False
            dig_requested_time = 0
            place_requested_time = 0
            revealed, reveal_counts = init_reveal_state()
            bombs = []
            bomb_positions = set()
            bombs_left = MAX_BOMBS_PER_LIFE
            if level > record_level:
                record_level = level
                save_record(record_score, record_level)
            continue

        if not dots and not power:
            running = False

        draw(screen, walls, dots, power, cyan, player, ghosts, score, lives, bombs_left, record_score, record_level, dig_timers, wall_color, level, reveal_image, revealed, reveal_error, bombs, break_blocks)
        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
