import copy
import math
import random
from collections import deque


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

BASE_GHOST_SPEED = 2
MAX_GHOST_SPEED = 6
MOVE_DIRS = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
ACTIONS = ["none", "dig", "place", "bomb"]
ACTION_SPACE = [(move, act) for move in MOVE_DIRS for act in ACTIONS]
ACTION_INDEX = {pair: idx for idx, pair in enumerate(ACTION_SPACE)}


def rects_intersect(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by


class Player:
    def __init__(self, x, y, speed=4):
        self.x = x
        self.y = y
        self.dir = (0, 0)
        self.speed = speed

    def rect(self, tile):
        return (self.x, self.y, tile, tile)

    def update(self, env, desired_dir):
        tile = env.tile
        walls = env.walls
        break_blocks = env.break_blocks
        dig_timers = env.dig_timers
        revealed = env.revealed
        reveal_counts = env.reveal_counts
        block_required = env.block_required

        cx = (self.x // tile) * tile + tile // 2
        cy = (self.y // tile) * tile + tile // 2
        px = self.x + tile // 2
        py = self.y + tile // 2
        snap = 14

        if self.dir == (0, 0):
            tx = (self.x + tile // 2) // tile
            ty = (self.y + tile // 2) // tile
            self.x = tx * tile
            self.y = ty * tile
            cx = (self.x // tile) * tile + tile // 2
            cy = (self.y // tile) * tile + tile // 2
            px = self.x + tile // 2
            py = self.y + tile // 2

        def can_move(d):
            nx = self.x + d[0] * self.speed
            ny = self.y + d[1] * self.speed
            return not env.hits_wall(nx, ny, walls)

        def record_reveal(tx, ty):
            key = (tx // env.reveal_block, ty // env.reveal_block)
            if key not in revealed:
                reveal_counts[key] = reveal_counts.get(key, 0) + 1
                if reveal_counts[key] >= block_required.get(key, env.reveal_block * env.reveal_block):
                    revealed.add(key)

        if desired_dir != (0, 0) and desired_dir != self.dir:
            if desired_dir[0] != 0 and abs(py - cy) <= snap:
                self.y = cy - tile // 2
                if can_move(desired_dir):
                    self.dir = desired_dir
            elif desired_dir[1] != 0 and abs(px - cx) <= snap:
                self.x = cx - tile // 2
                if can_move(desired_dir):
                    self.dir = desired_dir

        if self.dir == (0, 0) and break_blocks and desired_dir != (0, 0):
            tx = (self.x + tile // 2) // tile + desired_dir[0]
            ty = (self.y + tile // 2) // tile + desired_dir[1]
            if (tx, ty) in walls:
                walls.remove((tx, ty))
                record_reveal(tx, ty)
                if (tx, ty) in dig_timers:
                    del dig_timers[(tx, ty)]
                if can_move(desired_dir):
                    self.dir = desired_dir

        if self.dir[0] != 0 and abs(py - cy) <= snap:
            self.y = cy - tile // 2
        if self.dir[1] != 0 and abs(px - cx) <= snap:
            self.x = cx - tile // 2

        nx, ny = self.x + self.dir[0] * self.speed, self.y + self.dir[1] * self.speed
        if not env.hits_wall(nx, ny, walls):
            self.x, self.y = nx, ny
        else:
            if break_blocks and self.dir != (0, 0):
                tx = (self.x + tile // 2) // tile + self.dir[0]
                ty = (self.y + tile // 2) // tile + self.dir[1]
                if (tx, ty) in walls:
                    walls.remove((tx, ty))
                    record_reveal(tx, ty)
                    if (tx, ty) in dig_timers:
                        del dig_timers[(tx, ty)]
                    if not env.hits_wall(nx, ny, walls):
                        self.x, self.y = nx, ny
                        return
            self.dir = (0, 0)
            tx = (self.x + tile // 2) // tile
            ty = (self.y + tile // 2) // tile
            self.x = tx * tile
            self.y = ty * tile


class Ghost:
    def __init__(self, x, y, color, speed, rng):
        self.x = x
        self.y = y
        self.color = color
        self.dir = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        self.speed = speed
        self.recent_tiles = deque(maxlen=6)
        self.alive = True

    def rect(self, tile):
        return (self.x, self.y, tile, tile)

    def update(self, env, player_pos, player_rect=None):
        if not self.alive:
            return False
        tile = env.tile
        walls = env.walls
        rng = env.rng
        cx = (self.x // tile) * tile + tile // 2
        cy = (self.y // tile) * tile + tile // 2
        px = self.x + tile // 2
        py = self.y + tile // 2
        snap = 6
        if self.dir[0] != 0 and abs(py - cy) <= snap:
            self.y = cy - tile // 2
        if self.dir[1] != 0 and abs(px - cx) <= snap:
            self.x = cx - tile // 2

        if abs(px - cx) <= snap and abs(py - cy) <= snap:
            tx = (self.x + tile // 2) // tile
            ty = (self.y + tile // 2) // tile
            if not self.recent_tiles or self.recent_tiles[-1] != (tx, ty):
                self.recent_tiles.append((tx, ty))
            possible = []
            for d in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                if (tx + d[0], ty + d[1]) not in walls:
                    possible.append(d)

            if possible:
                ppx, ppy = player_pos

                def manhattan(d):
                    return abs((tx + d[0]) * tile - ppx) + abs((ty + d[1]) * tile - ppy)

                reverse = (-self.dir[0], -self.dir[1])
                candidates = [d for d in possible if d != reverse] or possible

                def target_tile(d):
                    return (tx + d[0], ty + d[1])

                filtered = [d for d in candidates if target_tile(d) not in self.recent_tiles]
                choices = filtered or candidates

                if len(choices) > 1 and rng.random() < 0.25:
                    self.dir = rng.choice(choices)
                else:
                    best = min(manhattan(d) for d in choices)
                    near_best = [d for d in choices if manhattan(d) <= best + tile]
                    self.dir = rng.choice(near_best) if near_best else min(choices, key=manhattan)
        else:
            nx, ny = self.x + self.dir[0] * self.speed, self.y + self.dir[1] * self.speed
            if env.hits_wall(nx, ny, walls):
                tx = (self.x + tile // 2) // tile
                ty = (self.y + tile // 2) // tile
                possible = []
                for d in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    if (tx + d[0], ty + d[1]) not in walls:
                        possible.append(d)
                if possible:
                    self.dir = rng.choice(possible)

        steps = max(1, int(math.ceil(self.speed)))
        step = self.speed / steps
        for _ in range(steps):
            nx, ny = self.x + self.dir[0] * step, self.y + self.dir[1] * step
            if env.hits_wall(nx, ny, walls):
                break
            self.x, self.y = nx, ny
            if player_rect and rects_intersect(self.rect(tile), player_rect):
                return True
        return False


class PacEnv:
    def __init__(self, config, base_maze=None, seed=None):
        self.config = config
        self.tile = int(config["tile_size"])
        self.bomb_timer_ms = int(config["bomb_timer_ms"])
        self.reveal_block = int(config["reveal_block"])
        self.bomb_size = int(config["bomb_size"])
        self.cyan_effect_ms = int(config["cyan_effect_ms"])
        self.ghosts_first_level = int(config["ghosts_first_level"])
        self.ghosts_per_level = int(config["ghosts_per_level"])
        self.ghost_speed_multiplier = float(config["ghost_speed_multiplier"])
        self.max_bombs_per_life = int(config["max_bombs_per_life"])
        self.points_per_level = int(config["points_per_level"])

        self.base_maze = base_maze or BASE_MAZE
        self.grid_w = len(self.base_maze[0])
        self.grid_h = len(self.base_maze)
        self.rng = random.Random(seed)

        self.reset_all()

    def clone(self):
        cloned = copy.deepcopy(self)
        cloned.rng.setstate(self.rng.getstate())
        return cloned

    def legal_actions(self):
        return ACTION_SPACE

    def action_count(self):
        return len(ACTION_SPACE)

    def index_to_action(self, idx):
        return ACTION_SPACE[idx]

    def action_to_index(self, move_dir, action):
        return ACTION_INDEX[(move_dir, action)]

    def legal_action_mask(self):
        mask = [1] * len(ACTION_SPACE)
        px = int((self.player.x + self.tile // 2) // self.tile)
        py = int((self.player.y + self.tile // 2) // self.tile)
        for i, (move, act) in enumerate(ACTION_SPACE):
            if act == "bomb":
                if self.bombs_left <= 0 or (px, py) in self.bomb_positions:
                    mask[i] = 0
                continue
            if act in ("dig", "place") and move == (0, 0):
                mask[i] = 0
                continue
            if act == "dig":
                tx = px + move[0]
                ty = py + move[1]
                if (tx, ty) not in self.walls:
                    mask[i] = 0
            elif act == "place":
                tx = px + move[0]
                ty = py + move[1]
                if not (0 <= tx < self.grid_w and 0 <= ty < self.grid_h):
                    mask[i] = 0
                    continue
                if (tx, ty) in self.walls or (tx, ty) in self.dig_timers:
                    mask[i] = 0
                    continue
                if not self.can_place_wall(tx, ty):
                    mask[i] = 0
        return mask

    def hits_wall(self, px, py, walls):
        left = px // self.tile
        right = (px + self.tile - 1) // self.tile
        top = py // self.tile
        bottom = (py + self.tile - 1) // self.tile
        for gx in (left, right):
            for gy in (top, bottom):
                if gx < 0 or gy < 0 or gx >= self.grid_w or gy >= self.grid_h:
                    return True
                if (gx, gy) in walls:
                    return True
        return False

    def make_level_maze(self, level):
        maze = [list(row) for row in self.base_maze]
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

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if (x, y) == p or (x, y) in ghosts:
                    continue
                if p and abs(x - p[0]) <= 2 and abs(y - p[1]) <= 2:
                    continue
                if maze[y][x] == ".":
                    if self.rng.random() < 0.08:
                        maze[y][x] = "#"
                elif maze[y][x] == "#":
                    if self.rng.random() < 0.04:
                        maze[y][x] = "."

        return ["".join(r) for r in maze]

    def ghost_speed_for_level(self, level):
        return min(MAX_GHOST_SPEED, BASE_GHOST_SPEED * (self.ghost_speed_multiplier ** (level - 1)))

    def wall_color_for_level(self, level):
        return WALL_COLORS[(level - 1) % len(WALL_COLORS)]

    def ghost_count_for_level(self, level):
        return max(1, self.ghosts_first_level + (level - 1) * self.ghosts_per_level)

    def required_score_for_next_level(self, level):
        return level * self.points_per_level

    def build_level(self, maze):
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
                wx, wy = x * self.tile, y * self.tile
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
            player_start = (self.tile, self.tile)
        px = player_start[0] // self.tile
        py = player_start[1] // self.tile
        for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
            walls.discard((px + dx, py + dy))
        if not ghost_starts:
            ghost_starts = [
                (self.grid_w // 2 * self.tile - self.tile, self.grid_h // 2 * self.tile - self.tile),
                (self.grid_w // 2 * self.tile + self.tile, self.grid_h // 2 * self.tile - self.tile),
                (self.grid_w // 2 * self.tile - self.tile, self.grid_h // 2 * self.tile + self.tile),
                (self.grid_w // 2 * self.tile + self.tile, self.grid_h // 2 * self.tile + self.tile),
            ]
        return walls, dots, power, cyan, player_start, ghost_starts

    def add_extra_ghosts(self, maze, base_starts, extra_count):
        if extra_count <= 0:
            return list(base_starts)
        h = len(maze)
        w = len(maze[0])
        p = None
        base_tiles = {(x // self.tile, y // self.tile) for (x, y) in base_starts}
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
        self.rng.shuffle(open_cells)
        extras = open_cells[:extra_count]
        return list(base_starts) + [(x * self.tile, y * self.tile) for (x, y) in extras]

    def is_surrounded(self):
        px = (self.player.x + self.tile // 2) // self.tile
        py = (self.player.y + self.tile // 2) // self.tile
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = px + dx, py + dy
            if 0 <= nx < self.grid_w and 0 <= ny < self.grid_h:
                if (nx, ny) not in self.walls:
                    return False
        return True

    def build_block_requirements(self):
        req = {}
        bx_max = (self.grid_w + self.reveal_block - 1) // self.reveal_block
        by_max = (self.grid_h + self.reveal_block - 1) // self.reveal_block
        for by in range(by_max):
            for bx in range(bx_max):
                x0 = bx * self.reveal_block
                y0 = by * self.reveal_block
                w = min(self.reveal_block, self.grid_w - x0)
                h = min(self.reveal_block, self.grid_h - y0)
                req[(bx, by)] = w * h
        return req

    def register_clear(self, x, y):
        key = (x // self.reveal_block, y // self.reveal_block)
        if key in self.revealed:
            return
        self.reveal_counts[key] = self.reveal_counts.get(key, 0) + 1
        if self.reveal_counts[key] >= self.block_required.get(key, self.reveal_block * self.reveal_block):
            self.revealed.add(key)

    def reset_ghosts(self):
        for g, start in zip(self.ghosts, self.ghost_starts):
            if g.alive:
                g.x, g.y = start
                g.dir = self.rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])

    def apply_player_hit(self):
        self.lives -= 1
        self.player.x, self.player.y = self.player_start
        self.player.dir = (0, 0)
        self.break_blocks = False
        self.break_timer = 0
        self.bombs_left = self.max_bombs_per_life
        self.reset_ghosts()
        self.invuln_timer = 2000
        if self.lives <= 0:
            self.game_over = True

    def init_reveal_state(self):
        rc = {}
        rev = set()
        for y in range(self.grid_h):
            for x in range(self.grid_w):
                if (x, y) in self.walls:
                    continue
                if (x, y) in self.dots or (x, y) in self.power or (x, y) in self.cyan:
                    continue
                key = (x // self.reveal_block, y // self.reveal_block)
                rc[key] = rc.get(key, 0) + 1
        for key, count in rc.items():
            if count >= self.block_required.get(key, self.reveal_block * self.reveal_block):
                rev.add(key)
        return rev, rc

    def reset_all(self):
        self.level = 1
        self.score = 0
        self.lives = 3
        self._init_level(self.level)

    def continue_after_game_over(self):
        self.lives = 3
        self.invuln_timer = 2000
        self.break_blocks = False
        self.break_timer = 0
        self.bombs = []
        self.bomb_positions = set()
        self.bombs_left = self.max_bombs_per_life
        self.game_won = False
        self.game_over = False

    def _init_level(self, level):
        self.wall_color = self.wall_color_for_level(level)
        self.current_maze = self.make_level_maze(level)
        self.walls, self.dots, self.power, self.cyan, self.player_start, ghost_starts_all = self.build_level(
            self.current_maze
        )
        base_starts = ghost_starts_all[:1]
        self.ghost_starts = self.add_extra_ghosts(
            self.current_maze, base_starts, self.ghost_count_for_level(level) - len(base_starts)
        )
        self.player = Player(*self.player_start)
        self.ghosts = [
            Ghost(x, y, GHOST_COLORS[i % len(GHOST_COLORS)], self.ghost_speed_for_level(level), self.rng)
            for i, (x, y) in enumerate(self.ghost_starts)
        ]
        self.power_timer = 0
        self.invuln_timer = 0
        self.dig_timers = {}
        self.place_cooldown = 0
        self.break_blocks = False
        self.break_timer = 0
        self.bombs = []
        self.bomb_positions = set()
        self.bombs_left = self.max_bombs_per_life
        self.block_required = self.build_block_requirements()
        self.revealed, self.reveal_counts = self.init_reveal_state()
        self.total_reveal_blocks = len(self.block_required)
        self.game_over = False
        self.game_won = False

    def can_place_wall(self, tx, ty):
        tile_rect = (tx * self.tile, ty * self.tile, self.tile, self.tile)
        if rects_intersect(self.player.rect(self.tile), tile_rect):
            return False
        for g in self.ghosts:
            if g.alive and rects_intersect(g.rect(self.tile), tile_rect):
                return False
        return True

    def snap_player_to_tile(self):
        tx = (self.player.x + self.tile // 2) // self.tile
        ty = (self.player.y + self.tile // 2) // self.tile
        if (tx, ty) in self.walls:
            return False
        self.player.x = tx * self.tile
        self.player.y = ty * self.tile
        return True

    def attempt_dig(self, d):
        if d == (0, 0):
            return
        if not self.snap_player_to_tile():
            return
        px = self.player.x // self.tile
        py = self.player.y // self.tile
        tx = px + d[0]
        ty = py + d[1]
        if (tx, ty) in self.walls and (tx, ty) not in self.dig_timers:
            self.dig_timers[(tx, ty)] = 500

    def attempt_place(self, d):
        if d == (0, 0):
            return False
        if not self.snap_player_to_tile():
            return False
        px = self.player.x // self.tile
        py = self.player.y // self.tile
        tx = px + d[0]
        ty = py + d[1]
        if not (0 <= tx < self.grid_w and 0 <= ty < self.grid_h):
            return False
        if (tx, ty) in self.walls or (tx, ty) in self.dig_timers:
            return False
        if not self.can_place_wall(tx, ty):
            return False
        self.dots.discard((tx, ty))
        self.power.discard((tx, ty))
        self.cyan.discard((tx, ty))
        self.walls.add((tx, ty))
        tile_rect = (tx * self.tile, ty * self.tile, self.tile, self.tile)
        if self.hits_wall(self.player.x, self.player.y, self.walls):
            self.walls.remove((tx, ty))
            return False
        for g in self.ghosts:
            if rects_intersect(g.rect(self.tile), tile_rect):
                self.walls.remove((tx, ty))
                return False
        return True

    def drop_bomb(self):
        bx = (self.player.x + self.tile // 2) // self.tile
        by = (self.player.y + self.tile // 2) // self.tile
        if self.bombs_left <= 0:
            return False
        if (bx, by) in self.bomb_positions:
            return False
        self.bombs.append({"x": bx, "y": by, "timer": self.bomb_timer_ms})
        self.bomb_positions.add((bx, by))
        self.bombs_left -= 1
        return True

    def step(self, move_dir, do_dig=False, do_place=False, do_bomb=False, dt_ms=16):
        if self.game_over or self.game_won:
            return

        if do_dig and move_dir != (0, 0):
            self.attempt_dig(move_dir)
        if do_place and move_dir != (0, 0) and self.place_cooldown <= 0:
            if self.attempt_place(move_dir):
                self.place_cooldown = 200
        if do_bomb:
            self.drop_bomb()

        if move_dir == (0, 0):
            self.player.dir = (0, 0)

        self.player.update(self, move_dir)

        px = (self.player.x + self.tile // 2) // self.tile
        py = (self.player.y + self.tile // 2) // self.tile
        if (px, py) in self.walls:
            self.walls.remove((px, py))
            if (px, py) in self.dig_timers:
                del self.dig_timers[(px, py)]
            self.register_clear(px, py)

        hit_ghost = None
        player_rect = self.player.rect(self.tile)
        for ghost in self.ghosts:
            if ghost.update(self, (self.player.x, self.player.y), player_rect):
                hit_ghost = ghost
                break

        if self.bombs:
            exploded = []
            for b in self.bombs:
                b["timer"] -= dt_ms
                if b["timer"] <= 0:
                    exploded.append(b)
            player_hit_by_bomb = False
            for b in exploded:
                self.bombs.remove(b)
                self.bomb_positions.discard((b["x"], b["y"]))
                start_x = b["x"] - (self.bomb_size // 2)
                start_y = b["y"] - (self.bomb_size // 2)
                end_x = start_x + self.bomb_size
                end_y = start_y + self.bomb_size
                for y in range(start_y, end_y):
                    for x in range(start_x, end_x):
                        if x < 0 or y < 0 or x >= self.grid_w or y >= self.grid_h:
                            continue
                        if (x, y) in self.walls:
                            self.walls.remove((x, y))
                            self.register_clear(x, y)
                        if (x, y) in self.dots:
                            self.dots.remove((x, y))
                            self.register_clear(x, y)
                        if (x, y) in self.power:
                            self.power.remove((x, y))
                            self.register_clear(x, y)
                        if (x, y) in self.cyan:
                            self.cyan.remove((x, y))
                            self.register_clear(x, y)
                        if (x, y) in self.dig_timers:
                            del self.dig_timers[(x, y)]

                for g in self.ghosts:
                    if not g.alive:
                        continue
                    gx = (g.x + self.tile // 2) // self.tile
                    gy = (g.y + self.tile // 2) // self.tile
                    if start_x <= gx < end_x and start_y <= gy < end_y:
                        g.alive = False

                if not player_hit_by_bomb and self.invuln_timer <= 0:
                    px = (self.player.x + self.tile // 2) // self.tile
                    py = (self.player.y + self.tile // 2) // self.tile
                    if start_x <= px < end_x and start_y <= py < end_y:
                        self.apply_player_hit()
                        player_hit_by_bomb = True

        grid_x, grid_y = self.player.x // self.tile, self.player.y // self.tile
        if (grid_x, grid_y) in self.dots:
            self.dots.remove((grid_x, grid_y))
            self.score += 10
            self.register_clear(grid_x, grid_y)
        if (grid_x, grid_y) in self.power:
            self.power.remove((grid_x, grid_y))
            self.score += 50
            self.power_timer = 3000
            self.register_clear(grid_x, grid_y)
        if (grid_x, grid_y) in self.cyan:
            self.cyan.remove((grid_x, grid_y))
            self.score += 50
            self.break_blocks = True
            self.break_timer = self.cyan_effect_ms
            self.register_clear(grid_x, grid_y)

        if self.power_timer > 0:
            self.power_timer -= dt_ms
        if self.invuln_timer > 0:
            self.invuln_timer -= dt_ms
        if self.break_timer > 0:
            self.break_timer -= dt_ms
            if self.break_timer <= 0:
                self.break_blocks = False
        if self.place_cooldown > 0:
            self.place_cooldown -= dt_ms
        if self.dig_timers:
            done = []
            for (x, y), remaining in list(self.dig_timers.items()):
                remaining -= dt_ms
                if remaining <= 0:
                    done.append((x, y))
                else:
                    self.dig_timers[(x, y)] = remaining
            for pos in done:
                if pos in self.walls:
                    self.walls.remove(pos)
                    self.register_clear(pos[0], pos[1])
                if pos in self.dig_timers:
                    del self.dig_timers[pos]

        if hit_ghost is None:
            for ghost in self.ghosts:
                if ghost.alive and rects_intersect(self.player.rect(self.tile), ghost.rect(self.tile)):
                    hit_ghost = ghost
                    break

        if hit_ghost:
            if self.power_timer > 0:
                self.score += 200
                hit_ghost.x, hit_ghost.y = self.rng.choice(self.ghost_starts)
            elif self.invuln_timer <= 0:
                self.apply_player_hit()

        if self.score >= self.required_score_for_next_level(self.level) and self.is_surrounded():
            self.level += 1
            self._init_level(self.level)
            return

        if not self.game_won and len(self.revealed) >= self.total_reveal_blocks:
            self.game_won = True

    def step_action(self, action_idx, dt_ms=16):
        move_dir, act = ACTION_SPACE[action_idx]
        prev_score = self.score
        prev_lives = self.lives
        prev_level = self.level
        self.step(
            move_dir,
            do_dig=(act == "dig"),
            do_place=(act == "place"),
            do_bomb=(act == "bomb"),
            dt_ms=dt_ms,
        )
        return {
            "score_delta": self.score - prev_score,
            "lives_delta": self.lives - prev_lives,
            "level_delta": self.level - prev_level,
            "done": self.game_over or self.game_won,
            "game_over": self.game_over,
            "game_won": self.game_won,
        }

    def is_terminal(self):
        return self.game_over or self.game_won

    def terminal_value(self):
        if self.game_won:
            return 1.0
        if self.game_over:
            return -1.0
        return 0.0

    def get_state(self):
        return {
            "grid_w": self.grid_w,
            "grid_h": self.grid_h,
            "level": self.level,
            "score": self.score,
            "lives": self.lives,
            "player": {
                "x": self.player.x,
                "y": self.player.y,
                "dir": self.player.dir,
                "speed": self.player.speed,
            },
            "ghosts": [
                {"x": g.x, "y": g.y, "dir": g.dir, "speed": g.speed, "alive": g.alive} for g in self.ghosts
            ],
            "walls": list(self.walls),
            "dots": list(self.dots),
            "power": list(self.power),
            "cyan": list(self.cyan),
            "dig_timers": {f"{k[0]},{k[1]}": v for k, v in self.dig_timers.items()},
            "bombs": [dict(b) for b in self.bombs],
            "bombs_left": self.bombs_left,
            "power_timer": self.power_timer,
            "invuln_timer": self.invuln_timer,
            "break_blocks": self.break_blocks,
            "break_timer": self.break_timer,
            "place_cooldown": self.place_cooldown,
            "revealed": list(self.revealed),
            "reveal_counts": {f"{k[0]},{k[1]}": v for k, v in self.reveal_counts.items()},
            "game_over": self.game_over,
            "game_won": self.game_won,
        }

    def get_observation(self):
        def blank():
            return [[0 for _ in range(self.grid_w)] for _ in range(self.grid_h)]

        walls = blank()
        dots = blank()
        power = blank()
        cyan = blank()
        dig = blank()
        bombs = blank()
        bomb_timer = blank()
        player = blank()
        ghost = blank()
        ghost_alive = blank()

        for x, y in self.walls:
            walls[y][x] = 1
        for x, y in self.dots:
            dots[y][x] = 1
        for x, y in self.power:
            power[y][x] = 1
        for x, y in self.cyan:
            cyan[y][x] = 1
        for (x, y), remaining in self.dig_timers.items():
            dig[y][x] = 1
        for b in self.bombs:
            bombs[b["y"]][b["x"]] = 1
            bomb_timer[b["y"]][b["x"]] = max(0.0, min(1.0, b["timer"] / float(self.bomb_timer_ms)))

        px = (self.player.x + self.tile // 2) // self.tile
        py = (self.player.y + self.tile // 2) // self.tile
        if 0 <= px < self.grid_w and 0 <= py < self.grid_h:
            player[py][px] = 1

        for g in self.ghosts:
            gx = int((g.x + self.tile // 2) // self.tile)
            gy = int((g.y + self.tile // 2) // self.tile)
            if 0 <= gx < self.grid_w and 0 <= gy < self.grid_h:
                ghost[gy][gx] = 1
                if g.alive:
                    ghost_alive[gy][gx] = 1

        planes = [
            walls,
            dots,
            power,
            cyan,
            dig,
            bombs,
            bomb_timer,
            player,
            ghost,
            ghost_alive,
        ]
        labels = [
            "walls",
            "dots",
            "power",
            "cyan",
            "dig",
            "bombs",
            "bomb_timer",
            "player",
            "ghost",
            "ghost_alive",
        ]
        scalars = {
            "level": self.level,
            "score": self.score,
            "lives": self.lives,
            "bombs_left": self.bombs_left,
            "power_timer": self.power_timer / 3000.0 if self.power_timer > 0 else 0.0,
            "break_timer": self.break_timer / float(self.cyan_effect_ms) if self.break_timer > 0 else 0.0,
            "invuln_timer": self.invuln_timer / 2000.0 if self.invuln_timer > 0 else 0.0,
            "game_over": 1.0 if self.game_over else 0.0,
            "game_won": 1.0 if self.game_won else 0.0,
            "reveal_progress": len(self.revealed) / float(self.total_reveal_blocks) if self.total_reveal_blocks else 0.0,
        }
        return planes, labels, scalars
