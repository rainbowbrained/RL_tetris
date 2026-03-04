import numpy as np
from typing import Optional, Tuple, Dict, List


TETROMINOS: Dict[str, List[np.ndarray]] = {}

def _rotations(shape: np.ndarray) -> List[np.ndarray]:
    rots = []
    s = shape.copy()
    seen = set()
    for _ in range(4):
        s = s - s.min(axis=0)
        key = tuple(sorted(map(tuple, s)))
        if key not in seen:
            seen.add(key)
            rots.append(s.copy())
        s = np.column_stack([s[:, 1], s[:, 0].max() - s[:, 0]])
    return rots

TETROMINOS["I"] = _rotations(np.array([[0, 0], [0, 1], [0, 2], [0, 3]]))
TETROMINOS["O"] = _rotations(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))
TETROMINOS["T"] = _rotations(np.array([[0, 0], [0, 1], [0, 2], [1, 1]]))
TETROMINOS["S"] = _rotations(np.array([[0, 1], [0, 2], [1, 0], [1, 1]]))
TETROMINOS["Z"] = _rotations(np.array([[0, 0], [0, 1], [1, 1], [1, 2]]))
TETROMINOS["L"] = _rotations(np.array([[0, 0], [1, 0], [2, 0], [2, 1]]))
TETROMINOS["J"] = _rotations(np.array([[0, 1], [1, 1], [2, 0], [2, 1]]))

PIECE_NAMES = list(TETROMINOS.keys())
NUM_PIECES  = len(PIECE_NAMES)

REWARD_WEIGHTS = {
    "line_clear_scale": 60.0,
    "survival_bonus": 1.0,
    "hole_penalty": 1.5,
    "hole_removal_bonus": 0.5,
    "bumpiness_penalty": 0.3,
    "height_penalty": 0.3,
    "game_over_penalty": 10.0,
}


class TetrisLiteEnv:
    def __init__(
        self,
        width: int = 6,
        height: int = 20,
        max_steps: Optional[int] = 500,
        seed: Optional[int] = None,
        include_next: bool = True,
    ):
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.include_next = include_next
        self.rng = np.random.RandomState(seed)

        self._build_action_table()

        self.board: np.ndarray = np.zeros((height, width), dtype=np.int8)
        self.current_piece: int = 0
        self.next_piece: int = 0
        self.step_count: int = 0
        self.total_lines: int = 0
        self.done: bool = False

        self._prev_holes: int = 0
        self._prev_bumpiness: float = 0.0
        self._prev_height: float = 0.0

        self.last_reward_info: Dict[str, float] = {}

    def _build_action_table(self):
        self.piece_actions: Dict[int, List[Tuple[int, int, np.ndarray]]] = {}
        max_actions = 0
        for pid, name in enumerate(PIECE_NAMES):
            acts = []
            for rid, shape in enumerate(TETROMINOS[name]):
                piece_width = shape[:, 1].max() + 1
                for x in range(self.width - piece_width + 1):
                    acts.append((rid, x, shape))
            self.piece_actions[pid] = acts
            max_actions = max(max_actions, len(acts))
        self.max_actions = max_actions

    def num_actions(self) -> int:
        return self.max_actions

    def obs_size(self) -> int:
        return self.width * 3 + 3 + NUM_PIECES * 2

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.board = np.zeros((self.height, self.width), dtype=np.int8)
        self.current_piece = self.rng.randint(NUM_PIECES)
        self.next_piece = self.rng.randint(NUM_PIECES)
        self.step_count = 0
        self.total_lines = 0
        self.done = False
        self._prev_holes = 0
        self._prev_bumpiness = 0.0
        self._prev_height = 0.0
        self.last_reward_info = {}
        return self._obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        assert not self.done, "Episode is done. Call reset()."

        actions = self.piece_actions[self.current_piece]
        if action >= len(actions):
            action = len(actions) - 1

        rid, x, shape = actions[action]

        placed = self._drop(shape, x)

        if not placed:
            self.done = True
            self.last_reward_info = {
                "line_clear": 0.0,
                "survival": 0.0,
                "hole_penalty": 0.0,
                "hole_removal": 0.0,
                "bump_penalty": 0.0,
                "height_penalty": 0.0,
                "game_over": -REWARD_WEIGHTS["game_over_penalty"],
                "clear_type": 0,
            }
            return self._obs(), -REWARD_WEIGHTS["game_over_penalty"], True, {
                "lines": 0, "total_lines": self.total_lines,
            }

        lines_cleared = self._clear_lines()
        self.total_lines += lines_cleared

        holes = self._count_holes()
        bumpiness = self._bumpiness()
        agg_height = self._aggregate_height()

        delta_holes = holes - self._prev_holes
        delta_bump = bumpiness - self._prev_bumpiness
        delta_height = agg_height - self._prev_height

        w = REWARD_WEIGHTS
        if lines_cleared > 0:
            line_reward = lines_cleared * lines_cleared * w["line_clear_scale"]
        else:
            line_reward = 0.0

        survival    = w["survival_bonus"]
        hole_pen    = -w["hole_penalty"] * max(delta_holes, 0)
        hole_rem    = w["hole_removal_bonus"] * max(-delta_holes, 0)
        bump_pen    = -w["bumpiness_penalty"] * max(delta_bump, 0)
        height_pen  = -w["height_penalty"] * max(delta_height, 0)

        reward = line_reward + survival + hole_pen + hole_rem + bump_pen + height_pen

        self._prev_holes = holes
        self._prev_bumpiness = bumpiness
        self._prev_height = agg_height

        self.current_piece = self.next_piece
        self.next_piece = self.rng.randint(NUM_PIECES)
        self.step_count += 1

        game_over_pen = 0.0
        if self.max_steps and self.step_count >= self.max_steps:
            self.done = True

        if not self.done and not self._has_legal_placement():
            self.done = True
            game_over_pen = -w["game_over_penalty"]
            reward += game_over_pen

        self.last_reward_info = {
            "line_clear": line_reward,
            "survival": survival,
            "hole_penalty": hole_pen,
            "hole_removal": hole_rem,
            "bump_penalty": bump_pen,
            "height_penalty": height_pen,
            "game_over": game_over_pen,
            "clear_type": lines_cleared,
        }

        return self._obs(), reward, self.done, {
            "lines": lines_cleared,
            "total_lines": self.total_lines,
        }

    def _can_place_fast(self, shape: np.ndarray, x: int, hmap: np.ndarray) -> bool:
        cols = shape[:, 1] + x
        rows_offset = shape[:, 0]
        min_start = self.height
        for ro, c in zip(rows_offset, cols):
            first_occ = self.height - int(hmap[c])
            limit = first_occ - ro
            if limit < min_start:
                min_start = limit
        final_row = min(min_start - 1, self.height - 1 - rows_offset.max())
        return not np.any(rows_offset + final_row < 0)

    def _drop(self, shape: np.ndarray, x: int) -> bool:
        cols = shape[:, 1] + x
        rows_offset = shape[:, 0]

        for start_row in range(-(rows_offset.max()), self.height):
            abs_rows = rows_offset + start_row
            collision = False
            for r, c in zip(abs_rows, cols):
                if r >= self.height:
                    collision = True
                    break
                if r >= 0 and self.board[r, c] != 0:
                    collision = True
                    break
            if collision:
                final_row = start_row - 1
                break
        else:
            final_row = self.height - 1 - rows_offset.max()

        abs_rows = rows_offset + final_row
        if np.any(abs_rows < 0):
            return False

        for r, c in zip(abs_rows, cols):
            self.board[r, c] = self.current_piece + 1

        return True

    def _clear_lines(self) -> int:
        full = np.all(self.board != 0, axis=1)
        n = int(full.sum())
        if n > 0:
            keep = self.board[~full]
            self.board = np.vstack([
                np.zeros((n, self.width), dtype=np.int8),
                keep,
            ])
        return n

    def _height_map(self) -> np.ndarray:
        has_block = (self.board != 0)
        first_occ = has_block.argmax(axis=0)
        col_has_any = has_block.any(axis=0)
        hmap = np.where(col_has_any, self.height - first_occ, 0).astype(np.float32)
        return hmap

    def _aggregate_height(self) -> float:
        return float(self._height_map().sum())

    def _count_holes(self) -> int:
        holes = 0
        for c in range(self.width):
            block_found = False
            for r in range(self.height):
                if self.board[r, c] != 0:
                    block_found = True
                elif block_found:
                    holes += 1
        return holes

    def _bumpiness(self) -> float:
        hmap = self._height_map()
        return float(np.sum(np.abs(np.diff(hmap))))

    def _max_height(self) -> float:
        return float(self._height_map().max())

    def _has_legal_placement(self) -> bool:
        hmap = self._height_map()
        for rid, x, shape in self.piece_actions[self.current_piece]:
            if self._can_place_fast(shape, x, hmap):
                return True
        return False

    def _holes_per_column(self) -> np.ndarray:
        hpc = np.zeros(self.width, dtype=np.float32)
        for c in range(self.width):
            block_found = False
            for r in range(self.height):
                if self.board[r, c] != 0:
                    block_found = True
                elif block_found:
                    hpc[c] += 1
        return hpc

    def _row_fill(self, n_rows: int) -> np.ndarray:
        fills = np.zeros(n_rows, dtype=np.float32)
        for i in range(n_rows):
            r = self.height - 1 - i
            if r >= 0:
                fills[i] = np.sum(self.board[r] != 0) / self.width
        return fills

    def _obs(self) -> np.ndarray:
        hmap = self._height_map() / self.height
        hpc = self._holes_per_column() / self.height
        row_fills = self._row_fill(self.width)

        scalar = np.array([
            self._max_height() / self.height,
            self._count_holes() / (self.height * self.width),
            self._bumpiness() / self.height,
        ], dtype=np.float32)

        piece_oh = np.zeros(NUM_PIECES, dtype=np.float32)
        piece_oh[self.current_piece] = 1.0

        next_oh = np.zeros(NUM_PIECES, dtype=np.float32)
        if self.include_next:
            next_oh[self.next_piece] = 1.0

        return np.concatenate([hmap, hpc, row_fills, scalar, piece_oh, next_oh])

    def legal_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.max_actions, dtype=np.float32)
        hmap = self._height_map()
        for i, (rid, x, shape) in enumerate(self.piece_actions[self.current_piece]):
            if self._can_place_fast(shape, x, hmap):
                mask[i] = 1.0
        return mask

    def action_stats(self) -> Dict[str, int]:
        actions = self.piece_actions[self.current_piece]
        n_template = len(actions)
        hmap = self._height_map()
        n_placeable = sum(1 for _, x, s in actions if self._can_place_fast(s, x, hmap))
        return {"n_template": n_template, "n_placeable": n_placeable}

    def get_cnn_obs(self) -> Tuple[np.ndarray, np.ndarray]:
        board_2d = (self.board != 0).astype(np.float32)[np.newaxis, :, :]
        piece_vec = np.zeros(NUM_PIECES * 2, dtype=np.float32)
        piece_vec[self.current_piece] = 1.0
        if self.include_next:
            piece_vec[NUM_PIECES + self.next_piece] = 1.0
        return board_2d, piece_vec

    def render_board(self) -> np.ndarray:
        return self.board.copy()

    def get_board_rgb(self, cell_size: int = 20) -> np.ndarray:
        COLORS = [
            [30,  30,  30],
            [0,   240, 240],
            [240, 240, 0],
            [160, 0,   240],
            [0,   240, 0],
            [240, 0,   0],
            [240, 160, 0],
            [0,   0,   240],
        ]
        img = np.zeros((self.height * cell_size, self.width * cell_size, 3),
                        dtype=np.uint8)
        for r in range(self.height):
            for c in range(self.width):
                color = COLORS[int(self.board[r, c])]
                r0, r1 = r * cell_size, (r + 1) * cell_size
                c0, c1 = c * cell_size, (c + 1) * cell_size
                img[r0+1:r1, c0+1:c1] = color
        return img
