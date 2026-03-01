"""
Tetris-Lite (Placement) Environment
====================================
A simplified Tetris MDP where the agent places one tetromino per step.

Action  = (x_position, rotation)  →  flattened into a single discrete index
State   = (height_map, current_piece_id, next_piece_id)
Reward  = line_clears * 10  – holes_penalty – bumpiness_penalty + small_step_reward

Implemented from first principles using only NumPy (no Gymnasium).
"""

import numpy as np
from typing import Optional, Tuple, Dict, List


# ──────────────────────────────────────────────────────────────
# Tetromino definitions  (each rotation is a list of (row, col) offsets)
# ──────────────────────────────────────────────────────────────

TETROMINOS: Dict[str, List[np.ndarray]] = {}

def _rotations(shape: np.ndarray) -> List[np.ndarray]:
    """Generate all unique rotations (0°, 90°, 180°, 270°) of a shape."""
    rots = []
    s = shape.copy()
    seen = set()
    for _ in range(4):
        # Normalise so min row/col = 0
        s = s - s.min(axis=0)
        key = tuple(sorted(map(tuple, s)))
        if key not in seen:
            seen.add(key)
            rots.append(s.copy())
        # 90° clockwise rotation: (r,c) -> (c, max_r - r)
        s = np.column_stack([s[:, 1], s[:, 0].max() - s[:, 0]])
    return rots

# I-piece
TETROMINOS["I"] = _rotations(np.array([[0, 0], [0, 1], [0, 2], [0, 3]]))
# O-piece
TETROMINOS["O"] = _rotations(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))
# T-piece
TETROMINOS["T"] = _rotations(np.array([[0, 0], [0, 1], [0, 2], [1, 1]]))
# S-piece
TETROMINOS["S"] = _rotations(np.array([[0, 1], [0, 2], [1, 0], [1, 1]]))
# Z-piece
TETROMINOS["Z"] = _rotations(np.array([[0, 0], [0, 1], [1, 1], [1, 2]]))
# L-piece
TETROMINOS["L"] = _rotations(np.array([[0, 0], [1, 0], [2, 0], [2, 1]]))
# J-piece
TETROMINOS["J"] = _rotations(np.array([[0, 1], [1, 1], [2, 0], [2, 1]]))

PIECE_NAMES = list(TETROMINOS.keys())  # deterministic order
NUM_PIECES  = len(PIECE_NAMES)


# ──────────────────────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────────────────────

class TetrisLiteEnv:
    """
    Tetris-Lite (Placement) environment.

    Parameters
    ----------
    width       : board width  (default 10)
    height      : board height (default 20)
    max_steps   : episode horizon (None = unlimited)
    seed        : random seed
    include_next: whether observation includes the next piece
    """

    def __init__(
        self,
        width: int = 10,
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

        # Pre-compute all (piece, rotation, x) actions and cache shapes
        self._build_action_table()

        # State
        self.board: np.ndarray = np.zeros((height, width), dtype=np.int8)
        self.current_piece: int = 0
        self.next_piece: int = 0
        self.step_count: int = 0
        self.total_lines: int = 0
        self.done: bool = False
        # Track previous metrics for delta-based reward shaping
        self._prev_holes: int = 0
        self._prev_bumpiness: float = 0.0
        self._prev_max_height: float = 0.0

    # ── action table ──────────────────────────────────────────
    def _build_action_table(self):
        """Pre-compute every legal (rotation, x) for every piece."""
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
        self.max_actions = max_actions        # for network output dim

    def num_actions(self) -> int:
        return self.max_actions

    def obs_size(self) -> int:
        """Flat observation: board_rows * width + width (hmap) + 1 (piece) [+ 1 (next)]."""
        # We include the top `visible_rows` of the board so the agent can
        # see the actual cell layout (where the gaps are) — critical for
        # learning to complete rows.
        return self._visible_rows * self.width + self.width + 1 + (1 if self.include_next else 0)

    @property
    def _visible_rows(self) -> int:
        """How many rows from the top of the stack we include in obs."""
        return min(self.height, 10)  # top-10 rows of effective board

    # ── reset / step ──────────────────────────────────────────
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
        self._prev_max_height = 0.0
        return self._obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        assert not self.done, "Episode is done. Call reset()."

        actions = self.piece_actions[self.current_piece]
        # Mask illegal actions → clamp
        if action >= len(actions):
            action = len(actions) - 1

        rid, x, shape = actions[action]

        # ── drop piece ────────────────────────────────────────
        placed = self._drop(shape, x)

        if not placed:
            # Game over: can't place the piece
            self.done = True
            return self._obs(), -5.0, True, {"lines": 0, "total_lines": self.total_lines}

        # ── clear lines ───────────────────────────────────────
        lines_cleared = self._clear_lines()
        self.total_lines += lines_cleared

        # ── reward shaping (delta-based) ─────────────────────
        holes = self._count_holes()
        bumpiness = self._bumpiness()
        max_height = self._max_height()

        delta_holes = holes - self._prev_holes
        delta_bump = bumpiness - self._prev_bumpiness
        delta_height = max_height - self._prev_max_height

        # Row-completion bonus: smooth gradient towards line clears.
        # For each row, reward proportional to (filled / width)^2 so the
        # agent is pulled towards almost-full rows.
        row_fill_bonus = 0.0
        for r in range(self.height):
            filled = int(np.sum(self.board[r] != 0))
            if filled > 0:
                frac = filled / self.width
                row_fill_bonus += frac * frac  # quadratic: 0.5→0.25, 0.8→0.64, 0.9→0.81

        # Super-linear line clear bonus (1→50, 2→200, 3→450, 4→800)
        line_clear_reward = lines_cleared * lines_cleared * 50.0

        reward = (
            + line_clear_reward         # dominant signal: clear lines!
            + 0.5 * row_fill_bonus      # shaping: reward dense rows
            + 0.1                       # small survival bonus
            - 0.7 * delta_holes         # penalise NEW holes
            - 0.3 * delta_bump          # penalise increased bumpiness
            - 0.15 * delta_height       # penalise height increase
        )

        self._prev_holes = holes
        self._prev_bumpiness = bumpiness
        self._prev_max_height = max_height

        # ── advance piece ─────────────────────────────────────
        self.current_piece = self.next_piece
        self.next_piece = self.rng.randint(NUM_PIECES)
        self.step_count += 1

        if self.max_steps and self.step_count >= self.max_steps:
            self.done = True

        # Check if new piece has at least one legal placement
        if not self.done and not self._has_legal_placement():
            self.done = True
            reward -= 5.0

        return self._obs(), reward, self.done, {
            "lines": lines_cleared,
            "total_lines": self.total_lines,
        }

    # ── internal helpers ──────────────────────────────────────
    def _drop(self, shape: np.ndarray, x: int) -> bool:
        """Hard-drop a piece at column x.  Returns True if placed."""
        cols = shape[:, 1] + x
        rows_offset = shape[:, 0]

        # For each column the piece occupies, find the landing row
        drop_row = self.height  # start from bottom
        for i in range(len(cols)):
            c = cols[i]
            r_off = rows_offset[i]
            # Find highest occupied cell in this column
            col_data = self.board[:, c]
            occupied = np.where(col_data != 0)[0]
            if len(occupied) > 0:
                land = occupied.min() - 1 - r_off
            else:
                land = self.height - 1 - r_off
            # We need the piece bottom in this column to sit just above
            # the highest block.
            # Actually let's compute differently: find the lowest row we can
            # place the piece root (row 0 of shape) such that no collision.
            drop_row = min(drop_row, land)

        # The piece's row-0 will be at drop_row, offset rows at drop_row + r_off
        # Correction: let me use a simpler approach — slide down from top
        # until collision.
        piece_rows = rows_offset
        piece_cols = cols

        # Find the valid drop position by sliding from top
        for start_row in range(-(piece_rows.max()), self.height):
            abs_rows = piece_rows + start_row
            # Check if any cell is below the board or collides
            collision = False
            for r, c in zip(abs_rows, piece_cols):
                if r >= self.height:
                    collision = True
                    break
                if r >= 0 and self.board[r, c] != 0:
                    collision = True
                    break
            if collision:
                # Place at start_row - 1
                final_row = start_row - 1
                break
        else:
            final_row = self.height - 1 - piece_rows.max()

        abs_rows = piece_rows + final_row
        # Check all cells are on the board
        if np.any(abs_rows < 0):
            return False  # game over — piece sticks out above board

        for r, c in zip(abs_rows, piece_cols):
            self.board[r, c] = self.current_piece + 1  # piece IDs 1-7

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
        """Height of each column (0 = empty column)."""
        hmap = np.zeros(self.width, dtype=np.float32)
        for c in range(self.width):
            occupied = np.where(self.board[:, c] != 0)[0]
            if len(occupied) > 0:
                hmap[c] = self.height - occupied.min()
        return hmap

    def _count_holes(self) -> int:
        """A hole is an empty cell with at least one filled cell above it."""
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
        return len(self.piece_actions[self.current_piece]) > 0

    def _obs(self) -> np.ndarray:
        hmap = self._height_map() / self.height          # normalise [0,1]
        pid = np.array([self.current_piece / NUM_PIECES], dtype=np.float32)

        # ── board bitmap: top `visible_rows` rows of the occupied zone ──
        # This lets the agent see *which cells* are filled — essential for
        # learning to complete rows rather than just stacking blindly.
        vr = self._visible_rows
        # Find the top-most occupied row
        occupied_rows = np.where(np.any(self.board != 0, axis=1))[0]
        if len(occupied_rows) > 0:
            top = occupied_rows.min()
            # Extract rows from `top` to `top + vr`, clipped to board
            end = min(top + vr, self.height)
            actual = end - top
            board_slice = (self.board[top:end] != 0).astype(np.float32).flatten()
            # Pad with zeros if we got fewer than vr rows
            if actual < vr:
                pad = np.zeros((vr - actual) * self.width, dtype=np.float32)
                board_slice = np.concatenate([pad, board_slice])
        else:
            board_slice = np.zeros(vr * self.width, dtype=np.float32)

        if self.include_next:
            nid = np.array([self.next_piece / NUM_PIECES], dtype=np.float32)
            return np.concatenate([board_slice, hmap, pid, nid])
        return np.concatenate([board_slice, hmap, pid])

    # ── legal action mask ─────────────────────────────────────
    def legal_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.max_actions, dtype=np.float32)
        n = len(self.piece_actions[self.current_piece])
        mask[:n] = 1.0
        return mask

    # ── rendering ─────────────────────────────────────────────
    def render_board(self) -> np.ndarray:
        """Return the board as a numpy array (height x width) for visualisation."""
        return self.board.copy()

    def get_board_rgb(self, cell_size: int = 20) -> np.ndarray:
        """Return an RGB image (H, W, 3) of the current board."""
        COLORS = [
            [30,  30,  30],   # 0 = empty (dark grey)
            [0,   240, 240],  # 1 = I  (cyan)
            [240, 240, 0],    # 2 = O  (yellow)
            [160, 0,   240],  # 3 = T  (purple)
            [0,   240, 0],    # 4 = S  (green)
            [240, 0,   0],    # 5 = Z  (red)
            [240, 160, 0],    # 6 = L  (orange)
            [0,   0,   240],  # 7 = J  (blue)
        ]
        img = np.zeros((self.height * cell_size, self.width * cell_size, 3),
                        dtype=np.uint8)
        for r in range(self.height):
            for c in range(self.width):
                color = COLORS[int(self.board[r, c])]
                r0, r1 = r * cell_size, (r + 1) * cell_size
                c0, c1 = c * cell_size, (c + 1) * cell_size
                img[r0+1:r1, c0+1:c1] = color  # 1px grid gap
        return img
