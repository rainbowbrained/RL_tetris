"""
Visualization & GIF Utilities for Tetris-Lite RL
=================================================
Functions that generate frames and save animated GIFs showing
the agent playing, learning curves overlaid on gameplay, etc.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend for GIF generation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
import imageio
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from tetris_env import TetrisLiteEnv, PIECE_NAMES

# Tetromino colours (RGB float)
PIECE_COLORS = [
    (0.12, 0.12, 0.12),   # 0  empty
    (0.00, 0.94, 0.94),   # 1  I  cyan
    (0.94, 0.94, 0.00),   # 2  O  yellow
    (0.63, 0.00, 0.94),   # 3  T  purple
    (0.00, 0.94, 0.00),   # 4  S  green
    (0.94, 0.00, 0.00),   # 5  Z  red
    (0.94, 0.63, 0.00),   # 6  L  orange
    (0.00, 0.00, 0.94),   # 7  J  blue
]


# ──────────────────────────────────────────────────────────────
# Core board-to-image
# ──────────────────────────────────────────────────────────────

def board_to_rgb(board: np.ndarray, cell: int = 20) -> np.ndarray:
    """Convert an integer board (H,W) to an RGB uint8 image."""
    H, W = board.shape
    img = np.zeros((H * cell, W * cell, 3), dtype=np.uint8)
    for r in range(H):
        for c in range(W):
            color = PIECE_COLORS[int(board[r, c])]
            r0, r1 = r * cell, (r + 1) * cell
            c0, c1 = c * cell, (c + 1) * cell
            img[r0 + 1 : r1, c0 + 1 : c1] = [int(v * 255) for v in color]
    return img


# ──────────────────────────────────────────────────────────────
# Play one episode and collect frames
# ──────────────────────────────────────────────────────────────

def play_episode(env: TetrisLiteEnv, agent, max_frames: int = 300, device="cpu"):
    """
    Run the agent in the environment for one episode.
    Returns: frames (list of RGB arrays), total_reward, total_lines
    """
    obs = env.reset()
    frames = [board_to_rgb(env.render_board())]
    total_reward = 0.0
    total_lines = 0
    for _ in range(max_frames):
        mask = env.legal_action_mask()
        action, _, _ = agent.select_action(obs, mask, env=env)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        total_lines += info["lines"]
        frames.append(board_to_rgb(env.render_board()))
        if done:
            break
    return frames, total_reward, total_lines


def play_episode_random(env: TetrisLiteEnv, max_frames: int = 300):
    """Run a random policy."""
    obs = env.reset()
    frames = [board_to_rgb(env.render_board())]
    total_reward = 0.0
    total_lines = 0
    for _ in range(max_frames):
        mask = env.legal_action_mask()
        n_legal = int(mask.sum())
        action = np.random.randint(n_legal) if n_legal > 0 else 0
        obs, reward, done, info = env.step(action)
        total_reward += reward
        total_lines += info["lines"]
        frames.append(board_to_rgb(env.render_board()))
        if done:
            break
    return frames, total_reward, total_lines


# ──────────────────────────────────────────────────────────────
# Save GIF
# ──────────────────────────────────────────────────────────────

def save_gif(frames: List[np.ndarray], path: str, fps: int = 8):
    """Save a list of RGB frames as an animated GIF."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, loop=0)
    print(f"Saved GIF → {path}  ({len(frames)} frames)")


# ──────────────────────────────────────────────────────────────
# Annotated frame (board + HUD overlay)
# ──────────────────────────────────────────────────────────────

def annotated_frame(
    board: np.ndarray,
    step: int,
    reward: float,
    lines: int,
    piece_name: str = "",
    cell: int = 20,
) -> np.ndarray:
    """Render the board with a small matplotlib HUD and return as RGB array."""
    img = board_to_rgb(board, cell)
    H_px, W_px = img.shape[:2]

    dpi = 80
    fig_w = (W_px + 160) / dpi
    fig_h = H_px / dpi
    fig, (ax_board, ax_info) = plt.subplots(
        1, 2, figsize=(fig_w, fig_h), dpi=dpi,
        gridspec_kw={"width_ratios": [W_px, 160]},
    )

    ax_board.imshow(img)
    ax_board.axis("off")

    ax_info.axis("off")
    txt = (
        f"Step  {step}\n"
        f"Reward {reward:+.1f}\n"
        f"Lines  {lines}\n"
    )
    if piece_name:
        txt += f"Piece  {piece_name}"
    ax_info.text(
        0.1, 0.7, txt, fontsize=11, fontfamily="monospace",
        verticalalignment="top", transform=ax_info.transAxes,
        color="white",
    )
    fig.patch.set_facecolor("black")
    fig.tight_layout(pad=0.3)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    rgb = buf[:, :, :3].copy()
    plt.close(fig)
    return rgb


def play_episode_annotated(
    env: TetrisLiteEnv,
    agent,
    max_frames: int = 300,
    device="cpu",
) -> List[np.ndarray]:
    """Play an episode returning annotated frames (board + HUD)."""
    obs = env.reset()
    frames = []
    total_reward = 0.0
    total_lines = 0
    step = 0

    frames.append(annotated_frame(
        env.render_board(), step, total_reward, total_lines,
        PIECE_NAMES[env.current_piece],
    ))

    for _ in range(max_frames):
        mask = env.legal_action_mask()
        action, _, _ = agent.select_action(obs, mask, env=env)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        total_lines += info["total_lines"]
        step += 1

        frames.append(annotated_frame(
            env.render_board(), step, total_reward, total_lines,
            PIECE_NAMES[env.current_piece] if not done else "—",
        ))
        if done:
            break
    return frames


# ──────────────────────────────────────────────────────────────
# Learning-curve plot (static)
# ──────────────────────────────────────────────────────────────

def plot_learning_curves(
    metrics: Dict[str, List[float]],
    title: str = "Training Progress",
    window: int = 20,
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot smoothed learning curves.
    metrics: {"label": [values_per_iteration], ...}
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, vals in metrics.items():
        vals = np.array(vals, dtype=np.float64)
        if len(vals) >= window:
            smooth = np.convolve(vals, np.ones(window) / window, mode="valid")
            ax.plot(smooth, label=label, linewidth=2)
        else:
            ax.plot(vals, label=label, linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
        print(f"Saved plot → {save_path}")
    return fig


# ──────────────────────────────────────────────────────────────
# Animated learning-curve GIF
# ──────────────────────────────────────────────────────────────

def learning_curve_gif(
    all_rewards: Dict[str, List[float]],
    path: str,
    window: int = 10,
    step: int = 5,
    fps: int = 12,
):
    """
    Create an animated GIF of the learning curve being drawn over time.
    all_rewards: {"REINFORCE": [...], "PPO": [...]}
    """
    max_len = max(len(v) for v in all_rewards.values())
    frames = []
    colors = {"REINFORCE": "#e74c3c", "PPO": "#2ecc71"}

    for end in range(window, max_len + 1, step):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for label, vals in all_rewards.items():
            vals = np.array(vals[:end], dtype=np.float64)
            if len(vals) >= window:
                smooth = np.convolve(vals, np.ones(window) / window, mode="valid")
                ax.plot(smooth, label=label, color=colors.get(label, None), linewidth=2)
            else:
                ax.plot(vals, label=label, color=colors.get(label, None), linewidth=2)
        ax.set_xlim(0, max_len)
        y_all = np.concatenate([np.array(v) for v in all_rewards.values()])
        ax.set_ylim(np.percentile(y_all, 1) - 5, np.percentile(y_all, 99) + 5)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Reward")
        ax.set_title("Learning Curves (animated)")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        frames.append(buf[:, :, :3].copy())
        plt.close(fig)

    save_gif(frames, path, fps=fps)


# ──────────────────────────────────────────────────────────────
# Side-by-side comparison GIF
# ──────────────────────────────────────────────────────────────

def side_by_side_gif(
    frames_left: List[np.ndarray],
    frames_right: List[np.ndarray],
    label_left: str,
    label_right: str,
    path: str,
    fps: int = 6,
):
    """Create a side-by-side GIF comparing two agents."""
    max_len = max(len(frames_left), len(frames_right))
    combined = []

    for i in range(max_len):
        fl = frames_left[min(i, len(frames_left) - 1)]
        fr = frames_right[min(i, len(frames_right) - 1)]

        # Resize to same height
        h = max(fl.shape[0], fr.shape[0])
        if fl.shape[0] < h:
            pad = np.zeros((h - fl.shape[0], fl.shape[1], 3), dtype=np.uint8)
            fl = np.vstack([pad, fl])
        if fr.shape[0] < h:
            pad = np.zeros((h - fr.shape[0], fr.shape[1], 3), dtype=np.uint8)
            fr = np.vstack([pad, fr])

        # Add labels
        dpi = 80
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, h / dpi + 0.6), dpi=dpi)
        ax1.imshow(fl); ax1.set_title(label_left, fontsize=13, color="white")
        ax1.axis("off")
        ax2.imshow(fr); ax2.set_title(label_right, fontsize=13, color="white")
        ax2.axis("off")
        fig.patch.set_facecolor("black")
        fig.tight_layout(pad=0.5)

        fig.canvas.draw()
        w2, h2 = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h2, w2, 4)
        combined.append(buf[:, :, :3].copy())
        plt.close(fig)

    save_gif(combined, path, fps=fps)


# ──────────────────────────────────────────────────────────────
# Training-snapshot GIF (board at checkpoints during training)
# ──────────────────────────────────────────────────────────────

def training_snapshots_gif(
    snapshot_boards: List[Tuple],
    path: str,
    fps: int = 2,
):
    """
    snapshot_boards: list of (iteration, board_array, reward, lines)
    Creates a GIF cycling through board snapshots taken at different
    stages of training.
    """
    frames = []
    for (iteration, board, reward, lines) in snapshot_boards:
        img = board_to_rgb(board, cell=20)
        H_px, W_px = img.shape[:2]
        dpi = 80
        fig, (ax_board, ax_info) = plt.subplots(
            1, 2, figsize=((W_px + 200) / dpi, H_px / dpi), dpi=dpi,
            gridspec_kw={"width_ratios": [W_px, 200]},
        )
        ax_board.imshow(img); ax_board.axis("off")
        ax_info.axis("off")
        txt = (
            f"Training iter {iteration}\n\n"
            f"Ep. reward  {reward:+.1f}\n"
            f"Lines cleared {lines}\n"
        )
        ax_info.text(
            0.05, 0.75, txt, fontsize=12, fontfamily="monospace",
            verticalalignment="top", transform=ax_info.transAxes,
            color="white",
        )
        fig.patch.set_facecolor("black")
        fig.tight_layout(pad=0.3)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        frames.append(buf[:, :, :3].copy())
        plt.close(fig)

    save_gif(frames, path, fps=fps)


