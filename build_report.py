#!/usr/bin/env python3
"""
build_report.py
===============
Generate a report from a saved checkpoint (.pt) and training metrics JSON.

The script is model-aware and supports current training artifacts:
- REINFORCE checkpoints saved with key: "net"
- PPO checkpoints saved with keys: "policy", "value"

It also tries to infer policy/value architecture from checkpoint payloads:
- Policy: MLP or CNN
- Value: linear (closed-form weights) or MLP
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

if "MPLCONFIGDIR" not in os.environ:
    mpl_cache = Path("/tmp/matplotlib_cache")
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache)

import matplotlib
import numpy as np
import torch
from torch.distributions import Categorical

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lib.agents import (
    CNNPolicyNetwork,
    LinearValueEstimator,
    MLPValueEstimator,
    PolicyNetwork,
    ReinforceAgent,
)
from lib.env import NUM_PIECES, TetrisLiteEnv
from lib.visualize import play_episode, play_episode_random, side_by_side_gif


# --------------------------- IO / parsing helpers --------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a training report from checkpoint and metrics JSON."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to checkpoint .pt file",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        required=True,
        help="Path to training metrics JSON file",
    )
    parser.add_argument(
        "--metrics-key",
        type=str,
        default=None,
        help=(
            "Top-level key inside metrics JSON (e.g. reinforce, ppo, ppo_cnn_linear). "
            "If omitted, script tries to infer."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for report artifacts. "
            "Default: reports/<checkpoint_stem>_<metrics_key_or_inferred>"
        ),
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional custom report title",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=15,
        help="Window size for moving-average overlays",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=None,
        help=(
            "Seed for gameplay-comparison GIFs. If omitted, script auto-selects "
            "a median-performance seed."
        ),
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=200,
        help="Max steps/frames for gameplay-comparison GIFs (legacy default: 200).",
    )
    parser.add_argument(
        "--gif-fps",
        type=int,
        default=6,
        help="FPS for gameplay-comparison GIFs (legacy default: 6).",
    )
    parser.add_argument(
        "--seed-range-start",
        type=int,
        default=5000,
        help="Start seed (inclusive) for automatic median-seed selection.",
    )
    parser.add_argument(
        "--seed-range-count",
        type=int,
        default=50,
        help="Number of seeds for automatic median-seed selection.",
    )
    parser.add_argument(
        "--reinforce-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional REINFORCE checkpoint for comparison GIFs. "
            "If omitted, script tries <checkpoint_dir>/reinforce.pt."
        ),
    )
    parser.add_argument(
        "--skip-gifs",
        action="store_true",
        help="Disable gameplay-comparison GIF generation.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def load_checkpoint(path: Path) -> Dict[str, Any]:
    # weights_only=False is needed when checkpoint includes non-tensor objects.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Older torch without weights_only argument.
        return torch.load(path, map_location="cpu")


def to_float_array(values: Any) -> np.ndarray:
    if not isinstance(values, list):
        return np.array([], dtype=np.float64)
    out: List[float] = []
    for v in values:
        if v is None:
            out.append(np.nan)
        else:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                out.append(np.nan)
    return np.array(out, dtype=np.float64)


def valid_values(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    return arr[~np.isnan(arr)]


def first_valid(arr: np.ndarray) -> Optional[float]:
    vv = valid_values(arr)
    return float(vv[0]) if vv.size else None


def last_valid(arr: np.ndarray) -> Optional[float]:
    vv = valid_values(arr)
    return float(vv[-1]) if vv.size else None


def max_valid(arr: np.ndarray) -> Optional[float]:
    vv = valid_values(arr)
    return float(np.max(vv)) if vv.size else None


def mean_valid(arr: np.ndarray) -> Optional[float]:
    vv = valid_values(arr)
    return float(np.mean(vv)) if vv.size else None


def fmt(v: Optional[float], digits: int = 3) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}"


def smooth_series(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or arr.size == 0:
        return arr.copy()
    out = np.full_like(arr, np.nan, dtype=np.float64)
    for i in range(arr.size):
        lo = max(0, i - window + 1)
        chunk = arr[lo : i + 1]
        vv = valid_values(chunk)
        if vv.size:
            out[i] = float(np.mean(vv))
    return out


# -------------------------- model / metrics inference ----------------------- #


@dataclass
class ModelInfo:
    family: str
    policy_arch: str
    value_arch: str
    variant_id: str
    label: str


def infer_policy_arch_from_state_dict(sd: Dict[str, Any]) -> str:
    keys = list(sd.keys())
    if any(k.startswith("conv.") or ".conv." in k for k in keys):
        return "cnn"
    if any(k.startswith("shared.") or ".shared." in k for k in keys):
        return "mlp"
    if any(k.startswith("fc.") or ".fc." in k for k in keys):
        # Could still be CNN+FC, but if no conv keys exist it's most likely MLP.
        return "mlp"
    return "unknown"


def infer_value_arch(value_payload: Any) -> str:
    if isinstance(value_payload, dict):
        keys = list(value_payload.keys())
        if "w" in keys and len(keys) <= 2:
            return "linear"
        if any(k.endswith(".weight") or ".weight" in k for k in keys):
            return "mlp"
    if isinstance(value_payload, np.ndarray):
        return "linear"
    return "unknown"


def detect_model_info(ckpt: Dict[str, Any]) -> ModelInfo:
    if "net" in ckpt and "policy" not in ckpt:
        return ModelInfo(
            family="reinforce",
            policy_arch="mlp",
            value_arch="implicit_mean_baseline",
            variant_id="reinforce",
            label="REINFORCE (MLP policy)",
        )
    if "policy" in ckpt:
        policy_arch = "unknown"
        if isinstance(ckpt["policy"], dict):
            policy_arch = infer_policy_arch_from_state_dict(ckpt["policy"])
        value_arch = infer_value_arch(ckpt.get("value"))
        p_lbl = policy_arch.upper() if policy_arch != "unknown" else "Unknown"
        v_lbl = value_arch.upper() if value_arch != "unknown" else "Unknown"
        variant_id = f"ppo_{policy_arch}_{value_arch}"
        return ModelInfo(
            family="ppo",
            policy_arch=policy_arch,
            value_arch=value_arch,
            variant_id=variant_id,
            label=f"PPO ({p_lbl} policy, {v_lbl} value)",
        )
    return ModelInfo(
        family="unknown",
        policy_arch="unknown",
        value_arch="unknown",
        variant_id="unknown",
        label="Unknown model",
    )


def looks_like_metrics_leaf(obj: Dict[str, Any]) -> bool:
    keys = set(obj.keys())
    known = {
        "rewards",
        "lines",
        "iter_mean_reward",
        "iter_mean_lines",
        "policy_loss",
        "entropy",
    }
    return len(keys & known) >= 2


def select_metrics_section(
    metrics_root: Dict[str, Any],
    metrics_key: Optional[str],
    model_info: ModelInfo,
) -> Tuple[str, Dict[str, Any]]:
    if metrics_key:
        if metrics_key not in metrics_root:
            raise KeyError(
                f"metrics key '{metrics_key}' not found. Available keys: "
                f"{sorted(metrics_root.keys())}"
            )
        section = metrics_root[metrics_key]
        if not isinstance(section, dict):
            raise ValueError(f"metrics section '{metrics_key}' must be a JSON object")
        return metrics_key, section

    # Case 1: JSON itself already looks like a metrics leaf.
    if looks_like_metrics_leaf(metrics_root):
        return "<root>", metrics_root

    if not isinstance(metrics_root, dict):
        raise ValueError("metrics JSON must be an object")

    # Case 2: standard keys.
    preferred = model_info.family
    if preferred in metrics_root and isinstance(metrics_root[preferred], dict):
        return preferred, metrics_root[preferred]

    # Case 3: fuzzy by family token.
    if model_info.family in {"ppo", "reinforce"}:
        candidates = [
            k
            for k, v in metrics_root.items()
            if isinstance(v, dict) and model_info.family in k.lower()
        ]
        if len(candidates) == 1:
            k = candidates[0]
            return k, metrics_root[k]

    # Case 4: single top-level entry.
    object_keys = [k for k, v in metrics_root.items() if isinstance(v, dict)]
    if len(object_keys) == 1:
        k = object_keys[0]
        return k, metrics_root[k]

    raise ValueError(
        "Could not infer metrics section. Pass --metrics-key explicitly. "
        f"Top-level keys: {sorted(metrics_root.keys())}"
    )


# ------------------------------ metric extractors --------------------------- #


@dataclass
class CoreSeries:
    reward: np.ndarray
    lines: np.ndarray
    length: np.ndarray
    fixed_eval_x: np.ndarray
    fixed_eval_reward: np.ndarray
    fixed_eval_lines: np.ndarray
    x_label: str


def extract_core_series(metrics: Dict[str, Any], family: str) -> CoreSeries:
    if family == "reinforce":
        return CoreSeries(
            reward=to_float_array(metrics.get("rewards", [])),
            lines=to_float_array(metrics.get("lines", [])),
            length=to_float_array(metrics.get("steps_survived", [])),
            fixed_eval_x=to_float_array(metrics.get("fixed_eval_episodes", [])),
            fixed_eval_reward=to_float_array(metrics.get("fixed_eval_mean_reward", [])),
            fixed_eval_lines=to_float_array(metrics.get("fixed_eval_mean_lines", [])),
            x_label="Episode",
        )
    return CoreSeries(
        reward=to_float_array(metrics.get("iter_mean_reward", metrics.get("ep_rewards", []))),
        lines=to_float_array(metrics.get("iter_mean_lines", metrics.get("ep_lines", []))),
        length=to_float_array(metrics.get("ep_length_mean", metrics.get("ep_steps", []))),
        fixed_eval_x=to_float_array(metrics.get("fixed_eval_iters", [])),
        fixed_eval_reward=to_float_array(metrics.get("fixed_eval_mean_reward", [])),
        fixed_eval_lines=to_float_array(metrics.get("fixed_eval_mean_lines", [])),
        x_label="Iteration",
    )


def correlation(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if a.size == 0 or b.size == 0:
        return None
    n = min(a.size, b.size)
    aa = a[:n]
    bb = b[:n]
    mask = (~np.isnan(aa)) & (~np.isnan(bb))
    aa = aa[mask]
    bb = bb[mask]
    if aa.size < 3:
        return None
    if float(np.std(aa)) < 1e-12 or float(np.std(bb)) < 1e-12:
        return None
    return float(np.corrcoef(aa, bb)[0, 1])


def series_summary(arr: np.ndarray) -> Dict[str, Optional[float]]:
    return {
        "first": first_valid(arr),
        "last": last_valid(arr),
        "max": max_valid(arr),
        "mean": mean_valid(arr),
        "delta": (
            (last_valid(arr) - first_valid(arr))
            if first_valid(arr) is not None and last_valid(arr) is not None
            else None
        ),
    }


# ---------------------------------- plotting -------------------------------- #


def save_core_plot(
    core: CoreSeries,
    out_path: Path,
    smooth_window: int,
    title_prefix: str,
):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    # Reward
    x = np.arange(1, core.reward.size + 1)
    axes[0].plot(x, core.reward, alpha=0.35, color="#1f77b4", label="raw")
    if core.reward.size:
        axes[0].plot(
            x,
            smooth_series(core.reward, smooth_window),
            color="#1f77b4",
            linewidth=2,
            label=f"smoothed ({smooth_window})",
        )
    axes[0].set_title("Mean Reward")
    axes[0].set_xlabel(core.x_label)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    # Lines
    x = np.arange(1, core.lines.size + 1)
    axes[1].plot(x, core.lines, alpha=0.35, color="#2ca02c", label="raw")
    if core.lines.size:
        axes[1].plot(
            x,
            smooth_series(core.lines, smooth_window),
            color="#2ca02c",
            linewidth=2,
            label=f"smoothed ({smooth_window})",
        )
    axes[1].set_title("Mean Lines Cleared")
    axes[1].set_xlabel(core.x_label)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")

    # Episode length
    x = np.arange(1, core.length.size + 1)
    axes[2].plot(x, core.length, alpha=0.35, color="#9467bd", label="raw")
    if core.length.size:
        axes[2].plot(
            x,
            smooth_series(core.length, smooth_window),
            color="#9467bd",
            linewidth=2,
            label=f"smoothed ({smooth_window})",
        )
    axes[2].set_title("Episode Length")
    axes[2].set_xlabel(core.x_label)
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(loc="best")

    # Fixed-seed eval
    if core.fixed_eval_x.size and (
        core.fixed_eval_reward.size or core.fixed_eval_lines.size
    ):
        ax = axes[3]
        if core.fixed_eval_reward.size:
            ax.plot(
                core.fixed_eval_x,
                core.fixed_eval_reward,
                "o-",
                color="#ff7f0e",
                linewidth=2,
                markersize=4,
                label="Eval reward",
            )
        if core.fixed_eval_lines.size:
            ax2 = ax.twinx()
            ax2.plot(
                core.fixed_eval_x,
                core.fixed_eval_lines,
                "s-",
                color="#17becf",
                linewidth=2,
                markersize=4,
                label="Eval lines",
            )
            ax2.set_ylabel("Lines")
            # Combined legend
            lines_l, labels_l = ax.get_legend_handles_labels()
            lines_r, labels_r = ax2.get_legend_handles_labels()
            ax.legend(lines_l + lines_r, labels_l + labels_r, loc="best")
        else:
            ax.legend(loc="best")
        ax.set_title("Fixed-Seed Evaluation")
        ax.set_xlabel(core.x_label)
        ax.set_ylabel("Reward")
        ax.grid(True, alpha=0.25)
    else:
        axes[3].axis("off")
        axes[3].text(
            0.5,
            0.5,
            "No fixed-seed eval metrics",
            ha="center",
            va="center",
            fontsize=11,
        )

    fig.suptitle(f"{title_prefix} — Core Training Curves", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def save_diagnostics_plot(
    metrics: Dict[str, Any],
    family: str,
    out_path: Path,
    x_label: str,
):
    if family == "ppo":
        candidates = [
            ("policy_loss", "Policy Loss", "#1f77b4"),
            ("value_loss", "Value Loss", "#ff7f0e"),
            ("entropy", "Entropy", "#2ca02c"),
            ("approx_kl", "Approx KL", "#17becf"),
            ("clip_fraction", "Clip Fraction", "#bcbd22"),
            ("gradient_norm", "Gradient Norm", "#9467bd"),
            ("explained_variance", "Explained Variance", "#d62728"),
            ("ratio_max", "Ratio Max", "#8c564b"),
        ]
    else:
        candidates = [
            ("policy_loss", "Policy Loss", "#1f77b4"),
            ("entropy", "Entropy", "#2ca02c"),
            ("gradient_norm", "Gradient Norm", "#9467bd"),
        ]

    available = []
    for key, title, color in candidates:
        arr = to_float_array(metrics.get(key, []))
        if arr.size:
            available.append((key, title, color, arr))
    if not available:
        return

    cols = 3
    rows = math.ceil(len(available) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.3 * rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.reshape(-1)

    for ax, (_, title, color, arr) in zip(axes, available):
        x = np.arange(1, arr.size + 1)
        ax.plot(x, arr, color=color, linewidth=1.7)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.grid(True, alpha=0.25)

    for ax in axes[len(available) :]:
        ax.axis("off")

    fig.suptitle("Diagnostics", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def save_reward_decomposition_plot(
    metrics: Dict[str, Any],
    out_path: Path,
    x_label: str,
    smooth_window: int,
):
    keys = [
        ("reward_line_clear", "Line clear", "#2ecc71"),
        ("reward_survival", "Survival", "#3498db"),
        ("reward_hole_penalty", "Hole penalty", "#e74c3c"),
        ("reward_hole_removal", "Hole removal", "#27ae60"),
        ("reward_bump_penalty", "Bump penalty", "#e67e22"),
        ("reward_height_penalty", "Height penalty", "#9b59b6"),
    ]
    series = []
    for k, label, color in keys:
        arr = to_float_array(metrics.get(k, []))
        if arr.size:
            series.append((label, color, arr))
    if not series:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    for label, color, arr in series:
        x = np.arange(1, arr.size + 1)
        ax.plot(
            x,
            smooth_series(arr, smooth_window),
            label=label,
            color=color,
            linewidth=1.8,
        )
    ax.set_title("Reward Decomposition")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Component value")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def save_per_piece_plot(metrics: Dict[str, Any], out_path: Path, x_label: str):
    ppr = metrics.get("per_piece_rewards")
    if not isinstance(ppr, dict) or not ppr:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    plotted = False
    for piece, vals in ppr.items():
        arr = to_float_array(vals)
        if arr.size == 0:
            continue
        x = np.arange(1, arr.size + 1)
        ax.plot(x, arr, label=str(piece), linewidth=1.4)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_title("Per-Piece Mean Reward")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Mean reward")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", ncol=4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def save_value_calibration_plot(metrics: Dict[str, Any], out_path: Path):
    iters = metrics.get("value_calibration_iters", [])
    pred = metrics.get("value_calibration_pred", [])
    actual = metrics.get("value_calibration_actual", [])
    if not (isinstance(iters, list) and isinstance(pred, list) and isinstance(actual, list)):
        return
    n = min(len(iters), len(pred), len(actual))
    if n == 0:
        return

    cols = min(4, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.7 * cols, 3.9 * rows), squeeze=False)

    for i in range(n):
        ax = axes[i // cols][i % cols]
        p = to_float_array(pred[i])
        a = to_float_array(actual[i])
        m = min(p.size, a.size)
        if m == 0:
            ax.set_title(f"Iter {iters[i]} (empty)")
            ax.axis("off")
            continue
        p = p[:m]
        a = a[:m]
        mask = (~np.isnan(p)) & (~np.isnan(a))
        p = p[mask]
        a = a[mask]
        if p.size == 0:
            ax.set_title(f"Iter {iters[i]} (empty)")
            ax.axis("off")
            continue
        ax.scatter(a, p, alpha=0.28, s=11, color="#1f77b4")
        lo = float(min(np.min(a), np.min(p)))
        hi = float(max(np.max(a), np.max(p)))
        ax.plot([lo, hi], [lo, hi], "r--", alpha=0.6, linewidth=1)
        ax.set_title(f"Iter {iters[i]}")
        ax.set_xlabel("Actual return")
        ax.set_ylabel("Predicted V(s)")
        ax.grid(True, alpha=0.25)

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.suptitle("Value Calibration")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ------------------------------- GIF generation ----------------------------- #


def _int_cfg(d: Dict[str, Any], key: str, default: int) -> int:
    try:
        return int(d.get(key, default))
    except (TypeError, ValueError):
        return default


def make_env_from_checkpoint(
    ckpt: Dict[str, Any],
    seed: int,
    max_steps: Optional[int] = None,
) -> TetrisLiteEnv:
    env_cfg = ckpt.get("env_config", {}) if isinstance(ckpt, dict) else {}
    w = _int_cfg(env_cfg, "board_w", 6)
    h = _int_cfg(env_cfg, "board_h", 20)
    ms = _int_cfg(env_cfg, "max_steps", 500) if max_steps is None else int(max_steps)
    return TetrisLiteEnv(width=w, height=h, max_steps=ms, seed=seed)


def load_reinforce_agent_for_eval(ckpt: Dict[str, Any]) -> ReinforceAgent:
    if "net" not in ckpt:
        raise ValueError("Checkpoint does not contain REINFORCE weights ('net').")
    hp = ckpt.get("hyperparams", {}) if isinstance(ckpt.get("hyperparams"), dict) else {}
    env = make_env_from_checkpoint(ckpt, seed=0, max_steps=1)
    agent = ReinforceAgent(
        obs_dim=env.obs_size(),
        act_dim=env.num_actions(),
        lr=float(hp.get("lr", 3e-4)),
        gamma=float(hp.get("gamma", 0.99)),
        hidden=int(hp.get("hidden", 128)),
        ent_coef=0.0,
        epsilon=0.0,
        device="cpu",
    )
    agent.net.load_state_dict(ckpt["net"])
    agent.epsilon = 0.0
    agent.net.eval()
    return agent


class PPOInferenceAgent:
    """Lightweight inference-only PPO agent wrapper for gameplay GIFs."""

    def __init__(self, policy: torch.nn.Module, value: Any, policy_arch: str, device: str = "cpu"):
        self.policy = policy
        self.value = value
        self.policy_arch = policy_arch
        self.device = device

    def select_action(self, obs: np.ndarray, mask: np.ndarray, env=None):
        if self.value is not None and hasattr(self.value, "predict_single"):
            value = float(self.value.predict_single(obs))
        else:
            value = 0.0

        if self.policy_arch == "cnn":
            if env is None:
                raise ValueError("PPOInferenceAgent(cnn) requires env for CNN observation.")
            board_2d, piece_vec = env.get_cnn_obs()
            board_t = torch.as_tensor(
                board_2d, dtype=torch.float32, device=self.device
            ).unsqueeze(0)  # (1,1,H,W)
            piece_t = torch.as_tensor(
                piece_vec, dtype=torch.float32, device=self.device
            ).unsqueeze(0)  # (1,P)
            with torch.no_grad():
                logits = self.policy(board_t, piece_t)[0]
        elif self.policy_arch == "mlp":
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                logits, _ = self.policy(obs_t, None)
            logits = logits[0]
        else:
            raise ValueError(f"Unsupported PPO policy architecture '{self.policy_arch}'.")

        mask_t = torch.as_tensor(mask, dtype=torch.float32, device=self.device)
        logits = logits + (1 - mask_t) * (-1e8)
        dist = Categorical(logits=logits)
        action_t = dist.sample()
        log_prob = dist.log_prob(action_t).item()
        return action_t.item(), log_prob, value


def load_ppo_agent_for_eval(ckpt: Dict[str, Any], model_info: ModelInfo) -> PPOInferenceAgent:
    if "policy" not in ckpt:
        raise ValueError("Checkpoint does not contain PPO policy weights ('policy').")

    hp = ckpt.get("hyperparams", {}) if isinstance(ckpt.get("hyperparams"), dict) else {}
    env = make_env_from_checkpoint(ckpt, seed=0, max_steps=1)
    obs_dim = env.obs_size()
    act_dim = env.num_actions()
    env_cfg = ckpt.get("env_config", {}) if isinstance(ckpt, dict) else {}
    board_h = _int_cfg(env_cfg, "board_h", 20)
    board_w = _int_cfg(env_cfg, "board_w", 6)

    policy_arch = model_info.policy_arch
    if policy_arch == "unknown":
        policy_arch = infer_policy_arch_from_state_dict(ckpt["policy"])

    if policy_arch == "cnn":
        policy = CNNPolicyNetwork(
            board_h=board_h,
            board_w=board_w,
            num_pieces=NUM_PIECES,
            act_dim=act_dim,
            n_filters=int(hp.get("n_filters", 32)),
            hidden=int(hp.get("hidden", 128)),
        ).to("cpu")
    elif policy_arch == "mlp":
        policy = PolicyNetwork(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden=int(hp.get("hidden", 128)),
        ).to("cpu")
    else:
        raise ValueError(f"Unsupported PPO policy architecture '{policy_arch}'.")

    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    value_payload = ckpt.get("value", None)
    value_arch = model_info.value_arch
    if value_arch == "unknown" and isinstance(value_payload, dict):
        value_arch = infer_value_arch(value_payload)

    value: Any = None
    if value_payload is not None:
        if value_arch == "linear" or (
            isinstance(value_payload, dict) and "w" in value_payload
        ):
            value = LinearValueEstimator(obs_dim=obs_dim)
            value.load_state_dict(value_payload)
        elif value_arch == "mlp":
            value = MLPValueEstimator(
                obs_dim=obs_dim,
                hidden1=int(hp.get("value_hidden1", 128)),
                hidden2=int(hp.get("value_hidden2", 64)),
                lr=float(hp.get("value_lr", 1e-3)),
                device="cpu",
            )
            value.load_state_dict(value_payload)
            value.eval()

    return PPOInferenceAgent(
        policy=policy,
        value=value,
        policy_arch=policy_arch,
        device="cpu",
    )


def _play_random_seeded(
    env: TetrisLiteEnv,
    max_frames: int,
    seed: int,
) -> Tuple[List[np.ndarray], float, int]:
    # play_episode_random uses global np.random choice; seed it temporarily.
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        return play_episode_random(env, max_frames=max_frames)
    finally:
        np.random.set_state(state)


def _play_episode_seeded(
    env: TetrisLiteEnv,
    agent: Any,
    max_frames: int,
    seed: int,
) -> Tuple[List[np.ndarray], float, int]:
    # Keep stochastic policies reproducible when evaluating candidate seeds.
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        return play_episode(env, agent, max_frames=max_frames, device="cpu")
    finally:
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)


def _select_median_seed_for_agent(
    agent: Any,
    ckpt: Dict[str, Any],
    max_frames: int,
    seed_start: int,
    seed_count: int,
) -> Tuple[int, Dict[str, float]]:
    records: List[Tuple[int, float, float, int]] = []  # (seed, lines, reward, steps)
    for seed in range(seed_start, seed_start + max(seed_count, 1)):
        env = make_env_from_checkpoint(ckpt, seed=seed, max_steps=max_frames)
        frames, reward, lines = _play_episode_seeded(
            env, agent, max_frames=max_frames, seed=seed
        )
        steps = max(len(frames) - 1, 0)
        records.append((seed, float(lines), float(reward), steps))

    if not records:
        raise RuntimeError("No candidate seeds available for median seed selection.")

    ranked = sorted(records, key=lambda x: (x[1], x[2], x[0]))
    mid = len(ranked) // 2
    chosen = ranked[mid]
    all_lines = np.array([r[1] for r in ranked], dtype=np.float64)
    all_rewards = np.array([r[2] for r in ranked], dtype=np.float64)
    meta = {
        "chosen_seed": int(chosen[0]),
        "chosen_lines": float(chosen[1]),
        "chosen_reward": float(chosen[2]),
        "chosen_steps": int(chosen[3]),
        "median_lines_pool": float(np.median(all_lines)),
        "median_reward_pool": float(np.median(all_rewards)),
        "p25_lines_pool": float(np.percentile(all_lines, 25)),
        "p75_lines_pool": float(np.percentile(all_lines, 75)),
    }
    return int(chosen[0]), meta


def _resolve_reinforce_checkpoint(
    main_checkpoint: Path,
    explicit_path: Optional[Path],
) -> Optional[Path]:
    if explicit_path is not None:
        p = explicit_path.resolve()
        return p if p.exists() else None
    candidate = (main_checkpoint.parent / "reinforce.pt").resolve()
    return candidate if candidate.exists() else None


def generate_comparison_gifs(
    out_dir: Path,
    checkpoint_path: Path,
    ckpt: Dict[str, Any],
    model_info: ModelInfo,
    reinforce_checkpoint: Optional[Path],
    eval_seed: Optional[int],
    eval_steps: int,
    gif_fps: int,
    seed_range_start: int,
    seed_range_count: int,
) -> Tuple[List[Path], List[str]]:
    """
    Reproduce legacy notebook-style comparison GIFs:
    - Random vs REINFORCE
    - REINFORCE vs PPO
    """
    gif_paths: List[Path] = []
    notes: List[str] = []

    if model_info.family == "reinforce":
        try:
            rf_agent = load_reinforce_agent_for_eval(ckpt)
            if eval_seed is None:
                chosen_seed, meta = _select_median_seed_for_agent(
                    agent=rf_agent,
                    ckpt=ckpt,
                    max_frames=eval_steps,
                    seed_start=seed_range_start,
                    seed_count=seed_range_count,
                )
                notes.append(
                    "Auto-selected median seed for GIFs "
                    f"(REINFORCE over [{seed_range_start}, {seed_range_start + seed_range_count - 1}]): "
                    f"seed={chosen_seed}, lines={meta['chosen_lines']:.1f}, "
                    f"reward={meta['chosen_reward']:+.1f}, "
                    f"pool median lines={meta['median_lines_pool']:.1f}."
                )
                seed_for_gif = chosen_seed
            else:
                seed_for_gif = eval_seed

            env_rand = make_env_from_checkpoint(ckpt, seed=seed_for_gif, max_steps=eval_steps)
            env_rf = make_env_from_checkpoint(ckpt, seed=seed_for_gif, max_steps=eval_steps)
            frames_rand, rew_rand, lines_rand = _play_random_seeded(
                env_rand, max_frames=eval_steps, seed=seed_for_gif
            )
            frames_rf, rew_rf, lines_rf = _play_episode_seeded(
                env_rf, rf_agent, max_frames=eval_steps, seed=seed_for_gif
            )
            out = out_dir / "gif_random_vs_reinforce.gif"
            side_by_side_gif(
                frames_rand,
                frames_rf,
                "Random Agent",
                "REINFORCE",
                str(out),
                fps=gif_fps,
            )
            gif_paths.append(out)
            notes.append(
                "GIF comparison (Random vs REINFORCE): "
                f"seed={seed_for_gif}, "
                f"random lines={lines_rand}, reinforce lines={lines_rf}, "
                f"random reward={rew_rand:+.1f}, reinforce reward={rew_rf:+.1f}."
            )
        except Exception as e:
            notes.append(f"Could not generate Random vs REINFORCE GIF: {e}")
        return gif_paths, notes

    if model_info.family == "ppo":
        try:
            ppo_agent = load_ppo_agent_for_eval(ckpt, model_info)
        except Exception as e:
            notes.append(f"Could not load PPO agent for GIFs: {e}")
            return gif_paths, notes

        rf_ckpt_path = _resolve_reinforce_checkpoint(checkpoint_path, reinforce_checkpoint)
        if rf_ckpt_path is None:
            notes.append(
                "Skipped REINFORCE comparison GIFs: no REINFORCE checkpoint found. "
                "Pass --reinforce-checkpoint to enable."
            )
            return gif_paths, notes

        try:
            rf_ckpt = load_checkpoint(rf_ckpt_path)
            rf_agent = load_reinforce_agent_for_eval(rf_ckpt)

            if eval_seed is None:
                chosen_seed, meta = _select_median_seed_for_agent(
                    agent=ppo_agent,
                    ckpt=ckpt,
                    max_frames=eval_steps,
                    seed_start=seed_range_start,
                    seed_count=seed_range_count,
                )
                notes.append(
                    "Auto-selected median seed for GIFs "
                    f"(PPO over [{seed_range_start}, {seed_range_start + seed_range_count - 1}]): "
                    f"seed={chosen_seed}, lines={meta['chosen_lines']:.1f}, "
                    f"reward={meta['chosen_reward']:+.1f}, "
                    f"pool median lines={meta['median_lines_pool']:.1f}."
                )
                seed_for_gif = chosen_seed
            else:
                seed_for_gif = eval_seed

            # Random vs REINFORCE (legacy notebook GIF 7)
            env_rand = make_env_from_checkpoint(rf_ckpt, seed=seed_for_gif, max_steps=eval_steps)
            env_rf = make_env_from_checkpoint(rf_ckpt, seed=seed_for_gif, max_steps=eval_steps)
            frames_rand, rew_rand, lines_rand = _play_random_seeded(
                env_rand, max_frames=eval_steps, seed=seed_for_gif
            )
            frames_rf, rew_rf, lines_rf = _play_episode_seeded(
                env_rf, rf_agent, max_frames=eval_steps, seed=seed_for_gif
            )
            out_rr = out_dir / "gif_random_vs_reinforce.gif"
            side_by_side_gif(
                frames_rand,
                frames_rf,
                "Random Agent",
                "REINFORCE",
                str(out_rr),
                fps=gif_fps,
            )
            gif_paths.append(out_rr)
            notes.append(
                "GIF comparison (Random vs REINFORCE): "
                f"seed={seed_for_gif}, "
                f"random lines={lines_rand}, reinforce lines={lines_rf}, "
                f"random reward={rew_rand:+.1f}, reinforce reward={rew_rf:+.1f}."
            )

            # REINFORCE vs PPO (legacy notebook GIF 8)
            env_rf_2 = make_env_from_checkpoint(rf_ckpt, seed=seed_for_gif, max_steps=eval_steps)
            env_ppo = make_env_from_checkpoint(ckpt, seed=seed_for_gif, max_steps=eval_steps)
            frames_rf2, rew_rf2, lines_rf2 = _play_episode_seeded(
                env_rf_2, rf_agent, max_frames=eval_steps, seed=seed_for_gif
            )
            frames_ppo, rew_ppo, lines_ppo = _play_episode_seeded(
                env_ppo, ppo_agent, max_frames=eval_steps, seed=seed_for_gif
            )
            out_rp = out_dir / "gif_reinforce_vs_ppo.gif"
            side_by_side_gif(
                frames_rf2,
                frames_ppo,
                "REINFORCE",
                "PPO",
                str(out_rp),
                fps=gif_fps,
            )
            gif_paths.append(out_rp)
            notes.append(
                "GIF comparison (REINFORCE vs PPO): "
                f"seed={seed_for_gif}, "
                f"reinforce lines={lines_rf2}, ppo lines={lines_ppo}, "
                f"reinforce reward={rew_rf2:+.1f}, ppo reward={rew_ppo:+.1f}."
            )
        except Exception as e:
            notes.append(f"Could not generate REINFORCE/PPO comparison GIFs: {e}")
        return gif_paths, notes

    return gif_paths, notes


# ------------------------------ report writing ------------------------------ #


def summarize_checkpoint(ckpt: Dict[str, Any], model_info: ModelInfo) -> Dict[str, Any]:
    return {
        "model_family": model_info.family,
        "policy_arch": model_info.policy_arch,
        "value_arch": model_info.value_arch,
        "variant_id": model_info.variant_id,
        "label": model_info.label,
        "env_config": ckpt.get("env_config", {}),
        "hyperparams": ckpt.get("hyperparams", {}),
        "reward_weights": ckpt.get("reward_weights", {}),
        "final_metrics_from_checkpoint": ckpt.get("final_metrics", {}),
    }


def derive_findings(
    model_info: ModelInfo,
    core: CoreSeries,
    metrics: Dict[str, Any],
) -> List[str]:
    findings: List[str] = []

    reward_sum = series_summary(core.reward)
    lines_sum = series_summary(core.lines)
    eval_lines = series_summary(core.fixed_eval_lines)
    eval_rew = series_summary(core.fixed_eval_reward)

    if lines_sum["delta"] is not None and lines_sum["delta"] > 0:
        findings.append(
            f"Training improved lines per {core.x_label.lower()}: "
            f"{fmt(lines_sum['first'], 2)} -> {fmt(lines_sum['last'], 2)}."
        )
    if reward_sum["delta"] is not None and reward_sum["delta"] > 0:
        findings.append(
            f"Training reward trend is positive: "
            f"{fmt(reward_sum['first'], 2)} -> {fmt(reward_sum['last'], 2)}."
        )
    if eval_lines["delta"] is not None and eval_lines["delta"] > 0:
        findings.append(
            "Fixed-seed evaluation improved, indicating better cross-seed behavior: "
            f"{fmt(eval_lines['first'], 2)} -> {fmt(eval_lines['last'], 2)} lines."
        )
    if eval_rew["delta"] is not None and eval_rew["delta"] > 0:
        findings.append(
            f"Fixed-seed eval reward improved: "
            f"{fmt(eval_rew['first'], 2)} -> {fmt(eval_rew['last'], 2)}."
        )

    if model_info.family == "ppo":
        ev = to_float_array(metrics.get("explained_variance", []))
        kl = to_float_array(metrics.get("approx_kl", []))
        clip = to_float_array(metrics.get("clip_fraction", []))
        topout = to_float_array(metrics.get("topout_rate", []))

        ev_last = last_valid(ev)
        kl_last = last_valid(kl)
        clip_last = last_valid(clip)
        topout_last = last_valid(topout)

        if ev_last is not None and ev_last < 0.2:
            findings.append(
                f"Critic fit is weak (explained variance last={fmt(ev_last, 3)}), "
                "which can limit PPO sample efficiency."
            )
        if kl_last is not None and clip_last is not None and (
            kl_last > 0.03 or clip_last > 0.25
        ):
            findings.append(
                f"Late training updates are aggressive (KL={fmt(kl_last, 3)}, "
                f"clip fraction={fmt(clip_last, 3)})."
            )
        if topout_last is not None and topout_last > 0.8:
            findings.append(
                f"Top-out rate remains high (last={fmt(topout_last, 3)}), "
                "so policy still operates near failure boundary."
            )

    corr_rl = correlation(core.reward, core.lines)
    if corr_rl is not None:
        findings.append(
            f"Reward/lines correlation over training: r={fmt(corr_rl, 3)}."
        )

    if not findings:
        findings.append("Insufficient metrics to derive robust findings.")
    return findings


def write_markdown_report(
    out_path: Path,
    title: str,
    checkpoint_path: Path,
    metrics_path: Path,
    metrics_key: str,
    ckpt_summary: Dict[str, Any],
    core: CoreSeries,
    metrics: Dict[str, Any],
    plot_files: List[Path],
    findings: List[str],
):
    reward_sum = series_summary(core.reward)
    lines_sum = series_summary(core.lines)
    len_sum = series_summary(core.length)
    eval_rew = series_summary(core.fixed_eval_reward)
    eval_lines = series_summary(core.fixed_eval_lines)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    hp = ckpt_summary.get("hyperparams", {})
    env_cfg = ckpt_summary.get("env_config", {})
    rw = ckpt_summary.get("reward_weights", {})
    fm = ckpt_summary.get("final_metrics_from_checkpoint", {})

    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"_Generated: {now}_")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- Checkpoint: `{checkpoint_path}`")
    lines.append(f"- Metrics JSON: `{metrics_path}`")
    lines.append(f"- Metrics key: `{metrics_key}`")
    lines.append("")
    lines.append("## Model")
    lines.append(f"- Family: `{ckpt_summary.get('model_family', 'n/a')}`")
    lines.append(f"- Policy architecture: `{ckpt_summary.get('policy_arch', 'n/a')}`")
    lines.append(f"- Value architecture: `{ckpt_summary.get('value_arch', 'n/a')}`")
    lines.append(f"- Variant ID: `{ckpt_summary.get('variant_id', 'n/a')}`")
    lines.append(f"- Label: `{ckpt_summary.get('label', 'n/a')}`")
    lines.append("")
    lines.append("## Headline Metrics")
    lines.append("| Metric | First | Last | Best | Delta |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(
        f"| Reward | {fmt(reward_sum['first'],2)} | {fmt(reward_sum['last'],2)} | "
        f"{fmt(reward_sum['max'],2)} | {fmt(reward_sum['delta'],2)} |"
    )
    lines.append(
        f"| Lines | {fmt(lines_sum['first'],2)} | {fmt(lines_sum['last'],2)} | "
        f"{fmt(lines_sum['max'],2)} | {fmt(lines_sum['delta'],2)} |"
    )
    lines.append(
        f"| Episode length | {fmt(len_sum['first'],2)} | {fmt(len_sum['last'],2)} | "
        f"{fmt(len_sum['max'],2)} | {fmt(len_sum['delta'],2)} |"
    )
    lines.append(
        f"| Fixed eval reward | {fmt(eval_rew['first'],2)} | {fmt(eval_rew['last'],2)} | "
        f"{fmt(eval_rew['max'],2)} | {fmt(eval_rew['delta'],2)} |"
    )
    lines.append(
        f"| Fixed eval lines | {fmt(eval_lines['first'],2)} | {fmt(eval_lines['last'],2)} | "
        f"{fmt(eval_lines['max'],2)} | {fmt(eval_lines['delta'],2)} |"
    )
    lines.append("")
    lines.append("## Automatic Findings")
    for f in findings:
        lines.append(f"- {f}")
    lines.append("")
    lines.append("## Config Snapshot")
    lines.append("### Environment")
    lines.append("```json")
    lines.append(json.dumps(env_cfg, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("### Hyperparameters")
    lines.append("```json")
    lines.append(json.dumps(hp, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("### Reward Weights")
    lines.append("```json")
    lines.append(json.dumps(rw, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("### Final Metrics (from checkpoint)")
    lines.append("```json")
    lines.append(json.dumps(fm, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    lines.append("## Figures")
    for p in plot_files:
        rel = p.name
        lines.append(f"### {p.stem.replace('_', ' ').title()}")
        lines.append(f"![{p.stem}]({rel})")
        lines.append("")
    lines.append("## Raw Metric Keys")
    lines.append("```json")
    lines.append(json.dumps(sorted(metrics.keys()), indent=2))
    lines.append("```")
    lines.append("")

    out_path.write_text("\n".join(lines))


def write_summary_json(
    out_path: Path,
    checkpoint_path: Path,
    metrics_path: Path,
    metrics_key: str,
    model_info: ModelInfo,
    core: CoreSeries,
    metrics: Dict[str, Any],
):
    data: Dict[str, Any] = {
        "inputs": {
            "checkpoint": str(checkpoint_path),
            "metrics": str(metrics_path),
            "metrics_key": metrics_key,
        },
        "model": {
            "family": model_info.family,
            "policy_arch": model_info.policy_arch,
            "value_arch": model_info.value_arch,
            "variant_id": model_info.variant_id,
            "label": model_info.label,
        },
        "summary": {
            "reward": series_summary(core.reward),
            "lines": series_summary(core.lines),
            "length": series_summary(core.length),
            "fixed_eval_reward": series_summary(core.fixed_eval_reward),
            "fixed_eval_lines": series_summary(core.fixed_eval_lines),
            "reward_lines_corr": correlation(core.reward, core.lines),
        },
    }

    # Include common diagnostics if present.
    for k in [
        "entropy",
        "policy_loss",
        "value_loss",
        "approx_kl",
        "clip_fraction",
        "gradient_norm",
        "explained_variance",
        "topout_rate",
        "ratio_mean",
        "ratio_max",
    ]:
        arr = to_float_array(metrics.get(k, []))
        if arr.size:
            data["summary"][k] = series_summary(arr)

    out_path.write_text(json.dumps(data, indent=2))


def main():
    args = parse_args()

    checkpoint_path = args.checkpoint.resolve()
    metrics_path = args.metrics.resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics file not found: {metrics_path}")

    ckpt = load_checkpoint(checkpoint_path)
    model_info = detect_model_info(ckpt)

    metrics_root = load_json(metrics_path)
    metrics_key, metrics = select_metrics_section(metrics_root, args.metrics_key, model_info)
    core = extract_core_series(metrics, model_info.family)

    if args.output_dir is None:
        default_dir = Path("reports") / f"{checkpoint_path.stem}_{metrics_key.replace('/', '_')}"
        out_dir = default_dir.resolve()
    else:
        out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    title = args.title or f"{model_info.label} Report"
    ckpt_summary = summarize_checkpoint(ckpt, model_info)
    findings = derive_findings(model_info, core, metrics)

    plot_paths: List[Path] = []
    core_plot = out_dir / "core_training_curves.png"
    save_core_plot(core, core_plot, args.smooth_window, model_info.label)
    plot_paths.append(core_plot)

    diag_plot = out_dir / "diagnostics.png"
    save_diagnostics_plot(metrics, model_info.family, diag_plot, core.x_label)
    if diag_plot.exists():
        plot_paths.append(diag_plot)

    rew_plot = out_dir / "reward_decomposition.png"
    save_reward_decomposition_plot(metrics, rew_plot, core.x_label, args.smooth_window)
    if rew_plot.exists():
        plot_paths.append(rew_plot)

    piece_plot = out_dir / "per_piece_reward.png"
    save_per_piece_plot(metrics, piece_plot, core.x_label)
    if piece_plot.exists():
        plot_paths.append(piece_plot)

    cal_plot = out_dir / "value_calibration.png"
    save_value_calibration_plot(metrics, cal_plot)
    if cal_plot.exists():
        plot_paths.append(cal_plot)

    if not args.skip_gifs:
        gif_paths, gif_notes = generate_comparison_gifs(
            out_dir=out_dir,
            checkpoint_path=checkpoint_path,
            ckpt=ckpt,
            model_info=model_info,
            reinforce_checkpoint=args.reinforce_checkpoint,
            eval_seed=args.eval_seed,
            eval_steps=args.eval_steps,
            gif_fps=args.gif_fps,
            seed_range_start=args.seed_range_start,
            seed_range_count=args.seed_range_count,
        )
        plot_paths.extend(gif_paths)
        findings.extend(gif_notes)

    report_md = out_dir / "report.md"
    write_markdown_report(
        out_path=report_md,
        title=title,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        metrics_key=metrics_key,
        ckpt_summary=ckpt_summary,
        core=core,
        metrics=metrics,
        plot_files=plot_paths,
        findings=findings,
    )

    summary_json = out_dir / "summary.json"
    write_summary_json(
        out_path=summary_json,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        metrics_key=metrics_key,
        model_info=model_info,
        core=core,
        metrics=metrics,
    )

    print(f"Model detected: {model_info.label}")
    print(f"Metrics section: {metrics_key}")
    print(f"Report written: {report_md}")
    print(f"Summary JSON:   {summary_json}")
    if plot_paths:
        print("Artifacts:")
        for p in plot_paths:
            print(f"  - {p}")


if __name__ == "__main__":
    main()
