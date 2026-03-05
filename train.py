#!/usr/bin/env python3

import argparse
import json
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from lib.env import TetrisLiteEnv, PIECE_NAMES, NUM_PIECES, REWARD_WEIGHTS
from lib.agents import RolloutBuffer, ReinforceAgent, PPOAgent
from lib.visualize import (
    play_episode, play_episode_random, save_gif,
    play_episode_annotated, plot_learning_curves,
    learning_curve_gif, side_by_side_gif, training_snapshots_gif,
)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = get_device()

BOARD_W   = 6
BOARD_H   = 20
MAX_STEPS = 500
HIDDEN    = 256

GIF_DIR  = Path("gifs");        GIF_DIR.mkdir(exist_ok=True)
CKPT_DIR = Path("checkpoints"); CKPT_DIR.mkdir(exist_ok=True)
LOG_DIR  = Path("logs");        LOG_DIR.mkdir(exist_ok=True)


def smooth(values, window=20):
    arr = np.array(values, dtype=np.float64)
    if len(arr) < window:
        return arr.tolist()
    return np.convolve(arr, np.ones(window) / window, mode="valid").tolist()


EVAL_SEEDS = list(range(10))


def compute_board_stats(env: TetrisLiteEnv) -> dict:
    return {
        "holes": env._count_holes(),
        "bumpiness": env._bumpiness(),
        "max_height": env._max_height(),
    }


def fixed_seed_eval(agent, seeds=EVAL_SEEDS, max_steps=500):
    results = []
    for s in seeds:
        env = TetrisLiteEnv(width=BOARD_W, height=BOARD_H, max_steps=max_steps, seed=s)
        obs = env.reset()
        total_reward, total_lines, steps = 0.0, 0, 0
        done = False
        while not done:
            mask = env.legal_action_mask()
            action, _, _ = agent.select_action(obs, mask, env=env)
            obs, rew, done, info = env.step(action)
            total_reward += rew
            total_lines += info["lines"]
            steps += 1
        results.append({"reward": total_reward, "lines": total_lines, "steps": steps})
    return results


def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def train_reinforce():
    N_EPISODES = 2000
    LR         = 5e-4
    GAMMA      = 0.99
    ENT_START, ENT_END = 0.05, 0.002
    EPS_START, EPS_END = 0.10, 0.00
    SNAPSHOT_EVERY = 250
    FIXED_EVAL_EVERY = 250

    env = TetrisLiteEnv(width=BOARD_W, height=BOARD_H, max_steps=MAX_STEPS)
    agent = ReinforceAgent(
        obs_dim=env.obs_size(), act_dim=env.num_actions(),
        lr=LR, gamma=GAMMA, hidden=HIDDEN,
        ent_coef=ENT_START, epsilon=EPS_START, device=DEVICE,
    )

    metrics = {
        "rewards": [], "lines": [], "steps_survived": [],
        "entropy": [], "policy_loss": [], "gradient_norm": [],
        "per_piece_rewards": {n: [] for n in PIECE_NAMES},
        "board_holes": [], "board_bumpiness": [], "board_max_height": [],
        "reward_line_clear": [], "reward_survival": [],
        "reward_hole_penalty": [], "reward_hole_removal": [],
        "reward_bump_penalty": [], "reward_height_penalty": [],
        "singles": [], "doubles": [], "triples": [], "quads": [],
        "topout": [],
        "n_placeable_mean": [],
        "fixed_eval_episodes": [],
        "fixed_eval_mean_reward": [],
        "fixed_eval_mean_lines": [],
    }
    snapshots = []

    pbar = tqdm(range(1, N_EPISODES + 1), desc="REINFORCE")
    for ep in pbar:
        progress = ep / N_EPISODES
        agent.ent_coef = ENT_START + (ENT_END - ENT_START) * progress
        agent.epsilon  = EPS_START + (EPS_END  - EPS_START) * progress

        buf = RolloutBuffer()
        obs = env.reset()
        done = False
        ep_reward = 0.0
        ep_lines = 0
        ep_steps = 0
        piece_rewards = {n: [] for n in PIECE_NAMES}
        rc = {k: 0.0 for k in [
            "line_clear", "survival", "hole_penalty",
            "hole_removal", "bump_penalty", "height_penalty",
        ]}
        clear_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        placeable_counts = []

        while not done:
            mask = env.legal_action_mask()
            piece_name = PIECE_NAMES[env.current_piece]
            placeable_counts.append(int(mask.sum()))
            action, log_prob, value = agent.select_action(obs, mask, env=env)
            next_obs, reward, done, info = env.step(action)
            buf.store(obs, action, reward, done, log_prob, value, mask)
            ep_reward += reward
            ep_lines += info["lines"]
            ep_steps += 1
            piece_rewards[piece_name].append(reward)
            for k, v in env.last_reward_info.items():
                if k in rc:
                    rc[k] += v
            ct = env.last_reward_info.get("clear_type", 0)
            if ct in clear_counts:
                clear_counts[ct] += 1
            obs = next_obs

        board_stats = compute_board_stats(env)
        stats = agent.update(buf)
        topped_out = env.board[0].any()

        if ep_steps > 0:
            for k in rc:
                rc[k] /= ep_steps

        metrics["rewards"].append(ep_reward)
        metrics["lines"].append(ep_lines)
        metrics["steps_survived"].append(ep_steps)
        metrics["entropy"].append(stats["entropy"])
        metrics["policy_loss"].append(stats["policy_loss"])
        metrics["gradient_norm"].append(stats["grad_norm"])
        metrics["board_holes"].append(board_stats["holes"])
        metrics["board_bumpiness"].append(board_stats["bumpiness"])
        metrics["board_max_height"].append(board_stats["max_height"])
        metrics["reward_line_clear"].append(rc["line_clear"])
        metrics["reward_survival"].append(rc["survival"])
        metrics["reward_hole_penalty"].append(rc["hole_penalty"])
        metrics["reward_hole_removal"].append(rc["hole_removal"])
        metrics["reward_bump_penalty"].append(rc["bump_penalty"])
        metrics["reward_height_penalty"].append(rc["height_penalty"])
        metrics["singles"].append(clear_counts[1])
        metrics["doubles"].append(clear_counts[2])
        metrics["triples"].append(clear_counts[3])
        metrics["quads"].append(clear_counts[4])
        metrics["topout"].append(1 if topped_out else 0)
        metrics["n_placeable_mean"].append(float(np.mean(placeable_counts)) if placeable_counts else 0.0)

        for name in PIECE_NAMES:
            v = np.mean(piece_rewards[name]) if piece_rewards[name] else float("nan")
            metrics["per_piece_rewards"][name].append(v)

        if ep % SNAPSHOT_EVERY == 0 or ep == 1:
            snapshots.append((ep, env.render_board().copy(), ep_reward, ep_lines))

        if ep % FIXED_EVAL_EVERY == 0 or ep == 1:
            eval_res = fixed_seed_eval(agent)
            metrics["fixed_eval_episodes"].append(ep)
            metrics["fixed_eval_mean_reward"].append(float(np.mean([r["reward"] for r in eval_res])))
            metrics["fixed_eval_mean_lines"].append(float(np.mean([r["lines"] for r in eval_res])))

        if ep % 100 == 0:
            r = metrics["rewards"][-100:]
            pbar.set_postfix(rew=f"{np.mean(r):.1f}", ent=f"{stats['entropy']:.3f}")

    hyperparams = {
        "lr": LR, "gamma": GAMMA, "hidden": HIDDEN,
        "ent_start": ENT_START, "ent_end": ENT_END,
        "eps_start": EPS_START, "eps_end": EPS_END,
        "n_episodes": N_EPISODES,
    }
    return agent, metrics, snapshots, hyperparams


def train_ppo(policy_type="cnn", value_type="mlp"):
    N_ITERS       = 1500
    ROLLOUT_STEPS = 4096
    LR            = 3e-4
    LR_LATE       = 1e-4
    GAMMA         = 0.99
    LAM           = 0.95
    CLIP_EPS      = 0.2
    EPOCHS        = 4
    EPOCHS_LATE   = 2
    MINIBATCH     = 256
    N_FILTERS     = 32
    VALUE_LR      = 1e-3
    VALUE_LR_LATE = 3e-4
    TARGET_KL     = 0.02
    LATE_PHASE_START = 1000
    ENT_START, ENT_END = 0.20, 0.02
    SNAPSHOT_EVERY = 50
    FIXED_EVAL_EVERY = 50

    env = TetrisLiteEnv(width=BOARD_W, height=BOARD_H, max_steps=MAX_STEPS)
    agent = PPOAgent(
        obs_dim=env.obs_size(), act_dim=env.num_actions(),
        board_h=BOARD_H, board_w=BOARD_W, num_pieces=NUM_PIECES,
        lr=LR, gamma=GAMMA, lam=LAM, clip_eps=CLIP_EPS,
        epochs=EPOCHS, minibatch_size=MINIBATCH, ent_coef=ENT_START,
        n_filters=N_FILTERS, hidden=HIDDEN, value_lr=VALUE_LR,
        target_kl=TARGET_KL, device=DEVICE,
        policy_type=policy_type, value_type=value_type,
    )

    metrics = {
        "ep_rewards": [], "ep_lines": [], "ep_steps": [],
        "iter_mean_reward": [], "iter_mean_lines": [],
        "policy_loss": [], "value_loss": [], "entropy": [], "gradient_norm": [],
        "advantage_mean": [], "advantage_std": [], "advantage_min": [], "advantage_max": [],
        "ratio_mean": [], "ratio_max": [], "clip_fraction": [],
        "explained_variance": [], "approx_kl": [], "epochs_completed": [],
        "ep_length_mean": [], "ep_length_std": [],
        "board_holes_mean": [], "board_bumpiness_mean": [], "board_max_height_mean": [],
        "per_piece_rewards": {n: [] for n in PIECE_NAMES},
        "reward_line_clear": [], "reward_survival": [],
        "reward_hole_penalty": [], "reward_hole_removal": [],
        "reward_bump_penalty": [], "reward_height_penalty": [],
        "value_calibration_iters": [],
        "value_calibration_pred": [],
        "value_calibration_actual": [],
        "singles": [], "doubles": [], "triples": [], "quads": [],
        "topout_rate": [],
        "n_placeable_mean": [],
        "fixed_eval_iters": [],
        "fixed_eval_mean_reward": [],
        "fixed_eval_mean_lines": [],
    }
    snapshots = []

    late_phase_entered = False

    pbar = tqdm(range(1, N_ITERS + 1), desc="PPO")
    for it in pbar:
        progress = it / N_ITERS
        agent.ent_coef = ENT_START + (ENT_END - ENT_START) * progress

        if it > LATE_PHASE_START and not late_phase_entered:
            late_phase_entered = True
            for pg in agent.optimizer.param_groups:
                pg["lr"] = LR_LATE
            if value_type == "mlp":
                for pg in agent.value.optimizer.param_groups:
                    pg["lr"] = VALUE_LR_LATE
            agent.epochs = EPOCHS_LATE
            tqdm.write(
                f"[iter {it}] Late phase: policy_lr={LR_LATE}, "
                f"value_lr={VALUE_LR_LATE}, epochs={EPOCHS_LATE}"
            )

        buf = RolloutBuffer()
        obs = env.reset()
        ep_rewards_iter = []
        ep_lines_iter = []
        ep_steps_iter = []
        piece_rewards = {n: [] for n in PIECE_NAMES}
        board_stats_list = []
        ep_reward_components = []
        cur_rc = {k: 0.0 for k in [
            "line_clear", "survival", "hole_penalty",
            "hole_removal", "bump_penalty", "height_penalty",
        ]}
        cur_rew, cur_lines, cur_steps = 0.0, 0, 0
        done = False
        iter_clear_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        iter_placeable = []
        iter_topouts = 0
        iter_n_episodes = 0

        for _ in range(ROLLOUT_STEPS):
            mask = env.legal_action_mask()
            if agent.uses_cnn:
                board_obs, piece_obs = env.get_cnn_obs()
            else:
                board_obs, piece_obs = None, None
            piece_name = PIECE_NAMES[env.current_piece]
            iter_placeable.append(int(mask.sum()))
            action, lp, val = agent.select_action(
                obs, mask, env=env if agent.uses_cnn else None)
            next_obs, rew, done, info = env.step(action)
            buf.store(obs, action, rew, done, lp, val, mask,
                      board_obs=board_obs, piece_obs=piece_obs)
            cur_rew += rew
            cur_lines += info["lines"]
            cur_steps += 1
            piece_rewards[piece_name].append(rew)
            for k, v in env.last_reward_info.items():
                if k in cur_rc:
                    cur_rc[k] += v
            ct = env.last_reward_info.get("clear_type", 0)
            if ct in iter_clear_counts:
                iter_clear_counts[ct] += 1
            obs = next_obs
            if done:
                board_stats_list.append(compute_board_stats(env))
                ep_rewards_iter.append(cur_rew)
                ep_lines_iter.append(cur_lines)
                ep_steps_iter.append(cur_steps)
                if cur_steps > 0:
                    normalized_rc = {k: v / cur_steps for k, v in cur_rc.items()}
                else:
                    normalized_rc = dict(cur_rc)
                ep_reward_components.append(normalized_rc)
                iter_n_episodes += 1
                if env.board[0].any():
                    iter_topouts += 1
                cur_rew, cur_lines, cur_steps = 0.0, 0, 0
                cur_rc = {k: 0.0 for k in cur_rc}
                obs = env.reset()

        last_val = 0.0
        if not done:
            mask = env.legal_action_mask()
            _, _, last_val = agent.select_action(
                obs, mask, env=env if agent.uses_cnn else None)

        stats = agent.update(buf, last_value=last_val)

        metrics["ep_rewards"].extend(ep_rewards_iter)
        metrics["ep_lines"].extend(ep_lines_iter)
        metrics["ep_steps"].extend(ep_steps_iter)

        mean_rew = np.mean(ep_rewards_iter) if ep_rewards_iter else 0
        mean_lines = np.mean(ep_lines_iter) if ep_lines_iter else 0
        metrics["iter_mean_reward"].append(float(mean_rew))
        metrics["iter_mean_lines"].append(float(mean_lines))

        for key in ["policy_loss", "value_loss", "entropy", "gradient_norm",
                     "advantage_mean", "advantage_std", "advantage_min", "advantage_max",
                     "ratio_mean", "ratio_max", "clip_fraction", "explained_variance",
                     "approx_kl", "epochs_completed"]:
            k_stats = "grad_norm" if key == "gradient_norm" else key
            metrics[key].append(stats.get(k_stats, 0.0))

        for name in PIECE_NAMES:
            v = np.mean(piece_rewards[name]) if piece_rewards[name] else float("nan")
            metrics["per_piece_rewards"][name].append(v)

        if ep_steps_iter:
            metrics["ep_length_mean"].append(float(np.mean(ep_steps_iter)))
            metrics["ep_length_std"].append(float(np.std(ep_steps_iter)))
        else:
            metrics["ep_length_mean"].append(0.0)
            metrics["ep_length_std"].append(0.0)

        if board_stats_list:
            metrics["board_holes_mean"].append(float(np.mean([s["holes"] for s in board_stats_list])))
            metrics["board_bumpiness_mean"].append(float(np.mean([s["bumpiness"] for s in board_stats_list])))
            metrics["board_max_height_mean"].append(float(np.mean([s["max_height"] for s in board_stats_list])))
        else:
            metrics["board_holes_mean"].append(0.0)
            metrics["board_bumpiness_mean"].append(0.0)
            metrics["board_max_height_mean"].append(0.0)

        if ep_reward_components:
            for key in ["line_clear", "survival", "hole_penalty",
                        "hole_removal", "bump_penalty", "height_penalty"]:
                metrics[f"reward_{key}"].append(
                    float(np.mean([rc[key] for rc in ep_reward_components]))
                )
        else:
            for key in ["line_clear", "survival", "hole_penalty",
                        "hole_removal", "bump_penalty", "height_penalty"]:
                metrics[f"reward_{key}"].append(0.0)

        metrics["singles"].append(iter_clear_counts[1])
        metrics["doubles"].append(iter_clear_counts[2])
        metrics["triples"].append(iter_clear_counts[3])
        metrics["quads"].append(iter_clear_counts[4])
        metrics["topout_rate"].append(iter_topouts / max(iter_n_episodes, 1))
        metrics["n_placeable_mean"].append(float(np.mean(iter_placeable)) if iter_placeable else 0.0)

        if it % 50 == 0 or it == 1:
            obs_np = np.array(buf.obs)
            returns_np = stats.get("mean_return", 0.0)
            n_sample = min(200, len(obs_np))
            idx = np.random.choice(len(obs_np), n_sample, replace=False)
            pred = agent.value.predict(obs_np[idx])
            _, _, rew_t, done_t, _, val_t, _ = buf.to_tensors("cpu")
            from lib.agents import compute_gae
            _, rets = compute_gae(rew_t, val_t, done_t, agent.gamma, agent.lam, last_value=last_val)
            actual = rets.numpy()[idx]
            metrics["value_calibration_iters"].append(it)
            metrics["value_calibration_pred"].append(pred.tolist())
            metrics["value_calibration_actual"].append(actual.tolist())

        if it % FIXED_EVAL_EVERY == 0 or it == 1:
            eval_res = fixed_seed_eval(agent)
            metrics["fixed_eval_iters"].append(it)
            metrics["fixed_eval_mean_reward"].append(float(np.mean([r["reward"] for r in eval_res])))
            metrics["fixed_eval_mean_lines"].append(float(np.mean([r["lines"] for r in eval_res])))

        if it % SNAPSHOT_EVERY == 0 or it == 1:
            snapshots.append((it, env.render_board().copy(), float(mean_rew), int(mean_lines)))

        if it % 10 == 0:
            recent = metrics["ep_rewards"][-300:] if metrics["ep_rewards"] else [0]
            pbar.set_postfix(rew=f"{np.mean(recent):.1f}", ent=f"{stats['entropy']:.3f}")

    hyperparams = {
        "lr": LR, "lr_late": LR_LATE,
        "gamma": GAMMA, "lam": LAM, "clip_eps": CLIP_EPS,
        "epochs": EPOCHS, "epochs_late": EPOCHS_LATE,
        "minibatch_size": MINIBATCH, "hidden": HIDDEN,
        "n_filters": N_FILTERS, "ent_start": ENT_START, "ent_end": ENT_END,
        "n_iters": N_ITERS, "rollout_steps": ROLLOUT_STEPS,
        "value_lr": VALUE_LR, "value_lr_late": VALUE_LR_LATE,
        "target_kl": TARGET_KL, "late_phase_start": LATE_PHASE_START,
        "policy_type": policy_type, "value_type": value_type,
    }
    return agent, metrics, snapshots, hyperparams


def generate_visualizations(rf_agent, rf_metrics, rf_snapshots,
                            ppo_agent, ppo_metrics, ppo_snapshots):
    print("\n=== Generating visualizations ===\n")

    min_len = min(len(rf_metrics["rewards"]), len(ppo_metrics["ep_rewards"]))
    plot_learning_curves(
        {"REINFORCE": rf_metrics["rewards"][:min_len],
         "PPO": ppo_metrics["ep_rewards"][:min_len]},
        title="Episode Reward: REINFORCE vs PPO",
        window=30,
        save_path=str(GIF_DIR / "learning_curves_reward.png"),
    )
    plt.close("all")

    plot_learning_curves(
        {"REINFORCE": rf_metrics["lines"][:min_len],
         "PPO": ppo_metrics["ep_lines"][:min_len]},
        title="Lines Cleared: REINFORCE vs PPO",
        window=30,
        save_path=str(GIF_DIR / "learning_curves_lines.png"),
    )
    plt.close("all")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(smooth(rf_metrics["steps_survived"], 30),
            label="REINFORCE", color="#e74c3c", linewidth=2)
    ax.plot(smooth(ppo_metrics["ep_steps"], 30),
            label="PPO", color="#2ecc71", linewidth=2)
    ax.set_xlabel("Episode"); ax.set_ylabel("Steps Survived")
    ax.set_title("Episode Length Over Training"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(GIF_DIR / "steps_survived_comparison.png"), dpi=120)
    plt.close(fig)
    print(f"Saved plot → {GIF_DIR / 'steps_survived_comparison.png'}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(ppo_metrics["policy_loss"], alpha=0.7, color="steelblue")
    axes[0, 0].set_title("Policy Loss"); axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(ppo_metrics["value_loss"], alpha=0.7, color="coral")
    axes[0, 1].set_title("Value Loss (residual MSE)"); axes[0, 1].grid(True, alpha=0.3)
    axes[1, 0].plot(ppo_metrics["entropy"], alpha=0.7, color="mediumseagreen")
    axes[1, 0].set_title("Entropy"); axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].plot(ppo_metrics["clip_fraction"], alpha=0.7, color="darkorange")
    axes[1, 1].set_title("Clip Fraction"); axes[1, 1].grid(True, alpha=0.3)
    fig.suptitle("PPO Training Diagnostics", fontsize=14)
    fig.tight_layout(); fig.savefig(str(GIF_DIR / "ppo_diagnostics_panel.png"), dpi=120)
    plt.close(fig)
    print(f"Saved plot → {GIF_DIR / 'ppo_diagnostics_panel.png'}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    am = ppo_metrics["advantage_mean"]
    astd = ppo_metrics["advantage_std"]
    x_range = range(len(am))
    axes[0, 0].fill_between(x_range,
        [m - s for m, s in zip(am, astd)],
        [m + s for m, s in zip(am, astd)],
        alpha=0.3, color="steelblue")
    axes[0, 0].plot(am, color="steelblue")
    axes[0, 0].set_title("Advantage (mean +/- std)"); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(ppo_metrics["ratio_mean"], label="mean", color="teal")
    axes[0, 1].plot(ppo_metrics["ratio_max"], label="max", color="red", alpha=0.6)
    axes[0, 1].legend(); axes[0, 1].set_title("PPO Ratio"); axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(ppo_metrics["gradient_norm"], alpha=0.7, color="purple")
    axes[0, 2].set_title("Gradient Norm"); axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(ppo_metrics["explained_variance"], alpha=0.7, color="darkgreen")
    axes[1, 0].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    axes[1, 0].set_title("Explained Variance"); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(ppo_metrics["ep_length_mean"], alpha=0.7, color="navy")
    axes[1, 1].set_title("Mean Episode Length"); axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(ppo_metrics["board_holes_mean"], label="holes", alpha=0.7)
    axes[1, 2].plot(ppo_metrics["board_bumpiness_mean"], label="bumpiness", alpha=0.7)
    axes[1, 2].plot(ppo_metrics["board_max_height_mean"], label="max height", alpha=0.7)
    axes[1, 2].legend(); axes[1, 2].set_title("Board End-State Stats"); axes[1, 2].grid(True, alpha=0.3)

    fig.suptitle("PPO Extended Diagnostics", fontsize=14)
    fig.tight_layout(); fig.savefig(str(GIF_DIR / "ppo_extended_diagnostics.png"), dpi=120)
    plt.close(fig)
    print(f"Saved plot → {GIF_DIR / 'ppo_extended_diagnostics.png'}")

    fig, ax = plt.subplots(figsize=(12, 5))
    for name in PIECE_NAMES:
        vals = ppo_metrics["per_piece_rewards"][name]
        if len(vals) > 20:
            ax.plot(smooth(vals, 20), label=name, linewidth=1.5)
    ax.set_title("PPO: Per-Piece-Type Mean Reward")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Mean Reward")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(str(GIF_DIR / "ppo_per_piece_rewards.png"), dpi=120)
    plt.close(fig)
    print(f"Saved plot → {GIF_DIR / 'ppo_per_piece_rewards.png'}")

    fig, ax = plt.subplots(figsize=(12, 5))
    iters = range(len(ppo_metrics["reward_line_clear"]))
    components = [
        ("Line clear",    ppo_metrics["reward_line_clear"],    "#2ecc71"),
        ("Survival",      ppo_metrics["reward_survival"],      "#3498db"),
        ("Hole penalty",  ppo_metrics["reward_hole_penalty"],  "#e74c3c"),
        ("Hole removal",  ppo_metrics["reward_hole_removal"],  "#27ae60"),
        ("Bump penalty",  ppo_metrics["reward_bump_penalty"],  "#e67e22"),
        ("Height penalty",ppo_metrics["reward_height_penalty"],"#9b59b6"),
    ]
    for label, vals, color in components:
        ax.plot(smooth(vals, 10), label=label, color=color, linewidth=1.5, alpha=0.8)
    ax.set_title("PPO: Reward Decomposition (per-step mean)")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Reward Component")
    ax.legend(loc="best"); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(str(GIF_DIR / "ppo_reward_decomposition.png"), dpi=120)
    plt.close(fig)
    print(f"Saved plot → {GIF_DIR / 'ppo_reward_decomposition.png'}")

    fig, ax = plt.subplots(figsize=(12, 5))
    rf_components = [
        ("Line clear",    rf_metrics["reward_line_clear"],    "#2ecc71"),
        ("Survival",      rf_metrics["reward_survival"],      "#3498db"),
        ("Hole penalty",  rf_metrics["reward_hole_penalty"],  "#e74c3c"),
        ("Hole removal",  rf_metrics["reward_hole_removal"],  "#27ae60"),
        ("Bump penalty",  rf_metrics["reward_bump_penalty"],  "#e67e22"),
        ("Height penalty",rf_metrics["reward_height_penalty"],"#9b59b6"),
    ]
    for label, vals, color in rf_components:
        ax.plot(smooth(vals, 30), label=label, color=color, linewidth=1.5, alpha=0.8)
    ax.set_title("REINFORCE: Reward Decomposition (per-step mean)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Reward Component")
    ax.legend(loc="best"); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(str(GIF_DIR / "reinforce_reward_decomposition.png"), dpi=120)
    plt.close(fig)
    print(f"Saved plot → {GIF_DIR / 'reinforce_reward_decomposition.png'}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(smooth(rf_metrics["entropy"], 30), color="mediumseagreen")
    axes[0].set_title("Entropy"); axes[0].grid(True, alpha=0.3)
    axes[1].plot(smooth(rf_metrics["policy_loss"], 30), color="steelblue")
    axes[1].set_title("Policy Loss"); axes[1].grid(True, alpha=0.3)
    axes[2].plot(smooth(rf_metrics["gradient_norm"], 30), color="purple")
    axes[2].set_title("Gradient Norm"); axes[2].grid(True, alpha=0.3)
    fig.suptitle("REINFORCE Diagnostics", fontsize=14)
    fig.tight_layout(); fig.savefig(str(GIF_DIR / "reinforce_diagnostics.png"), dpi=120)
    plt.close(fig)
    print(f"Saved plot → {GIF_DIR / 'reinforce_diagnostics.png'}")

    cal_iters = ppo_metrics.get("value_calibration_iters", [])
    if cal_iters:
        n_cal = len(cal_iters)
        cols = min(n_cal, 4)
        rows = (n_cal + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
        for idx, (it_num, pred, actual) in enumerate(zip(
            cal_iters,
            ppo_metrics["value_calibration_pred"],
            ppo_metrics["value_calibration_actual"],
        )):
            ax = axes[idx // cols][idx % cols]
            ax.scatter(actual, pred, alpha=0.3, s=10, color="steelblue")
            lims = [min(min(actual), min(pred)), max(max(actual), max(pred))]
            ax.plot(lims, lims, "r--", alpha=0.5, linewidth=1)
            ax.set_title(f"Iter {it_num}")
            ax.set_xlabel("Actual Return"); ax.set_ylabel("Predicted V(s)")
            ax.grid(True, alpha=0.3)
        for idx in range(n_cal, rows * cols):
            axes[idx // cols][idx % cols].set_visible(False)
        fig.suptitle("PPO: Value Calibration (V(s) vs actual return)", fontsize=14)
        fig.tight_layout()
        fig.savefig(str(GIF_DIR / "ppo_value_calibration.png"), dpi=120)
        plt.close(fig)
        print(f"Saved plot → {GIF_DIR / 'ppo_value_calibration.png'}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(ppo_metrics.get("approx_kl", []), alpha=0.7, color="teal")
    ax1.set_title("Approximate KL Divergence"); ax1.set_xlabel("Iteration")
    ax1.set_ylabel("KL"); ax1.grid(True, alpha=0.3)
    ax2.plot(ppo_metrics["clip_fraction"], alpha=0.7, color="darkorange")
    ax2.set_title("Clip Fraction"); ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Fraction"); ax2.grid(True, alpha=0.3)
    fig.suptitle("PPO: KL & Clipping", fontsize=14)
    fig.tight_layout(); fig.savefig(str(GIF_DIR / "ppo_kl_and_clip.png"), dpi=120)
    plt.close(fig)
    print(f"Saved plot → {GIF_DIR / 'ppo_kl_and_clip.png'}")

    singles = ppo_metrics.get("singles", [])
    if singles:
        fig, ax = plt.subplots(figsize=(12, 5))
        s1 = smooth(singles, 10)
        s2 = smooth(ppo_metrics["doubles"], 10)
        s3 = smooth(ppo_metrics["triples"], 10)
        s4 = smooth(ppo_metrics["quads"], 10)
        ax.stackplot(range(len(s1)), s1, s2, s3, s4,
                     labels=["Singles", "Doubles", "Triples", "Quads"],
                     colors=["#3498db", "#2ecc71", "#e67e22", "#e74c3c"],
                     alpha=0.8)
        ax.set_title("PPO: Line-Clear Composition per Iteration")
        ax.set_xlabel("Iteration"); ax.set_ylabel("Count")
        ax.legend(loc="upper left"); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(str(GIF_DIR / "ppo_line_composition.png"), dpi=120)
        plt.close(fig)
        print(f"Saved plot → {GIF_DIR / 'ppo_line_composition.png'}")

    n_place = ppo_metrics.get("n_placeable_mean", [])
    if n_place:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(n_place, alpha=0.7, color="steelblue")
        ax.set_title("PPO: Mean Truly-Placeable Actions per Step")
        ax.set_xlabel("Iteration"); ax.set_ylabel("Placeable Actions")
        ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(str(GIF_DIR / "ppo_action_interface.png"), dpi=120)
        plt.close(fig)
        print(f"Saved plot → {GIF_DIR / 'ppo_action_interface.png'}")

    topout = ppo_metrics.get("topout_rate", [])
    if topout:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(smooth(topout, 10), alpha=0.7, color="#e74c3c")
        ax.set_title("PPO: Top-Out Rate per Iteration")
        ax.set_xlabel("Iteration"); ax.set_ylabel("Fraction")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(str(GIF_DIR / "ppo_topout_rate.png"), dpi=120)
        plt.close(fig)
        print(f"Saved plot → {GIF_DIR / 'ppo_topout_rate.png'}")

    ppo_fe_iters = ppo_metrics.get("fixed_eval_iters", [])
    if ppo_fe_iters:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(ppo_fe_iters, ppo_metrics["fixed_eval_mean_reward"],
                 "o-", color="#2ecc71", linewidth=2, markersize=4)
        ax1.set_title("PPO Fixed-Seed Eval: Reward")
        ax1.set_xlabel("Iteration"); ax1.set_ylabel("Mean Reward"); ax1.grid(True, alpha=0.3)
        ax2.plot(ppo_fe_iters, ppo_metrics["fixed_eval_mean_lines"],
                 "o-", color="#2ecc71", linewidth=2, markersize=4)
        ax2.set_title("PPO Fixed-Seed Eval: Lines")
        ax2.set_xlabel("Iteration"); ax2.set_ylabel("Mean Lines"); ax2.grid(True, alpha=0.3)
        fig.suptitle("PPO: Fixed-Seed Evaluation", fontsize=14)
        fig.tight_layout(); fig.savefig(str(GIF_DIR / "ppo_fixed_eval.png"), dpi=120)
        plt.close(fig)
        print(f"Saved plot → {GIF_DIR / 'ppo_fixed_eval.png'}")

    rf_fe_eps = rf_metrics.get("fixed_eval_episodes", [])
    if rf_fe_eps:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(rf_fe_eps, rf_metrics["fixed_eval_mean_reward"],
                 "o-", color="#e74c3c", linewidth=2, markersize=4)
        ax1.set_title("REINFORCE Fixed-Seed Eval: Reward")
        ax1.set_xlabel("Episode"); ax1.set_ylabel("Mean Reward"); ax1.grid(True, alpha=0.3)
        ax2.plot(rf_fe_eps, rf_metrics["fixed_eval_mean_lines"],
                 "o-", color="#e74c3c", linewidth=2, markersize=4)
        ax2.set_title("REINFORCE Fixed-Seed Eval: Lines")
        ax2.set_xlabel("Episode"); ax2.set_ylabel("Mean Lines"); ax2.grid(True, alpha=0.3)
        fig.suptitle("REINFORCE: Fixed-Seed Evaluation", fontsize=14)
        fig.tight_layout(); fig.savefig(str(GIF_DIR / "reinforce_fixed_eval.png"), dpi=120)
        plt.close(fig)
        print(f"Saved plot → {GIF_DIR / 'reinforce_fixed_eval.png'}")

    if rf_snapshots:
        training_snapshots_gif(rf_snapshots, str(GIF_DIR / "reinforce_training_snapshots.gif"), fps=1)
    if ppo_snapshots:
        training_snapshots_gif(ppo_snapshots, str(GIF_DIR / "ppo_training_snapshots.gif"), fps=1)

    env_rf = TetrisLiteEnv(width=BOARD_W, height=BOARD_H, max_steps=MAX_STEPS, seed=99)
    frames_rf = play_episode_annotated(env_rf, rf_agent, max_frames=200, device=DEVICE)
    save_gif(frames_rf, str(GIF_DIR / "reinforce_gameplay.gif"), fps=6)

    env_ppo = TetrisLiteEnv(width=BOARD_W, height=BOARD_H, max_steps=MAX_STEPS, seed=99)
    frames_ppo = play_episode_annotated(env_ppo, ppo_agent, max_frames=200, device=DEVICE)
    save_gif(frames_ppo, str(GIF_DIR / "ppo_gameplay.gif"), fps=6)

    EVAL_SEED = 2024
    env1 = TetrisLiteEnv(width=BOARD_W, height=BOARD_H, max_steps=200, seed=EVAL_SEED)
    frames_rf_sbs, _, _ = play_episode(env1, rf_agent, max_frames=200, device=DEVICE)
    env2 = TetrisLiteEnv(width=BOARD_W, height=BOARD_H, max_steps=200, seed=EVAL_SEED)
    frames_ppo_sbs, _, _ = play_episode(env2, ppo_agent, max_frames=200, device=DEVICE)
    side_by_side_gif(frames_rf_sbs, frames_ppo_sbs,
                     "REINFORCE", "PPO",
                     str(GIF_DIR / "reinforce_vs_ppo.gif"), fps=6)

    N_EVAL = 50
    eval_results = {"Random": [], "REINFORCE": [], "PPO": []}
    eval_lines = {"Random": [], "REINFORCE": [], "PPO": []}

    for seed in tqdm(range(N_EVAL), desc="Evaluation"):
        s = seed + 5000
        e = TetrisLiteEnv(width=BOARD_W, height=BOARD_H, max_steps=MAX_STEPS, seed=s)
        _, r, l = play_episode_random(e, max_frames=MAX_STEPS)
        eval_results["Random"].append(r); eval_lines["Random"].append(l)

        e = TetrisLiteEnv(width=BOARD_W, height=BOARD_H, max_steps=MAX_STEPS, seed=s)
        _, r, l = play_episode(e, rf_agent, max_frames=MAX_STEPS, device=DEVICE)
        eval_results["REINFORCE"].append(r); eval_lines["REINFORCE"].append(l)

        e = TetrisLiteEnv(width=BOARD_W, height=BOARD_H, max_steps=MAX_STEPS, seed=s)
        _, r, l = play_episode(e, ppo_agent, max_frames=MAX_STEPS, device=DEVICE)
        eval_results["PPO"].append(r); eval_lines["PPO"].append(l)

    print("\n=== Evaluation Results (50 episodes) ===")
    for name in ["Random", "REINFORCE", "PPO"]:
        rews = eval_results[name]
        lns = eval_lines[name]
        print(f"{name:12s}  reward: {np.mean(rews):+7.1f} +/- {np.std(rews):5.1f}  |  "
              f"lines: {np.mean(lns):5.1f} +/- {np.std(lns):4.1f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    labels = list(eval_results.keys())
    colors_box = ["#95a5a6", "#e74c3c", "#2ecc71"]
    bp1 = ax1.boxplot([eval_results[k] for k in labels], tick_labels=labels, patch_artist=True)
    for patch, c in zip(bp1["boxes"], colors_box):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax1.set_title("Episode Reward Distribution"); ax1.grid(True, alpha=0.3)
    bp2 = ax2.boxplot([eval_lines[k] for k in labels], tick_labels=labels, patch_artist=True)
    for patch, c in zip(bp2["boxes"], colors_box):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax2.set_title("Lines Cleared Distribution"); ax2.grid(True, alpha=0.3)
    fig.suptitle("Multi-Seed Evaluation (50 episodes)", fontsize=14)
    fig.tight_layout(); fig.savefig(str(GIF_DIR / "evaluation_boxplot.png"), dpi=120)
    plt.close(fig)
    print(f"Saved plot → {GIF_DIR / 'evaluation_boxplot.png'}")

    learning_curve_gif(
        {"REINFORCE": rf_metrics["rewards"][:min_len],
         "PPO": ppo_metrics["ep_rewards"][:min_len]},
        str(GIF_DIR / "learning_curve_animated.gif"),
        window=20, step=10, fps=10,
    )


def save_checkpoints(rf_agent, rf_metrics, rf_hyperparams,
                     ppo_agent, ppo_metrics, ppo_hyperparams):
    env_config = {"board_w": BOARD_W, "board_h": BOARD_H, "max_steps": MAX_STEPS}

    rf_final = {
        "mean_reward": float(np.mean(rf_metrics["rewards"][-100:])),
        "mean_lines": float(np.mean(rf_metrics["lines"][-100:])),
    }
    torch.save({
        "net": rf_agent.net.state_dict(),
        "hyperparams": rf_hyperparams,
        "reward_weights": dict(REWARD_WEIGHTS),
        "env_config": env_config,
        "device": DEVICE,
        "final_metrics": rf_final,
    }, str(CKPT_DIR / "reinforce.pt"))
    print(f"Saved checkpoint → {CKPT_DIR / 'reinforce.pt'}")

    ppo_final = {
        "mean_reward": float(np.mean(ppo_metrics["ep_rewards"][-100:])) if ppo_metrics["ep_rewards"] else 0,
        "mean_lines": float(np.mean(ppo_metrics["ep_lines"][-100:])) if ppo_metrics["ep_lines"] else 0,
    }
    torch.save({
        "policy": ppo_agent.policy.state_dict(),
        "value": ppo_agent.value.state_dict(),
        "policy_type": ppo_agent.policy_type,
        "value_type": ppo_agent.value_type,
        "hyperparams": ppo_hyperparams,
        "reward_weights": dict(REWARD_WEIGHTS),
        "env_config": env_config,
        "device": DEVICE,
        "final_metrics": ppo_final,
    }, str(CKPT_DIR / "ppo.pt"))
    print(f"Saved checkpoint → {CKPT_DIR / 'ppo.pt'}")


def save_metrics(rf_metrics, ppo_metrics):
    data = {
        "reinforce": sanitize_for_json(rf_metrics),
        "ppo": sanitize_for_json(ppo_metrics),
    }
    path = LOG_DIR / "training_metrics.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved metrics → {path}")


def main():
    parser = argparse.ArgumentParser(description="Tetris-Lite RL Training")
    parser.add_argument("--policy", choices=["cnn", "mlp"], default="cnn",
                        help="PPO policy network type (default: cnn)")
    parser.add_argument("--value", choices=["mlp", "linear"], default="mlp",
                        help="PPO value estimator type (default: mlp)")
    args = parser.parse_args()

    print("=" * 60)
    print("Tetris-Lite RL Training")
    print(f"Board: {BOARD_H}x{BOARD_W}, max_steps={MAX_STEPS}")
    print(f"Device: {DEVICE}")
    print(f"PPO config: policy={args.policy}, value={args.value}")
    print("=" * 60)

    t0 = time.time()

    print("\n>>> Training REINFORCE <<<")
    rf_agent, rf_metrics, rf_snapshots, rf_hp = train_reinforce()

    print("\n>>> Training PPO <<<")
    ppo_agent, ppo_metrics, ppo_snapshots, ppo_hp = train_ppo(
        policy_type=args.policy, value_type=args.value,
    )

    save_checkpoints(rf_agent, rf_metrics, rf_hp,
                     ppo_agent, ppo_metrics, ppo_hp)
    save_metrics(rf_metrics, ppo_metrics)

    generate_visualizations(
        rf_agent, rf_metrics, rf_snapshots,
        ppo_agent, ppo_metrics, ppo_snapshots,
    )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"Checkpoints: {CKPT_DIR}/")
    print(f"GIFs/plots:  {GIF_DIR}/")
    print(f"Metrics:     {LOG_DIR}/")


if __name__ == "__main__":
    main()
