"""
Policy-Gradient Agents — REINFORCE and PPO
===========================================
Implemented from first principles in PyTorch.
No stable-baselines / CleanRL / SpinningUp.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import List, Tuple, Optional


# ──────────────────────────────────────────────────────────────
# Shared Policy / Value Network
# ──────────────────────────────────────────────────────────────

class PolicyNetwork(nn.Module):
    """Simple MLP policy (shared backbone, separate heads)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, act_dim)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        h = self.shared(x)
        logits = self.policy_head(h)
        if mask is not None:
            logits = logits + (1 - mask) * (-1e8)   # mask illegal actions
        value = self.value_head(h).squeeze(-1)
        return logits, value

    def get_action(self, obs: np.ndarray, mask: np.ndarray, device="cpu"):
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mask_t = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)
            logits, value = self.forward(obs_t, mask_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()


# ──────────────────────────────────────────────────────────────
# Rollout Buffer
# ──────────────────────────────────────────────────────────────

class RolloutBuffer:
    def __init__(self):
        self.obs: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.masks: List[np.ndarray] = []

    def store(self, obs, action, reward, done, log_prob, value, mask):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.masks.append(mask)

    def clear(self):
        self.__init__()

    def to_tensors(self, device="cpu"):
        return (
            torch.tensor(np.array(self.obs), dtype=torch.float32, device=device),
            torch.tensor(self.actions, dtype=torch.long, device=device),
            torch.tensor(self.rewards, dtype=torch.float32, device=device),
            torch.tensor(self.dones, dtype=torch.float32, device=device),
            torch.tensor(self.log_probs, dtype=torch.float32, device=device),
            torch.tensor(self.values, dtype=torch.float32, device=device),
            torch.tensor(np.array(self.masks), dtype=torch.float32, device=device),
        )


# ──────────────────────────────────────────────────────────────
# Generalised Advantage Estimation (GAE)
# ──────────────────────────────────────────────────────────────

def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
    last_value: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE-Lambda advantages and discounted returns."""
    T = len(rewards)
    advantages = torch.zeros(T, dtype=torch.float32)
    last_adv = 0.0
    next_value = last_value

    for t in reversed(range(T)):
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        advantages[t] = last_adv = delta + gamma * lam * non_terminal * last_adv
        next_value = values[t].item()

    returns = advantages + values
    return advantages, returns


# ──────────────────────────────────────────────────────────────
# REINFORCE Agent
# ──────────────────────────────────────────────────────────────

class ReinforceAgent:
    """
    Vanilla REINFORCE with optional baseline (mean return).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        hidden: int = 128,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.device = device
        self.net = PolicyNetwork(obs_dim, act_dim, hidden).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def select_action(self, obs: np.ndarray, mask: np.ndarray):
        return self.net.get_action(obs, mask, self.device)

    def update(self, buffer: RolloutBuffer) -> dict:
        obs_t, act_t, rew_t, done_t, old_lp_t, val_t, mask_t = buffer.to_tensors(self.device)

        # Compute discounted returns (no GAE — plain MC returns)
        T = len(rew_t)
        returns = torch.zeros(T)
        G = 0.0
        for t in reversed(range(T)):
            if done_t[t]:
                G = 0.0
            G = rew_t[t] + self.gamma * G
            returns[t] = G
        returns = returns.to(self.device)

        # Baseline = mean return (simple variance reduction)
        baseline = returns.mean()

        # Forward pass
        logits, _ = self.net(obs_t, mask_t)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(act_t)

        # Policy loss: -E[log π(a|s) * (G - b)]
        policy_loss = -(log_probs * (returns - baseline)).mean()

        self.optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "mean_return": returns.mean().item(),
        }


# ──────────────────────────────────────────────────────────────
# PPO Agent
# ──────────────────────────────────────────────────────────────

class PPOAgent:
    """
    Proximal Policy Optimization (clip variant) with GAE.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        epochs: int = 4,
        minibatch_size: int = 64,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        hidden: int = 128,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.device = device

        self.net = PolicyNetwork(obs_dim, act_dim, hidden).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def select_action(self, obs: np.ndarray, mask: np.ndarray):
        return self.net.get_action(obs, mask, self.device)

    def update(self, buffer: RolloutBuffer) -> dict:
        obs_t, act_t, rew_t, done_t, old_lp_t, val_t, mask_t = buffer.to_tensors(self.device)

        # ── GAE ───────────────────────────────────────────────
        advantages, returns = compute_gae(
            rew_t, val_t, done_t, self.gamma, self.lam
        )
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)

        # Normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── PPO update epochs ─────────────────────────────────
        T = len(obs_t)
        indices = np.arange(T)
        total_pg_loss = 0.0
        total_vf_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.epochs):
            np.random.shuffle(indices)
            for start in range(0, T, self.minibatch_size):
                end = start + self.minibatch_size
                mb = indices[start:end]

                mb_obs = obs_t[mb]
                mb_act = act_t[mb]
                mb_old_lp = old_lp_t[mb]
                mb_adv = advantages[mb]
                mb_ret = returns[mb]
                mb_mask = mask_t[mb]

                logits, values = self.net(mb_obs, mb_mask)
                dist = Categorical(logits=logits)
                new_lp = dist.log_prob(mb_act)
                entropy = dist.entropy().mean()

                # Clipped surrogate objective
                ratio = torch.exp(new_lp - mb_old_lp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_adv
                pg_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                vf_loss = F.mse_loss(values, mb_ret)

                loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optimizer.step()

                total_pg_loss += pg_loss.item()
                total_vf_loss += vf_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        return {
            "policy_loss": total_pg_loss / max(n_updates, 1),
            "value_loss": total_vf_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
            "mean_return": returns.mean().item(),
        }
