"""
Policy-Gradient Agents — REINFORCE and PPO
===========================================
Implemented from first principles in PyTorch.
No stable-baselines / CleanRL / SpinningUp.

Architecture (PPO):
  • Policy  — CNN reads the raw board grid (1×H×W) + piece one-hots → logits
  • Value   — Linear model V(s) = w^T φ(s) + b  on the compact flat features,
              fitted by closed-form ridge regression (least squares).
              Completely separate from the policy — no shared backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import List, Tuple, Optional


# ──────────────────────────────────────────────────────────────
# MLP Policy / Value Network  (used by REINFORCE)
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
# CNN Policy Network  (used by PPO)
# ──────────────────────────────────────────────────────────────

class CNNPolicyNetwork(nn.Module):
    """
    CNN policy that reads the raw board grid (1×H×W) plus a piece-info
    vector (one-hot current + next piece).

    Architecture
    ------------
    board (1,H,W) → Conv2d layers → flatten
    concat with piece_info → FC → action logits
    """

    def __init__(
        self,
        board_h: int,
        board_w: int,
        num_pieces: int,
        act_dim: int,
        n_filters: int = 32,
        hidden: int = 128,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, n_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_filters * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_filters * 2, n_filters * 2, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # Compute conv output size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, board_h, board_w)
            conv_out = self.conv(dummy)
            self._conv_flat = conv_out.view(1, -1).shape[1]

        piece_dim = num_pieces * 2  # current + next piece one-hot
        self.fc = nn.Sequential(
            nn.Linear(self._conv_flat + piece_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, board: torch.Tensor, piece_info: torch.Tensor) -> torch.Tensor:
        """
        board      : (B, 1, H, W) float tensor
        piece_info : (B, num_pieces*2) float tensor
        Returns    : (B, act_dim) logits
        """
        x = self.conv(board)            # (B, C, h', w')
        x = x.view(x.size(0), -1)      # (B, conv_flat)
        x = torch.cat([x, piece_info], dim=1)
        return self.fc(x)


# ──────────────────────────────────────────────────────────────
# Linear Value Estimator  (used by PPO — fitted by least squares)
# ──────────────────────────────────────────────────────────────

class LinearValueEstimator:
    """
    V(s) = φ(s)^T w + b

    A single linear layer fitted by ridge regression (closed-form OLS).
    Uses the compact flat observation (e.g. 35-dim) as feature vector φ(s).

    No neural network, no gradient descent — just the normal equation:

        w* = (X^T X + λI)^{-1} X^T y
    """

    def __init__(self, obs_dim: int, reg: float = 1e-4):
        self.obs_dim = obs_dim
        self.reg = reg
        # Weight vector including bias: [w_1, ..., w_d, b]
        self.w = np.zeros(obs_dim + 1, dtype=np.float64)

    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        """Append a column of ones for the bias term."""
        return np.column_stack([X, np.ones(len(X), dtype=np.float64)])

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Predict values for a batch.  obs: (N, obs_dim) → (N,) float32."""
        X = self._add_bias(obs.astype(np.float64))
        return (X @ self.w).astype(np.float32)

    def predict_single(self, obs: np.ndarray) -> float:
        """Predict value for one observation.  obs: (obs_dim,) → scalar."""
        x = np.append(obs.astype(np.float64), 1.0)
        return float(x @ self.w)

    def fit(self, obs: np.ndarray, returns: np.ndarray):
        """
        Fit by ridge regression (closed-form least squares).

            w* = (X^T X + λI)^{-1} X^T y

        obs     : (N, obs_dim)
        returns : (N,) target values (GAE returns)
        """
        X = self._add_bias(obs.astype(np.float64))
        y = returns.astype(np.float64)
        A = X.T @ X + self.reg * np.eye(X.shape[1])
        b = X.T @ y
        self.w = np.linalg.solve(A, b)

    def state_dict(self) -> dict:
        """Return weights for serialisation."""
        return {"w": self.w.copy()}

    def load_state_dict(self, d: dict):
        """Load weights from dict."""
        self.w = d["w"].copy()


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
        # CNN observations (optional — used by PPO with CNN policy)
        self.board_obs: List[np.ndarray] = []
        self.piece_obs: List[np.ndarray] = []

    def store(self, obs, action, reward, done, log_prob, value, mask,
              board_obs=None, piece_obs=None):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.masks.append(mask)
        if board_obs is not None:
            self.board_obs.append(board_obs)
        if piece_obs is not None:
            self.piece_obs.append(piece_obs)

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

    def cnn_tensors(self, device="cpu"):
        """Return CNN observations as tensors: (board_t, piece_t)."""
        board_t = torch.tensor(
            np.array(self.board_obs), dtype=torch.float32, device=device
        )
        piece_t = torch.tensor(
            np.array(self.piece_obs), dtype=torch.float32, device=device
        )
        return board_t, piece_t


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
    # Work on CPU to avoid device mismatches; caller moves results to device
    rewards = rewards.cpu()
    values = values.cpu()
    dones = dones.cpu()
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
    Vanilla REINFORCE with optional baseline (mean return) and entropy regularisation.

    ent_coef : weight on entropy bonus — keep high early to prevent mode collapse,
               anneal to ~0 over training (set agent.ent_coef externally).
    epsilon  : prob of taking a uniformly-random legal action (epsilon-greedy).
               Anneal to 0 over training (set agent.epsilon externally).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        hidden: int = 128,
        ent_coef: float = 0.05,
        epsilon: float = 0.1,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.ent_coef = ent_coef
        self.epsilon = epsilon
        self.device = device
        self.net = PolicyNetwork(obs_dim, act_dim, hidden).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def select_action(self, obs: np.ndarray, mask: np.ndarray, env=None):
        """Select action.  `env` is accepted for interface compatibility but ignored."""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.net(obs_t, mask_t)
            dist = Categorical(logits=logits)
            if self.epsilon > 0 and np.random.random() < self.epsilon:
                legal = np.where(mask)[0]
                action_t = torch.tensor([int(np.random.choice(legal))], device=self.device)
            else:
                action_t = dist.sample()
            log_prob = dist.log_prob(action_t)
        return action_t.item(), log_prob.item(), value.item()

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
        entropy = dist.entropy().mean()

        # Policy loss: -E[log π(a|s) * (G - b)] - ent_coef * H[π]
        policy_loss = -(log_probs * (returns - baseline)).mean()
        loss = policy_loss - self.ent_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "entropy": entropy.item(),
            "mean_return": returns.mean().item(),
            "grad_norm": grad_norm.item(),
        }


# ──────────────────────────────────────────────────────────────
# PPO Agent  (CNN policy  +  linear value estimator)
# ──────────────────────────────────────────────────────────────

class PPOAgent:
    """
    Proximal Policy Optimization (clip variant) with GAE.

    Architecture
    ------------
    • Policy  – CNN reads the raw board (1×H×W) + piece one-hots → logits
    • Value   – Linear model V(s) = w^T φ(s) + b  on the 35-dim flat features,
                fitted once per rollout by closed-form ridge regression (least
                squares).  No gradient descent, no shared backbone.

    Separating policy and value this way:
      1. Eliminates the gradient conflict in a shared backbone
      2. Gives the policy a spatial (CNN) inductive bias for the grid
      3. Keeps value estimation simple and numerically stable
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        board_h: int = 20,
        board_w: int = 6,
        num_pieces: int = 7,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        epochs: int = 4,
        minibatch_size: int = 64,
        ent_coef: float = 0.05,
        n_filters: int = 32,
        hidden: int = 128,
        value_reg: float = 1e-4,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.ent_coef = ent_coef
        self.device = device
        self.epsilon = 0.0          # no ε-greedy by default for PPO

        # ── CNN policy ────────────────────────────────────────
        self.policy = CNNPolicyNetwork(
            board_h, board_w, num_pieces, act_dim,
            n_filters=n_filters, hidden=hidden,
        ).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # ── Linear value estimator (least squares) ───────────
        self.value = LinearValueEstimator(obs_dim, reg=value_reg)

    # ── action selection ──────────────────────────────────────

    def select_action(self, obs: np.ndarray, mask: np.ndarray, env=None):
        """
        Parameters
        ----------
        obs  : flat observation (obs_dim,) — for the linear value estimator
        mask : legal action mask (act_dim,)
        env  : TetrisLiteEnv — REQUIRED for the CNN policy (provides board + piece)

        Returns
        -------
        (action: int, log_prob: float, value: float)
        """
        assert env is not None, (
            "PPOAgent.select_action requires env for CNN observation"
        )

        # Value from linear estimator (pure numpy, no gradient)
        value = self.value.predict_single(obs)

        # CNN policy (torch, no_grad for inference)
        board_2d, piece_vec = env.get_cnn_obs()
        board_t = torch.as_tensor(
            board_2d, dtype=torch.float32, device=self.device
        ).unsqueeze(0)                                         # (1, 1, H, W)
        piece_t = torch.as_tensor(
            piece_vec, dtype=torch.float32, device=self.device
        ).unsqueeze(0)                                         # (1, 14)

        with torch.no_grad():
            logits = self.policy(board_t, piece_t)             # (1, act_dim)

        # Mask illegal actions
        mask_t = torch.as_tensor(mask, dtype=torch.float32, device=self.device)
        logits = logits[0] + (1 - mask_t) * (-1e8)
        dist = Categorical(logits=logits)

        # ε-greedy (usually 0 for PPO)
        if self.epsilon > 0 and np.random.random() < self.epsilon:
            legal = np.where(mask)[0]
            action = int(np.random.choice(legal))
            log_prob = dist.log_prob(
                torch.tensor(action, device=self.device)
            ).item()
        else:
            action_t = dist.sample()
            log_prob = dist.log_prob(action_t).item()
            action = action_t.item()

        return action, log_prob, value

    # ── PPO update ────────────────────────────────────────────

    def update(self, buffer: RolloutBuffer, last_value: float = 0.0) -> dict:
        obs_t, act_t, rew_t, done_t, old_lp_t, val_t, mask_t = \
            buffer.to_tensors(self.device)
        board_t, piece_t = buffer.cnn_tensors(self.device)

        # ── GAE ───────────────────────────────────────────────
        advantages, returns = compute_gae(
            rew_t, val_t, done_t, self.gamma, self.lam,
            last_value=last_value,
        )
        advantages = advantages.to(self.device)
        returns    = returns.to(self.device)

        # ── Fit value estimator by least squares ──────────────
        obs_np     = obs_t.cpu().numpy()
        returns_np = returns.cpu().numpy()
        self.value.fit(obs_np, returns_np)

        # ── Advantage stats (before normalisation) ────────────
        adv_mean = advantages.mean().item()
        adv_std  = advantages.std().item()
        adv_min  = advantages.min().item()
        adv_max  = advantages.max().item()

        # ── Normalise advantages ──────────────────────────────
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── PPO policy update (multiple epochs) ───────────────
        N = len(obs_t)
        indices = np.arange(N)
        total_pg_loss = 0.0
        total_entropy = 0.0
        total_grad_norm = 0.0
        total_ratio_sum = 0.0
        max_ratio = 0.0
        total_clipped = 0
        total_samples = 0
        total_kl = 0.0
        n_updates = 0

        for _ in range(self.epochs):
            np.random.shuffle(indices)
            for start in range(0, N, self.minibatch_size):
                end = start + self.minibatch_size
                mb = indices[start:end]

                # Forward through CNN policy
                logits = self.policy(board_t[mb], piece_t[mb])

                # Mask illegal actions
                logits = logits + (1 - mask_t[mb]) * (-1e8)
                dist = Categorical(logits=logits)

                new_lp  = dist.log_prob(act_t[mb])
                entropy = dist.entropy().mean()

                # Clipped surrogate objective
                ratio = torch.exp(new_lp - old_lp_t[mb])
                mb_adv = advantages[mb]
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(
                    ratio, 1 - self.clip_eps, 1 + self.clip_eps
                ) * mb_adv
                pg_loss = -torch.min(surr1, surr2).mean()

                # Policy loss only — no vf_loss through the CNN!
                loss = pg_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                gn = nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                # Accumulate diagnostics
                total_pg_loss += pg_loss.item()
                total_entropy += entropy.item()
                total_grad_norm += gn.item()
                total_ratio_sum += ratio.mean().item()
                max_ratio = max(max_ratio, ratio.max().item())
                total_clipped += ((ratio - 1.0).abs() > self.clip_eps).sum().item()
                total_samples += len(mb)
                total_kl += (old_lp_t[mb] - new_lp).mean().item()
                n_updates += 1

        # ── Value loss for logging (post-fit residual MSE) ────
        with torch.no_grad():
            fitted_vals = torch.as_tensor(
                self.value.predict(obs_np),
                dtype=torch.float32, device=self.device,
            )
            vf_loss = F.mse_loss(fitted_vals, returns).item()

        # ── Explained variance ────────────────────────────────
        fitted_vals_np = self.value.predict(obs_np)
        var_true = np.var(returns_np)
        explained_var = 1.0 - np.var(returns_np - fitted_vals_np) / max(var_true, 1e-8)

        return {
            "policy_loss": total_pg_loss / max(n_updates, 1),
            "value_loss":  vf_loss,
            "entropy":     total_entropy / max(n_updates, 1),
            "mean_return": returns.mean().item(),
            "grad_norm":   total_grad_norm / max(n_updates, 1),
            "advantage_mean": adv_mean,
            "advantage_std":  adv_std,
            "advantage_min":  adv_min,
            "advantage_max":  adv_max,
            "ratio_mean":     total_ratio_sum / max(n_updates, 1),
            "ratio_max":      max_ratio,
            "clip_fraction":  total_clipped / max(total_samples, 1),
            "explained_variance": explained_var,
            "approx_kl":      total_kl / max(n_updates, 1),
        }
