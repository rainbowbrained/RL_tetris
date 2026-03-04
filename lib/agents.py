import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import List, Tuple, Optional


class PolicyNetwork(nn.Module):
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
            logits = logits + (1 - mask) * (-1e8)
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


class MLPPolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class CNNPolicyNetwork(nn.Module):
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
        with torch.no_grad():
            dummy = torch.zeros(1, 1, board_h, board_w)
            conv_out = self.conv(dummy)
            self._conv_flat = conv_out.view(1, -1).shape[1]

        piece_dim = num_pieces * 2
        self.fc = nn.Sequential(
            nn.Linear(self._conv_flat + piece_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, board: torch.Tensor, piece_info: torch.Tensor) -> torch.Tensor:
        x = self.conv(board)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, piece_info], dim=1)
        return self.fc(x)


class LinearValueEstimator:
    def __init__(self, obs_dim: int, reg: float = 1e-4):
        self.obs_dim = obs_dim
        self.reg = reg
        self.w = np.zeros(obs_dim + 1, dtype=np.float64)

    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([X, np.ones(len(X), dtype=np.float64)])

    def predict(self, obs: np.ndarray) -> np.ndarray:
        X = self._add_bias(obs.astype(np.float64))
        return (X @ self.w).astype(np.float32)

    def predict_single(self, obs: np.ndarray) -> float:
        x = np.append(obs.astype(np.float64), 1.0)
        return float(x @ self.w)

    def fit(self, obs: np.ndarray, returns: np.ndarray):
        X = self._add_bias(obs.astype(np.float64))
        y = returns.astype(np.float64)
        A = X.T @ X + self.reg * np.eye(X.shape[1])
        b = X.T @ y
        self.w = np.linalg.solve(A, b)

    def state_dict(self) -> dict:
        return {"w": self.w.copy()}

    def load_state_dict(self, d: dict):
        self.w = d["w"].copy()


class MLPValueEstimator(nn.Module):
    def __init__(self, obs_dim: int, hidden1: int = 128, hidden2: int = 64,
                 lr: float = 1e-3, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        ).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def forward(self, obs_t: torch.Tensor) -> torch.Tensor:
        return self.net(obs_t).squeeze(-1)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            vals = self.forward(obs_t)
        return vals.cpu().numpy().astype(np.float32)

    def predict_single(self, obs: np.ndarray) -> float:
        with torch.no_grad():
            obs_t = torch.as_tensor(
                obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            val = self.forward(obs_t)
        return val.item()

    def train_on_batch(self, obs_t: torch.Tensor, returns_t: torch.Tensor) -> float:
        pred = self.forward(obs_t)
        loss = F.mse_loss(pred, returns_t)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optimizer.step()
        return loss.item()


class RolloutBuffer:
    def __init__(self):
        self.obs: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.masks: List[np.ndarray] = []
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
        board_t = torch.tensor(
            np.array(self.board_obs), dtype=torch.float32, device=device
        )
        piece_t = torch.tensor(
            np.array(self.piece_obs), dtype=torch.float32, device=device
        )
        return board_t, piece_t


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
    last_value: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    T = len(rewards)
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


class ReinforceAgent:
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

        T = len(rew_t)
        returns = torch.zeros(T)
        G = 0.0
        for t in reversed(range(T)):
            if done_t[t]:
                G = 0.0
            G = rew_t[t] + self.gamma * G
            returns[t] = G
        returns = returns.to(self.device)

        baseline = returns.mean()

        logits, _ = self.net(obs_t, mask_t)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(act_t)
        entropy = dist.entropy().mean()

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


class PPOAgent:
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
        value_lr: float = 1e-3,
        target_kl: float = 0.02,
        device: str = "cpu",
        policy_type: str = "cnn",
        value_type: str = "mlp",
    ):
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.ent_coef = ent_coef
        self.target_kl = target_kl
        self.device = device
        self.epsilon = 0.0
        self.policy_type = policy_type
        self.value_type = value_type

        if policy_type == "cnn":
            self.policy = CNNPolicyNetwork(
                board_h, board_w, num_pieces, act_dim,
                n_filters=n_filters, hidden=hidden,
            ).to(device)
        elif policy_type == "mlp":
            self.policy = MLPPolicyNetwork(
                obs_dim, act_dim, hidden=hidden,
            ).to(device)
        else:
            raise ValueError(f"Unknown policy_type: {policy_type!r}")
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        if value_type == "mlp":
            self.value = MLPValueEstimator(obs_dim, lr=value_lr, device=device)
        elif value_type == "linear":
            self.value = LinearValueEstimator(obs_dim)
        else:
            raise ValueError(f"Unknown value_type: {value_type!r}")

    @property
    def uses_cnn(self) -> bool:
        return self.policy_type == "cnn"

    def select_action(self, obs: np.ndarray, mask: np.ndarray, env=None):
        value = self.value.predict_single(obs)

        with torch.no_grad():
            if self.policy_type == "cnn":
                assert env is not None, (
                    "PPOAgent with CNN policy requires env for get_cnn_obs()"
                )
                board_2d, piece_vec = env.get_cnn_obs()
                board_t = torch.as_tensor(
                    board_2d, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                piece_t = torch.as_tensor(
                    piece_vec, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                logits = self.policy(board_t, piece_t)
            else:
                obs_t = torch.as_tensor(
                    obs, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                logits = self.policy(obs_t)

        mask_t = torch.as_tensor(mask, dtype=torch.float32, device=self.device)
        logits = logits[0] + (1 - mask_t) * (-1e8)
        dist = Categorical(logits=logits)

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

    def update(self, buffer: RolloutBuffer, last_value: float = 0.0) -> dict:
        obs_t, act_t, rew_t, done_t, old_lp_t, val_t, mask_t = \
            buffer.to_tensors(self.device)

        if self.policy_type == "cnn":
            board_t, piece_t = buffer.cnn_tensors(self.device)

        advantages, returns = compute_gae(
            rew_t, val_t, done_t, self.gamma, self.lam,
            last_value=last_value,
        )
        advantages = advantages.to(self.device)
        returns    = returns.to(self.device)

        adv_mean = advantages.mean().item()
        adv_std  = advantages.std().item()
        adv_min  = advantages.min().item()
        adv_max  = advantages.max().item()

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        if self.value_type == "linear":
            self.value.fit(obs_t.cpu().numpy(), returns.cpu().numpy())

        N = len(obs_t)
        indices = np.arange(N)
        total_pg_loss = 0.0
        total_vf_loss = 0.0
        total_entropy = 0.0
        total_grad_norm = 0.0
        total_ratio_sum = 0.0
        max_ratio = 0.0
        total_clipped = 0
        total_samples = 0
        total_kl = 0.0
        n_updates = 0
        epochs_completed = 0

        for _epoch in range(self.epochs):
            epoch_kl = 0.0
            epoch_batches = 0

            np.random.shuffle(indices)
            for start in range(0, N, self.minibatch_size):
                end = start + self.minibatch_size
                mb = indices[start:end]

                if self.policy_type == "cnn":
                    logits = self.policy(board_t[mb], piece_t[mb])
                else:
                    logits = self.policy(obs_t[mb])

                logits = logits + (1 - mask_t[mb]) * (-1e8)
                dist = Categorical(logits=logits)

                new_lp  = dist.log_prob(act_t[mb])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_lp - old_lp_t[mb])
                mb_adv = advantages[mb]
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(
                    ratio, 1 - self.clip_eps, 1 + self.clip_eps
                ) * mb_adv
                pg_loss = -torch.min(surr1, surr2).mean()

                loss = pg_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                gn = nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                if self.value_type == "mlp":
                    vf_loss = self.value.train_on_batch(obs_t[mb], returns[mb])
                else:
                    with torch.no_grad():
                        pred = torch.as_tensor(
                            self.value.predict(obs_t[mb].cpu().numpy()),
                            device=self.device,
                        )
                        vf_loss = ((pred - returns[mb]) ** 2).mean().item()

                approx_kl_mb = (old_lp_t[mb] - new_lp).mean().item()
                total_pg_loss += pg_loss.item()
                total_vf_loss += vf_loss
                total_entropy += entropy.item()
                total_grad_norm += gn.item()
                total_ratio_sum += ratio.mean().item()
                max_ratio = max(max_ratio, ratio.max().item())
                total_clipped += ((ratio - 1.0).abs() > self.clip_eps).sum().item()
                total_samples += len(mb)
                total_kl += approx_kl_mb
                epoch_kl += approx_kl_mb
                epoch_batches += 1
                n_updates += 1

            epochs_completed += 1

            if self.target_kl and epoch_kl / max(epoch_batches, 1) > self.target_kl:
                break

        obs_np = obs_t.cpu().numpy()
        returns_np = returns.cpu().numpy()
        fitted_vals_np = self.value.predict(obs_np)
        var_true = np.var(returns_np)
        explained_var = 1.0 - np.var(returns_np - fitted_vals_np) / max(var_true, 1e-8)

        return {
            "policy_loss": total_pg_loss / max(n_updates, 1),
            "value_loss":  total_vf_loss / max(n_updates, 1),
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
            "epochs_completed": epochs_completed,
        }
