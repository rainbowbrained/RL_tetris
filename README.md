## Methods

### Environment (Tetris-lite Placement MDP)
We implement a minimal Tetris-style MDP where **one decision = one piece placement**.

#### State (observations)
Compact representation to keep learning stable and fast:
- `height_map` — vector of column heights (board width `W`)
- `piece_id` — current tetromino type (one-hot or integer embedding)
- `next_piece_id` — optional (next tetromino), for reduced partial observability

> Optionally, we also log/compute auxiliary board stats (holes, bumpiness) for reward shaping.

#### Actions
- **One-shot placement**: `(x, rotation)` followed by an immediate **hard drop**
- Action space is **piece-dependent** (some rotations invalid, some x positions illegal)
- We enumerate legal placements for the current piece and use **action masking** during sampling

#### Transition
Deterministic:
1. Apply rotation
2. Drop piece until collision
3. Lock piece into board
4. Clear completed lines
5. Spawn next piece
6. Terminal if no legal placements exist (game over)

#### Reward
- Main signal: `+ lines_cleared` (e.g., 1/2/3/4)
- Shaping to reduce sparsity and discourage bad geometry:
  - `- holes` (empty cells with a filled cell above in the same column)
  - `- bumpiness` (sum of `|h[i] - h[i+1]|` over columns)
- Optional: small step penalty to encourage efficiency

---

### Policy Parameterization (Discrete masked policy)
- We model a **categorical policy** over the set of legal placements for the current piece:
  - network outputs logits for all placements (or per-(x,rot) grid), then we apply a **legal mask**
- Benefits:
  - avoids wasting probability mass on illegal actions
  - keeps gradients well-behaved
  - simplifies evaluation: greedy vs stochastic

---

### Baseline: REINFORCE (Monte-Carlo Policy Gradient)
We implement REINFORCE as a baseline for comparison.

- Roll out episodes with the current policy
- Compute returns `G_t` (discounted sum of rewards)
- Policy gradient update:
  - maximize `E[ log π(a_t|s_t) * (G_t - b_t) ]`
- We use a **value baseline** (learned critic) or moving-average baseline to reduce variance

---

### Advanced Policy Gradient: PPO (Actor–Critic, from first principles)
We implement PPO to improve stability vs REINFORCE.

#### Actor–Critic architecture
- Shared encoder over state features
- Two heads:
  - **policy head** → action logits (with masking)
  - **value head** → `V(s)`

#### Advantage estimation (GAE-λ)
- Compute TD residuals `δ_t = r_t + γ V(s_{t+1}) - V(s_t)`
- Advantages via GAE:
  - `A_t = Σ (γλ)^k δ_{t+k}`

#### PPO clipped objective
- Ratio: `r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)`
- Optimize:
  - `E[ min(r_t A_t, clip(r_t, 1-ε, 1+ε) A_t) ]`
- Add:
  - value loss: `||V(s_t) - target||^2`
  - entropy bonus for exploration
  - optional KL penalty / early stopping by KL to prevent collapse

---

### Training Protocol
- On-policy rollouts: collect trajectories with current policy
- Normalize advantages (per batch)
- Multiple PPO epochs over the same batch (minibatches)
- Evaluation snapshots:
  - greedy (argmax) policy performance
  - stochastic policy performance across multiple seeds

---

### Evaluation Metrics
- Average episode return
- Mean lines cleared per episode
- Game length (pieces placed)
- Board quality diagnostics:
  - holes, bumpiness, max height
- Sample efficiency: curves vs environment steps
- Stability: mean ± std over several random seeds

---

### (Optional) Roadmap Extensions
- Add `next_piece` to compare MDP vs more POMDP-like setting
- Curriculum over board width / piece set
- Ablations: reward shaping weights, masking strategy, entropy coefficient
