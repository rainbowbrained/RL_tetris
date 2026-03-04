"""Quick training test to verify the agent learns to clear lines."""
from tetris_env import TetrisLiteEnv, PIECE_NAMES, NUM_PIECES
from agents import PPOAgent, RolloutBuffer
import numpy as np

W, H = 6, 20
env = TetrisLiteEnv(width=W, height=H, max_steps=500, seed=None)
print(f"obs_size={env.obs_size()}, actions={env.num_actions()}")

ppo = PPOAgent(
    obs_dim=env.obs_size(), act_dim=env.num_actions(),
    board_h=H, board_w=W, num_pieces=NUM_PIECES,
    lr=3e-4, gamma=0.99, lam=0.95, clip_eps=0.2,
    epochs=4, minibatch_size=256, hidden=256, device='cpu',
)

for it in range(400):
    buf = RolloutBuffer()
    obs = env.reset()
    ep_rews, ep_lines, ep_steps = [], [], []
    cur_rew, cur_lines, cur_steps = 0, 0, 0
    done = False
    for _ in range(4096):
        mask = env.legal_action_mask()
        board_obs, piece_obs = env.get_cnn_obs()
        action, lp, val = ppo.select_action(obs, mask, env=env)
        next_obs, rew, done, info = env.step(action)
        buf.store(obs, action, rew, done, lp, val, mask,
                  board_obs=board_obs, piece_obs=piece_obs)
        cur_rew += rew; cur_lines += info['lines']; cur_steps += 1
        obs = next_obs
        if done:
            ep_rews.append(cur_rew); ep_lines.append(cur_lines); ep_steps.append(cur_steps)
            cur_rew, cur_lines, cur_steps = 0, 0, 0
            obs = env.reset()
    # Bootstrap value for incomplete episode at end of rollout
    if done:
        last_val = 0.0
    else:
        mask = env.legal_action_mask()
        _, _, last_val = ppo.select_action(obs, mask, env=env)
    ppo.update(buf, last_value=last_val)
    if (it+1) % 10 == 0:
        print(f"Iter {it+1:3d}: rew={np.mean(ep_rews):+7.1f}  lines={np.mean(ep_lines):.2f}  "
              f"steps/ep={np.mean(ep_steps):.0f}  eps={len(ep_rews)}")

# Play one episode and show board
print("\n=== EPISODE REPLAY ===")
env2 = TetrisLiteEnv(width=W, height=H, max_steps=500, seed=42)
obs = env2.reset()
total_lines = 0
for step in range(60):
    mask = env2.legal_action_mask()
    action, _, _ = ppo.select_action(obs, mask, env=env2)
    acts = env2.piece_actions[env2.current_piece]
    rid, x, shape = acts[min(action, len(acts)-1)]
    piece = PIECE_NAMES[env2.current_piece]
    obs, rew, done, info = env2.step(action)
    total_lines += info['lines']
    if info['lines'] > 0:
        print(f"  Step {step+1:2d}: piece={piece} rot={rid} x={x} → LINE CLEAR! ({info['lines']} lines, total={total_lines})")
    if done:
        print(f"  DONE at step {step+1}, total_lines={total_lines}")
        break

print(f"\nTotal lines cleared: {total_lines}")
print("\nFinal board (bottom 12 rows):")
board = env2.render_board()
for r in range(max(0, env2.height-12), env2.height):
    row_str = ''
    for c in range(env2.width):
        row_str += '#' if board[r,c] != 0 else '.'
    print(f"  {row_str}")
