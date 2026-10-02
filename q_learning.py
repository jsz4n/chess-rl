import random

import chess
import numpy as np

from chess_gym.env import ChessEnv

EPISODES = 300
ALPHA = 0.1
GAMMA = 0.99
EPSILON = 0.2
MAX_STEPS = 60


def q_values(q, state_key, legal_actions):
    row = q.get(state_key, {})
    return np.array([row.get(a, 0.0) for a in legal_actions], dtype=float)


def choose_action(q, state_key, mask):
    legal_actions = np.flatnonzero(mask)
    if random.random() < EPSILON:
        return int(random.choice(legal_actions))
    values = q_values(q, state_key, legal_actions)
    return int(legal_actions[int(np.argmax(values))])


def update(q, state_key, action, reward, next_key, next_mask, done):
    row = q.setdefault(state_key, {})
    if done:
        best_next = 0.0
    else:
        next_legal = np.flatnonzero(next_mask)
        best_next = float(np.max(q_values(q, next_key, next_legal)))
    old = row.get(action, 0.0)
    row[action] = old + ALPHA * (reward + GAMMA * best_next - old)


def play_episode(q, env):
    obs, _ = env.reset()
    state_key = env.board.fen()
    done = False
    while not done:
        action = choose_action(q, state_key, obs["action_mask"])
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if not done:
            opp_action = env.action_space.sample()
            obs, opp_reward, terminated, truncated, _ = env.step(opp_action)
            reward += opp_reward
            done = terminated or truncated
        next_key = env.board.fen()
        update(q, state_key, action, reward, next_key, obs["action_mask"], done)
        state_key = next_key
    outcome = env.board.outcome()
    if outcome is None or outcome.winner is None:
        return "draw"
    return "win" if outcome.winner == chess.WHITE else "loss"


def main():
    env = ChessEnv(max_steps=MAX_STEPS, render_mode="human")
    q = {}
    stats = {"win": 0, "loss": 0, "draw": 0}
    for ep in range(1, EPISODES + 1):
        result = play_episode(q, env)
        stats[result] += 1
        if ep % 50 == 0:
            print(f"episode {ep}: {stats}")
    print(f"states visited: {len(q)}")
    env.render_mode = "unicode"
    env.render()
    env.close()


if __name__ == "__main__":
    main()
