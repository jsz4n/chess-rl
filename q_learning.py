import argparse
import pickle
import random
from pathlib import Path

import chess
import numpy as np

from chess_gym.env import ChessEnv

DEFAULT_FEN_WHITE = "6k1/5ppp/8/8/8/8/8/4R2K w - - 0 1"
DEFAULT_FEN_BLACK = "4r2k/8/8/8/8/8/5PPP/6K1 b - - 0 1"
ALPHA = 0.1
GAMMA = 0.99
EPSILON = 0.2
MAX_STEPS = 60
CHECKPOINT_DIR = Path("checkpoints")


def q_values(q, state_key, legal_actions):
    row = q.get(state_key, {})
    return np.array([row.get(a, 0.0) for a in legal_actions], dtype=float)


def choose_action(q, state_key, mask, epsilon):
    legal_actions = np.flatnonzero(mask)
    if random.random() < epsilon:
        return int(random.choice(legal_actions))
    values = q_values(q, state_key, legal_actions)
    return int(legal_actions[int(np.argmax(values))])


def greedy_action(q, state_key, mask):
    legal_actions = np.flatnonzero(mask)
    if legal_actions.size == 0:
        return None
    values = q_values(q, state_key, legal_actions)
    best = legal_actions[values == values.max()]
    return int(random.choice(best))


def update(q, state_key, action, reward, next_key, next_mask, done):
    row = q.setdefault(state_key, {})
    if done:
        best_next = 0.0
    else:
        next_legal = np.flatnonzero(next_mask)
        best_next = float(np.max(q_values(q, next_key, next_legal)))
    old = row.get(action, 0.0)
    row[action] = old + ALPHA * (reward + GAMMA * best_next - old)


def play_episode(q, env, color, epsilon=EPSILON, seed=None):
    obs, _ = env.reset(seed=seed)
    state_key = env.board.fen()
    done = False
    while not done:
        if env.board.turn == color:
            action = choose_action(q, state_key, obs["action_mask"], epsilon)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if not done:
                obs, opp_reward, terminated, truncated, _ = env.step(
                    env.action_space.sample()
                )
                done = terminated or truncated
                if terminated:
                    reward -= opp_reward
            update(
                q,
                state_key,
                action,
                reward,
                env.board.fen(),
                obs["action_mask"],
                done,
            )
            state_key = env.board.fen()
        else:
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            done = terminated or truncated
            state_key = env.board.fen()
    outcome = env.board.outcome()
    if outcome is None or outcome.winner is None:
        return "draw"
    return "win" if outcome.winner == color else "loss"


def save_checkpoint(path, q, color, fen) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"q": q, "color": color, "fen": fen}, f)


def load_checkpoint(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tabular Q-learning from a fixed FEN")
    parser.add_argument("--fen")
    parser.add_argument("--color", choices=["white", "black"], default="white")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--out")
    args = parser.parse_args()

    color = chess.WHITE if args.color == "white" else chess.BLACK
    fen = args.fen or (DEFAULT_FEN_WHITE if color else DEFAULT_FEN_BLACK)
    if chess.Board(fen).turn != color:
        parser.error(
            f"FEN has {'white' if chess.Board(fen).turn else 'black'} to move, "
            f"but --color {args.color} was requested"
        )

    env = ChessEnv(fen=fen, max_steps=MAX_STEPS)
    q = {}
    stats = {"win": 0, "loss": 0, "draw": 0}
    for ep in range(1, args.episodes + 1):
        stats[play_episode(q, env, color)] += 1
        if ep % 50 == 0:
            print(f"episode {ep}: {stats}")
    print(f"states visited: {len(q)}")

    out = args.out or (CHECKPOINT_DIR / f"q_{args.color}.pkl")
    save_checkpoint(out, q, color, fen)
    print(f"checkpoint saved: {out}")
    env.close()


if __name__ == "__main__":
    main()
