import argparse
import random

import chess

from arena import play_game
from chess_gym.env import ChessEnv
from q_learning import (
    CHECKPOINT_DIR,
    DEFAULT_FEN_BLACK,
    DEFAULT_FEN_WHITE,
    EPSILON,
    MAX_STEPS,
    play_episode,
    save_checkpoint,
)


def train(q, color, fen, episodes, seed):
    rng_state = random.getstate()
    random.seed(seed)
    env = ChessEnv(fen=fen, max_steps=MAX_STEPS)
    env.action_space.seed(seed)
    stats = {"win": 0, "loss": 0, "draw": 0}
    for ep in range(episodes):
        stats[play_episode(q, env, color, EPSILON, seed=seed * 100_000 + ep)] += 1
    env.close()
    random.setstate(rng_state)
    return stats


def aggregate(tables):
    sums = {}
    counts = {}
    for table in tables:
        for fen, row in table.items():
            acc = sums.setdefault(fen, {})
            cnt = counts.setdefault(fen, {})
            for action, value in row.items():
                acc[action] = acc.get(action, 0.0) + value
                cnt[action] = cnt.get(action, 0) + 1
    return {
        fen: {a: total / counts[fen][a] for a, total in row.items()}
        for fen, row in sums.items()
    }


def federated_train(color, episodes, clients, group_size, seed_base=0):
    fen = DEFAULT_FEN_WHITE if color else DEFAULT_FEN_BLACK
    name = "white" if color else "black"

    round1 = []
    for i in range(1, clients + 1):
        q = {}
        stats = train(q, color, fen, episodes, seed_base + i)
        round1.append(q)
        print(f"[{name}] client {i:2d}/{clients}: {stats} states={len(q)}")

    groups = [round1[i : i + group_size] for i in range(0, clients, group_size)]
    round2 = []
    for i, q in enumerate([aggregate(group) for group in groups], start=1):
        stats = train(q, color, fen, episodes, seed_base + 1_000 + i)
        round2.append(q)
        print(f"[{name}] retrain {i}/{len(groups)}: {stats} states={len(q)}")

    final = aggregate(round2)
    print(f"[{name}] final model: {len(final)} states")
    return final


def evaluate(q, color, fen, games):
    stats = {"win": 0, "loss": 0, "draw": 0}
    tables = {
        chess.WHITE: q if color == chess.WHITE else {},
        chess.BLACK: q if color == chess.BLACK else {},
    }
    for _ in range(games):
        outcome, _steps = play_game(
            tables[chess.WHITE], tables[chess.BLACK], fen, max_steps=MAX_STEPS
        )
        if outcome is None or outcome.winner is None:
            stats["draw"] += 1
        elif outcome.winner == color:
            stats["win"] += 1
        else:
            stats["loss"] += 1
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Federated tabular Q-learning (train → FedAvg → retrain → FedAvg)"
    )
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--clients", type=int, default=15)
    parser.add_argument("--group-size", type=int, default=3)
    parser.add_argument("--color", choices=["white", "black", "both"], default="both")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-games", type=int, default=30)
    args = parser.parse_args()
    if args.clients < 1 or args.group_size < 1:
        parser.error("--clients and --group-size must be positive")
    if args.clients % args.group_size:
        parser.error("--clients must be divisible by --group-size")

    colors = {
        "white": [chess.WHITE],
        "black": [chess.BLACK],
        "both": [chess.WHITE, chess.BLACK],
    }[args.color]

    for color in colors:
        name = "white" if color else "black"
        fen = DEFAULT_FEN_WHITE if color else DEFAULT_FEN_BLACK
        q = federated_train(
            color, args.episodes, args.clients, args.group_size, args.seed
        )
        out = CHECKPOINT_DIR / f"fed_q_{name}.pkl"
        save_checkpoint(out, q, color, fen)
        print(f"[{name}] checkpoint saved: {out}")
        print(
            f"[{name}] final eval vs random: {evaluate(q, color, fen, args.eval_games)}"
        )


if __name__ == "__main__":
    main()
