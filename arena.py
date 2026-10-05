import argparse
import time

import chess
import pygame

from chess_gym.env import ChessEnv
from q_learning import greedy_action, load_checkpoint, resolve_fen


def load_agent(path, expected_color, parser):
    ckpt = load_checkpoint(path)
    if ckpt["color"] != expected_color:
        parser.error(
            f"{path} holds a {'white' if ckpt['color'] else 'black'} agent; "
            f"expected {'white' if expected_color else 'black'}"
        )
    return ckpt


def play_game(
    white_q, black_q, fen, max_steps=200, render_mode=None, move_delay=0.0, env=None
):
    owns_env = env is None
    if env is None:
        env = ChessEnv(fen=fen, max_steps=max_steps, render_mode=render_mode)
        obs, _ = env.reset()
    else:
        obs, _ = env.reset(options={"fen": fen})
    tables = {chess.WHITE: white_q, chess.BLACK: black_q}
    board = env.board
    done = board.is_game_over()
    while not done:
        if render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return board.outcome(), env.step_count
            time.sleep(move_delay)
        elif render_mode == "unicode":
            env.render()
            time.sleep(move_delay)
        action = greedy_action(tables[board.turn], board.fen(), obs["action_mask"])
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    outcome = board.outcome()
    steps = env.step_count
    if render_mode == "human":
        result = outcome.termination.name if outcome else "MAX_STEPS"
        pygame.display.set_caption(f"Chess-v0 — {result}")
        time.sleep(max(move_delay, 1.0))
    if owns_env:
        env.close()
    return outcome, steps


def main() -> None:
    parser = argparse.ArgumentParser(description="Pit two Q-agents against each other")
    parser.add_argument("--white", default="checkpoints/q_white.pkl")
    parser.add_argument("--black", default="checkpoints/q_black.pkl")
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument(
        "--fen",
        help="start position FEN, or 'start' for the opening "
        "(default: white agent's training FEN)",
    )
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--render", choices=["human", "unicode"])
    parser.add_argument("--move-delay", type=float, default=0.35)
    args = parser.parse_args()

    white = load_agent(args.white, chess.WHITE, parser)
    black = load_agent(args.black, chess.BLACK, parser)
    fen = resolve_fen(args.fen) or white["fen"]
    print(f"start position: {fen}")

    env = None
    if args.render == "human":
        env = ChessEnv(fen=fen, max_steps=args.max_steps, render_mode="human")

    stats = {"white": 0, "black": 0, "draw": 0}
    total_steps = 0
    for _ in range(args.games):
        outcome, steps = play_game(
            white["q"],
            black["q"],
            fen,
            args.max_steps,
            args.render,
            args.move_delay,
            env,
        )
        if outcome is not None and outcome.winner is not None:
            stats["white" if outcome.winner == chess.WHITE else "black"] += 1
        else:
            stats["draw"] += 1
        total_steps += steps

    print(f"{args.games} games from: {fen}")
    print(f"white: {stats['white']}  black: {stats['black']}  draw: {stats['draw']}")
    print(f"avg game length: {total_steps / args.games:.1f} plies")
    if env is not None:
        env.close()


if __name__ == "__main__":
    main()
