import argparse
import random

import chess
import numpy as np

from arena import play_game
from chess_gym.env import ChessEnv
from q_learning import (
    CHECKPOINT_DIR,
    load_checkpoint,
    load_fens,
    resolve_fen,
    save_checkpoint,
)


class ChessAgent:
    def __init__(self, color, alpha=0.1, gamma=0.99, epsilon=0.2, q=None):
        self.q = {} if q is None else q
        self.color = color
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def _values(self, fen, legal):
        row = self.q.get(fen, {})
        return np.array([row.get(a, 0.0) for a in legal], dtype=float)

    def act(self, board, mask):
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            return None
        values = self._values(board.fen(), legal)
        best = legal[values == values.max()]
        return int(random.choice(best))

    def choose(self, board, mask):
        legal = np.flatnonzero(mask)
        if random.random() < self.epsilon:
            return int(random.choice(legal))
        return self.act(board, mask)

    def learn(self, fen, action, reward, next_board=None, next_mask=None, done=True):
        row = self.q.setdefault(fen, {})
        if done or next_board is None:
            target = reward
        else:
            next_legal = np.flatnonzero(next_mask)
            target = reward + self.gamma * float(
                self._values(next_board.fen(), next_legal).max()
            )
        old = row.get(action, 0.0)
        row[action] = old + self.alpha * (target - old)


def self_play_episode(env, white, black, fen=None):
    if fen is None:
        obs, _ = env.reset()
    else:
        obs, _ = env.reset(options={"fen": fen})
    board = env.board
    agents = {chess.WHITE: white, chess.BLACK: black}
    pending = None
    done = False
    while not done:
        agent = agents[board.turn]
        s = board.fen()
        a = agent.choose(board, obs["action_mask"])
        obs, r, term, trunc, _ = env.step(a)
        done = term or trunc
        if pending is not None:
            p_agent, p_s, p_a = pending
            if term:
                p_agent.learn(p_s, p_a, -r, done=True)
            elif trunc:
                p_agent.learn(p_s, p_a, 0.0, done=True)
            else:
                p_agent.learn(p_s, p_a, 0.0, board, obs["action_mask"], done=False)
            pending = None
        if done:
            agent.learn(s, a, r, done=True)
            break
        pending = (agent, s, a)
    outcome = board.outcome()
    if outcome is None or outcome.winner is None:
        return "draw"
    return "win" if outcome.winner == chess.WHITE else "loss"


def warm_start(white_path=None, black_path=None):
    qs = {chess.WHITE: None, chess.BLACK: None}
    fen = None
    fens = None
    for color, path in ((chess.WHITE, white_path), (chess.BLACK, black_path)):
        if path is None:
            continue
        ckpt = load_checkpoint(path)
        if ckpt["color"] != color:
            raise ValueError(
                f"{path} holds a {'white' if ckpt['color'] else 'black'} agent"
            )
        qs[color] = ckpt["q"]
        if color == chess.WHITE:
            fen = ckpt["fen"]
            fens = ckpt.get("fens")
    return qs[chess.WHITE], qs[chess.BLACK], fen, fens


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-agent self-play Q-learning")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument(
        "--fen",
        help="start position FEN, or 'start' for the opening "
        "(default: standard game start)",
    )
    parser.add_argument(
        "--fen-file",
        help="file with one FEN per line; each episode starts from a random entry",
    )
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--white-checkpoint")
    parser.add_argument("--black-checkpoint")
    parser.add_argument(
        "--render",
        choices=["human", "unicode"],
        help="play one exhibition episode (greedy, no training) instead of training",
    )
    parser.add_argument("--move-delay", type=float, default=0.35)
    args = parser.parse_args()

    if args.fen and args.fen_file:
        parser.error("use either --fen or --fen-file, not both")

    try:
        white_q, black_q, ckpt_fen, ckpt_fens = warm_start(
            args.white_checkpoint, args.black_checkpoint
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.fen_file:
        try:
            fens = load_fens(args.fen_file)
        except ValueError as exc:
            parser.error(str(exc))
    elif args.fen:
        fens = [resolve_fen(args.fen)]
    elif ckpt_fens:
        fens = ckpt_fens
    else:
        fens = [ckpt_fen or chess.STARTING_FEN]
    if len(fens) > 1:
        print(f"start positions: {len(fens)} (random per episode)")
    else:
        print(f"start position: {fens[0]}")
    white = ChessAgent(chess.WHITE, q=white_q)
    black = ChessAgent(chess.BLACK, q=black_q)

    if args.render:
        fen = random.choice(fens)
        outcome, steps = play_game(
            white.q,
            black.q,
            fen,
            max_steps=args.max_steps,
            render_mode=args.render,
            move_delay=args.move_delay,
        )
        if outcome is None or outcome.winner is None:
            print(f"exhibition: draw after {steps} plies")
        else:
            winner = "white" if outcome.winner == chess.WHITE else "black"
            print(
                f"exhibition: {winner} wins by "
                f"{outcome.termination.name} in {steps} plies"
            )
        return

    env = ChessEnv(fen=fens[0], max_steps=args.max_steps)
    if white_q is not None or black_q is not None:
        print(f"resuming from: {args.white_checkpoint} / {args.black_checkpoint}")
    print(f"start states — white: {len(white.q)}  black: {len(black.q)}")

    stats = {"win": 0, "loss": 0, "draw": 0}
    for ep in range(1, args.episodes + 1):
        stats[self_play_episode(env, white, black, fen=random.choice(fens))] += 1
        if ep % 50 == 0:
            print(f"episode {ep}: {stats}")
    env.close()
    print(f"states — white: {len(white.q)}  black: {len(black.q)}")
    for agent, name in ((white, "sp_white"), (black, "sp_black")):
        out = CHECKPOINT_DIR / f"{name}.pkl"
        save_checkpoint(
            out, agent.q, agent.color, fens[0], fens=fens if len(fens) > 1 else None
        )
        print(f"checkpoint saved: {out}")


if __name__ == "__main__":
    main()
