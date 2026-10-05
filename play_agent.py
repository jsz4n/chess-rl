import argparse
import time

import chess
import gymnasium as gym
import pygame

import chess_gym
from chess_gym.render import pixel_to_square
from q_learning import greedy_action, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Play against a trained Q-agent")
    parser.add_argument("--checkpoint", default="checkpoints/q_white.pkl")
    args = parser.parse_args()

    ckpt = load_checkpoint(args.checkpoint)
    q, agent_color, fen = ckpt["q"], ckpt["color"], ckpt["fen"]
    human_color = not agent_color

    env = gym.make("Chess-v0", render_mode="human", fen=fen, max_steps=200)
    obs, info = env.reset()
    board = env.unwrapped.board
    renderer = env.unwrapped._ensure_renderer()
    pygame.display.set_caption(
        f"Chess-v0 — agent: {'white' if agent_color else 'black'}, "
        f"you: {'white' if human_color else 'black'}"
    )

    running = True
    selected = None
    result_msg = None

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif (
                event.type == pygame.MOUSEBUTTONDOWN
                and result_msg is None
                and board.turn == human_color
            ):
                square = pixel_to_square(*event.pos)
                if selected is not None and square in renderer.legal_targets:
                    moves = [
                        m
                        for m in board.legal_moves
                        if m.from_square == selected and m.to_square == square
                    ]
                    move = next(
                        (m for m in moves if m.promotion in (None, chess.QUEEN)),
                        moves[0],
                    )
                    action = chess_gym.env.move_to_action(move)
                    obs, reward, terminated, truncated, info = env.step(action)
                    selected = None
                    renderer.selected = None
                    renderer.legal_targets = set()
                elif (
                    board.piece_at(square) is not None
                    and board.piece_at(square).color == human_color
                ):
                    selected = square
                    renderer.selected = square
                    renderer.legal_targets = {
                        m.to_square
                        for m in board.legal_moves
                        if m.from_square == square
                    }
                else:
                    selected = None
                    renderer.selected = None
                    renderer.legal_targets = set()

        board = env.unwrapped.board
        if result_msg is None and board.is_game_over():
            outcome = board.outcome()
            result_msg = outcome.termination.name if outcome else "MAX_STEPS"
        elif result_msg is None and board.turn == agent_color:
            time.sleep(0.2)
            action = greedy_action(q, board.fen(), obs["action_mask"])
            if action is not None:
                obs, reward, terminated, truncated, info = env.step(action)
                board = env.unwrapped.board
                if terminated or truncated:
                    outcome = board.outcome()
                    result_msg = outcome.termination.name if outcome else "MAX_STEPS"

        env.render()
        if result_msg is not None:
            pygame.display.set_caption(f"Chess-v0 — {result_msg}")

    env.close()


if __name__ == "__main__":
    main()
