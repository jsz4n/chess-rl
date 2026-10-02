import time

import chess
import gymnasium as gym
import numpy as np
import pygame

import chess_gym
from chess_gym.render import pixel_to_square

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def greedy_action(env) -> int:
    board = env.unwrapped.board
    obs = env.unwrapped._get_obs()
    legal = np.flatnonzero(obs["action_mask"])
    best, best_score = None, -np.inf
    for action in legal:
        move = chess_gym.env.action_to_move(int(action), board)
        captured = board.piece_type_at(move.to_square)
        score = PIECE_VALUES.get(captured, 0) + np.random.random()
        if score > best_score:
            best, best_score = int(action), score
    return best


def main() -> None:
    env = gym.make("Chess-v0", render_mode="human", max_steps=200)
    obs, info = env.reset()
    board = env.unwrapped.board
    renderer = env.unwrapped._ensure_renderer()

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
                and board.turn == chess.WHITE
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
                    and board.piece_at(square).color == chess.WHITE
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
        elif result_msg is None and board.turn == chess.BLACK:
            time.sleep(0.2)
            action = greedy_action(env)
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
