import chess
import numpy as np
import pytest

import chess_gym
from chess_gym.env import ChessEnv
from play import greedy_action


@pytest.fixture
def env() -> ChessEnv:
    env = ChessEnv()
    env.reset()
    return env


def legal_actions(env: ChessEnv) -> np.ndarray:
    return np.flatnonzero(env.unwrapped._get_obs()["action_mask"])


class TestGreedyAction:
    def test_returns_legal_action(self, env: ChessEnv):
        action = greedy_action(env)
        assert action in set(legal_actions(env).tolist())

    def test_prefers_higher_value_capture(self):
        # Knight d4 can take pawn c6 (1) or queen e6 (9); must pick queen.
        env = ChessEnv(fen="k7/8/2p1q3/8/3N4/8/8/K7 w - - 0 1")
        env.reset()
        action = greedy_action(env)
        move = chess_gym.env.action_to_move(action, env.unwrapped.board)
        assert move.to_square == chess.E6

    def test_prefers_capture_over_quiet_move(self):
        env = ChessEnv(fen="k7/8/2p5/8/3N4/8/8/K7 w - - 0 1")
        env.reset()
        action = greedy_action(env)
        move = chess_gym.env.action_to_move(action, env.unwrapped.board)
        assert move.to_square == chess.C6

    def test_returns_none_when_no_legal_moves(self):
        # Regression: play.py used to crash with action=None here.
        env = ChessEnv(fen="7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
        env.reset()
        assert env.unwrapped.board.is_checkmate()
        assert greedy_action(env) is None
