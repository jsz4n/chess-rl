import chess
import numpy as np
import pytest

from arena import play_game
from chess_gym.env import ChessEnv
from q_learning import (
    DEFAULT_FEN_BLACK,
    DEFAULT_FEN_WHITE,
    greedy_action,
    load_checkpoint,
    play_episode,
    save_checkpoint,
)


@pytest.fixture
def env() -> ChessEnv:
    env = ChessEnv(fen=DEFAULT_FEN_WHITE)
    env.reset()
    return env


def legal_actions(env: ChessEnv) -> np.ndarray:
    return np.flatnonzero(mask(env))


def mask(env: ChessEnv) -> np.ndarray:
    return env.unwrapped._get_obs()["action_mask"]


class TestGreedyAction:
    def test_returns_legal_action_on_empty_table(self, env: ChessEnv):
        fen = env.unwrapped.board.fen()
        action = greedy_action({}, fen, mask(env))
        assert action in set(legal_actions(env).tolist())

    def test_prefers_trained_move(self, env: ChessEnv):
        fen = env.unwrapped.board.fen()
        best = int(legal_actions(env)[0])
        q = {fen: {best: 5.0}}
        for _ in range(10):
            assert greedy_action(q, fen, mask(env)) == best

    def test_returns_none_when_no_legal_moves(self):
        env = ChessEnv(fen="7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
        env.reset()
        empty_mask = np.zeros(20480, dtype=np.int8)
        assert greedy_action({}, env.unwrapped.board.fen(), empty_mask) is None


class TestTraining:
    def test_play_episode_white_populates_q(self, env: ChessEnv):
        q = {}
        result = play_episode(q, env, chess.WHITE)
        assert result in {"win", "loss", "draw"}
        assert len(q) > 0

    def test_play_episode_black_populates_q(self):
        env = ChessEnv(fen=DEFAULT_FEN_BLACK, max_steps=60)
        q = {}
        result = play_episode(q, env, chess.BLACK)
        assert result in {"win", "loss", "draw"}
        assert len(q) > 0
        env.close()


class TestCheckpoint:
    def test_roundtrip(self, tmp_path, env: ChessEnv):
        q = {}
        play_episode(q, env, chess.WHITE)
        path = tmp_path / "q_white.pkl"
        save_checkpoint(path, q, chess.WHITE, DEFAULT_FEN_WHITE)

        assert path.exists()
        ckpt = load_checkpoint(path)
        assert ckpt["color"] == chess.WHITE
        assert ckpt["fen"] == DEFAULT_FEN_WHITE
        assert ckpt["q"] == q

    def test_greedy_uses_loaded_table(self, tmp_path, env: ChessEnv):
        fen = env.unwrapped.board.fen()
        best = int(legal_actions(env)[0])
        save_checkpoint(tmp_path / "ckpt.pkl", {fen: {best: 5.0}}, chess.WHITE, fen)
        ckpt = load_checkpoint(tmp_path / "ckpt.pkl")
        assert greedy_action(ckpt["q"], fen, mask(env)) == best


class TestArena:
    def test_game_between_untrained_agents_terminates(self):
        outcome, steps = play_game({}, {}, DEFAULT_FEN_WHITE, max_steps=30)
        assert 1 <= steps <= 30
        if outcome is not None:
            assert outcome.winner in (None, chess.WHITE, chess.BLACK)

    def test_trained_white_mates_quickly(self):
        q = {}
        env = ChessEnv(fen=DEFAULT_FEN_WHITE, max_steps=60)
        for _ in range(150):
            play_episode(q, env, chess.WHITE)
        env.close()
        outcome, steps = play_game(q, {}, DEFAULT_FEN_WHITE, max_steps=30)
        assert outcome is not None and outcome.winner == chess.WHITE
        assert steps <= 5
