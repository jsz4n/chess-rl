import chess
import numpy as np
import pytest

from arena import play_game
from chess_gym.env import ChessEnv
from federated import train
from q_learning import (
    DEFAULT_FEN_BLACK,
    DEFAULT_FEN_WHITE,
    greedy_action,
    load_checkpoint,
    load_fens,
    play_episode,
    resolve_fen,
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


class TestLoadFens:
    def test_parses_and_skips_comments(self, tmp_path):
        fen_file = tmp_path / "fens.txt"
        fen_file.write_text(f"# openings\n{DEFAULT_FEN_WHITE}\n\n{DEFAULT_FEN_BLACK}\n")
        assert load_fens(fen_file) == [DEFAULT_FEN_WHITE, DEFAULT_FEN_BLACK]

    def test_invalid_fen_raises_with_line(self, tmp_path):
        fen_file = tmp_path / "fens.txt"
        fen_file.write_text(f"{DEFAULT_FEN_WHITE}\nnot a fen\n")
        with pytest.raises(ValueError, match="2:"):
            load_fens(fen_file)

    def test_empty_file_raises(self, tmp_path):
        fen_file = tmp_path / "fens.txt"
        fen_file.write_text("# nothing here\n\n")
        with pytest.raises(ValueError, match="no FENs"):
            load_fens(fen_file)

    def test_illegal_position_raises(self, tmp_path):
        fen_file = tmp_path / "fens.txt"
        fen_file.write_text("8/8/8/3k4/3K4/8/8/8 w - - 0 1\n")
        with pytest.raises(ValueError, match="illegal position \\(OPPOSITE_CHECK\\)"):
            load_fens(fen_file)


class TestResolveFen:
    def test_none_passthrough(self):
        assert resolve_fen(None) is None

    def test_start_token(self):
        assert resolve_fen("start") == chess.STARTING_FEN

    def test_fen_passthrough(self):
        assert resolve_fen(DEFAULT_FEN_WHITE) == DEFAULT_FEN_WHITE


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

    def test_roundtrip_with_fens(self, tmp_path):
        fens = [DEFAULT_FEN_WHITE, DEFAULT_FEN_BLACK]
        save_checkpoint(tmp_path / "c.pkl", {}, chess.WHITE, fens[0], fens=fens)
        ckpt = load_checkpoint(tmp_path / "c.pkl")
        assert ckpt["fen"] == fens[0]
        assert ckpt["fens"] == fens

    def test_play_episode_from_custom_fen(self):
        env = ChessEnv(fen=DEFAULT_FEN_WHITE, max_steps=60)
        q = {}
        play_episode(q, env, chess.BLACK, fen=DEFAULT_FEN_BLACK)
        assert DEFAULT_FEN_BLACK in q
        env.close()

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
        train(q, chess.WHITE, DEFAULT_FEN_WHITE, 150, seed=2)
        outcome, steps = play_game(q, {}, DEFAULT_FEN_WHITE, max_steps=30)
        assert outcome is not None and outcome.winner == chess.WHITE
        assert steps <= 5
