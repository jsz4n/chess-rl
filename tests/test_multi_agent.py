import sys

import chess
import numpy as np
import pytest

from chess_gym.env import ChessEnv, legal_action_mask, move_to_action
from multi_agent import ChessAgent, main, self_play_episode, warm_start
from q_learning import (
    DEFAULT_FEN_BLACK,
    DEFAULT_FEN_WHITE,
    load_checkpoint,
    save_checkpoint,
)


@pytest.fixture
def env() -> ChessEnv:
    env = ChessEnv(fen=DEFAULT_FEN_WHITE, max_steps=60)
    env.reset()
    return env


class TestChessAgent:
    def test_instances_do_not_share_table(self):
        white, black = ChessAgent(chess.WHITE), ChessAgent(chess.BLACK)
        white.q["fen"] = {1: 2.0}
        assert white.q is not black.q
        assert black.q == {}

    def test_act_returns_legal_action(self, env: ChessEnv):
        agent = ChessAgent(chess.WHITE)
        mask = legal_action_mask(env.board)
        action = agent.act(env.board, mask)
        assert action in set(np.flatnonzero(mask).tolist())

    def test_act_returns_none_when_no_legal_moves(self):
        env = ChessEnv(fen="7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
        env.reset()
        assert ChessAgent(chess.BLACK).act(env.board, np.zeros(20480, np.int8)) is None

    def test_learn_terminal_reward(self):
        agent = ChessAgent(chess.WHITE)
        agent.learn("fen", 5, 1.0, done=True)
        assert agent.q["fen"][5] == pytest.approx(0.1)

    def test_learn_bootstraps_from_next_state(self, env: ChessEnv):
        agent = ChessAgent(chess.WHITE)
        next_fen = env.board.fen()
        next_mask = legal_action_mask(env.board)
        agent.q[next_fen] = {int(np.flatnonzero(next_mask)[0]): 4.0}
        agent.learn("fen", 5, 0.0, env.board, next_mask, done=False)
        assert agent.q["fen"][5] == pytest.approx(0.1 * (0.99 * 4.0))


class TestSelfPlayEpisode:
    MATELESS_FEN = "k7/8/8/8/8/8/8/K6R w - - 0 1"

    def test_terminates_and_populates_both_tables(self):
        env = ChessEnv(fen=self.MATELESS_FEN, max_steps=10)
        white, black = ChessAgent(chess.WHITE), ChessAgent(chess.BLACK)
        for _ in range(3):
            assert self_play_episode(env, white, black) in {"win", "loss", "draw"}
        assert len(white.q) > 1
        assert len(black.q) > 1
        env.close()

    def test_both_agents_learn_non_terminal_moves(self):
        # Regression: black's table only grew on terminal transitions, so a
        # truncating episode taught black exactly one state.
        env = ChessEnv(fen=self.MATELESS_FEN, max_steps=4)
        white, black = ChessAgent(chess.WHITE), ChessAgent(chess.BLACK)
        self_play_episode(env, white, black)
        assert len(black.q) >= 2
        env.close()

    def test_own_move_mate_is_handled(self):
        env = ChessEnv(fen=DEFAULT_FEN_WHITE, max_steps=60)
        white = ChessAgent(chess.WHITE, epsilon=0.0)
        black = ChessAgent(chess.BLACK, epsilon=0.0)
        mate = move_to_action(chess.Move(chess.E1, chess.E8))
        white.q[DEFAULT_FEN_WHITE] = {mate: 10.0}
        assert self_play_episode(env, white, black) == "win"
        assert black.q == {}
        env.close()

    def test_truncation_on_own_move_is_handled(self):
        env = ChessEnv(fen=self.MATELESS_FEN, max_steps=1)
        white, black = ChessAgent(chess.WHITE), ChessAgent(chess.BLACK)
        assert self_play_episode(env, white, black) == "draw"
        assert len(white.q) == 1
        assert black.q == {}
        env.close()


class TestCheckpoints:
    def test_roundtrip(self, tmp_path):
        agent = ChessAgent(chess.BLACK)
        agent.q["fen"] = {3: 0.5}
        save_checkpoint(
            tmp_path / "sp_black.pkl", agent.q, chess.BLACK, DEFAULT_FEN_WHITE
        )
        ckpt = load_checkpoint(tmp_path / "sp_black.pkl")
        assert ckpt["q"] == agent.q
        assert ckpt["color"] == chess.BLACK


class TestWarmStart:
    def test_loads_both_agents_and_fen(self, tmp_path):
        white_q, black_q = {"wf": {1: 1.0}}, {"bf": {2: 2.0}}
        save_checkpoint(tmp_path / "w.pkl", white_q, chess.WHITE, DEFAULT_FEN_WHITE)
        save_checkpoint(tmp_path / "b.pkl", black_q, chess.BLACK, DEFAULT_FEN_WHITE)

        w_q, b_q, fen, fens = warm_start(
            str(tmp_path / "w.pkl"), str(tmp_path / "b.pkl")
        )
        assert w_q == white_q
        assert b_q == black_q
        assert fen == DEFAULT_FEN_WHITE
        assert fens is None

    def test_no_checkpoints_returns_fresh(self):
        assert warm_start() == (None, None, None, None)

    def test_color_mismatch_raises(self, tmp_path):
        save_checkpoint(tmp_path / "w.pkl", {}, chess.BLACK, DEFAULT_FEN_WHITE)
        with pytest.raises(ValueError, match="black agent"):
            warm_start(str(tmp_path / "w.pkl"))

    def test_resumes_fens_list(self, tmp_path):
        fens = [DEFAULT_FEN_WHITE, "k7/8/8/8/8/8/8/K6R w - - 0 1"]
        save_checkpoint(tmp_path / "w.pkl", {}, chess.WHITE, fens[0], fens=fens)
        _, _, fen, resumed_fens = warm_start(str(tmp_path / "w.pkl"))
        assert fen == fens[0]
        assert resumed_fens == fens

    def test_resumed_agents_keep_learning(self, tmp_path):
        save_checkpoint(tmp_path / "w.pkl", {}, chess.WHITE, DEFAULT_FEN_WHITE)
        save_checkpoint(tmp_path / "b.pkl", {}, chess.BLACK, DEFAULT_FEN_WHITE)
        w_q, b_q, fen, _fens = warm_start(
            str(tmp_path / "w.pkl"), str(tmp_path / "b.pkl")
        )
        white = ChessAgent(chess.WHITE, q=w_q)
        black = ChessAgent(chess.BLACK, q=b_q)
        env = ChessEnv(fen=fen, max_steps=10)
        self_play_episode(env, white, black)
        env.close()
        assert len(white.q) > 0


class TestExhibitionMode:
    def test_render_plays_one_episode_without_training(self, capsys, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "multi_agent.py",
                "--render",
                "unicode",
                "--episodes",
                "5",
                "--max-steps",
                "20",
            ],
        )
        main()
        out = capsys.readouterr().out
        assert "exhibition:" in out
        assert "checkpoint saved" not in out
        assert "episode 50" not in out

    def test_fen_start_token_overrides_checkpoint(self, capsys, monkeypatch, tmp_path):
        save_checkpoint(
            tmp_path / "w.pkl", {"x": {1: 1.0}}, chess.WHITE, DEFAULT_FEN_WHITE
        )
        save_checkpoint(
            tmp_path / "b.pkl", {"y": {2: 2.0}}, chess.BLACK, DEFAULT_FEN_WHITE
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "multi_agent.py",
                "--render",
                "unicode",
                "--fen",
                "start",
                "--max-steps",
                "4",
                "--white-checkpoint",
                str(tmp_path / "w.pkl"),
                "--black-checkpoint",
                str(tmp_path / "b.pkl"),
            ],
        )
        main()
        out = capsys.readouterr().out
        assert f"start position: {chess.STARTING_FEN}" in out


class TestFenFileRandomStarts:
    def test_self_play_episode_from_custom_fen(self):
        env = ChessEnv(fen=DEFAULT_FEN_WHITE, max_steps=10)
        white, black = ChessAgent(chess.WHITE), ChessAgent(chess.BLACK)
        self_play_episode(env, white, black, fen=DEFAULT_FEN_BLACK)
        env.close()
        assert DEFAULT_FEN_BLACK in black.q

    def test_main_trains_from_fen_file(self, capsys, monkeypatch, tmp_path):
        fen_file = tmp_path / "fens.txt"
        mateless = "k7/8/8/8/8/8/8/K6R w - - 0 1"
        fen_file.write_text(f"# comment\n{DEFAULT_FEN_WHITE}\n\n{mateless}\n")
        monkeypatch.setattr("multi_agent.CHECKPOINT_DIR", tmp_path)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "multi_agent.py",
                "--fen-file",
                str(fen_file),
                "--episodes",
                "2",
                "--max-steps",
                "4",
            ],
        )
        main()
        out = capsys.readouterr().out
        assert "start positions: 2 (random per episode)" in out
        ckpt = load_checkpoint(tmp_path / "sp_white.pkl")
        assert len(ckpt["fens"]) == 2

    def test_fen_and_fen_file_are_exclusive(self, capsys, monkeypatch, tmp_path):
        fen_file = tmp_path / "fens.txt"
        fen_file.write_text(DEFAULT_FEN_WHITE + "\n")
        monkeypatch.setattr(
            sys,
            "argv",
            ["multi_agent.py", "--fen", "start", "--fen-file", str(fen_file)],
        )
        with pytest.raises(SystemExit):
            main()
