import chess
import gymnasium as gym
import numpy as np
import pytest

import chess_gym  # noqa: F401  (registers Chess-v0)
from chess_gym.env import (
    NUM_ACTIONS,
    ChessEnv,
    action_to_move,
    action_to_uci,
    move_to_action,
    uci_to_action,
)


@pytest.fixture
def env() -> ChessEnv:
    env = ChessEnv(max_steps=100)
    env.reset()
    return env


def env_with_fen(fen: str, **kwargs) -> ChessEnv:
    env = ChessEnv(fen=fen, **kwargs)
    env.reset()
    return env


class TestActionConversions:
    def test_move_to_action_roundtrip(self):
        move = chess.Move(chess.E2, chess.E4)
        action = move_to_action(move)
        assert action == chess.E2 * 64 + chess.E4
        assert action_to_move(action, chess.Board()) == move

    def test_layer0_matches_legacy_encoding(self):
        for move in chess.Board().legal_moves:
            assert move_to_action(move) == move.from_square * 64 + move.to_square

    def test_promotion_roundtrip_all_pieces(self):
        for piece in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
            move = chess.Move(chess.A7, chess.A8, promotion=piece)
            action = move_to_action(move)
            assert action_to_move(action, chess.Board()) == move

    def test_underpromotion_distinct_actions(self):
        promos = {
            move_to_action(chess.Move(chess.A7, chess.A8, promotion=p))
            for p in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)
        }
        assert len(promos) == 4

    def test_action_to_move_no_promotion(self):
        board = chess.Board()
        move = action_to_move(chess.E2 * 64 + chess.E3, board)
        assert move.promotion is None


class TestUciHelpers:
    def test_action_to_uci(self):
        assert action_to_uci(uci_to_action("e2e4"), chess.Board()) == "e2e4"

    def test_uci_roundtrip_with_promotion(self):
        assert action_to_uci(uci_to_action("a7a8n"), chess.Board()) == "a7a8n"

    def test_uci_action_matches_move_action(self):
        move = chess.Move.from_uci("a7a8q")
        assert uci_to_action("a7a8q") == move_to_action(move)


class TestReset:
    def test_reset_starting_position(self, env: ChessEnv):
        assert env.board.fen() == chess.STARTING_FEN
        assert env.last_move is None
        assert env.step_count == 0

    def test_reset_with_custom_fen(self):
        env = env_with_fen(chess.Board("8/P7/8/8/8/8/7k/K7 w - - 0 1").fen())
        assert env.board.piece_at(chess.A7).piece_type == chess.PAWN

    def test_reset_returns_obs_and_info(self, env: ChessEnv):
        obs, info = ChessEnv().reset()
        assert set(obs) == {"board", "turn", "action_mask"}
        assert "fen" in info


class TestObservation:
    def test_obs_matches_spaces(self, env: ChessEnv):
        obs = env._get_obs()
        assert env.observation_space.contains(obs)

    def test_starting_board_planes(self, env: ChessEnv):
        planes = env._get_obs()["board"]
        # White pawns on rank 2 in plane 0
        assert planes[1, :, 0].sum() == 8
        # White rooks in plane 3
        assert planes[0, 0, 3] == 1 and planes[0, 7, 3] == 1
        # Black back rank on rank 8, planes 6..11 (white to move: no flip)
        assert planes[7, :, 6:].sum() == 8
        # Black pawns on rank 7, plane 6
        assert planes[6, :, 6].sum() == 8

    def test_turn_field(self):
        env = ChessEnv()
        env.reset()
        assert env._get_obs()["turn"] == int(chess.WHITE)
        env = env_with_fen("8/8/8/8/8/8/7k/K7 b - - 0 1")
        assert env._get_obs()["turn"] == int(chess.BLACK)

    def test_action_mask_counts_starting_moves(self, env: ChessEnv):
        assert env._get_obs()["action_mask"].sum() == 20

    def test_action_mask_covers_underpromotions(self):
        env = env_with_fen("8/P7/8/8/8/8/7k/K7 w - - 0 1")
        mask = env._get_obs()["action_mask"]
        for piece in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
            assert (
                mask[move_to_action(chess.Move(chess.A7, chess.A8, promotion=piece))]
                == 1
            )

    def test_action_mask_matches_legal_moves(self, env: ChessEnv):
        mask = env._get_obs()["action_mask"]
        for move in env.board.legal_moves:
            assert mask[move_to_action(move)] == 1


class TestStep:
    def test_legal_move_updates_state(self, env: ChessEnv):
        obs, reward, terminated, truncated, info = env.step(chess.E2 * 64 + chess.E4)
        assert reward == 0.0
        assert not terminated and not truncated
        assert (
            env.board.fen()
            == chess.Board(
                "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
            ).fen()
        )
        assert env.last_move == chess.Move(chess.E2, chess.E4)
        assert env.step_count == 1
        assert info["fen"] == env.board.fen()

    def test_illegal_move_terminates(self, env: ChessEnv):
        obs, reward, terminated, truncated, info = env.step(chess.E2 * 64 + chess.E5)
        assert reward == -1.0
        assert terminated
        assert not truncated
        assert env.board.fen() == chess.STARTING_FEN

    def test_checkmate_rewards_mover(self):
        # Fool's mate: black delivers mate on h4 -> +1 from mover's view.
        env = env_with_fen(
            "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b kq g3 0 2"
        )
        _obs, reward, terminated, _truncated, _info = env.step(chess.D8 * 64 + chess.H4)
        assert reward == 1.0
        assert terminated
        outcome = env.board.outcome()
        assert outcome is not None
        assert outcome.termination.name == "CHECKMATE"
        assert outcome.winner == chess.BLACK

    def test_checkmate_white_mover_gets_reward(self):
        # Back-rank mate: white plays Re8# -> +1.
        env = env_with_fen("6k1/5ppp/8/8/8/8/8/4R2K w - - 0 1")
        _obs, reward, terminated, _truncated, _info = env.step(chess.E1 * 64 + chess.E8)
        assert reward == 1.0
        assert terminated

    def test_stalemate_zero_reward(self):
        # White plays Qb6, stalemate -> 0, terminated.
        env = env_with_fen("k7/8/8/1Q6/8/8/8/K7 w - - 0 1")
        _obs, reward, terminated, _truncated, _info = env.step(chess.B5 * 64 + chess.B6)
        assert reward == 0.0
        assert terminated
        assert env.board.outcome().termination.name == "STALEMATE"

    def test_max_steps_truncation(self):
        env = ChessEnv(max_steps=1)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(chess.E2 * 64 + chess.E4)
        assert truncated and not terminated

    def test_obs_after_step_is_black_to_move(self, env: ChessEnv):
        obs, *_ = env.step(chess.E2 * 64 + chess.E4)
        assert obs["turn"] == int(chess.BLACK)


class TestActionSpace:
    def test_sample_returns_legal_move(self, env: ChessEnv):
        for _ in range(50):
            action = env.action_space.sample()
            assert action_to_move(action, env.board) in env.board.legal_moves

    def test_sample_returns_legal_move_midgame(self):
        env = env_with_fen("8/P7/8/8/8/8/7k/K7 w - - 0 1")
        env.step(chess.A7 * 64 + chess.A8)
        for _ in range(50):
            action = env.action_space.sample()
            assert action_to_move(action, env.board) in env.board.legal_moves

    def test_sample_can_return_promotions(self):
        env = env_with_fen("8/P7/8/8/8/8/7k/K7 w - - 0 1")
        samples = {env.action_space.sample() for _ in range(500)}
        assert any(
            samples
            & {
                move_to_action(chess.Move(chess.A7, chess.A8, promotion=p))
                for p in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)
            }
        )

    def test_sample_respects_explicit_mask(self, env: ChessEnv):
        mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
        mask[chess.E2 * 64 + chess.E4] = 1
        for _ in range(10):
            assert env.action_space.sample(mask=mask) == chess.E2 * 64 + chess.E4

    def test_sample_returns_plain_int(self, env: ChessEnv):
        assert type(env.action_space.sample()) is int

    def test_seeded_sample_reproducible(self):
        env_a, env_b = ChessEnv(), ChessEnv()
        env_a.reset(), env_b.reset()
        env_a.action_space.seed(42)
        env_b.action_space.seed(42)
        assert [env_a.action_space.sample() for _ in range(10)] == [
            env_b.action_space.sample() for _ in range(10)
        ]

    def test_sample_tracks_position_after_each_step(self, env: ChessEnv):
        for _ in range(10):
            action = env.action_space.sample()
            obs, _reward, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                break

    def test_sample_from_terminal_position_returns_valid_index(self):
        # Black to move is stalemated: no legal moves, mask all zeros.
        env = env_with_fen("k7/8/1Q6/8/8/8/8/K7 b - - 0 1")
        assert env.action_space.sample() in range(NUM_ACTIONS)

    def test_full_random_game_all_moves_legal(self):
        env = ChessEnv(max_steps=500)
        env.reset()
        for _ in range(1000):
            action = env.action_space.sample()
            move = action_to_move(action, env.board)
            assert move in env.board.legal_moves
            _obs, _reward, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                break
        env.close()


class TestRegistration:
    def test_gym_make(self):
        env = gym.make("Chess-v0")
        obs, info = env.reset()
        assert obs["action_mask"].shape == (NUM_ACTIONS,)
        assert env.action_space.n == NUM_ACTIONS
        env.close()

    def test_random_rollout(self):
        env = gym.make("Chess-v0", max_steps=20)
        env.reset()
        for _ in range(20):
            mask = env.unwrapped._get_obs()["action_mask"]
            legal = np.flatnonzero(mask)
            if legal.size == 0:
                break
            obs, reward, terminated, truncated, info = env.step(
                int(np.random.choice(legal))
            )
            if terminated or truncated:
                break
        env.close()
