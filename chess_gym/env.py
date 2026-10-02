from __future__ import annotations

from typing import Any, ClassVar

import chess
import gymnasium as gym
import numpy as np
from gymnasium import spaces

PIECE_PLANES = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

PROMOTIONS = (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)
PROMO_LAYER = {None: 0, **{piece: layer for layer, piece in enumerate(PROMOTIONS, 1)}}

NUM_SQUARES = 64
NUM_LAYERS = 1 + len(PROMOTIONS)
NUM_ACTIONS = NUM_SQUARES * NUM_SQUARES * NUM_LAYERS


def move_to_action(move: chess.Move) -> int:
    return PROMO_LAYER[move.promotion] * NUM_SQUARES * NUM_SQUARES + (
        move.from_square * NUM_SQUARES + move.to_square
    )


def action_to_move(action: int, board: chess.Board) -> chess.Move:
    layer, rest = divmod(action, NUM_SQUARES * NUM_SQUARES)
    from_square, to_square = divmod(rest, NUM_SQUARES)
    promotion = PROMOTIONS[layer - 1] if layer else None
    return chess.Move(from_square, to_square, promotion=promotion)


def action_to_uci(action: int, board: chess.Board) -> str:
    return action_to_move(action, board).uci()


def uci_to_action(uci: str) -> int:
    return move_to_action(chess.Move.from_uci(uci))


def legal_action_mask(board: chess.Board) -> np.ndarray:
    mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
    for move in board.legal_moves:
        mask[move_to_action(move)] = 1
    return mask


class LegalActionSpace(spaces.Discrete):
    """Discrete action space whose sample() only returns legal moves."""

    def __init__(self, env: ChessEnv) -> None:
        super().__init__(NUM_ACTIONS)
        self.env = env

    def sample(self, mask=None, probability=None) -> int:
        if mask is None and probability is None:
            mask = legal_action_mask(self.env.board)
        return int(super().sample(mask=mask, probability=probability))


class ChessEnv(gym.Env):
    metadata: ClassVar[dict] = {
        "render_modes": ["human", "unicode", "fen", "svg"],
        "render_fps": 60,
    }

    def __init__(
        self,
        max_steps: int = 500,
        render_mode: str | None = None,
        fen: str | None = None,
    ) -> None:
        super().__init__()
        if render_mode not in self.metadata["render_modes"] + [None]:
            raise ValueError(
                f"render_mode must be one of {self.metadata['render_modes']}, got {render_mode!r}"
            )
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.initial_fen = fen or chess.STARTING_FEN
        self.renderer = None

        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(low=0, high=1, shape=(8, 8, 12), dtype=np.uint8),
                "turn": spaces.Discrete(2),
                "action_mask": spaces.Box(
                    low=0, high=1, shape=(NUM_ACTIONS,), dtype=np.int8
                ),
            }
        )
        self.action_space = LegalActionSpace(self)

        self.board: chess.Board
        self.last_move: chess.Move | None
        self.step_count: int

    def _get_obs(self) -> dict[str, Any]:
        planes = np.zeros((8, 8, 12), dtype=np.uint8)
        white_to_move = self.board.turn == chess.WHITE
        for square, piece in self.board.piece_map().items():
            rank, file = chess.square_rank(square), chess.square_file(square)
            if not white_to_move:
                rank, file = 7 - rank, 7 - file
            plane = PIECE_PLANES[piece.piece_type] + (
                0 if piece.color == self.board.turn else 6
            )
            planes[rank, file, plane] = 1

        return {
            "board": planes,
            "turn": int(self.board.turn),
            "action_mask": legal_action_mask(self.board),
        }

    def _get_info(self) -> dict[str, Any]:
        return {"fen": self.board.fen()}

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        self.board = chess.Board(
            options.get("fen", self.initial_fen) if options else self.initial_fen
        )
        self.last_move = None
        self.step_count = 0
        self._clear_renderer_state()
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), self._get_info()

    def step(
        self, action: int
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        move = action_to_move(action, self.board)
        reward = 0.0
        terminated = False

        if move not in self.board.legal_moves:
            reward, terminated = -1.0, True
        else:
            self.board.push(move)
            self.last_move = move
            self.step_count += 1
            outcome = self.board.outcome()
            if outcome is not None:
                terminated = True
                mover = not self.board.turn  # side that just moved
                if outcome.winner is not None:
                    reward = 1.0 if outcome.winner == mover else -1.0

        truncated = self.step_count >= self.max_steps and not terminated
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _ensure_renderer(self):
        if self.renderer is None:
            from chess_gym.render import PygameRenderer

            self.renderer = PygameRenderer(fps=self.metadata["render_fps"])
        return self.renderer

    def _clear_renderer_state(self) -> None:
        if self.renderer is not None:
            self.renderer.selected = None
            self.renderer.legal_targets = set()

    def render(self) -> str | None:
        if self.render_mode == "human":
            self._ensure_renderer().draw(self.board, self.last_move)
            return None
        if self.render_mode == "fen":
            output = self.board.fen()
        elif self.render_mode == "svg":
            output = self.board.svg()
        else:
            output = self.board.unicode(empty_square=".")
            print(output)
        return output

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
