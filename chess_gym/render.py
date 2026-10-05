from __future__ import annotations

import contextlib
import ctypes
import sys

import chess
import pygame

SQUARE_SIZE = 80
BOARD_PX = SQUARE_SIZE * 8

LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
LAST_MOVE_COLOR = (205, 210, 130)
CHECK_COLOR = (235, 97, 80)
SELECTED_COLOR = (130, 202, 168)
DOT_COLOR = (106, 111, 69)

GLYPHS = {
    (chess.WHITE, chess.PAWN): "♙",
    (chess.WHITE, chess.KNIGHT): "♘",
    (chess.WHITE, chess.BISHOP): "♗",
    (chess.WHITE, chess.ROOK): "♖",
    (chess.WHITE, chess.QUEEN): "♕",
    (chess.WHITE, chess.KING): "♔",
    (chess.BLACK, chess.PAWN): "♟",
    (chess.BLACK, chess.KNIGHT): "♞",
    (chess.BLACK, chess.BISHOP): "♝",
    (chess.BLACK, chess.ROOK): "♜",
    (chess.BLACK, chess.QUEEN): "♛",
    (chess.BLACK, chess.KING): "♚",
}

PIECE_FONT_CANDIDATES = (
    "segoeuisymbol",
    "dejavusans",
    "arialunicodems",
    "apple symbols",
    "arial",
    "noto sans symbols",
    "freesans",
)


def _piece_font(size: int) -> pygame.font.Font:
    if not pygame.font.get_init():
        pygame.font.init()
    for name in PIECE_FONT_CANDIDATES:
        path = pygame.font.match_font(name)
        if path:
            return pygame.font.Font(path, size)
    return pygame.font.Font(None, size)


def _enable_windows_dpi_awareness() -> None:
    if sys.platform != "win32":
        return
    with contextlib.suppress(Exception):
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    with contextlib.suppress(Exception):
        ctypes.windll.user32.SetProcessDPIAware()


def square_to_pixel(square: chess.Square) -> tuple[int, int]:
    file = chess.square_file(square)
    rank = 7 - chess.square_rank(square)
    return file * SQUARE_SIZE, rank * SQUARE_SIZE


def pixel_to_square(x: int, y: int) -> chess.Square:
    file = x // SQUARE_SIZE
    rank = 7 - (y // SQUARE_SIZE)
    file = min(max(file, 0), 7)
    rank = min(max(rank, 0), 7)
    return chess.square(file, rank)


class PygameRenderer:
    def __init__(self, caption: str = "Chess-v0", fps: int = 60) -> None:
        _enable_windows_dpi_awareness()
        pygame.init()
        pygame.display.set_caption(caption)
        try:
            self.screen = pygame.display.set_mode((BOARD_PX, BOARD_PX), pygame.SCALED)
        except pygame.error:
            self.screen = pygame.display.set_mode((BOARD_PX, BOARD_PX))
        self.font = _piece_font(int(SQUARE_SIZE * 0.78))
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.selected: chess.Square | None = None
        self.legal_targets: set[chess.Square] = set()

    def draw(self, board: chess.Board, last_move: chess.Move | None = None) -> None:
        pygame.event.pump()
        check_square = board.king(board.turn) if board.is_check() else None
        for square in chess.SQUARES:
            rect = pygame.Rect(*square_to_pixel(square), SQUARE_SIZE, SQUARE_SIZE)
            color = (
                DARK_SQUARE
                if (chess.square_rank(square) + chess.square_file(square)) % 2 == 0
                else LIGHT_SQUARE
            )
            if last_move and square in (last_move.from_square, last_move.to_square):
                color = LAST_MOVE_COLOR
            if square == check_square:
                color = CHECK_COLOR
            pygame.draw.rect(self.screen, color, rect)

            piece = board.piece_at(square)
            if piece is not None:
                glyph = GLYPHS[(piece.color, piece.piece_type)]
                text = self.font.render(
                    glyph,
                    True,
                    (24, 24, 24) if piece.color == chess.BLACK else (250, 250, 250),
                )
                self.screen.blit(text, text.get_rect(center=rect.center))

            if square == self.selected:
                pygame.draw.rect(self.screen, SELECTED_COLOR, rect, width=6)
            if square in self.legal_targets:
                center = rect.center
                pygame.draw.circle(self.screen, DOT_COLOR, center, SQUARE_SIZE // 7)

        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self) -> None:
        self.selected = None
        self.legal_targets = set()
        pygame.display.quit()
        pygame.quit()
