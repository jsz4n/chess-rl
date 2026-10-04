import chess
import pytest

from chess_gym.render import BOARD_PX, SQUARE_SIZE, pixel_to_square, square_to_pixel

pytest.importorskip("pygame")


class TestSquareToPixel:
    def test_a8_top_left(self):
        assert square_to_pixel(chess.A8) == (0, 0)

    def test_h1_bottom_right(self):
        assert square_to_pixel(chess.H1) == (
            BOARD_PX - SQUARE_SIZE,
            BOARD_PX - SQUARE_SIZE,
        )

    def test_e4(self):
        assert square_to_pixel(chess.E4) == (4 * SQUARE_SIZE, 4 * SQUARE_SIZE)


class TestPixelToSquare:
    def test_roundtrip_all_squares(self):
        for square in chess.SQUARES:
            x, y = square_to_pixel(square)
            center = (x + SQUARE_SIZE // 2, y + SQUARE_SIZE // 2)
            assert pixel_to_square(*center) == square

    def test_clamps_out_of_bounds(self):
        assert pixel_to_square(-100, -100) == chess.A8
        assert pixel_to_square(BOARD_PX + 100, BOARD_PX + 100) == chess.H1

    def test_origin_is_a8(self):
        assert pixel_to_square(0, 0) == chess.A8
