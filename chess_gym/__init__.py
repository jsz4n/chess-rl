import gymnasium as gym

from chess_gym.env import ChessEnv

__all__ = ["ChessEnv"]

gym.register(id="Chess-v0", entry_point="chess_gym.env:ChessEnv")
