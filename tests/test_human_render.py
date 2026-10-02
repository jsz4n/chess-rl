import os

import gymnasium as gym
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import chess_gym  # noqa: F401  # registers Chess-v0


@pytest.fixture
def human_env():
    env = gym.make("Chess-v0", render_mode="human", max_steps=10)
    yield env
    env.close()


def test_renderer_created_on_reset(human_env):
    human_env.reset()
    assert human_env.unwrapped.renderer is not None


def test_renderer_reused_on_step(human_env):
    human_env.reset()
    renderer = human_env.unwrapped.renderer
    human_env.step(312)
    assert human_env.unwrapped.renderer is renderer


def test_render_fps_wired_from_metadata(human_env):
    human_env.reset()
    renderer = human_env.unwrapped.renderer
    assert renderer.fps == human_env.unwrapped.metadata["render_fps"] == 60


def test_reset_clears_selection_state(human_env):
    human_env.reset()
    renderer = human_env.unwrapped.renderer
    renderer.selected = 12
    renderer.legal_targets = {13, 14}
    human_env.reset()
    assert renderer.selected is None
    assert renderer.legal_targets == set()


def test_close_clears_renderer(human_env):
    human_env.reset()
    human_env.close()
    assert human_env.unwrapped.renderer is None


def test_no_renderer_in_non_human_mode():
    env = gym.make("Chess-v0", render_mode="unicode")
    env.reset()
    assert env.unwrapped.renderer is None
    env.close()
