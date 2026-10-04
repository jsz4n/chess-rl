import gymnasium as gym
import numpy as np

import chess_gym  # noqa: F401  (registers Chess-v0)


def main() -> None:
    env = gym.make("Chess-v0", render_mode="unicode")
    obs, info = env.reset(seed=42)
    assert env.observation_space.contains(obs), "invalid observation"

    done = False
    total_reward = 0.0
    steps = 0
    while not done:
        legal = np.flatnonzero(obs["action_mask"])
        action = int(env.np_random.choice(legal))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated
    env.render()
    print(
        f"steps={steps} total_reward={total_reward} result={info['fen'].split(' ')[-1]}"
    )


if __name__ == "__main__":
    main()
