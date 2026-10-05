# chess-gym

A [Gymnasium](https://gymnasium.farama.org/) environment for chess, built for deep reinforcement learning courses and experiments. Play against a simple greedy bot in a pygame window, or train an agent on a clean, fully-masked chess API. Built on [python-chess](https://python-chess.readthedocs.io/).

## Features

- **Standard Gymnasium API** — registered as `Chess-v0`, works with `gym.make`, wrappers, and vectorization.
- **Structured observations** — 8×8×12 piece planes, side-to-move flag, and a full legal-move action mask.
- **Flat 20,480-action space** — five `from_square * 64 + to_square` layers (normal moves, then =Q/=R/=B/=N promotions) with UCI-style helpers (`e2e4`, `a7a8n`); invalid moves are masked out.
- **Multiple render modes** — pygame window, unicode board, FEN, or SVG.
- **Interactive play** — click-to-move pygame UI against a greedy-capture opponent (`play.py`).

## Installation

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Usage

### Quick demo (random agents)

Runs a full game where both sides play uniformly random legal moves:

```bash
uv run main.py
```

### Play in a window

You play White by clicking a piece (legal destinations are highlighted), then clicking a target square. Promotions auto-queen. Black is answered by a greedy bot that captures the most valuable piece available:

```bash
uv run play.py
```

Close the window or finish the game to exit; the result appears in the window title.

### Train a Q-learning agent

Tabular Q-learning (ε-greedy, opponent plays uniform random legal moves) from a fixed position. The trained Q-table is saved to `checkpoints/` as a pickle holding `{"q", "color", "fen"}`:

```bash
uv run q_learning.py --color white --episodes 300   # → checkpoints/q_white.pkl
uv run q_learning.py --color black --episodes 300   # → checkpoints/q_black.pkl
```

Defaults train from a rook endgame with a back-rank mate in one (`--fen` to override; it must match the `--color` side to move). The agent only knows positions reachable from that FEN — that's inherent to tabular Q-learning over exact FEN keys.

### Play against your agent

Loads a checkpoint, starts from its position, and opens the click-to-move pygame window. You take the color the agent didn't train as:

```bash
uv run play_agent.py --checkpoint checkpoints/q_white.pkl   # agent White, you Black
uv run play_agent.py --checkpoint checkpoints/q_black.pkl   # agent Black, you White
```

The agent plays greedily (ε=0), with a random tie-break in states it never visited during training.

### Agent vs agent

Headless match between two checkpoints, greedy policies, N games:

```bash
uv run arena.py --white checkpoints/q_white.pkl --black checkpoints/q_black.pkl --games 50
```

Prints the W/L/D table and average game length. `--fen` overrides the start position (default: the white agent's training FEN). Note that knowledge doesn't transfer across positions — for meaningful games, both agents should have trained from the same position tree.

Watch a match live in the pygame window (add `--games 1` for a single game, `--move-delay` to slow it down):

```bash
uv run arena.py --white checkpoints/fed_q_white.pkl --black checkpoints/fed_q_black.pkl --render human --games 1
```

### Federated training

FedAvg-style tabular Q-learning in `federated.py`: 15 independent client trainings → averaged (mean) in groups of 3 → the 5 aggregated models are retrained → the 5 retrained models are averaged into the final agent. Runs for both colors and saves standard checkpoints (`fed_q_white.pkl`, `fed_q_black.pkl`) usable in `play_agent.py` and `arena.py`:

```bash
uv run federated.py                        # 15×3 groups, both colors, 300 episodes each
uv run federated.py --clients 9 --group-size 3 --episodes 200 --color white
```

Averaging is per state-action entry over the group members that visited it; unseen entries stay absent rather than being diluted toward zero. `--seed` makes the whole pipeline reproducible. Each final agent is evaluated greedily against a random opponent.

### Use the environment in your own code

```python
import gymnasium as gym
import numpy as np

import chess_gym  # registers Chess-v0

env = gym.make("Chess-v0", render_mode="unicode", max_steps=500)
obs, info = env.reset(seed=42)

done = False
while not done:
    legal = np.flatnonzero(obs["action_mask"])
    action = int(env.np_random.choice(legal))
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.render()
print(info["fen"])
```

### Custom start position

```python
env = gym.make("Chess-v0", fen="6k1/5ppp/8/8/8/8/8/4R2K w - - 0 1")
```

`reset(options={"fen": ...})` also works for per-episode positions.

## API

### Action space

`spaces.Discrete(20480)` — five layers over all 64×64 from→to pairs, matching the coordinate notation of an electronic chessboard (lift a piece on a square, drop it on another, promotion suffix included):

| Layer | Meaning |
| ----- | ------- |
| 0 | normal moves, action = `from_square * 64 + to_square` |
| 1–4 | promotions to Q, R, B, N: action = `layer * 4096 + from_square * 64 + to_square` |

Layer 0 is identical to the plain 4096 encoding. Helpers use [python-chess](https://python-chess.readthedocs.io/) square indices (a1 = 0, h8 = 63): `chess_gym.env.move_to_action` / `action_to_move` convert moves in both directions, and `action_to_uci` / `uci_to_action` map to coordinate strings like `"e2e4"` or `"a7a8n"`.

```python
from chess_gym.env import action_to_uci, uci_to_action

uci_to_action("e2e4")   # 796 — layer 0, plain e2→e4
uci_to_action("a7a8n")  # knight underpromotion
action_to_uci(796, board)  # "e2e4"
```

### Observation space

A `spaces.Dict` with:

| Key           | Shape / type  | Description                                                                                                            |
| ------------- | ------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `board`       | `(8, 8, 12)` uint8 | One-hot piece planes. Planes 0–5: current side's pawn→king, planes 6–11: opponent's. Board is flipped so the side to move sits on ranks 1–2. |
| `turn`        | `Discrete(2)` | `0` = white to move, `1` = black to move.                                                                              |
| `action_mask` | `(20480,)` int8 | `1` for legal actions, `0` elsewhere. 20 at the starting position.                                     |

### Rewards

| Outcome                    | Reward |
| -------------------------- | ------ |
| Checkmate delivered        | `+1.0` |
| Checkmated                 | `-1.0` |
| Draw / stalemate / timeout | `0.0`  |
| Illegal move (terminates)  | `-1.0` |

Rewards are from the moving side's perspective. Episodes are truncated (not terminated) after `max_steps` moves (default `500`).

### Info

`info = {"fen": ...}` on every `reset` and `step`.

### Render modes

- `human` — pygame window with last-move highlight, check indicator, and click support. The window opens on `reset()` and refreshes after every `step()` automatically — no manual `render()` calls needed. Frame rate follows `metadata["render_fps"]` (60).
- `unicode` — prints the board as unicode characters to stdout.
- `fen` — returns the FEN string.
- `svg` — returns an SVG string of the board.

## Testing

```bash
uv run pytest
```

The suite covers action encoding/decoding (promotions and UCI helpers included), observations, rewards, termination vs. truncation, human-mode auto-rendering, and gym registration.

## Project layout

```
chess_gym/
  env.py      # ChessEnv, action encoding, observation building
  render.py   # PygameRenderer, pixel<->square helpers
main.py       # random-vs-random rollout demo
play.py       # human (White) vs greedy bot (Black) in pygame
q_learning.py # tabular Q-learning training (saves checkpoints/)
federated.py  # FedAvg-style multi-training + aggregation, both colors
play_agent.py # human vs trained Q-agent in pygame
arena.py      # Q-agent vs Q-agent, headless
tests/        # pytest suite
```

## License

[MIT](LICENSE)
