# rl-frozenlake

Utilities for exploring reinforcement learning on the classic
[FrozenLake](https://www.gymlibrary.dev/environments/toy_text/frozen_lake/) task.
The original repository contained a short exploratory script.  The project now
ships a small but well-documented tool that can be reused from the command line
or imported as a module.

## Features

- Modern Gymnasium-based implementation compatible with the latest environment
  APIs.
- Reusable :class:`RandomAgent` and :func:`run_random_agent` utilities for
  programmatic experimentation.
- Rich :class:`EpisodeStats` container exposing common metrics.
- Optional Matplotlib visualisations with support for saving plots to disk.
- Reproducible experiments through deterministic seeding.

## Installation

This repository targets Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The dependencies are lightweight and limited to research tooling:

- [gymnasium](https://github.com/Farama-Foundation/Gymnasium)
- [numpy](https://numpy.org)
- [matplotlib](https://matplotlib.org)

## Usage

The command line entry point offers several switches to control the number of
episodes, random seeds, moving-average windows, and output plots.

```bash
python frozenlake1.py --episodes 2000 --moving-average-window 100 --save-plot plots/frozenlake.png
```

Headless environments can disable interactive plotting while still saving the
figure:

```bash
python frozenlake1.py --episodes 500 --no-plot --save-plot frozenlake.png
```

The module can also be imported directly.  The snippet below executes a quick
simulation and inspects summary statistics:

```python
from frozenlake1 import run_random_agent

stats = run_random_agent(episodes=250, moving_average_window=25, seed=42)
print(f"Mean reward: {stats.mean_reward:.3f} ± {stats.std_reward:.3f}")
```

## Development

Linting and formatting can be added depending on your workflow, but no
additional tooling is required for basic experimentation.  Pull requests and
issues are always welcome.
