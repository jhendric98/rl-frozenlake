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
- Rich :class:`EpisodeStats` container exposing common metrics, success rates,
  and episode lengths.
- Optional Matplotlib visualisations with support for saving plots to disk and
  overlaying episode lengths.
- JSON exports for downstream analyses or reproducible experiment tracking.
- Reproducible experiments through deterministic seeding.

## Installation

This repository targets Python 3.10+ and relies on
[`uv`](https://github.com/astral-sh/uv) for dependency management.  Installing
the dependencies is as simple as:

```bash
uv sync
source .venv/bin/activate
```

`uv sync` resolves the dependency graph, creates a virtual environment, and
installs the pinned packages.  Subsequent commands can be executed with
`uv run`, which automatically reuses the managed environment:

```bash
uv run python frozenlake1.py --episodes 500 --no-plot
```

If you prefer manual activation, the virtual environment lives in `.venv/` by
default.  Dependency updates can be applied with `uv lock --upgrade` followed by
`uv sync`.

## Usage

The command line entry point offers switches to control the number of episodes,
random seeds, moving-average windows, episode-length overlays, and output
artefacts.

```bash
uv run python frozenlake1.py \
  --episodes 2000 \
  --moving-average-window 100 \
  --plot-lengths \
  --save-plot plots/frozenlake.png
```

Headless environments can disable interactive plotting while still saving the
figure and exporting JSON statistics:

```bash
uv run python frozenlake1.py --episodes 500 --no-plot --save-plot frozenlake.png --save-stats results.json
```

The module can also be imported directly.  The snippet below executes a quick
simulation and inspects summary statistics:

```python
from frozenlake1 import run_random_agent

stats = run_random_agent(episodes=250, moving_average_window=25, seed=42)
print(f"Success rate: {stats.success_rate:.3f}")
print(f"Average episode length: {stats.mean_episode_length:.2f} steps")
```

## Development

Linting and formatting can be added depending on your workflow, but no
additional tooling is required for basic experimentation.  Pull requests and
issues are always welcome.
