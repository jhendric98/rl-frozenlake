"""rl_frozenlake.random_agent
================================

A small utility module for running a random agent on OpenAI's FrozenLake
environment.  The original repository contained a short script for the first
exercise of a reinforcement learning course.  The code has since been expanded
into a reusable module and command line interface with richer documentation,
modern Gymnasium APIs, and optional visualisation utilities.

Examples
--------
Run 200 episodes without plotting::

    python frozenlake1.py --episodes 200 --no-plot

Plot the moving average of the win rate and save it to a file::

    python frozenlake1.py --episodes 500 --moving-average-window 50 \
        --save-plot results.png

"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# Gymnasium and Matplotlib are relatively heavy imports; declare the public API
# explicitly to make `from frozenlake1 import *` predictable.
__all__ = [
    "EpisodeStats",
    "RandomAgent",
    "plot_episode_stats",
    "run_random_agent",
]


@dataclass(slots=True)
class EpisodeStats:
    """Statistics describing a collection of FrozenLake episodes.

    Parameters
    ----------
    episode_rewards:
        The raw reward obtained at the end of each episode.
    moving_average:
        The moving average of rewards computed over the trailing ``window``
        number of episodes.  The window size is stored as ``moving_average_window``.
    moving_average_window:
        The number of episodes used to compute the moving average.  A value of
        ``0`` indicates that no moving average was calculated.
    """

    episode_rewards: list[float]
    moving_average: list[float]
    moving_average_window: int

    @property
    def total_episodes(self) -> int:
        """Return the number of simulated episodes."""

        return len(self.episode_rewards)

    @property
    def mean_reward(self) -> float:
        """Return the mean reward across all episodes.

        The FrozenLake environment provides a reward of ``1.0`` for reaching the
        goal and ``0.0`` otherwise, therefore the mean reward is equivalent to
        the win rate of the random agent.
        """

        if not self.episode_rewards:
            return float("nan")
        return float(np.mean(self.episode_rewards))

    @property
    def std_reward(self) -> float:
        """Return the standard deviation of rewards across all episodes."""

        if not self.episode_rewards:
            return float("nan")
        return float(np.std(self.episode_rewards))


class RandomAgent:
    """A minimalistic agent that samples uniformly from the action space."""

    def __init__(self, action_space: gym.Space, *, seed: Optional[int] = None) -> None:
        self.action_space = action_space
        if seed is not None:
            # Seeding the action space ensures reproducible action choices.
            self.action_space.seed(seed)

    def act(self, _observation: np.ndarray | int | float | None) -> int:
        """Return a random action from the agent's action space."""

        return int(self.action_space.sample())


def _moving_average(values: Iterable[float], window: int) -> list[float]:
    """Compute a simple moving average over ``values`` with a specific window.

    Parameters
    ----------
    values:
        Sequence of numeric values.
    window:
        The number of values to include in the rolling window.  A non-positive
        value disables the moving average.
    """

    if window <= 0:
        return []

    values_list = list(values)
    averages: list[float] = []
    for index in range(len(values_list)):
        start = max(index + 1 - window, 0)
        window_slice = values_list[start : index + 1]
        averages.append(float(np.mean(window_slice)))
    return averages


def run_random_agent(
    *,
    env_name: str = "FrozenLake-v1",
    episodes: int = 1_000,
    moving_average_window: int = 50,
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
) -> EpisodeStats:
    """Simulate ``episodes`` of a random agent in the specified environment.

    Parameters
    ----------
    env_name:
        The Gymnasium environment identifier to load.
    episodes:
        Number of episodes to simulate.  Must be a positive integer.
    moving_average_window:
        Window size for the moving average calculation.  A value of ``0``
        disables the moving average.
    seed:
        Optional seed for both environment and agent reproducibility.  The seed
        is applied to the environment reset operation as well as the agent's
        action space.
    render_mode:
        Render mode passed directly to :func:`gymnasium.make`.  Common values are
        ``"human"`` and ``"ansi"``; ``None`` (the default) disables rendering.

    Returns
    -------
    EpisodeStats
        Container with the episodic rewards and their moving averages.
    """

    if episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    if moving_average_window < 0:
        raise ValueError("moving_average_window must be non-negative")

    env = gym.make(env_name, render_mode=render_mode)

    # Gymnasium exposes `env.reset(seed=...)`, but the action space also needs to
    # be seeded to ensure reproducibility of sampled actions.
    agent = RandomAgent(env.action_space, seed=seed)

    episode_rewards: list[float] = []

    try:
        for episode in range(episodes):
            observation, _info = env.reset(seed=None if seed is None else seed + episode)
            terminated = False
            truncated = False
            reward_total = 0.0

            while not (terminated or truncated):
                action = agent.act(observation)
                observation, reward, terminated, truncated, _info = env.step(action)
                reward_total += float(reward)

            episode_rewards.append(reward_total)
    finally:
        env.close()

    moving_average = _moving_average(episode_rewards, moving_average_window)

    return EpisodeStats(
        episode_rewards=episode_rewards,
        moving_average=moving_average,
        moving_average_window=moving_average_window,
    )


def plot_episode_stats(
    stats: EpisodeStats,
    *,
    show: bool = True,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot episodic rewards and optional moving averages.

    Parameters
    ----------
    stats:
        The :class:`EpisodeStats` generated by :func:`run_random_agent`.
    show:
        Whether to call :func:`matplotlib.pyplot.show` before returning.  Set to
        ``False`` when running in headless or automated environments.
    save_path:
        Optional path to save the generated figure.  The parent directory is
        created automatically if it does not already exist.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure for further customisation.
    """

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(stats.episode_rewards, label="Episode reward", color="tab:blue")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title(
        "Random agent on FrozenLake"
        if stats.moving_average_window == 0
        else (
            "Random agent on FrozenLake"
            f" (moving average window = {stats.moving_average_window})"
        )
    )

    if stats.moving_average:
        ax.plot(
            stats.moving_average,
            label=f"Moving average ({stats.moving_average_window})",
            color="tab:orange",
        )

    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a random agent on the FrozenLake environment and plot the results.",
    )
    parser.add_argument(
        "--env-name",
        default="FrozenLake-v1",
        help="Gymnasium environment identifier (default: %(default)s)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1_000,
        help="Number of episodes to run (default: %(default)s)",
    )
    parser.add_argument(
        "--moving-average-window",
        type=int,
        default=50,
        help="Window size for moving average computation; 0 disables it (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed applied to the environment and agent",
    )
    parser.add_argument(
        "--render-mode",
        default=None,
        help="Render mode passed to gymnasium.make (default: %(default)s)",
    )
    parser.add_argument(
        "--save-plot",
        type=Path,
        default=None,
        help="Optional path to save the generated plot as an image",
    )
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Display the Matplotlib figure (use --no-plot for headless runs)",
    )
    return parser


def main(args: Optional[Iterable[str]] = None) -> EpisodeStats:
    """Entry point used by the command line interface.

    The function returns the :class:`EpisodeStats` instance produced during the
    simulation, which can be useful for unit tests or downstream analyses.
    """

    parser = _build_parser()
    parsed_args = parser.parse_args(args=args)

    stats = run_random_agent(
        env_name=parsed_args.env_name,
        episodes=parsed_args.episodes,
        moving_average_window=parsed_args.moving_average_window,
        seed=parsed_args.seed,
        render_mode=parsed_args.render_mode,
    )

    if parsed_args.plot:
        plot_episode_stats(stats, show=True, save_path=parsed_args.save_plot)
    elif parsed_args.save_plot is not None:
        # Allow saving plots without showing them via ``--no-plot --save-plot``.
        plot_episode_stats(stats, show=False, save_path=parsed_args.save_plot)

    return stats


if __name__ == "__main__":
    main()
