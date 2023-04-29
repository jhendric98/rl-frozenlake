import gym
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("FrozenLake-v1")

n_games = 1000
win_pct = []
scores = []
for i in range(n_games):
    terminated = False
    obs = env.reset()
    score = 0
    while not terminated:
        observation, reward, terminated, truncated, info = env.step(
            env.action_space.sample()
        )
        score += reward
    scores.append(score)

    if i % 10 == 0:
        average = np.mean(scores[-10:])
        win_pct.append(average)
plt.plot(win_pct)
plt.show()
