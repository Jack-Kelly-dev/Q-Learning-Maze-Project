# quick_test.py
import numpy as np
from env import MazeEnv, small_maze, Rewards, UP, DOWN, LEFT, RIGHT

grid, start = small_maze()
env = MazeEnv(grid, start, rewards=Rewards(), slip_prob=0.0, random_start=False, seed=42)

s = env.reset()
done = False
total = 0
while not done:
    # naive/rightward policy to see it move
    a = RIGHT if s[0] < env.drop_pos[0] else DOWN
    s, r, done, info = env.step(a)
    total += r

print("Return:", total, "Delivered:", info["deliver"])
