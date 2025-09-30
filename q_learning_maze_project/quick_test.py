# quick_test.py
import numpy as np
import sys
import matplotlib.pyplot as plt
from env import environment, rewards, UP, DOWN, LEFT, RIGHT

#build environment
env = environment(package_location=(12,12),drop_off_location=(24,24))

state = env.reset()

plt.ion()
fig, ax = plt.subplots(figsize=(10, 10))

def greedy_towards(target, pos):
    x, y = pos
    tx, ty = target
    # simple greedy: move horizontally first, then vertically
    if x < tx: return RIGHT
    if x > tx: return LEFT
    if y < ty: return DOWN
    if y > ty: return UP
    return RIGHT  # arbitrary when already there




done = False
total_reward = 0.0
steps = 0

while not done and steps < env.max_steps:
    #checks if the agent is carrying which changes the target from the
    #package pickup to the drop-off
    target = env.package if state[2] == 0 else env.drop_off
    action = greedy_towards(target,(state[0],state[1]))

    #step
    state,reward,done,info = env.step(action)
    total_reward += reward
    steps +=1

    #mechanism to render once every N steps

    if steps % 1 ==0:
        ax.clear()
        env.render(ax=ax)
        ax.set_title(f"GREEDY AGENT |  step={steps} pos =[{state[0]},{state[1]}] target = {target} carry={state[2]} reward={reward:.2f}")
        plt.pause(0.01)



plt.ioff()
print(f"Return: {total_reward:.2f} | Steps: {steps} | Delivered: {info.get('deliver', False)}")
plt.show()



