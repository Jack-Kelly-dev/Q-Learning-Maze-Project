import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Any,Optional
from env import environment, rewards, UP, DOWN, LEFT, RIGHT, FREE,WALL,PACKAGE,DROPOFF

from dijkstras import pathfinder



# main playing loop

plt.ion()
fig,ax = plt.subplots(figsize=(10,10))

env = environment(6,6,package_location=(4,4),drop_off_location=(5,5))
state = env.reset()
done = False
dijkstra_agent = pathfinder(env)
dijkstra_dist,dijkstra_parent = dijkstra_agent.dijkstra_path(env.START[0],env.START[1],
                                                            env.drop_off[0],env.drop_off[1],
                                                            )
path = dijkstra_agent.reconstruct_path(dijkstra_parent,
                                       env.START[0],env.START[1],
                                       env.drop_off[0],env.drop_off[1]
                                       )
print(path)

# while not done and dijkstra_agent.current_steps < env.max_steps:
#     target

