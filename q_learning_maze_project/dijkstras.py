
from numpy import argmax
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from env import environment, rewards, UP, DOWN, LEFT, RIGHT, FREE,WALL,PACKAGE,DROPOFF
import heapq
# import imageio


class pathfinder():
    def __init__(
            self,
            env:environment,
            ):
        self.env = env
        self.current_goal = None
        self.current_path = []
        self.current_steps = 0
        #eight neighbours
        self.NEIGHBOURS = [(0,1,1),(0,-1,1),(1,0,1),(-1,0,1),
              (1,1,np.sqrt(2)),(-1,-1,np.sqrt(2)),(-1,1,np.sqrt(2)),(1,-1,np.sqrt(2))]
        
    def cost_fn(self,x,y,nx,ny,w):
        return w

    def not_in_bounds(self,x,y):
        if x > self.env.width or y > self.env.height:
            return True
        else:
            return False

    def dijkstra_path(self,start_x,start_y,goal_x,goal_y,):
        
        INF = float('inf')
        env = self.env

        dist = [[INF]*env.width for _ in range(env.height)]
        parent = [[None]*env.width for _ in range(env.height)]
        

        pq = [(0,start_x,start_y)]

        while not (len(pq) == 0):
            d, x, y = heapq.heappop(pq)
            if d > dist[y][x]:
                continue 
            if (x, y) == (goal_x, goal_y):
                break

            for (dx, dy , w ) in self.NEIGHBOURS:
                nx,ny = x+dx,y+dy
                if self.not_in_bounds(nx,ny,env):continue
                if env.grid[nx,ny] == WALL:continue

                step_cost = self.cost_fn(x,y,nx,ny,w)
                nd = d + step_cost
                if nd < dist[ny][nx]:
                    dist[ny][nx] = nd
                    parent[ny][nx] = (x, y)
                    heapq.heappush(pq, (nd, nx, ny))

        return dist,parent

    def reconstruct_path(self,parent,start,goal):
        path = []
        x,y = goal
        while True:
            path.append((x,y))
            if (x,y) == start:
                break
            p = parent[y][x]
            if p is None:
                return []
            x,y =p

        path.reverse()
        return path
    
    def get_next_target(self):
        return None
    

        


















if __name__ == "__main__":
    env = environment(width = 12,height = 12, package_location=(4,4), dropoff_location=(0,0))
    dist,parent = dijkstra_path(0,0,4,4,cost_fn,env)
    print(reconstruct_path(parent,(0,0),(4,4)))


