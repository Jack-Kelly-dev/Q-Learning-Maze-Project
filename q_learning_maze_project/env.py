import numpy as np
import matplotlib
from dataclasses import dataclass
from typing import Any,Optional

FREE,WALL,PACKAGE,DESTINATION = 0,1,2,3
# 0,0 is top left
ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]

UP,DOWN,LEFT,RIGHT = 0,1,2,3


grid = [[],
        [],
        [],
        [],]


@dataclass
class rewards():
    step:float = -0.05
    wall:float = -2.0
    pickup:float = 1.0
    deliver:float = 10.0

class environment():
    def __init__(
            self,
            package: tuple(int,int),
            drop_off: tuple(int,int),
            start:tuple(int,int),
    ):
        
        self.H,self.W = 100,50
        self.grid = np.zeros(self.H,self.W)
        self.rewards = rewards()
        self.max_steps = 400

        self.start = (0,0)
        self.package = package
        self.drop_off = drop_off
        self.start = start
        self.pos = self.start
        self.carrying = False

        self.step_number =0

    def reset(self) -> tuple(int,int,int):
        self.pos = self.start
        self.step_number = 0
        return self.state()


    def step(self,action) -> tuple(tuple(int,int,int),float,bool,dict[str,Any]):
        reward = self.rewards.step
        nx,ny = self.move(self.pos,action)

        if self._wall_ahead(nx,ny):
            reward += self.rewards.wall
            next_pos = self.pos
        else:
            next_pos = (nx,ny)

        type_of_next_square = self._next_square_type(next_pos) 
        pickup_event = False
        deliver_event = False

        if type_of_next_square is PACKAGE and not self.carrying:
            self.carrying = True
            reward += self.rewards.pickup
            pickup_event = True
        if type_of_next_square is DESTINATION and self.carrying:
            self.carrying = False
            reward += self.rewards.deliver
            deliver_event = True
        
        self.pos = next_pos

        done = False
        if deliver_event:
            done = True
        elif self.step_number >= self.max_steps:
            done = True

        info = {
            "t": self._t,
            "pickup": pickup_event,
            "deliver": deliver_event,
            "pos": self.pos,
        }


        return (self.state(), reward, done, info)


        

    def render(self, ax=None, policy: Optional[np.ndarray] = None):
        """
        Quick matplotlib render. Optionally overlay a policy (W,H,2 -> action index).
        """
        import matplotlib.pyplot as plt
        show_now = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(4,4))
            show_now = True

        # base map: walls dark, free light; payload/drop colored
        img = np.zeros((self.H, self.W, 3), dtype=float)
        img[self.grid == FREE] = (0.95, 0.95, 0.95)
        img[self.grid == WALL] = (0.2, 0.2, 0.2)
        img[self.grid == PAYLOAD] = (0.2, 0.6, 1.0)
        img[self.grid == DROP] = (1.0, 0.5, 0.2)

        ax.imshow(img, origin="upper")
        ax.set_xticks(range(self.W)); ax.set_yticks(range(self.H))
        ax.grid(True, linewidth=0.5, color=(0,0,0,0.2))
        ax.set_xticklabels([]); ax.set_yticklabels([])

        # agent
        x, y = self.pos
        ax.scatter([x], [y], s=120, marker="o", edgecolors="k",
                    facecolors=(0.1, 0.8, 0.3) if self.carry else (0.9, 0.1, 0.3), zorder=3)

        # optional policy arrows for carry=0 (or both if you want)
        if policy is not None:
            from matplotlib.patches import FancyArrow
            for yy in range(self.H):
                for xx in range(self.W):
                    if self.grid[yy, xx] == WALL: 
                        continue
                    a = int(np.argmax(policy[xx, yy, 0]))  # show for carry=0
                    dx, dy = ACTIONS[a]
                    ax.add_artist(FancyArrow(xx, yy, dx*0.3, dy*0.3, width=0.02, color="k", alpha=0.5))

        if show_now:
            plt.tight_layout(); plt.show()


        #--------------------------------------------------------------------------------------------
        # smaller helper functions

    def _state(self) -> Tuple[int, int, int]:
        x, y = self.pos
        return (x, y, int(self.carry))

    def _tile(self, pos: Tuple[int, int]) -> int:
        x, y = pos
        return int(self.grid[y, x])

    def _is_wall(self, x: int, y: int) -> bool:
        return not (0 <= x < self.W and 0 <= y < self.H) or self.grid[y, x] == WALL

    def _move(self, pos: Tuple[int, int], a: int) -> Tuple[int, int]:
        dx, dy = ACTIONS[a]
        x, y = pos
        return (x + dx, y + dy)

    def _sample_free_cell(self, exclude=()) -> Tuple[int, int]:
        mask = (self.grid == FREE)
        for ex in exclude:
            exx, exy = ex
            if 0 <= exx < self.W and 0 <= exy < self.H:
                mask[exy, exx] = False
        ys, xs = np.where(mask)
        idx = self.rng.integers(0, len(xs))
        return (int(xs[idx]), int(ys[idx]))

        


        



        



        
        