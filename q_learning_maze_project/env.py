import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Any,Optional


FREE,WALL,PACKAGE,DROPOFF = 0,1,2,3
# 0,0 is top left
ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]

UP,DOWN,LEFT,RIGHT = 0,1,2,3

@dataclass
class rewards():
    step:float = -0.05
    wall:float = -2.0
    pickup:float = 1.0
    deliver:float = 10.0

class environment():
    def __init__(
            self,
            width : int,
            height: int,
            package_location: tuple[int,int],
            drop_off_location: tuple[int,int],
            seed : Optional[int] = None,
    ):
        
        self.width,self.height = width,height
        self.grid = np.zeros((self.height,self.width))
        self.rewards = rewards()
        self.max_steps = 50

        self.START = (0,0)
        self.package = package_location
        self.drop_off = drop_off_location

        self.grid[package_location[1],package_location[0]] = PACKAGE
        self.grid[drop_off_location[1],drop_off_location[0]] = DROPOFF

        self.pos = self.START
        self.carrying = False

        self.step_number = 0
        #for the random free space method
        self.rng = np.random.default_rng(seed)

    def reset(self) -> tuple[int,int,int]:
        self.pos = self.START
        self.step_number = 0
        self.carrying = False
        x, y = self.pos
        return (x, y, int(self.carrying))

    
    def _get_state(self) -> tuple[int, int, int]:
        x, y = self.pos
        return (x, y, int(self.carrying))

    def _tile(self, pos: tuple[int, int]) -> int:
        x, y = pos
        return int(self.grid[y, x])

    def _wall_ahead(self, x: int, y: int) -> bool:
        return not (0 <= x < self.width and 0 <= y < self.height) or self.grid[y, x] == WALL

    def _move(self, pos: tuple[int, int], a: int) -> tuple[int, int]:
        dx, dy = ACTIONS[a]
        x, y = pos
        return (x + dx, y + dy)

    def _next_square_type(self,next_pos: tuple[int,int]) -> int:
        x,y = next_pos
        # treat edges and walls
        if x < 0 or x > self.width or y < 0 or y > self.height:
            return WALL
        else:
            return int(self.grid[x,y])
        
    def step(self,action) -> tuple[tuple[int,int,int],float,bool,dict[str,Any]]:
        print("step")
        reward = self.rewards.step
        nx,ny = self._move(self.pos,action)

        if self._wall_ahead(nx,ny):
            reward += self.rewards.wall
            next_pos = self.pos
        else:
            next_pos = (nx,ny)

        type_of_next_square = self._next_square_type(next_pos) 
        pickup_event = False
        deliver_event = False
        print(type_of_next_square)

        if type_of_next_square == PACKAGE and not self.carrying:
            print("found package")
            self.carrying = True
            reward += self.rewards.pickup
            pickup_event = True
        if type_of_next_square == DROPOFF and self.carrying:
            print("dropped_ off")
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
            "t": "thisis deprecated",
            "step": self.step_number,
            "pickup": pickup_event,
            "deliver": deliver_event,
            "pos": self.pos,
        }


        return (self._get_state(), reward, done, info)
    
    # random free cell picker
    def _sample_free_cell(self, exclude=()) -> tuple[int, int]:
        mask = (self.grid == FREE)
        for ex in exclude:
            exx, exy = ex
            if 0 <= exx < self.W and 0 <= exy < self.H:
                mask[exy, exx] = False
        ys, xs = np.where(mask)
        idx = self.rng.integers(0, len(xs))
        return (int(xs[idx]), int(ys[idx]))


        

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
        img = np.zeros((self.height, self.width, 3), dtype=float)
        img[self.grid == FREE] = (0.95, 0.95, 0.95)
        img[self.grid == WALL] = (0.2, 0.2, 0.2)
        img[self.grid == PACKAGE] = (0.2, 0.6, 1.0)
        img[self.grid == DROPOFF] = (1.0, 0.5, 0.2)

        ax.imshow(img, origin="upper")
        ax.set_xticks(range(self.width)); ax.set_yticks(range(self.height))
        ax.grid(True, linewidth=0.5, color=(0,0,0,0.2))
        ax.set_xticklabels([]); ax.set_yticklabels([])

        # agent
        x, y = self.pos

        ax.scatter([x], [y], s=120, marker="o", edgecolors="k",
                    facecolors=(0.1, 0.8, 0.3) if self.carrying else (0.9, 0.1, 0.3), zorder=3)
        #gonna plot package and dropoff
        ax.scatter([self.package[0]], [self.package[1]], s=120, marker="o", edgecolors="k",
                    facecolors=(0.1, 0.2, 0.3) if self.carrying else (0.9, 0.9, 0.3), zorder=3)
        ax.scatter([self.drop_off[0]], [self.drop_off[1]], s=120, marker="o", edgecolors="k",
                    facecolors=(0.1, 0.8, 0.9) if self.carrying else (0.9, 0.1, 0.9), zorder=3)
        # optional policy arrows for carry=0 (or both if you want)
        if policy is not None:
            from matplotlib.patches import FancyArrow
            for yy in range(self.height):
                for xx in range(self.width):
                    if self.grid[yy, xx] == WALL: 
                        continue
                    a = int(np.argmax(policy[xx, yy, 0]))  # show for carry=0
                    dx, dy = ACTIONS[a]
                    ax.add_artist(FancyArrow(xx, yy, dx*0.3, dy*0.3, width=0.02, color="k", alpha=0.5))

        if show_now:
            plt.tight_layout(); plt.show()




        
        