from networkx import descendants
from numpy import argmax
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from env import environment, rewards, UP, DOWN, LEFT, RIGHT

#need to intitialise the action value function to 0 for every possible permutation

#constants
ALPHA = 0.5 #learning rate
GAMMA = 0.5 #discount factor
EPSILON = 0.5 # exploration rate
NUM_EPISODES = 100 #how many times to train


#build environment
env = environment(12,12,package_location=(6,6),drop_off_location=(11,11))
state = env.reset()

#list of actions
A = [UP,DOWN,LEFT,RIGHT]
#list of states
S = [(x, y, carrying)
     for x in range(env.width)
     for y in range(env.height)
     for carrying in range(0,2)
     ]


#state_action function initialised
Q = {(s,a): 0.0 for s in S for a in A}



def set_q(s,a,val):
    Q[(s,a)] = val 


def e_greedy_action(state, e = EPSILON):

    #picks E-greedy action
    if rnd.random() < e:
        return rnd.choice(A)
    else:
        q_vals = [Q[(state, a)] for a in A]
        print(f"q_vals: {q_vals}")
        max_q  = max(q_vals)
        print(f"max q: {max_q}")

        best_actions = [a for a, q in zip(A, q_vals) if q == max_q]
        print(f"best action: {best_actions}")

        action = rnd.choice(best_actions)
        print(f"action: {action}")

        return action
            
    

for episode in range(NUM_EPISODES):
    state = env.reset()
    done = False
    steps = 0

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 10))

    while not done and steps <= 50:
        
        action = e_greedy_action(state)
        next_state, reward, done, info = env.step(action)
        print(next_state)
        print(steps)

        if steps % 1 ==0:
            ax.clear()
            env.render(ax=ax)
            ax.set_title(f"GREEDY AGENT")
            plt.pause(0.01)

        if done:
            target = reward
        else:
            target = reward + GAMMA*max(Q[(next_state,a)] for a in A)

        #q-learning update

        old = Q[(state,action)]
        Q[(state,action)] = old + ALPHA*(target - old)

        steps +=1




plt.ioff()
print(f"Return: {total_reward:.2f} | Steps: {steps} | Delivered: {info.get('deliver', False)}")
plt.show()


