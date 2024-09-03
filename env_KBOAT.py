import os
import gym
import numpy as np
import pandas as pd
from gym import spaces
from collections import deque


class SimpleEnv(gym.Env):
    def __init__(self):
        super(SimpleEnv, self).__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.return_que = deque()

    def step(self, action):
        if action == 0:
            if np.where(self.prob == 1)[0] == np.where(self.rank == 1)[0]:
                score = 2
            else:
                score = -1

        elif action == 1:
            if ((np.where(self.prob == 1)[0] == np.where(self.rank == 1)[0]) or
                    (np.where(self.prob == 2)[0] == np.where(self.rank == 2)[0])):
                score = 1
            else:
                score = -0.5

        elif action == 2:
            state_indices = set(np.where((self.prob == 1) | (self.prob == 2))[0])
            rank_indices = set(np.where((self.rank == 1) | (self.rank == 2))[0])

            if state_indices == rank_indices:
                score = 5
            else:
                score = -2.5

        elif action == 3:
            if ((np.where(self.prob == 1)[0] == np.where(self.rank == 1)[0]) and
                    (np.where(self.prob == 2)[0] == np.where(self.rank == 2)[0])):
                score = 10
            else:
                score = -5

        elif action == 4:
            state_indices = set(np.where((self.prob == 1) | (self.prob == 2) | (self.prob == 3))[0])
            rank_indices = set(np.where((self.rank == 1) | (self.rank == 2) | (self.rank == 3))[0])

            if state_indices == rank_indices:
                score = 7
            else:
                score = -3.5

        elif action == 5:
            state_indices = set(np.where((self.prob == 2) | (self.prob == 3))[0])
            rank_indices = set(np.where((self.rank == 2) | (self.rank == 3))[0])

            if ((state_indices == rank_indices) and
                    (np.where(self.prob == 1)[0] == np.where(self.rank == 1)[0])):
                score = 20
            else:
                score = -10

        elif action == 6:
            if ((np.where(self.prob == 1)[0] == np.where(self.rank == 1)[0]) and
                    (np.where(self.prob == 2)[0] == np.where(self.rank == 2)[0]) and
                    (np.where(self.prob == 3)[0] == np.where(self.rank == 3)[0])):
                score = 40
            else:
                score = -20
        self.score += score
        self.state[-1] = self.score
        return self.state, score, False, False

    def reset(self, probabilities, rank):
        self.prob = np.argsort(-probabilities) + 1
        self.rank = rank
        self.score = 0
        self.state = np.append(probabilities, 0)
        return self.state

    def render(self):
        print(f"State: {self.state}")
