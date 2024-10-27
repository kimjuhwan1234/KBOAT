import gym
import numpy as np


class SimpleEnv(gym.Env):
    def __init__(self):
        super(SimpleEnv, self).__init__()

    def step(self, action):
        # # 단승식
        # if action == 0:
        #     if np.where(self.prob == 1)[0] == np.where(self.rank == 1)[0]:
        #         score = 1.5
        #         state = 0
        #     else:
        #         score = -2
        #         state = -1

        # 연승식
        if action == 1:
            state_indices = set(np.where((self.prob == 1) | (self.prob == 2))[0])
            rank_indices = set(np.where((self.rank == 1) | (self.rank == 2))[0])

            if len(state_indices.intersection(rank_indices)) >= 1:
                score = 1.2
                state = 1
            else:
                score = -2
                state = -1

        # 복승식
        elif action == 2:
            state_indices = set(np.where((self.prob == 1) | (self.prob == 2))[0])
            rank_indices = set(np.where((self.rank == 1) | (self.rank == 2))[0])

            if state_indices == rank_indices:
                score = 1.8
                state = 2
            else:
                score = -2
                state = -1
        # # 쌍승식
        # elif action == 3:
        #     if ((np.where(self.prob == 1)[0] == np.where(self.rank == 1)[0]) and
        #             (np.where(self.prob == 2)[0] == np.where(self.rank == 2)[0])):
        #         score = 2.5
        #         state = 3
        #     else:
        #         score = -2
        #         state = -1
        # 삼복승식
        elif action == 0:
            state_indices = set(np.where((self.prob == 1) | (self.prob == 2) | (self.prob == 3))[0])
            rank_indices = set(np.where((self.rank == 1) | (self.rank == 2) | (self.rank == 3))[0])

            if state_indices == rank_indices:
                score = 3
                state = 4
            else:
                score = -2
                state = -1
        # # 쌍복승식
        # elif action == 5:
        #     state_indices = set(np.where((self.prob == 2) | (self.prob == 3))[0])
        #     rank_indices = set(np.where((self.rank == 2) | (self.rank == 3))[0])
        #
        #     if ((state_indices == rank_indices) and
        #             (np.where(self.prob == 1)[0] == np.where(self.rank == 1)[0])):
        #         score = 5
        #         state = 5
        #     else:
        #         score = -2
        #         state = -1
        # # 삼쌍승식
        # elif action == 6:
        #     if ((np.where(self.prob == 1)[0] == np.where(self.rank == 1)[0]) and
        #             (np.where(self.prob == 2)[0] == np.where(self.rank == 2)[0]) and
        #             (np.where(self.prob == 3)[0] == np.where(self.rank == 3)[0])):
        #         score = 7
        #         state = 6
        #     else:
        #         score = -2
        #         state = -1

        self.state[-1] = state
        return self.state, score, False, False

    def reset(self, probabilities, rank):
        self.prob = np.argsort(-probabilities) + 1
        self.rank = rank
        self.state = np.append(probabilities, -1)
        return self.state

    def render(self):
        print(f"State: {self.state}")
