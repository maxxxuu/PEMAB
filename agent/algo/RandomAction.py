import random
import numpy as np
import logging

from agent.algo.base_algo.BaseAlgo import BaseAlgo


class RandomAction(BaseAlgo):
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    def choose_action(self, state, *args, **kwargs):
        action_ = np.random.randint(0, self.out_dim)
        return action_

    def fit(self):
        pass
