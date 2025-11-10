import random
import numpy as np
import math
import logging

from agent.algo.base_algo.BaseAlgo import BaseAlgo


class EpsilonGreedy(BaseAlgo):
    """
    This algo is designed for MAB env
    """

    def __init__(self, max_epsilon=1, min_epsilon=0, decay=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decay_step = 0
        self.max_horizon = None
        self.decay_step_counter = 0
        assert 1 >= max_epsilon >= 0, f"max_epsilon should be between [0, 1], get {max_epsilon}"
        assert max_epsilon >= min_epsilon >= 0, f"min_epsilon should be between [0, max_epsilon({max_epsilon})], " \
                                                f"get {min_epsilon}"
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.current_epsilon = max_epsilon

    def fit(self, env):
        self.max_horizon = env.horizon if env.horizon != 0 else 1
        if self.decay is not None:
            decay_time = math.ceil((self.max_epsilon - self.min_epsilon) / self.decay)
            self.decay_step = math.ceil(self.max_horizon / decay_time)
        else:
            self.decay_step = self.max_horizon
            self.decay = (self.max_epsilon - self.min_epsilon) / self.max_horizon
        self.decay_step_counter = 0

    def init_single_test(self):
        self.decay_step_counter = 0
        self.current_epsilon = self.max_epsilon

    def choose_action(self, state, env, *args, **kwargs):
        if random.random() < self.current_epsilon:
            action = np.random.randint(0, self.out_dim)
        else:
            states = env.get_raw_state()
            total_rewards = np.array([state['total_rewards'] for state in states])
            played_times = np.array([state["played"] for state in states])
            estimate_rewards = total_rewards / played_times
            action = np.argmax(estimate_rewards)

        self.decay_step_counter += 1
        if self.decay_step_counter % self.decay_step == 0:
            self.current_epsilon = self.current_epsilon - self.decay if \
                (self.current_epsilon - self.decay) > self.min_epsilon else self.min_epsilon

        return action
