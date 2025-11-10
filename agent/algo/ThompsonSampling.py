import random
import numpy as np
import math
import logging

from agent.algo.base_algo.BaseAlgo import BaseAlgo


class ThompsonSampling(BaseAlgo):

    def choose_action(self, state, env, *args, **kwargs):
        states = env.get_raw_state()
        reward_times = np.array([state['reward_times'] for state in states])
        played_times = np.array([state["played"] for state in states])
        loss_times = played_times - reward_times
        # Beta distri requires a and b to > 0
        reward_times = np.where(reward_times <= 0, 0.01, reward_times)
        loss_times = np.where(loss_times <= 0, 0.01, loss_times)
        estimate_reward_dis = np.array(
            [np.random.beta(reward_times[i], loss_times[i]) for i in range(len(reward_times))])
        action = np.argmax(estimate_reward_dis)
        return action

