import logging
import copy

import numpy as np
import itertools
# from statsmodels.stats.proportion import proportion_confint

from env.slot_machine.MultiArmedBandit import MultiArmedBandit
from env.abstract.Environment import AbstractEnvironment
from env.MultiArmedBanditEnvBase import MultiArmedBanditEnvBase


class MultiArmedBanditEnvForPESym(MultiArmedBanditEnvBase):
    # def __init__(self, play_nb=1, round_nb=10, machine_nb=1, reward_distris=None, reward=None):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # for each machine: played, reward_times, for env: horizon, round_nb
        # TODO: check if need to add total round
        self.state_size = len(self.state_keys) * self.state_repeat

        # self.state_space = (too complicated but not useful)

    # def reset_horizon(self):
    #     self.horizon = self.round_nb * self.play_nb

    def convert_state(self, raw_states, extra_info=None):
        # extra_info should be a list of number (int or float)
        # states = [] if extra_info is None else extra_info
        states = []
        for raw_state in raw_states:
            state = [
                # raw_state["played"],
                # raw_state["reward_times"],
                # (raw_state["reward_times"] + 1) / (raw_state["played"] + 1) if raw_state["played"] > 0 else 0,
                # raw_state["reward_times"] / raw_state["played"] if raw_state["played"] > 0 else -1,
                # raw_state["played"] / (self.round_nb - self.horizon) if (self.round_nb != self.horizon) else 0,
                # raw_state["played"] / self.round_nb,
                # (self.round_nb - self.horizon) / self.round_nb,
                # raw_state["total_rewards"],
                # Lower bound of confidence interval
                # proportion_confint(
                #     count=raw_state["reward_times"], nobs=raw_state["played"]
                # )[0] if raw_state["played"] > 0 else -1,
                # Upper bound of confidence interval
                # proportion_confint(
                #     count=raw_state["reward_times"], nobs=raw_state["played"]
                # )[1] if raw_state["played"] > 0 else -1,
                raw_state[key] for key in self.state_keys
            ]
            state = state * self.state_repeat
            states.append(state)

        return np.array(states)

