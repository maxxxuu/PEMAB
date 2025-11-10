import random
import numpy as np
import logging
from typing import Any, Union

from agent.algo.base_algo.BaseAlgo import BaseAlgo
from env.MultiArmedBanditEnvBase import MultiArmedBanditEnvBase


class UCB1(BaseAlgo):
    def fit(self, env: MultiArmedBanditEnvBase) -> None:
        pass

    def __init__(self, coef = np.sqrt(2), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coef = coef

    def choose_action(self, state: Any, env: MultiArmedBanditEnvBase, *args, **kwargs) -> Union[int, np.integer]:
        states = env.get_raw_state()
        total_rewards = np.array([state['total_rewards'] for state in states])
        played_times = np.array([state["played"] for state in states])
        total_played_times = np.sum(played_times)
        ucbs = total_rewards / played_times + self.coef * np.sqrt(np.log(total_played_times) / played_times)
        action = np.argmax(ucbs)
        return action
