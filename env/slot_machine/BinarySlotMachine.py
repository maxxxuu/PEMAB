import random
from typing import Union
from typing import Callable, Iterable, Union, Collection, Any, Optional

from utils.utils import random_verifier


class BinarySlotMachine:
    def __init__(self, index: Optional[int]=None, reward_distri: Optional[Union[Callable, float, int]]=None,
                 reward: Optional[Union[int, float]]=None) -> None:
        """


        """
        if reward_distri is not None:
            # If it is a number:
            if type(reward_distri) == float or type(reward_distri) == int:
                assert 0 <= reward_distri <= 1, \
                    f"the distribution of reward should be between [0,1], current value:{reward_distri}"
            # If reward distri is a random generator:
            elif hasattr(reward_distri, '__call__'):
                reward_distri = random_verifier(reward_distri, 0, 1)
            else:
                raise ValueError(f"Input invalid ({reward_distri})for reward_distri.")

        self.id = index
        self._reward_distri = reward_distri if reward_distri is not None else random.random()
        # self._reward_distri = random.uniform(0, reward_distri) if reward_distri is not None else random.random()
        self._reward = reward if reward is not None else 10
        self._played = 0
        self._last_played = 0
        self._reward_times = 0
        self._total_rewards: Union[float, int] = 0

    def play(self) -> Union[int, float, None]:
        epsilon = random.random()
        self._played += 1
        self._last_played = 1

        if epsilon <= self._reward_distri:
            reward = self._reward
            self._reward_times += 1
            self._total_rewards += reward
        else:
            reward = None

        return reward

    def not_played(self) -> None:
        self._last_played = 0

    def evolve_reward_distri(self, coef: float=1, diff: float=0):
        """
        Evolve the reward distri according to the coef and diff
        :param coef:
        :param diff:
        :return:
        """
        self._reward_distri = self._reward_distri * coef + diff
        # if self._reward_distri > 1:
        #     self._reward_distri = 1
        # if self._reward_distri < 0:
        #     self._reward_distri = 0

    def get_state(self) -> dict:
        state = {"played": self._played, "reward_times": self._reward_times, "total_rewards": self._total_rewards,
                 "last_played": self._last_played, }

        return state

    def get_reward_distri(self) -> float:
        return self._reward_distri

    def display(self) -> None:
        print(f"id: {self.id}")
        print(f"reward_distri: {self._reward_distri}")
        # print(f"reward: {self._reward}")
        print(f"played: {self._played}")
        print(f"reward times: {self._reward_times}")
        # print(f"total_rewards: {self._total_rewards}")
