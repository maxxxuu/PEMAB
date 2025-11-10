import logging
import copy

import numpy as np
import itertools

from slot_machine.MultiArmedBandit import MultiArmedBandit
from env.abstract.Environment import AbstractEnvironment

"""
This class needs to be updated after several changes been made in MAB env
"""

class MultiArmedBanditEnvForDis(AbstractEnvironment):
    def __init__(self, play_nb=1, round_nb=10, machine_nb=1, reward_distris=None, reward=None):
        self.play_nb = play_nb
        self.round_nb = round_nb
        self.horizon = round_nb
        self.machine_nb = machine_nb
        self.reward_distris = copy.deepcopy(reward_distris)
        self.reward = reward
        self.machines = MultiArmedBandit(machine_nb=machine_nb, reward_distris=self.reward_distris, reward=reward)
        self.history = []

        actions = [i for i in itertools.combinations([j for j in range(machine_nb)], play_nb)]
        self.action_space = len(actions)
        self.action_table = {i: actions[i] for i in range(len(actions))}
        # for each machine: played, reward_times, for env: horizon, round_nb
        # TODO: check if need to add total round
        self.state_size = 2 * machine_nb

        # self.state_space = (too complicated but not useful)

    def convert_state(self, raw_states, extra_info=None):
        # extra_info should be a list of number (int or float)
        # states = [] if extra_info is None else extra_info
        states = []
        for raw_state in raw_states:
            state = [
                # raw_state["played"],
                # raw_state["reward_times"],
                # (raw_state["reward_times"] + 1) / (raw_state["played"] + 1) if raw_state["played"] > 0 else 0,
                raw_state["reward_times"] / raw_state["played"] if raw_state["played"] > 0 else -1,
                # raw_state["played"] / (self.round_nb - self.horizon) if (self.round_nb != self.horizon) else 0,
                raw_state["played"] / self.round_nb,
                # raw_state["total_rewards"],
            ]
            states = states + state

        return np.array(states)

    def reset(self):
        self.machines = MultiArmedBandit(
            machine_nb=self.machine_nb,
            reward_distris=self.reward_distris,
            reward=self.reward
        )
        self.horizon = self.round_nb
        self.history = []
        return self.get_state()

    def step(self, action=None, display=True):
        reward = 0

        if action is not None:
            action = set(self.action_table[action])
        else:
            action = {}

        for i in action:
            assert 0 <= i < self.machine_nb

        if display:
            # print(f"Playing machine: {action}")
            pass

        step_record = []

        play_results = self.machines.play(action)
        self.horizon -= 1
        done = True if self.horizon <= 0 else False

        for played_machine in action:
            reward += play_results[played_machine] if play_results[played_machine] is not None else -10
            step_record.append(
                f"{played_machine}*" if play_results[played_machine] is not None else f"{played_machine}")

        self.history.append(f"({', '.join(step_record)})")
        state_output = self.get_state()

        if display:
            # self.render()
            if done:
                self.display_history()

        return state_output, reward, done

    def get_state(self):
        states = self.machines.get_state()

        state_output = self.convert_state(
            states, [(float(self.horizon) / float(self.round_nb)), self.round_nb, self.horizon])
        return state_output

    def render(self):
        print(f"No.{self.round_nb - self.horizon} round, {self.round_nb} rounds in total, Horizon: {self.horizon}")
        self.machines.display()

    def get_max_reward_distris(self):
        # Attention: when reward distris are equal, the function may choose some of them randomly
        reward_distri = np.array(self.machines.get_reward_distris())
        key_maximum = np.where(reward_distri == reward_distri.max())[0].tolist()
        if len(key_maximum) > self.play_nb:
            key_max_distris = key_maximum
        else:
            key_max_distris = np.argsort(self.machines.get_reward_distris())[-self.play_nb:].tolist()
            key_max_distris.reverse()
        return key_max_distris

    def display_history(self):
        logging.info(f"Max distris on: {self.get_max_reward_distris()}, reward "
                     f"distris:{self.machines.get_reward_distris()}, History:{', '.join(self.history)}")
        print(f"Max distris on: {self.get_max_reward_distris()}, reward "
              f"distris:{self.machines.get_reward_distris()}, History:{', '.join(self.history)}")

    # def get_state_space(self):
    #     return self.state_space

    def get_state_size(self):
        return self.state_size

    def get_action_space(self):
        return self.action_space

    def judge_win(self, done, total_reward):
        # TODO: how to define a expected rewards?
        return True


