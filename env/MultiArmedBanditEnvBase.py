import logging
import copy
import os
import csv

import numpy as np
import itertools
from typing import Callable, Iterable, Union, Collection, Optional
# from statsmodels.stats.proportion import proportion_confint

from slot_machine.MultiArmedBandit import MultiArmedBandit
from env.abstract.Environment import AbstractEnvironment
from agent.result_recorder.TestResult import TestResult
from agent.result_recorder.SingleResult import SingleResult
from agent.result_recorder.TrainResult import TrainResult


class PlayHistory:
    def __init__(self):
        self.actions = []
        self.rwd_dist_played_machine = []
        self.rwd_dist_max = []
        self.reward = []

    def __str__(self):
        return ",".join(self.actions)

    def update(self, action, rwd_dist=None, rwd_dist_max=None, reward=None):
        self.actions.append(str(action))
        self.rwd_dist_played_machine.append(rwd_dist)
        self.rwd_dist_max.append(rwd_dist_max)
        self.reward.append(reward)



class MultiArmedBanditEnvBase(AbstractEnvironment):
    def __init__(self, play_nb: int=1, round_nb: int=10, machine_nb: int=1,
                 reward_distris: Union[Callable[..., float],Collection[float],None]=None,
                 reward: Union[int, float, None]=None, evol_co: Union[int, float]=0, evol_diff: Union[int, float]=0,
                 state_keys: Union[list[str], None]=None, state_repeat: int=1, display: bool=False) -> None:
        self.play_nb = play_nb
        self.round_nb = round_nb
        self.horizon = round_nb * play_nb
        self.machine_nb = machine_nb
        self.reward_distris = copy.deepcopy(reward_distris)
        self.reward = reward if reward is not None else 1
        self.evol_co = evol_co
        self.evol_diff = evol_diff
        self.machines = MultiArmedBandit(
            machine_nb=machine_nb, reward_distris=self.reward_distris, reward=reward,
            evol_co=evol_co, evol_diff=evol_diff)
        self.history = PlayHistory()
        self.baseline_history: dict= {}
        self.display = display

        actions = [i for i in itertools.combinations([j for j in range(machine_nb)], 1)]
        # actions = [(i) for i in range(machine_nb)]
        self.action_space = len(actions)
        self.action_table = {i: actions[i] for i in range(len(actions))}

        self.state_keys = ["played", "reward_times"] if state_keys is None else state_keys

        assert type(state_repeat) is int and state_repeat > 0
        self.state_repeat = state_repeat
        self.state_size = len(self.state_keys) * state_repeat * machine_nb

# self.state_space = (too complicated but not useful)
    def env_snap(self, keys: Union[list[str], None]=None) -> dict:
        dict_env_snap_ = vars(self)
        dict_env_snap_["rwd_dists"] = self.get_rwd_dist()
        if keys is None:
            keys = list(dict_env_snap_.keys())
            keys_to_remove = ["machines", "history", "baseline_history"]
            keys = list(set(keys) - set(keys_to_remove))
        dict_env_snap = {key: dict_env_snap_[key] for key in keys}
        return dict_env_snap

    def reset_horizon(self) -> None:
        self.horizon = self.round_nb * self.play_nb

    def get_state_keys_len(self) -> int:
        return len(self.state_keys)

    def convert_state(self, raw_states: list[dict], extra_info: Optional[list]=None) -> np.ndarray:
        # extra_info should be a list of number (int or float)
        states = [] if extra_info is None else extra_info
        for raw_state in raw_states:
            state = [
                # raw_state["played"],
                # raw_state["reward_times"],
                # (raw_state["reward_times"] + 1) / (raw_state["played"] + 1) if raw_state["played"] > 0 else 0,
                # raw_state["reward_times"] / raw_state["played"] if raw_state["played"] > 0 else -1,
                # raw_state["played"] / self.round_nb,
                # raw_state["total_rewards"],
                raw_state[key] for key in self.state_keys
            ]
            state = state * self.state_repeat
            states = states + state

        return np.array(states)

    # def convert_state(self, raw_states, extra_info=None):
    #     # extra_info should be a list of number (int or float)
    #     # states = [] if extra_info is None else extra_info
    #     states = []
    #     for raw_state in raw_states:
    #         state = [
    #             raw_state["played"],
    #             raw_state["reward_times"],
    #             # (raw_state["reward_times"] + 1) / (raw_state["played"] + 1) if raw_state["played"] > 0 else 0,
    #             # raw_state["reward_times"] / raw_state["played"] if raw_state["played"] > 0 else -1,
    #             # raw_state["played"] / (self.round_nb - self.horizon) if (self.round_nb != self.horizon) else 0,
    #             # raw_state["played"] / self.round_nb,
    #             # (self.round_nb - self.horizon) / self.round_nb,
    #             # raw_state["total_rewards"],
    #             # Lower bound of confidence interval
    #             # proportion_confint(
    #             #     count=raw_state["reward_times"], nobs=raw_state["played"]
    #             # )[0] if raw_state["played"] > 0 else -1,
    #             # Upper bound of confidence interval
    #             # proportion_confint(
    #             #     count=raw_state["reward_times"], nobs=raw_state["played"]
    #             # )[1] if raw_state["played"] > 0 else -1,
    #         ]
    #         states.append(state)
    #
    #     return np.array(states)

    def reset(self) -> np.ndarray:
        self.machines = MultiArmedBandit(
            machine_nb=self.machine_nb,
            reward_distris=self.reward_distris,
            reward=self.reward,
            evol_co=self.evol_co,
            evol_diff=self.evol_diff,
        )
        self.reset_horizon()
        self.history = PlayHistory()
        return self.get_state()

    def get_machines(self) -> MultiArmedBandit:
        return self.machines

    def get_rwd_dist(self) -> list[float]:
        return self.machines.get_reward_distris()

    def reset_with_machines(self, machines: MultiArmedBandit) -> np.ndarray:
        self.machines = machines
        self.reset_horizon()
        self.history = PlayHistory()
        return self.get_state()

    def step(self, action: Union[int, np.integer, None], display: Union[bool, None]=None) -> (
            tuple)[np.ndarray, float, bool]:
        reward: Union[int, float] = 0

        if action is None:
            action_: set = set()
        else:
            action_ = set(self.action_table[int(action)])


        for i in action_:
            assert 0 <= i < self.machine_nb

        # if display:
        if self.display:
            print(f"Playing machine: {action_}")
            # pass

        step_record = []

        play_results = self.machines.play(action_)
        self.horizon -= 1
        done = True if self.horizon <= 0 else False
        # done = False

        for played_machine in action_:
            machine_reward = play_results[played_machine] if play_results[played_machine] is not None else - self.reward
            assert machine_reward is not None
            reward += machine_reward
            step_record.append(
                f"{played_machine}*" if play_results[played_machine] is not None else f"{played_machine}")

        self.history.update(
            action_,
            self.machines.get_ind_reward_distri(action_),
            self.machines.get_max_ind_reward_distri(),
            reward=reward
        )

        # self.history.append(f"{''.join(step_record)}")
        # if self.horizon % self.play_nb == 0:
        #     self.history.append("|")
        state_output = self.get_state()

        # if display:
        if self.display:
            # self.render()
            if done:
                self.display_history()

        return state_output, reward, done

    def get_state(self) -> np.ndarray:
        states = self.machines.get_state()
        global_info = {
            "horizon": self.horizon,
            "round_nb": self.round_nb,
            "total_played": self.round_nb - self.horizon,
        }
        states = [state | global_info for state in states]

        state_output = self.convert_state(states)
        return state_output

    def get_raw_state(self) -> list[dict]:
        """
        Return output of states as list of dict
        :return:
        """
        return self.machines.get_state()

    def render(self) -> None:
        print(f"No.{self.round_nb - self.horizon / self.play_nb} round, {self.round_nb} rounds in total, "
              f"Horizon: {self.horizon / self.play_nb}")
        self.machines.display()

    def get_max_reward_distris_keys(self, reverse: bool=False) -> np.ndarray:
        # If reverse = False: return the machine with min reward distri
        # Attention: when reward distris are equal, the function may choose some of them randomly
        reward_distri = np.array(self.machines.get_reward_distris())
        key_maximum = np.where(reward_distri == reward_distri.max())[0].tolist() if not reverse \
            else np.where(reward_distri == reward_distri.min())[0].tolist()
        if len(key_maximum) > self.play_nb:
            key_max_distris = key_maximum
        elif not reverse:
            key_max_distris = np.argsort(self.machines.get_reward_distris())[-self.play_nb:].tolist()
            key_max_distris.reverse()
        else:
            key_max_distris = np.argsort(self.machines.get_reward_distris())[:self.play_nb].tolist()
        return key_max_distris

    def display_history(self) -> None:
        logging.info(f"Max distris on: {self.get_max_reward_distris_keys()}, reward "
                     f"distris:{np.around(self.machines.get_reward_distris(),decimals=3)}, "
                     f"History:{self.history}")
        print(f"Max distris on: {self.get_max_reward_distris_keys()}, reward "
              f"distris:{np.around(self.machines.get_reward_distris(),decimals=3)}, History:{self.history}")

    # def get_state_space(self):
    #     return self.state_space

    def get_state_size(self) -> int:
        return self.state_size

    def get_action_space(self) -> int:
        return self.action_space

    def judge_win(self, done: bool, total_reward: Union[int, float]) -> bool:
        # TODO: how to define a expected rewards?
        return True

    def get_play_history(self) -> PlayHistory:
        return self.history

    def export_history_to_csv(self, train_result: TrainResult, path: str) -> None:

        """
        !Only works when play_nb == 1

        :param train_result:
        :param path:
        :param env_info_key:
        :return:
        """
        env_info_key = ["round_nb", ]
        file_exist = os.path.isfile(path)
        fieldnames = [
            "agent_name",
            "epoch",
            # "action",
            "rwd_dist",
            "rwd_dist_max",
            # "rwd_dist_regret",
            "reward",
        ]

        if not file_exist:
            f = open(path, 'w', newline='')
            fieldnames = fieldnames + env_info_key
        else:
            f = open(path, 'r')
            dict_reader = csv.DictReader(f)
            fieldnames = list(
                dict_reader.fieldnames if dict_reader.fieldnames is not None else fieldnames + env_info_key)
            f.close()
            f = open(path, 'a', newline='')

        env_info_dict = {key: vars(self)[key] for key in env_info_key}
        rows = [
            {"agent_name": train_result.get_summary_dict()["agent_name"],
             "round_nb": env_info_dict["round_nb"],
             "epoch": key, "rwd_dist": rwd_dist,
             "rwd_dist_max": rwd_dist_max, "reward": reward}
            for key, values in train_result.get_history().items()
            for rwd_dist, rwd_dist_max, reward in
            zip(values.rwd_dist_played_machine, values.rwd_dist_max, values.reward)]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exist:
            writer.writeheader()
        writer.writerows(rows)

        f.close()

    def export_q_values_to_csv(self, train_result: TrainResult, path: str) -> None:
        file_exist = os.path.isfile(path)
        env_info_key = ["rwd_dists", "round_nb"]
        fieldnames = [
            "agent_name",
            "epoch",
            # "action",
            # "rwd_dists",
            "q_values"
            # "rwd_dist_regret"
        ]

        if not file_exist:
            f = open(path, 'w', newline='')

            fieldnames = fieldnames + env_info_key
        else:
            f = open(path, 'r')
            dict_reader = csv.DictReader(f)
            fieldnames = list(
                dict_reader.fieldnames if dict_reader.fieldnames is not None else fieldnames + env_info_key)
            f.close()
            f = open(path, 'a', newline='')

        rows = [
            {"agent_name": train_result.get_summary_dict()["agent_name"],
             "rwd_dists": train_result.get_env_snaps()[key]["rwd_dists"],
             "round_nb": train_result.get_env_snaps()[key]["round_nb"],
             "epoch": key, "q_values": value}
            for key, values in train_result.get_q_values().items() if values is not None
            for value in values]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exist:
            writer.writeheader()
        writer.writerows(rows)

        f.close()
