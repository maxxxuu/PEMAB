import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import copy
import random
import logging
from collections import deque
from tqdm import tqdm

from agent.abstract.AbstractAlgo import AbstractAlgo
from agent.result_recorder.TestResult import TestResult
from agent.result_recorder.SingleResult import SingleResult
from abc import ABC


class BaseAlgo(AbstractAlgo, ABC):
    def __init__(self, in_dim=1, out_dim=1, input_shape=None, *args, **kwargs):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.input_shape = (1, self.in_dim) if input_shape is None else input_shape
        super().__init__(*args, **kwargs)

    def fit(self, env):
        pass

    def init_single_test(self):
        pass

    def single_test(self, env, init_state=None, max_moves=50, display=True, noise=True):
        # i = 0
        result = SingleResult()
        # if not baseline:
        #     state_ = env.reset()
        state_ = init_state if init_state is not None else env.reset()
        finish = False
        moves_count = 0
        # rewards = []
        done = False
        self.init_single_test()
        # if display:
        #     env.render()
        if display:
            logging.info(f"Model name:{type(self).__name__}")
            print(f"Model name:{type(self).__name__}")

        while not finish:

            action_ = self.choose_action(state_, env)

            state_, reward, done, *_ = env.step(action_, display)
            moves_count += 1
            # state_ = state_.reshape(1, self.in_dim)

            # rewards.append(reward)
            result.update_reward(reward)
            # if display:
            #     env.render()
            if done or moves_count >= max_moves:
                finish = True
                moves_count = 0

        total_reward = result.get_total_reward()
        if display:
            print(f"Total rewards: {total_reward}")

        # TODO: A better judge of win
        if env.judge_win(done, result.get_total_reward()):
            result.judged_win()
        result.update_history(env.get_play_history())
        return result

    def multi_test(self, env, max_games=1000, max_moves=50, display=True, noise=True):
        # wins = 0
        # total_rewards = 0
        algo_result = TestResult(type(self).__name__)
        for i in range(max_games):
            if display:
                print(f"Game # {i}:")
            test_result = self.single_test(env, max_moves, display, noise)
            algo_result.update_from_single_result(test_result)
            # if test_result[0]:
            #     wins += 1
            # total_rewards += test_result[1]

        win_perc, average_reward = algo_result.get_summary()
        algo_result.log(display=display)

        # win_perc = float(wins) / float(max_games)
        # average_reward = float(total_rewards) / float(max_games)
        #
        # logging.info(f"Games played: {max_games}, # of wins: {wins}")
        # logging.info(f"Win percentage: {100.0 * win_perc}%")
        # logging.info(f"Average reward :{average_reward}")
        # if display:
        #     print(f"Games played: {max_games}, # of wins: {wins}")
        #     print(f"Win percentage: {100.0 * win_perc}%")
        #     print(f"Average reward :{average_reward}")

        return algo_result
