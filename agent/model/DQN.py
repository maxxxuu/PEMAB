import numpy as np
import torch
import random
import copy
import logging
from tqdm import tqdm
import os

import torch.nn as nn
import torch.optim as optim

from collections import deque
from IPython.display import clear_output    # type: ignore
from utils.utils import count_parameters
from agent.model.base_model.BaseModel import BaseModel
from agent.result_recorder.TestResult import TestResult
from agent.result_recorder.SingleResult import SingleResult
from agent.result_recorder.TrainResult import TrainResult


# def baseline_test(env, baseline_models, max_moves=50, display=True, noise=True):
#     # temp_env = copy.deepcopy(env)
#     baseline_results = {}
#     for baseline_model in baseline_models:
#         baseline_results[type(baseline_model).__name__] = baseline_model.single_test(
#             env, max_moves=max_moves, display=display, noise=noise, baseline=True)
#
#     return baseline_results


class DQN(BaseModel):
    def __init__(self, batch_size=100, sync_freq=800, *args, **kwargs):
        self.batch_size = batch_size
        self.sync_freq = sync_freq
        self.temp_q = None
        super().__init__(*args, **kwargs)
        # self.set_experience_memory()

    def set_experience_memory(self):
        self.experiences = deque(maxlen=self.memory_size if self.memory_size > self.batch_size else self.batch_size * 2)

    def choose_action(self, state, epsilon=0.0, display=False):

        if random.random() < epsilon:
            action = np.random.randint(0, self.out_dim)
            if display:
                print(f"Random, Action: {action}")
            # self.temp_q = np.empty(self.out_dim)
            # self.temp_q.fill(np.nan)
            self.temp_q = {}
        else:
            with torch.no_grad():
                qval = self.torch_to_numpy(self.model(state))
            action = np.argmax(qval)
            if display:
                print(f"State: {state}, Q_values: {qval}, Action: {action}")
            # self.temp_q = qval.squeeze()
            temp_q = qval.squeeze().tolist()
            # if not hasattr(temp_q, "__iter__"):
            #     temp_q = [temp_q]
            self.temp_q = {f"q_{i}": temp_q[i] for i in range(len(temp_q))}

        return action

    def update_model(self, minibatch, *args, **kwargs):
        state1_batch, state2_batch, action_batch, reward_batch, done_batch = self.organise_experience(minibatch)

        Q1 = self.model(state1_batch)
        with torch.no_grad():
            Q2 = self.model(state2_batch)

        # For X and Y, squeeze() is added to handle the case of 2D input
        # Y = reward_batch + self.gamma * ((1 - done_batch) * torch.max(Q2, dim=1)[0])
        Y = reward_batch + self.gamma * ((1 - done_batch) * torch.max(Q2, dim=1)[0].squeeze())
        # X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
        X = Q1.squeeze().gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
        loss = self.loss_fn(X, Y.detach())

        #                     print(i, loss.item())

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)
        self.optimizer.step()
        self.update_counter += 1

        return loss

    def learn(self, env, epsilon=1, epochs=1000, max_moves=50, display=True, noise=True):
        train_result = TrainResult(agent_name=self.name)
        # experiences = deque(maxlen=self.memory_size if self.memory_size > self.batch_size else self.batch_size*2)
        # experiences = []
        # moves_count = 0
        # TODO: Is sync_freq and sync_count necessary here?
        sync_count = 0
        # TODO: think if 0 is the right value for epsilon when continous_train==1
        # Epsilon decay happens each time the model is back-propagated
        if self.continous_train:
            epsilon_decay = 0.99
        else:
            epsilon_decay = 1

        self.model.train()

        for i in tqdm(range(epochs)):
            q_values = []
            states = []
            # TODO: Make sure reset returns the state directly
            # state1_ = env.reset().reshape(1, self.in_dim)
            state1_ = env.reset().reshape(*self.input_shape)
            if noise:
                # state1_ = state1_ + np.random.rand(1, self.in_dim) / 100.0
                state1_ = self.add_noise(state1_)
            state1 = self.numpy_to_torch(state1_)
            finish = False
            moves_count = 0
            rewards = []

            if display:
                print(f"Epoch # {i}:")

            while not finish:
                sync_count += 1
                moves_count += 1

                action_ = self.choose_action(state1, epsilon, display=display)
                q_values.append(copy.deepcopy(self.temp_q))
                states.append(copy.deepcopy(self.torch_to_numpy(state1).tolist()))

                state2_, reward, done, *_ = env.step(action_, display)
                if display:
                    print(f"Step: {moves_count}, Reward: {reward}")
                # clear_output(wait=True)

                # state2_ = state2_.reshape(1, self.in_dim)
                state2_ = state2_.reshape(*self.input_shape)
                if noise:
                    state2_ = self.add_noise(state2_)
                state2 = self.numpy_to_torch(state2_)
                experience = (state1, action_, reward, state2, done)
                #                 print(state1.size())
                self.experiences.append(experience)
                # if len(experiences) <= self.memory_size:
                #     experiences.append(experience)
                # else:
                #     pass
                rewards.append(reward)
                state1 = state2

                if len(self.experiences) > self.batch_size or (self.continous_train and len(self.experiences) > 2):
                    # "Done step" reinforce
                    # !!ATTENTION!! with this reinforce, there will be 2 consecutive update every time
                    # if random.random() < epsilon:
                    #     minibatch_done = self.get_done_experience()
                    #     loss = self.update_model(minibatch_done, sync_count, self.sync_freq)
                    #     self.update_counter -= 1

                    minibatch_done = self.get_done_experience()
                    if minibatch_done:
                        loss = self.update_model(minibatch_done, sync_count, self.sync_freq)
                        self.update_counter -= 1

                    minibatch = random.sample(self.experiences, self.batch_size) \
                        if len(self.experiences) > self.batch_size else self.experiences
                    loss = self.update_model(minibatch, sync_count, self.sync_freq)
                    train_result.update(loss.item(), self.update_counter)

                    epsilon = epsilon * epsilon_decay

                if done or moves_count >= max_moves:
                    finish = True
                    moves_count = 0
                    train_result.update_play_history(epochs=i, play_history=env.get_play_history())
                    train_result.update_q_values(epochs=i, q_values=q_values)
                    train_result.update_env_snaps(epochs=i, env_snap=env.env_snap())

                # Can be added to comments
                if done and display:
                    print("DONE!")

            total_reward = sum(rewards)
            train_result.update_total_reward(total_reward)
            if display:
                print(f"Total rewards: {total_reward}")

            # TODO: Compare performance of changing epsilon and fixed epsilon
            # if epsilon > 0.1:
            #     epsilon -= (1 / epochs)
            epsilon = epsilon * 0.9


            # TODO: Shall we clear experiences when finish an epoch?

        return train_result

    def single_test(self, env, init_state, max_moves=50, display=True, noise=True):
        result = SingleResult()
        i = 0
        q_values = []
        states = []
        state_ = init_state
        # if baseline_models is not None:
        #     temp_env = copy.deepcopy(env)
        #     baseline_results = baseline_test(
        #         temp_env, baseline_models, max_moves=max_moves, display=display, noise=noise)
        # else:
        #     baseline_results = None
        state_ = state_.reshape(*self.input_shape)
        state = self.numpy_to_torch(state_)

        result.update_env_snap(env_snap=env.env_snap())
        finish = False
        moves_count = 0
        # rewards = []
        done = False
        # if display:
        #     env.render()

        self.model.eval()

        if display:
            logging.info(f"Model name:{type(self).__name__}")
            print(f"Model name:{type(self).__name__}")

        while not finish:
            action_ = self.choose_action(state, display=display)
            q_values.append(copy.deepcopy(self.temp_q))
            states.append(copy.deepcopy(self.torch_to_numpy(state).tolist()))
            # result.update_history(action_)
            state_, reward, done, *_ = env.step(action_, display)
            moves_count += 1
            # state_ = state_.reshape(1, self.in_dim)
            state_ = state_.reshape(*self.input_shape)
            #             print(f"Q_values: {qval_}, Action: {action_}")
            #             print(f"Step: {moves_count}, Reward: {reward}")
            clear_output(wait=True)

            if noise:
                # state_ = state_ + np.random.rand(1, self.in_dim) / 100.0
                state_ = self.add_noise(state_)
            state = self.numpy_to_torch(state_)

            # rewards.append(reward)
            result.update_reward(reward)
            # if display:
            #     env.render()
            if done or moves_count >= max_moves:
                finish = True
                moves_count = 0

        # total_reward = sum(rewards)
        if display:
            print(f"Total rewards: {result.get_total_reward()}")

        # TODO: A better judge of win
        if env.judge_win(done, result.get_total_reward()):
            result.judged_win()
        result.update_history(env.get_play_history())
        result.update_q_values(q_values)
        result.update_states(states)
        return result

    def multi_test(self, env, max_games=1000, max_moves=50, display=True, noise=True, baseline_models=None):
        """
        This function is going to be discarded. Multi tests should be handled by a test runner

        :param env:
        :param max_games:
        :param max_moves:
        :param display:
        :param noise:
        :param baseline_models:
        :return:
        """

        # wins = 0
        # total_rewards = 0
        model_result = TestResult(type(self).__name__)
        # if baseline_models is not None:
        #     baseline_results = {
        #         type(baseline_model).__name__: TestResult(type(baseline_model).__name__) for baseline_model in
        #         baseline_models}
        # else:
        #     baseline_results = {}
        for i in tqdm(range(max_games)):
            if display:
                print(f"Game # {i}:")
            state_ = env.reset()
            test_result = self.single_test(env, state_, max_moves, display, noise)
            # TODO: Modif when baseline discarded
            model_result.update_from_single_result(test_result)
            # if baseline_models is not None:
            #     for baseline_model in baseline_models:
            #         baseline_results[type(baseline_model).__name__].update(
            #             test_result[2][type(baseline_model).__name__][0],
            #             test_result[2][type(baseline_model).__name__][1])

        win_perc, average_reward = model_result.get_summary()
        model_result.log(display=display)
        # if baseline_models is not None:
        #     for value in baseline_results.values():
        #         value.log(display=display)

        # print(f"Average reward :{average_reward}")
        return model_result

    # def save_model(self, path="agent.pt"):
    #     torch.save(self.model, path)
    #     logging.info(f'Model saved at: {path}')
    #
    # def load_model(self, path="agent.pt"):
    #     self.model = torch.load(path)
    #     # self.in_dim = int(list(self.agent.parameters())[0].shape)
    #     # self.out_dim = int(list(self.agent.parameters())[-1].shape)
