import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from abc import ABC
import copy
import random
import logging
from collections import deque
from typing import Deque, Optional
from tqdm import tqdm

from agent.abstract.AbstractModel import AbstractModel
from env.abstract.Environment import AbstractEnvironment
from agent.result_recorder.TrainResult import TrainResult



class BaseModel(AbstractModel, nn.Module, ABC):
    def __init__(
            self, load_path=None, in_dim=1, out_dim=1, model=None, learning_rate=3e-2, gamma=0.9, input_shape=None,
            weight_decay=0, memory_size=1000, continous_train=False, max_norm=10,
            *args, **kwargs):

        super().__init__(*args, **kwargs)
        nn.Module.__init__(self)

        if load_path is None:
            if model is not None:
                self.model = copy.deepcopy(model)
            else:
                layers = [
                    nn.Linear(in_dim, 150),
                    nn.ReLU(),
                    nn.Linear(150, 100),
                    nn.ReLU(),
                    nn.Linear(100, out_dim),
                ]
                self.model = nn.Sequential(*layers)

        else:
            self.import_model(load_path)

        # if load_path is None:
        #     if model is not None:
        #         self.model = copy.deepcopy(model)
        #     else:
        #         layers = [
        #             nn.Linear(in_dim, 150),
        #             nn.ReLU(),
        #             nn.Linear(150, 100),
        #             nn.ReLU(),
        #             nn.Linear(100, out_dim),
        #         ]
        #         self.model = nn.Sequential(*layers)
        #
        # else:
        #     self.import_model(load_path)

        self.in_dim = in_dim
        self.input_shape = (1, self.in_dim) if input_shape is None else input_shape
        self.out_dim = out_dim

        # TODO: Check if Apple Chip is compatible
        # self.device = torch.device('mps' if torch.backends.mps.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.model.to(self.device)

        # TODO: Check if some of param should by given as input of __init__()
        self.loss_fn = nn.MSELoss()
        self.learning_rate = learning_rate
        self.optimizer: optim.Optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=weight_decay)
        self.update_counter = 0

        self.gamma = gamma
        self.epsilon = 1.0
        self.memory_size = memory_size
        self.continous_train = continous_train
        self.set_experience_memory()

        # For clip
        self.max_norm = max_norm

    def set_experience_memory(self):
        self.experiences: Deque[tuple] = deque(maxlen=self.memory_size)

    def choose_action(self, state, *args, **kwargs):

        action_ = np.argmax(self.torch_to_numpy(self.model(state)))

        return action_

    @staticmethod
    def add_noise(array, scale=1000):
        """
        Add a white noise to the input array
        :param array: array need noise
        :param scale: the scale of noise. Arrange of the noise is [0, 1/scale)
        :return: array with a white noise
        """
        return array + np.random.rand(*array.shape) / float(scale)

    def numpy_to_torch(self, array):
        return torch.from_numpy(array).float().to(self.device)

    @staticmethod
    def torch_to_numpy(array):
        return array.cpu().data.numpy()

    # def train_single_epoch(self, env, epoch_nb=0, display=True, noise=True):
    #     finish = False
    #     moves_count = 0
    #     rewards = []
    #     state1_ = env.get_state()
    #     if noise:
    #         state1_ = self.add_noise(state1_)
    #     state1 = self.numpy_to_torch(state1_)
    #     if display:
    #         print(f"Epoch # {epoch_nb}:")
    def get_done_experience(self):
        return [experience_ for experience_ in self.experiences if experience_[4]]

    def organise_experience(self, experience):
        if self.input_shape == (1, self.in_dim):
            state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in experience]).to(self.device)
            state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in experience]).to(self.device)
        else:
            state1_batch = torch.stack([s1 for (s1, a, r, s2, d) in experience]).to(self.device)
            state2_batch = torch.stack([s2 for (s1, a, r, s2, d) in experience]).to(self.device)

        action_batch = torch.Tensor([a for (s1, a, r, s2, d) in experience]).to(self.device)
        reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in experience]).to(self.device)

        done_batch = torch.Tensor([d for (s1, a, r, s2, d) in experience]).to(self.device)
        return state1_batch, state2_batch, action_batch, reward_batch, done_batch

    def learn(self, env: AbstractEnvironment, epsilon: float=1.0, epochs: int=1000,
              max_moves: int=50, batch_size: int=200, display: bool=True, noise: bool=True):
        losses = []

        # moves_count = 0
        sync_count = 0

        self.model.train()

        for i in tqdm(range(epochs)):
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
                qval = self.model(state1)
                qval_ = self.torch_to_numpy(qval)

                action_ = self.choose_action(qval_, epsilon)

                print(f"Q_values: {qval_}, Action: {action_}")

                state2_, reward, done, *_ = env.step(action_, display)
                print(f"Step: {moves_count}, Reward: {reward}")
                # clear_output(wait=True)

                # state2_ = state2_.reshape(1, self.in_dim)
                state2_ = state2_.reshape(*self.input_shape)
                if noise:
                    state2_ = state2_ + np.random.rand(1, self.in_dim) / 1000.0
                # state2 = torch.from_numpy(state2_).float().to(self.device)
                state2 = self.numpy_to_torch(state2_)
                experience = (state1, action_, reward, state2, done)
                #                 print(state1.size())
                self.experiences.append(experience)
                rewards.append(reward)
                state1 = state2

                if len(self.experiences) > batch_size or (self.continous_train and len(self.experiences) > 2):
                    minibatch = random.sample(self.experiences, batch_size) \
                        if len(self.experiences) > batch_size else self.experiences
                    state1_batch, state2_batch, action_batch, reward_batch, done_batch = \
                        self.organise_experience(minibatch)

                    #                     print(state1_batch.size())
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
                    losses.append(loss.item())

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)
                    self.optimizer.step()
                    self.update_counter += 1

                if done or moves_count >= max_moves:
                    finish = True
                    moves_count = 0

                # Can be added to comments
                if done:
                    print("DONE!")

            total_reward = sum(rewards)
            if display:
                print(f"Total rewards: {total_reward}")

            # TODO: Compare performance of changing epsilon and fixed epsilon
            if epsilon > 0.1:
                epsilon -= (1 / epochs)

            # TODO: Shall we clear experiences when finish an epoch?

        return np.array(losses)

    def export_model(self, path="agent.pt"):
        torch.save(self.model, path)
        logging.info(f'Model saved at: {path}')

    def import_model(self, path="agent.pt"):
        self.model = torch.load(path)

    def export_model_optimizer(self, path="checkpoint.pt"):
        torch.save(
            (self.model.state_dict(), self.optimizer.state_dict()), path)
        logging.info(f'Model and optimizer saved at: {path}')

    def import_model_optimizer(self, path="checkpoint.pt"):
        model_state, optimizer_state = torch.load(path)
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)
        logging.info(f'Model and optimizer loaded from: {path}')

