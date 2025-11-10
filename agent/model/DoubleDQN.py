import numpy as np
import torch
import random
import copy
import logging
from tqdm import tqdm

from collections import deque

from agent.model.DQN import DQN


class DoubleDQN(DQN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_target_model()

    def init_target_model(self):
        self.target_model = copy.deepcopy(self.model)
        self.target_model.to(self.device)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def update_model(self, minibatch, sync_count, *args, **kwargs):
        state1_batch, state2_batch, action_batch, reward_batch, done_batch = self.organise_experience(minibatch)

        self.optimizer.zero_grad()
        Q1 = self.model(state1_batch)
        with torch.no_grad():
            Q2 = self.target_model(state2_batch)

        # For X and Y, squeeze() is added to handle the case of 2D input
        # Y = reward_batch + self.gamma * ((1 - done_batch) * torch.max(Q2, dim=1)[0])
        # Y = reward_batch + self.gamma * ((1 - done_batch) * torch.max(Q2, dim=1)[0].squeeze())

        # Double DQN version
        Y = reward_batch + self.gamma * ((1 - done_batch) *
                                         Q2.gather(dim=1, index=torch.argmax(Q1, dim=1, keepdim=True)).squeeze())
        # X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
        X = Q1.squeeze(dim=-1).gather(dim=-1, index=action_batch.long().unsqueeze(dim=1)).squeeze()

        loss = self.loss_fn(X, Y.detach())

        loss.backward()
        if loss > 1000:
            pass

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)
        self.optimizer.step()
        self.update_counter += 1

        if self.update_counter % self.sync_freq == 0:
        # if self.update_counter % self.sync_freq == 0:
            self.update_target_model()
            print(f"Loss:{loss.item():.3f}")

        return loss

    def learn(self, *args, **kwargs):
        self.init_target_model()

        return super().learn(*args, **kwargs)
