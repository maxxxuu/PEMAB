import numpy as np
import torch
import random
import copy
import logging
from tqdm import tqdm

from collections import deque

from agent.model.DoubleDQN import DoubleDQN

class SingleMachineDQN(DoubleDQN):
    def __init__(self, machine_nb, *args, **kwargs):
        # The in_dim and out_dim should be the dim of each machine
        # The model should be the model predicting Q value of one machine
        super().__init__(*args, **kwargs)
        self.machine_nb = machine_nb

    def choose_action(self, state, epsilon=0.0, display=False):

        if random.random() < epsilon:
            action = np.random.randint(0, self.machine_nb)
            if display:
                print(f"Random, Action: {action}")
            # self.temp_q = np.empty((self.input_shape[0]))
            # self.temp_q.fill(np.nan)
            self.temp_q = {}
        else:
            qval = self.torch_to_numpy(self.model(state))
            action = np.argmax(qval)
            if display:
                print(f"Q_values: {qval}, Action: {action}")
            # self.temp_q = qval.squeeze()
            temp_q = qval.squeeze().tolist()
            self.temp_q = {f"q_{i}": temp_q[i] for i in range(len(temp_q))}

        return action

    def organise_experience(self, experience):
        if self.input_shape == (1, self.in_dim):
            state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in experience]).to(self.device)
            state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in experience]).to(self.device)
        else:
            state1_batch = torch.stack([s1[a] for (s1, a, r, s2, d) in experience]).to(self.device)
            state2_batch = torch.stack([s2[a] for (s1, a, r, s2, d) in experience]).to(self.device)

        action_batch = torch.Tensor([a for (s1, a, r, s2, d) in experience]).to(self.device)
        reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in experience]).to(self.device)

        done_batch = torch.Tensor([d for (s1, a, r, s2, d) in experience]).to(self.device)
        return state1_batch, state2_batch, action_batch, reward_batch, done_batch

    def update_model(self, minibatch, sync_count, *args, **kwargs):
        state1_batch, state2_batch, action_batch, reward_batch, done_batch = self.organise_experience(minibatch)

        Q1 = self.model(state1_batch)
        with torch.no_grad():
            Q2 = self.target_model(state2_batch)

        # For X and Y, squeeze() is added to handle the case of 2D input
        # Y = reward_batch + self.gamma * ((1 - done_batch) * torch.max(Q2, dim=1)[0])
        Y = reward_batch + self.gamma * ((1 - done_batch) * torch.max(Q2, dim=1)[0].squeeze())
        # X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
        X = Q1.squeeze()
        loss = self.loss_fn(X, Y.detach())

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)
        self.optimizer.step()
        self.update_counter += 1

        if self.update_counter % self.sync_freq == 0:
            self.update_target_model()
            print(f"Loss:{loss.item():.3f}")

        return loss