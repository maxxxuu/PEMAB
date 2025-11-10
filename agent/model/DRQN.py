import numpy as np
import torch
import random
import copy
import logging
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim

from agent.model.OneStepRNN import OneStepRNN

from collections import deque


class DRQN(nn.Module):
    def __init__(self, load_path=None, in_dim=1, out_dim=1, hidden_dim=1, rnn_layers=2, model=None, learning_rate=1e-3,
                 gamma=0.9):

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.rnn_layers = rnn_layers

        super().__init__()

        if model is not None:
            self.model = copy.deepcopy(model)
        else:
            # layers = [
            #     OneStepRNN(in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim, rnn_layers=rnn_layers)
            # ]
            # self.agent = nn.Sequential(*layers)
            self.model = OneStepRNN(in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim, rnn_layers=rnn_layers)

        if load_path is not None:
            self.load_model(load_path)

        # TODO: Check if some of param should by given as input of __init__()
        self.loss_fn = nn.MSELoss()
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.gamma = gamma
        self.epsilon = 1.0

    def train(self, env, epsilon=1, epochs=1000, memory_size=1000, batch_size=200,
              max_moves=50, sync_freq=500, display=True, noise=True):
        losses = []
        experiences = deque(maxlen=memory_size)
        # moves_count = 0
        sync_count = 0

        self.model.train()

        for i in tqdm(range(epochs)):
            # TODO: Make sure reset returns the state directly
            state1_ = env.reset().reshape(1, self.in_dim)
            if noise:
                state1_ = state1_ + np.random.rand(1, self.in_dim) / 100.0
            state1 = torch.from_numpy(state1_).float()
            finish = False
            moves_count = 0
            rewards = []
            hidden_state1 = torch.zeros(self.rnn_layers, self.hidden_dim, requires_grad=False)
            self.optimizer.zero_grad()

            if display:
                print(f"Epoch # {i}:")

            while not finish:
                sync_count += 1
                moves_count += 1
                qval, hidden_state2 = self.model(state1, hidden_state1)
                qval_ = qval.data.numpy()

                if random.random() < epsilon:
                    action_ = np.random.randint(0, self.out_dim)
                    if display:
                        print("random")
                else:
                    action_ = np.argmax(qval_)

                state2_, reward, done, *_ = env.step(action_, display)
                if display:
                    print(f"Q_values: {qval_}, Action: {action_}")
                    print(f"Step: {moves_count}, Reward: {reward}")
                # clear_output(wait=True)

                state2_ = state2_.reshape(1, self.in_dim)
                if noise:
                    state2_ = state2_ + np.random.rand(1, self.in_dim) / 100.0
                state2 = torch.from_numpy(state2_).float()
                # qval2, hidden_state3 = self.agent(state2, hidden_state2)
                # with torch.autograd.set_detect_anomaly(True):
                #
                #     # Y = reward + self.gamma * ((1 - done) * torch.max(qval2))
                #     # X = torch.index_select(qval, 1, torch.tensor([action_]))
                #     Y = qval.clone()
                #     Y[-1, action_] = reward + self.gamma * ((1 - done) * torch.max(qval2))
                #     X = qval.clone()
                #     loss = self.loss_fn(X, Y.detach())
                #
                #     loss.backward(retain_graph=True)
                #     losses.append(loss.item())
                #     self.optimizer.step()

                experience = (state1, hidden_state1.clone(), action_, reward, state2, hidden_state2.clone(), done)
                #                 print(state1.size())
                experiences.append(experience)
                rewards.append(reward)
                state1 = state2
                hidden_state1 = hidden_state2

                if len(experiences) > batch_size:
                    with torch.autograd.set_detect_anomaly(True):
                        minibatch = random.sample(experiences, batch_size)
                        state1_batch = torch.stack([s1.clone() for (s1, hs1, a, r, s2, hs2, d) in minibatch])
                        hidden_state1_batch = torch.stack([hs1.detach().clone().requires_grad_() for (s1, hs1, a, r, s2, hs2, d) in minibatch])
                        action_batch = torch.Tensor([a for (s1, hs1, a, r, s2, hs2, d) in minibatch])
                        reward_batch = torch.Tensor([r for (s1, hs1, a, r, s2, hs2, d) in minibatch])
                        state2_batch = torch.stack([s2.clone() for (s1, hs1, a, r, s2, hs2, d) in minibatch])
                        hidden_state2_batch = torch.stack([hs2.detach().clone() for (s1, hs1, a, r, s2, hs2, d) in minibatch])
                        done_batch = torch.Tensor([d for (s1, hs1, a, r, s2, hs2, d) in minibatch])

                        #                     print(state1_batch.size())
                        Q1, HS1 = self.model(state1_batch, torch.transpose(hidden_state1_batch, 0, 1))
                        with torch.no_grad():
                            # Q2, HS2 = self.agent(state2_batch, torch.transpose(hidden_state2_batch, 0, 1))
                            Q2, HS2 = self.model(state2_batch, HS1)

                        Y = reward_batch + self.gamma * ((1 - done_batch) * torch.max(Q2, dim=1)[0])
                        X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
                        loss = self.loss_fn(X, Y.detach())

                        #                     print(i, loss.item())

                        self.optimizer.zero_grad()
                        loss.backward()
                        losses.append(loss.item())
                        self.optimizer.step()
                        self.update_counter += 1

                if done or moves_count >= max_moves:
                    finish = True
                    moves_count = 0

                # Can be add to comments
                if done:
                    if display:
                        print("DONE!")

            total_reward = sum(rewards)
            if display:
                print(f"Total rewards: {total_reward}")

            # TODO: Compare performance of changing epsilon and fixed epsilon
            if epsilon > 0.1:
                epsilon -= (1 / epochs)

            # TODO: Shall we clear experiences when finish an epoch?

        return np.array(losses)

    def test_model(self, env, max_moves=50, display=True, noise=True):
        i = 0
        state_ = env.reset().reshape(1, self.in_dim)
        state = torch.from_numpy(state_).float()
        finish = False
        moves_count = 0
        rewards = []
        done = False
        hidden_state = torch.zeros(self.rnn_layers, self.hidden_dim)
        # if display:
        #     env.render()
        self.model.eval()
        while not finish:
            qval, hidden_state = self.model(state, hidden_state)
            qval_ = qval.data.numpy()
            action_ = np.argmax(qval_)

            state_, reward, done, *_ = env.step(action_, display)
            moves_count += 1
            state_ = state_.reshape(1, self.in_dim)
            #             print(f"Q_values: {qval_}, Action: {action_}")
            #             print(f"Step: {moves_count}, Reward: {reward}")
            # clear_output(wait=True)

            if noise:
                state_ = state_ + np.random.rand(1, self.in_dim) / 100.0
            state = torch.from_numpy(state_).float()
            rewards.append(reward)
            # if display:
            #     env.render()
            if done or moves_count >= max_moves:
                finish = True
                moves_count = 0

        total_reward = sum(rewards)
        if display:
            print(f"Total rewards: {total_reward}")

        # TODO: A better judge of win
        win = env.judge_win(done, total_reward)
        return win, total_reward

    def test(self, env, max_games=1000, max_moves=50, display=True, noise=True):
        wins = 0
        total_rewards = 0
        for i in tqdm(range(max_games)):
            if display:
                print(f"Game # {i}:")
            test_result = self.test_model(env, max_moves, display, noise)
            if test_result[0]:
                wins += 1
            total_rewards += test_result[1]

        win_perc = float(wins) / float(max_games)
        average_reward = float(total_rewards) / float(max_games)

        logging.info(f"Games played: {max_games}, # of wins: {wins}")
        logging.info(f"Win percentage: {100.0 * win_perc}%")
        logging.info(f"Average reward :{average_reward}")
        if display:
            print(f"Games played: {max_games}, # of wins: {wins}")
            print(f"Win percentage: {100.0 * win_perc}%")
            print(f"Average reward :{average_reward}")

        return win_perc, average_reward

    def save_model(self, path="agent.pt"):
        torch.save(self.model.state_dict(), path)
        logging.info(f'Model saved at: {path}')

    def load_model(self, path="agent.pt"):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)