import numpy as np
import torch
import random
import copy
import logging
from tqdm import tqdm

from collections import deque
from agent.model.DoubleDQN import DoubleDQN


class SumDQN(DoubleDQN):
    """
    SumDQN is adapted to problems where multiple actions are taken in each step, and the order of actions taken at one
    step causes no difference in result
    """

    def __init__(self, nb_action=1, **kwargs):
        super().__init__(**kwargs)

        self.nb_action = nb_action

    def choose_action(self, qval, epsilon=0):

        if random.random() < epsilon:
            # action_ = np.random.randint(0, self.out_dim, self.nb_action)
            action_ = random.sample(range(self.out_dim), self.nb_action)
            print("random")
        else:
            # action_ = np.argmax(qval)
            action_ = np.argsort(-qval.squeeze())[:self.nb_action].tolist()
        return action_

    def train1(self, env, epsilon=1, epochs=1000, memory_size=1000, batch_size=200,
              max_moves=50, sync_freq=500, display=True, noise=True):
        losses = []
        experiences = deque(maxlen=memory_size)
        # moves_count = 0
        sync_count = 0
        self.target_model = copy.deepcopy(self.model)
        self.target_model.to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        for i in tqdm(range(epochs)):
            # TODO: Make sure reset returns the state directly
            state1_ = env.reset().reshape(*self.input_shape)
            if noise:
                state1_ = state1_ + np.random.rand(*self.input_shape) / 1000.0
            state1 = torch.from_numpy(state1_).float().to(self.device)
            finish = False
            moves_count = 0
            rewards = []

            while not finish:
                sync_count += 1
                moves_count += 1
                qval = self.model(state1)
                qval_ = qval.cpu().data.numpy()

                actions_ = self.choose_action(qval_, epsilon)

                # print(f"Q_values: {qval_}, Action: {action_}")

                step_rewards = []
                for action_ in actions_:
                    state2_, reward, done, *_ = env.step(action_, display)
                    step_rewards.append(reward)
                if display:
                    print(f"Step: {moves_count}, Reward: {step_rewards}")

                state2_ = state2_.reshape(*self.input_shape)
                if noise:
                    state2_ = state2_ + np.random.rand(1, self.in_dim) / 1000.0
                state2 = torch.from_numpy(state2_).float().to(self.device)
                experience = (state1, actions_, step_rewards, state2, done)
                #                 print(state1.size())
                experiences.append(experience)
                rewards.append(sum(step_rewards))
                state1 = state2

                if len(experiences) > batch_size:
                    minibatch = random.sample(experiences, batch_size)
                    if self.input_shape == (1, self.in_dim):
                        state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in minibatch]).to(self.device)
                        state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in minibatch]).to(self.device)
                        action_batch = \
                            torch.Tensor([a for (s1, a, r, s2, d) in minibatch]).to(self.device).unsqueeze(dim=1)
                        reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in minibatch]).to(self.device)
                    else:
                        state1_batch = torch.stack([s1 for (s1, a, r, s2, d) in minibatch]).to(self.device)
                        state2_batch = torch.stack([s2 for (s1, a, r, s2, d) in minibatch]).to(self.device)
                        action_batch = \
                            torch.stack([torch.Tensor(a) for (s1, a, r, s2, d) in minibatch]).to(self.device)
                        reward_batch = \
                            torch.stack([torch.Tensor(r) for (s1, a, r, s2, d) in minibatch]).to(self.device)

                    done_batch = torch.Tensor([d for (s1, a, r, s2, d) in minibatch]).to(self.device)

                    #                     print(state1_batch.size())
                    Q1 = self.model(state1_batch)
                    with torch.no_grad():
                        Q2 = self.model(state2_batch)

                    # For X and Y, squeeze() is added to handle the case of 2D input
                    # Y = reward_batch + self.gamma * ((1 - done_batch) * torch.max(Q2, dim=1)[0])
                    Y = reward_batch.squeeze() + \
                        self.gamma * (
                                (1 - done_batch) * torch.topk(Q2, self.nb_action, dim=1)[0].mean(dim=1).squeeze()
                        ).unsqueeze(dim=1)
                    # X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
                    X = Q1.squeeze().gather(dim=1, index=action_batch.long()).squeeze()
                    loss = self.loss_fn(X, Y.detach())

                    #                     print(i, loss.item())

                    self.optimizer.zero_grad()
                    loss.backward()
                    losses.append(loss.item())
                    self.optimizer.step()
                    self.update_counter += 1

                    if sync_count % sync_freq == 0:
                        self.target_model.load_state_dict(self.model.state_dict())
                #                         print('updated')

                if done or moves_count >= max_moves:
                    finish = True
                    moves_count = 0

                # Can be added to comments
                if done:
                    logging.debug("DONE!")

            total_reward = sum(rewards)
            logging.debug(f"Epoch # {i}, Total rewards: {total_reward}")
            if display:
                print(f"Epoch # {i}, Total rewards: {total_reward}")

            # TODO: Compare performance of changing epsilon and fixed epsilon
            if epsilon > 0.1:
                epsilon -= (1 / epochs)

            # TODO: Shall we clear experiences when finish an epoch?

        return np.array(losses)

    def learn(self, env, epsilon=1, epochs=1000, memory_size=1000, batch_size=200,
              max_moves=50, sync_freq=500, display=True, noise=True):
        losses = []
        experiences = deque(maxlen=memory_size)
        # moves_count = 0
        sync_count = 0

        for i in tqdm(range(epochs)):
            # TODO: Make sure reset returns the state directly
            # state1_ = env.reset().reshape(1, self.in_dim)
            state1_ = env.reset().reshape(*self.input_shape)
            if noise:
                # state1_ = state1_ + np.random.rand(1, self.in_dim) / 100.0
                state1_ = state1_ + np.random.rand(*self.input_shape) / 1000.0
            state1 = torch.from_numpy(state1_).float().to(self.device)
            finish = False
            moves_count = 0
            rewards = []

            if display:
                print(f"Epoch # {i}:")

            while not finish:
                sync_count += 1
                moves_count += 1
                qval = self.model(state1)
                qval_ = qval.cpu().data.numpy()

                actions_ = self.choose_action(qval_, epsilon)

                # print(f"Q_values: {qval_}, Action: {actions_}")

                step_rewards = []
                for action_ in actions_:
                    state2_, reward, done, *_ = env.step(action_, display)
                    step_rewards.append(reward)
                # print(f"Step: {moves_count}, Reward: {step_rewards}")
                # clear_output(wait=True)

                # state2_ = state2_.reshape(1, self.in_dim)
                state2_ = state2_.reshape(*self.input_shape)
                if noise:
                    state2_ = state2_ + np.random.rand(1, self.in_dim) / 1000.0
                state2 = torch.from_numpy(state2_).float().to(self.device)
                experience = (state1, actions_, step_rewards, state2, done)
                #                 print(state1.size())
                experiences.append(experience)
                rewards.append(sum(step_rewards))
                state1 = state2

                if len(experiences) > batch_size:
                    minibatch = random.sample(experiences, batch_size)
                    if self.input_shape == (1, self.in_dim):
                        state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in minibatch]).to(self.device)
                        state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in minibatch]).to(self.device)
                        action_batch = \
                            torch.Tensor([a for (s1, a, r, s2, d) in minibatch]).to(self.device).unsqueeze(dim=1)
                        reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in minibatch]).to(self.device)
                    else:
                        state1_batch = torch.stack([s1 for (s1, a, r, s2, d) in minibatch]).to(self.device)
                        state2_batch = torch.stack([s2 for (s1, a, r, s2, d) in minibatch]).to(self.device)
                        action_batch = \
                            torch.stack([torch.Tensor(a) for (s1, a, r, s2, d) in minibatch]).to(self.device)
                        reward_batch = \
                            torch.stack([torch.Tensor(r) for (s1, a, r, s2, d) in minibatch]).to(self.device)

                    done_batch = torch.Tensor([d for (s1, a, r, s2, d) in minibatch]).to(self.device)

                    #                     print(state1_batch.size())
                    Q1 = self.model(state1_batch)
                    with torch.no_grad():
                        Q2 = self.model(state2_batch)

                    # For X and Y, squeeze() is added to handle the case of 2D input
                    # Y = reward_batch + self.gamma * ((1 - done_batch) * torch.max(Q2, dim=1)[0])
                    Y = reward_batch.squeeze() + \
                        self.gamma * (
                                (1 - done_batch) * torch.topk(Q2, self.nb_action, dim=1)[0].mean(dim=1).squeeze()
                        ).unsqueeze(dim=1)
                    # X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
                    X = Q1.squeeze().gather(dim=1, index=action_batch.long()).squeeze()
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
                if done and display:
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
        state_ = env.reset()
        state = torch.from_numpy(state_).float().to(self.device)
        finish = False
        moves_count = 0
        rewards = []
        done = False
        # if display:
        #     env.render()

        while not finish:
            qval = self.model(state)
            qval_ = qval.cpu().data.numpy()
            actions_ = self.choose_action(qval_)

            step_rewards = []
            for action_ in actions_:
                state_, reward, done, *_ = env.step(action_, display)
                step_rewards.append(reward)
            moves_count += 1
            # state_ = state_.reshape(1, self.in_dim)
            state_ = state_.reshape(*self.input_shape)
            #             print(f"Q_values: {qval_}, Action: {action_}")
            #             print(f"Step: {moves_count}, Reward: {reward}")
            # clear_output(wait=True)

            if noise:
                # state_ = state_ + np.random.rand(1, self.in_dim) / 100.0
                state_ = state_ + np.random.rand(*self.input_shape) / 1000.0
            state = torch.from_numpy(state_).float().to(self.device)

            rewards.append(sum(step_rewards))
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
