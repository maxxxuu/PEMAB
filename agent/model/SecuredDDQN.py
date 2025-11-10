import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy
import logging
from tqdm import tqdm
import os
import matplotlib.pyplot as plt # type: ignore

from collections import deque
from typing import Deque, Union, Iterable, Optional
from typing_extensions import override

from agent.model.DQN import DQN
from agent.model.base_model.BaseModel import BaseModel  # type: ignore
from agent.abstract.Agent import AbstractAgent
from env.abstract.Environment import AbstractEnvironment
from agent.result_recorder.TestResult import TestResult
from agent.result_recorder.SingleResult import SingleResult
from agent.result_recorder.TrainResult import TrainResult
from env.MultiArmedBanditEnvBase import MultiArmedBanditEnvBase

class QNetwork(nn.Module):
    def __init__(self, layers: list[nn.Module], batch_size: int, ind_nb: int, ind_input_len: int, name: str) -> None:
        super(QNetwork, self).__init__()
        self.name = name
        self.nn = nn.Sequential(*layers)
        self.batch_size = batch_size
        self.ind_nb = ind_nb
        self.ind_input_len = ind_input_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # e.g. [0,0,10]
        # 3 cases correspond to: training with batches, and choosing action on a single step
        assert (x.shape == (self.batch_size, self.ind_nb, self.ind_input_len) or
                x.shape == (self.ind_nb, self.ind_input_len))

        output: torch.Tensor = self.nn(x)
        # reveal_type(output)

        output_shape = (x.shape[0], x.shape[1], 1) if len(x.shape) == 3 else (x.shape[0], 1)

        assert output.shape == output_shape

        return output
class SecuredDoubleDQN(BaseModel):
    """
    This is a secured version of DDQN we have, overriding some functions with ones of miniQ
    """

    def __init__(self, layers: list[nn.Module], lr: float, gamma: float, ind_nb: int, batch_size: int,
                 ind_input_len: int, name: str, memory_size: int, update_period: int=10,
                 plot_path: Optional[str]=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.old_q_network: QNetwork = QNetwork(layers, batch_size, ind_nb, ind_input_len, name+"_copy")
        self.q_network: QNetwork = QNetwork(layers, batch_size, ind_nb, ind_input_len, name)
        self.optimizer: optim.Optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma: float = gamma
        self.ind_nb: int = ind_nb
        self.target_model = None
        self.batch_size = batch_size
        self.memory_size = memory_size
        # self.machine_nb = ind_nb
        # self.state_keys = state_keys
        self.ind_input_len = ind_input_len
        self.temp_q: dict[str, float] = {}
        self.update_period = update_period
        if plot_path is not None:
            assert os.path.exists(plot_path), f"Plot path ({plot_path}) doesn't exist."
        self.plot_path = plot_path
        self.name = name

    def choose_action(self, state: np.ndarray, epsilon: float=0.0) -> np.integer:
        # assert types
        assert type(state) == np.ndarray
        # print(state)
        # assert state.shape == (10,batch_size)  # why
        assert state.shape == (self.ind_nb, self.ind_input_len)
        assert type(epsilon) == float

        if np.random.rand() < epsilon:
            action = np.random.choice(range(self.ind_nb))
            self.temp_q = {}
        else:
            with torch.no_grad():
                # print(type(state))
                q_values = self.q_network(torch.tensor(state, dtype=torch.float32))
                assert q_values.shape == (self.ind_nb, 1)
                action = torch.argmax(q_values).item()
                action = np.int64(action)
                temp_q = q_values.squeeze().tolist()
                self.temp_q = {f"q_{i}": temp_q[i] for i in range(len(temp_q))}

        assert isinstance(action, np.integer)
        return action

    def organise_experience(self,
            experience: list[tuple[list[list[int]], int, Union[int, float], list[list[int]], Union[bool, int]]]) \
            -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # reveal_type(experience)
        assert len(experience) == self.batch_size
        state1_batch = torch.stack([torch.tensor(s1, dtype=torch.float32) for (s1, a, r, s2, d) in experience])
        state2_batch = torch.stack([torch.tensor(s2, dtype=torch.float32) for (s1, a, r, s2, d) in experience])
        assert state1_batch.shape == state2_batch.shape == (len(experience), self.ind_nb, self.ind_input_len)

        action_batch = torch.Tensor([a for (s1, a, r, s2, d) in experience])
        reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in experience])

        done_batch = torch.Tensor([d for (s1, a, r, s2, d) in experience])

        assert action_batch.shape == reward_batch.shape == done_batch.shape == (len(experience),)
        return state1_batch, state2_batch, action_batch, reward_batch, done_batch

    # def init_target_model(self):
    #     self.target_model = copy.deepcopy(self.model)
    #     self.target_model.to(self.device)
    #     self.update_target_model()

    # def update_target_model(self):
    #     self.target_model.load_state_dict(self.model.state_dict())
    def update_q_function(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor,
                          next_state: torch.Tensor, done: torch.Tensor) -> torch.Tensor:

        assert state.shape == (self.batch_size, self.ind_nb, self.ind_input_len)
        self.optimizer.zero_grad()
        s = state.clone().detach()
        assert s.shape == state.shape
        ns = next_state.clone().detach()
        assert ns.shape == state.shape
        # state_with_noise = s + torch.tensor(np.random.normal(0, 0.01, size=s.shape), dtype=torch.float32)

        q_values = self.q_network(s)

        q_shape = (state.shape[0], state.shape[1], 1)
        assert q_values.shape == q_shape

        with torch.no_grad():
            next_q_values = self.old_q_network(ns)
            assert next_q_values.shape == q_shape
        # if state == [0]:
        #     target_q_value = 0
        # if done:
        #     target_q_value = reward
        # else:
        target_q_value = reward + (1 - done) * self.gamma * torch.max(next_q_values, dim=-2)[0].squeeze()

        # Only the maximum among all actions are selected to be calculated as target Q
        assert target_q_value.shape == (state.shape[0],)

        # if not torch.is_tensor(target_q_value):
        #     target_q_value = torch.tensor([target_q_value], dtype=torch.float32, requires_grad=False) # Convert to tensor

        # if torch.is_tensor(action):
        selected_q_values = q_values.squeeze(dim=-1).gather(dim=-1, index=action.long().unsqueeze(dim=-1))
        target_q_value.unsqueeze_(dim=-1)
        assert target_q_value.shape == selected_q_values.shape == (self.batch_size, 1)

        loss = nn.MSELoss()(selected_q_values, target_q_value)
        # else:
        #     loss = nn.MSELoss()(q_values[action].squeeze(), target_q_value.squeeze())
        loss.backward()
        self.optimizer.step()
        assert loss.shape == ()
        self.update_counter += 1
        return loss

    @override
    def learn(self, env: MultiArmedBanditEnvBase, epsilon: float, epochs: int, max_moves: int, batch_size: int=200,
              display: bool=True, noise: bool=True) -> TrainResult:
        train_result = TrainResult(agent_name=self.q_network.name)
        # experiences = deque(maxlen=self.memory_size if self.memory_size > self.batch_size else self.batch_size*2)
        # experiences = []
        # moves_count = 0
        # TODO: Is sync_freq and sync_count necessary here?
        sync_count = 0
        # TODO: think if 0 is the right value for epsilon when continous_train==1

        experiences: Deque[tuple[list[list[int]], int, Union[int, float], list[list[int]], Union[bool, int]]] = (
            deque(maxlen=self.memory_size))
        for episode in tqdm(range(epochs)):
            state = env.reset()
            total_reward: Union[int, float] = 0
            done = False
            round_nr = 0
            total_loss: float = 0

            q_values: list[dict[str, float]] = []
            states: list[list] = []
            moves_count = 0
            # rewards = []

            while not done:
                sync_count += 1
                moves_count += 1
                assert state.shape == (self.ind_nb, self.ind_input_len)
                action = self.choose_action(state, epsilon)
                q_values.append(copy.deepcopy(self.temp_q))
                with torch.no_grad():
                    states.append(copy.deepcopy(state).tolist())
                next_state, reward, done, *_ = env.step(action=action, display=display)

                experience: tuple = (state, action, reward, next_state, done)
                experiences.append(experience)

                round_nr += 1
                if round_nr >= max_moves or done:
                    done = True
                    # Done reinforcement
                    state_, next_state_, action_, reward_, done_ = self.organise_experience(
                        [experience for i in range(self.batch_size)])
                    assert state_.shape == next_state_.shape == (self.batch_size, self.ind_nb, self.ind_input_len)
                    assert action_.shape == reward_.shape == done_.shape == (self.batch_size,)
                    loss = self.update_q_function(state_, action_, reward_, next_state_, done_)

                state = next_state
                total_reward += reward

                if len(experiences) >= self.batch_size:
                    minibatch = random.sample(experiences, self.batch_size)
                    state1_batch, state2_batch, action_batch, reward_batch, done_batch = (
                        self.organise_experience(minibatch))
                    assert (state1_batch.shape == state2_batch.shape ==
                            (self.batch_size, self.ind_nb, self.ind_input_len))
                    assert action_batch.shape == reward_batch.shape == done_batch.shape == (self.batch_size,)

                    loss = self.update_q_function(state1_batch, action_batch, reward_batch, state2_batch, done_batch)
                    train_result.update(loss.item(), self.update_counter)
                    total_loss += float(loss)

                if done:
                    finish = True
                    train_result.update_play_history(epochs=episode, play_history=env.get_play_history())
                    train_result.update_q_values(epochs=episode, q_values=q_values)
                    train_result.update_env_snaps(epochs=episode, env_snap=env.env_snap())

            train_result.update_total_reward(total_reward)
            if display:
                print(f"Train episode {episode + 1}, Total Reward: {total_reward}")

            if episode % 100 == 0:
                if self.plot_path:
                    self.generate_q_plot(env, episode, max_moves)
                print(f"Total loss: {total_loss}")

            if episode % 10 == 0:
                self.old_q_network = self.q_network

        return train_result

    def generate_q_plot(self, env, e: int, max_moves: int) -> None:
        q_plot = []
        # state = [R_NR]
        state = env.reset()
        total_reward: Union[int, float] = 0
        done = False
        round_nr = 0
        # TODO: replace with something sophisticated
        p = 0.9
        while not done:
            # Select action greedily based on Q-values
            with torch.no_grad():
                assert state.shape == (self.ind_nb, self.ind_input_len)
                q_values = self.q_network(torch.tensor(state, dtype=torch.float32))
                assert q_values.shape == (self.ind_nb, 1)
                q_plot += [q_values.numpy()]
                action = torch.argmax(q_values).item()
                assert type(action) == int
                assert self.ind_nb > action >= 0

            next_state, reward, done = env.step(action=action, display=self.ind_nb)

            # if action == 0:
            #     reward = 0
            # else:
            #     reward = 1 if np.random.rand() <= PWIN else 0

            # state = [R_NR - (round_nr + 1)]

            state = next_state

            total_reward += reward
            round_nr += 1
            if round_nr >= max_moves:
                done = True

        # best_machine = env.get_max_reward_distris_keys()[0]
        q_values_list: list[list[np.ndarray]] = [[q[i] for q in q_plot] for i in range(self.ind_nb)]
        assert len(q_values_list) == self.ind_nb and len(q_values_list[0]) == max_moves
        # q0_values = [q[0] for q in q_plot]
        # q1_values = [q[1] for q in q_plot]

        # Plot the line
        for i in range(len(q_values_list)):
            plt.plot(range(max_moves), q_values_list[i])
        # plt.plot(range(R_NR), q0_values, 'k.')
        # plt.plot(range(R_NR), q1_values, 'g.')

        # plt.plot(range(R_NR), [(2 * PWIN - 1) * (1 - gamma ** (R_NR - i)) / (1 - gamma) for i in range(R_NR)], 'b--')
        plt.plot(range(max_moves), [(2 * p - 1) * (1 - self.gamma ** (max_moves - i)) / (1 - self.gamma)
                                    for i in range(max_moves)], 'b--')
        plt.xlabel('round')
        plt.ylabel('Q')
        plt.savefig(f'{self.plot_path}/{self.name}_round_nb_{max_moves}_{e}.pdf')
        plt.close()

    @override
    def single_test(self, env: MultiArmedBanditEnvBase, init_state: np.ndarray, max_moves: int, display: bool=True,
                    noise: bool=True) -> SingleResult:
        result = SingleResult()
        i = 0
        q_values = []
        states = []
        state = init_state
        # if baseline_models is not None:
        #     temp_env = copy.deepcopy(env)
        #     baseline_results = baseline_test(
        #         temp_env, baseline_models, max_moves=max_moves, display=display, noise=noise)
        # else:
        #     baseline_results = None

        result.update_env_snap(env_snap=env.env_snap())
        moves_count = 0
        # rewards = []
        done = False

        # if display:
        #     env.render()

        self.model.eval()

        if display:
            logging.info(f"Model name:{type(self).__name__}")
            print(f"Model name:{type(self).__name__}")

        while not done:
            total_reward = 0.0
            round_nr = 0
            with torch.no_grad():
                assert state.shape == (self.ind_nb, self.ind_input_len)
                action = self.choose_action(state, epsilon=0.0)
                q_values.append(copy.deepcopy(self.temp_q))
                states.append(copy.deepcopy(state.tolist()))
                assert isinstance(action, np.integer)

            next_state, reward, done = env.step(action=action, display=display)

            state = next_state

            total_reward += reward
            round_nr += 1
            if round_nr >= max_moves:
                done = True

            # rewards.append(reward)
            result.update_reward(reward)

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

    @override
    def multi_test(self, env: MultiArmedBanditEnvBase, max_games: int=1000, max_moves: int=50, display: bool=True,
                   noise: bool=True):
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

