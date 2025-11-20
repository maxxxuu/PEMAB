import sys
import os
import logging
import csv

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt # type: ignore
import copy
from collections import deque
import random
from typing import Deque, Union, Iterable
from tqdm import tqdm
from pathlib import Path

# Ensure project root is in sys.path so imports like `agent.*` and `env.*` work
# when running this script directly. The project root (PEMAB) is 3 parents
# above this file: (.../run_script/vanilla/with_dr/vanilla_dr.py).
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from agent.model.layer.PESymetry import PESymetryMean
from env.MultiArmedBanditEnvForPESym import MultiArmedBanditEnvForPESym
from agent.algo.UCB import UCB1
from run_script.utils import prepare_results_directory

# TEST_DIR will be set in main(); keep as None so importing this module
# doesn't create directories or start training.
TEST_DIR = None

"""
Please note that the assertion in this script is specific to our current MAB problem in Q learning,
may need adjustment if the problem change
"""

R_NR = 200
# round_nbs = [1] + [i * 5 for i in range(1, 101)]
# round_nbs = [i * 10 for i in range(10, 21)]
round_nbs = [200]
PWIN = 0.9

# Environment parameters
lr = 0.00003  # Learning rate
gamma = 0.99  # Discount factor
machine_nb = 10
output_dim = machine_nb
reward_distris: list[float] = [0.1 for _ in range(machine_nb)]
reward_distris[5] = PWIN
# reward_distris = None
win_reward = 1
state_keys = ["played", "reward_times", "horizon"]
display = False
batch_size = 1
memory_size = batch_size
assert memory_size >= batch_size

# Training loop
epsilon = 0.1  # Exploration rate
num_episodes = 10000
repeat = 10

env_config = {
    "play_nb": 1,
    "round_nb": R_NR,
    "machine_nb": machine_nb,
    "reward_distris": reward_distris,
    "state_keys": state_keys,
    "reward": win_reward,
    "display": display,
}
# Note: `env` will be created inside `main()` so importing this module
# won't run the environment/training loops.

env_config_ucb = copy.deepcopy(env_config)
# env_config_ucb["reward"] = 10
# env_ucb = MultiArmedBanditEnvForPESym(
#     **env_config_ucb)


def _ensure_test_dirs_and_logging():
    """Create plots dir and configure logging when TEST_DIR is available."""
    global TEST_DIR
    if TEST_DIR is None:
        return

    plot_dir_path = os.path.join(TEST_DIR, 'plots')
    os.makedirs(plot_dir_path, exist_ok=True)

    log_path = os.path.join(TEST_DIR, 'record.log')
    file_exist = os.path.isfile(log_path)
    if not file_exist:
        # create an empty log file
        with open(log_path, 'w') as f_object:
            f_object.close()
    logging.basicConfig(filename=log_path, level=logging.DEBUG)

""" --- Pooling mix Idv ---"""
neuron_nb_factor = 5


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(QNetwork, self).__init__()
        fc_layers = [
            nn.Linear(input_dim, input_dim * neuron_nb_factor),
            nn.ELU(),
            # nn.Tanh(),
            nn.Linear(input_dim * neuron_nb_factor, input_dim * neuron_nb_factor),
            nn.ELU(),
            # PESymetryMean(in_dim * 5, in_dim * 5),
            # nn.LeakyReLU(),
            nn.Linear(input_dim * neuron_nb_factor, output_dim),
        ]
        self.nn = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # e.g. [0,0,10]
        # 3 cases correspond to: training with batches, and choosing action on a single step
        assert (x.shape == (batch_size, machine_nb * len(state_keys)) or x.shape == (machine_nb * len(state_keys),))

        output: torch.Tensor = self.nn(x)
        # reveal_type(output)

        output_shape = (x.shape[0], machine_nb) if len(x.shape) == 2 else (machine_nb,)

        assert output.shape == output_shape

        return output


class QLearningAgent:
    def __init__(self, input_dim: int, output_dim: int, lr: float, gamma: float, ind_nb: int =2) -> None:
        self.old_q_network: QNetwork = QNetwork(input_dim, output_dim)
        self.q_network: QNetwork = QNetwork(input_dim, output_dim)
        self.optimizer: optim.Optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma: float = gamma
        self.ind_nb: int = ind_nb


    def select_action(self, state: np.ndarray, epsilon: float) -> np.integer:
        # assert types
        assert type(state) == np.ndarray
        # print(state)
        # assert state.shape == (10,batch_size)  # check shape
        assert state.shape == (machine_nb * len(state_keys),)
        assert type(epsilon) == float
        
        if np.random.rand() < epsilon:
            action = np.random.choice(range(self.ind_nb))
        else:
            with torch.no_grad():
                # print(type(state))
                q_values = self.q_network(torch.tensor(state, dtype=torch.float32))
                assert q_values.shape == (machine_nb,)
                action = torch.argmax(q_values).item()
                action = np.int64(action)


        assert isinstance(action, np.integer)
        return action

    def update_q_function(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor,
                          next_state: torch.Tensor, done: torch.Tensor) -> torch.Tensor:

        assert state.shape == (batch_size, machine_nb * len(state_keys))
        self.optimizer.zero_grad()
        s = state.clone().detach()
        assert s.shape == state.shape
        ns = next_state.clone().detach()
        assert ns.shape == state.shape
        # state_with_noise = s + torch.tensor(np.random.normal(0, 0.01, size=s.shape), dtype=torch.float32)

        q_values = self.q_network(s)

        q_shape = (state.shape[0], machine_nb)
        assert q_values.shape == q_shape

        with torch.no_grad():
            next_q_values = self.old_q_network(ns)
            assert next_q_values.shape == q_shape
        # if state == [0]:
        #     target_q_value = 0
        # if done:
        #     target_q_value = reward
        # else:
        target_q_value = reward + (1 - done) * self.gamma * torch.max(next_q_values, dim=-1)[0].squeeze()

        # Only the maximum among all actions are selected to be calculated as target Q
        assert target_q_value.shape == (state.shape[0],)

        # if not torch.is_tensor(target_q_value):
        #     target_q_value = torch.tensor([target_q_value], dtype=torch.float32, requires_grad=False) # Convert to tensor

        # if torch.is_tensor(action):
        selected_q_values = q_values.squeeze(dim=-1).gather(dim=-1, index=action.long().unsqueeze(dim=-1))
        target_q_value.unsqueeze_(dim=-1)
        assert target_q_value.shape == selected_q_values.shape == (batch_size, 1)

        loss = nn.MSELoss()(selected_q_values, target_q_value)
        # else:
        #     loss = nn.MSELoss()(q_values[action].squeeze(), target_q_value.squeeze())
        loss.backward()
        self.optimizer.step()
        assert loss.shape == ()
        return loss

    def export_model(self, path="agent.pt"):
        # Save model into TEST_DIR by default
        if path is None or not os.path.isabs(path):
            if TEST_DIR is not None:
                path = os.path.join(TEST_DIR, path or 'agent.pt')
            else:
                path = os.path.abspath(path or 'agent.pt')
        torch.save(self.q_network.nn, path)
        logging.info(f'Model saved at: {path}')


def generate_q_plot(agent: QLearningAgent, e: int) -> None:
    q_plot = []
    # state = [R_NR]
    state = env.reset()
    total_reward: Union[int, float] = 0
    done = False
    round_nr = 0
    while not done:
        # Select action greedily based on Q-values
        with torch.no_grad():
            state = state.flatten()
            assert state.shape == (machine_nb * len(state_keys),)
            q_values = agent.q_network(torch.tensor(state, dtype=torch.float32))
            q_values.unsqueeze_(dim=-1)
            assert q_values.shape == (machine_nb,1)
            q_plot += [q_values.numpy()]
            action = torch.argmax(q_values).item()
            assert type(action) == int
            assert machine_nb > action >= 0

        next_state, reward, done = env.step(action=action, display=display)

        # if action == 0:
        #     reward = 0
        # else:
        #     reward = 1 if np.random.rand() <= PWIN else 0

        # state = [R_NR - (round_nr + 1)]

        state = next_state

        total_reward += reward
        round_nr += 1
        if round_nr >= R_NR:
            done = True

    # best_machine = env.get_max_reward_distris_keys()[0]
    q_values_list: list[list[np.ndarray]] = [[q[i] for q in q_plot] for i in range(machine_nb)]
    assert len(q_values_list) == machine_nb and len(q_values_list[0]) == R_NR
    # q0_values = [q[0] for q in q_plot]
    # q1_values = [q[1] for q in q_plot]

    # Plot the line
    for i in range(len(q_values_list)):
        plt.plot(range(R_NR), q_values_list[i])
    # plt.plot(range(R_NR), q0_values, 'k.')
    # plt.plot(range(R_NR), q1_values, 'g.')

    # plt.plot(range(R_NR), [(2 * PWIN - 1) * (1 - gamma ** (R_NR - i)) / (1 - gamma) for i in range(R_NR)], 'b--')
    plt.plot(range(R_NR), [(2 * PWIN - 1) * (1 - gamma ** (R_NR - i)) / (1 - gamma) for i in range(R_NR)], 'b--')
    plt.xlabel('round')
    plt.ylabel('Q')
    # Determine plot directory at runtime (TEST_DIR may be set later)
    plot_dir_path = os.path.join(TEST_DIR, 'plots') if TEST_DIR is not None else os.getcwd()
    os.makedirs(plot_dir_path, exist_ok=True)
    plt.savefig(os.path.join(plot_dir_path, f'rounds_{R_NR}_ep_{e}.pdf'))
    plt.close()


def organise_experience(
        experience: list[tuple[list[list[int]], int, Union[int, float], list[list[int]], Union[bool, int]]]) \
        -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # reveal_type(experience)
    assert len(experience) == batch_size
    state1_batch = torch.stack([torch.tensor(s1, dtype=torch.float32) for (s1, a, r, s2, d) in experience])
    state2_batch = torch.stack([torch.tensor(s2, dtype=torch.float32) for (s1, a, r, s2, d) in experience])
    assert state1_batch.shape == state2_batch.shape == (len(experience), machine_nb * len(state_keys))

    action_batch = torch.Tensor([a for (s1, a, r, s2, d) in experience])
    reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in experience])

    done_batch = torch.Tensor([d for (s1, a, r, s2, d) in experience])

    assert action_batch.shape == reward_batch.shape == done_batch.shape == (len(experience),)
    return state1_batch, state2_batch, action_batch, reward_batch, done_batch

# testing the agent

# Function to test the trained agent
def test_agent(agent: QLearningAgent, num_episodes: int = 10, train_epoch: Union[None, int]=None,
               path=None, repeat=None) -> float:
    if path is not None:
        # If a relative path is passed, place it under TEST_DIR (if set), else use relative path
        if not os.path.isabs(path):
            path = os.path.join(TEST_DIR, path) if TEST_DIR is not None else os.path.abspath(path)

        file_exist = os.path.isfile(path)
        fieldnames = [
            "agent_name",
            "epoch",
            "repeat",
            "best_action_count",
            "round_nb"
        ]

        if not file_exist:
            f = open(path, 'w', newline='')
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        else:
            f = open(path, 'r')
            dict_reader = csv.DictReader(f)
            fieldnames = list(
                dict_reader.fieldnames if dict_reader.fieldnames is not None else fieldnames)
            f.close()
            f = open(path, 'a', newline='')
            writer = csv.DictWriter(f, fieldnames=fieldnames)

    best_action_count = 0
    rwd_agent = 0
    rwd_baseline = 0.1
    for episode in range(num_episodes):
        # state = [R_NR]
        state = env.reset()
        env_config_ucb["reward_distris"] = env.get_rwd_dist()
        env_ucb = MultiArmedBanditEnvForPESym(
            **env_config_ucb)
        total_reward: Union[int, float] = 0
        done = False
        round_nr = 0
        while not done:
            # Select action greedily based on Q-values
            with torch.no_grad():
                state = state.flatten()
                assert state.shape == (machine_nb * len(state_keys),)
                q_values = agent.q_network(torch.tensor(state, dtype=torch.float32))
                q_values.unsqueeze_(dim=-1)
                assert q_values.shape == (machine_nb, 1)
                action = torch.argmax(q_values).item()
                assert type(action) == int

                if action == env.get_max_reward_distris_keys()[0]:
                    best_action_count += 1

            next_state, reward, done = env.step(action=action, display=display)

            # if action == 0:
            #     reward = 0
            # else:
            #     reward = 1 if np.random.rand() <= PWIN else 0

            # state = [R_NR - (round_nr + 1)]
            state = next_state

            total_reward += reward
            round_nr += 1
            if round_nr >= R_NR:
                done = True

        rwd_agent += total_reward
        logging.info(f"Test episode {episode + 1}, Total Reward: {total_reward}")
        print(f"Test episode {episode + 1}, Total Reward: {total_reward}")

        ''' --- UCB1 --- '''

        ucb1 = UCB1(
            coef=0.1 * np.sqrt(2),
            in_dim=env_ucb.get_state_size(),
            out_dim=env_ucb.get_action_space(),
            input_shape=(env_ucb.machine_nb, env_ucb.get_state_size())
        )
        ucb1_result = ucb1.single_test(env_ucb, max_moves=R_NR, display=False)

        rwd_baseline += ucb1_result.get_total_reward()
        logging.info(f"UCB1: {ucb1_result.get_total_reward()}")
        print(f"UCB1: {ucb1_result.get_total_reward()}")

        ''' --- ---- ---'''

        logging.info(f"rwd_agent / rwd_baseline = {rwd_agent / rwd_baseline}")
        print(f"rwd_agent / rwd_baseline = {rwd_agent / rwd_baseline}")

    if path is not None:
        rows = [
            {"agent_name": "FC",
             "epoch": train_epoch, "repeat": repeat,
             "best_action_count": best_action_count / num_episodes, "round_nb": R_NR}
        ]
        writer.writerows(rows)
        f.close()
    return rwd_agent / rwd_baseline

def main():
    global TEST_DIR
    global env

    # Prepare results directory and configure logging/plots
    TEST_DIR = prepare_results_directory()
    _ensure_test_dirs_and_logging()

    # Initialize environment and agent
    env = MultiArmedBanditEnvForPESym(**env_config)

    input_dim = machine_nb * env.get_state_size()
    agent = QLearningAgent(input_dim, output_dim, lr, gamma)
    experiences: Deque[tuple[list[list[int]], int, Union[int, float], list[list[int]], Union[bool, int]]] = (
        deque(maxlen=memory_size))

    max_rwd_perc = 0

    for rep in tqdm(range(repeat)):
        current_rwd_perc = 0
        for round_nb in round_nbs:
            for episode in range(num_episodes):
                env_config_train = copy.deepcopy(env_config)
                env_config_train["round_nb"] = round_nb
                env_train = MultiArmedBanditEnvForPESym(
                    **env_config_train)
                # state = [R_NR]  # np.random.rand(input_dim)  # Initial state
                state = env_train.reset()
                total_reward: Union[int, float] = 0
                done = False
                round_nr = 0
                total_loss: float = 0
                state = state.flatten()
                while not done:
                    assert state.shape == (machine_nb * len(state_keys),)
                    action = agent.select_action(state, epsilon)

                    next_state, reward, done = env_train.step(action=action, display=display)
                    next_state = next_state.flatten()
                    experience: tuple = (state, action, reward, next_state, done)
                    experiences.append(experience)

                    round_nr += 1
                    if round_nr >= round_nb:
                        done = True
                        # Done Reinforcement
                        # state_, next_state_, action_, reward_, done_ = organise_experience(
                        #     [experience for i in range(batch_size)])
                        # assert state_.shape == next_state_.shape == (batch_size, machine_nb * len(state_keys))
                        # assert action_.shape == reward_.shape == done_.shape == (batch_size,)
                        # loss = agent.update_q_function(state_, action_, reward_, next_state_, done_)

                    state = next_state
                    total_reward += reward


                    if len(experiences) >= batch_size:
                        minibatch = random.sample(experiences, batch_size)
                        state1_batch, state2_batch, action_batch, reward_batch, done_batch = organise_experience(minibatch)
                        assert state1_batch.shape == state2_batch.shape == (batch_size, machine_nb * len(state_keys))
                        assert action_batch.shape == reward_batch.shape == done_batch.shape == (batch_size,)

                        loss = agent.update_q_function(state1_batch, action_batch, reward_batch, state2_batch, done_batch)
                        total_loss += float(loss)

                # print(f"Train episode {episode + 1}, Total Reward: {total_reward}")
                if len(round_nbs) == 1:
                    if episode % 1000 == 0:
                        generate_q_plot(agent, episode)
                        # print(f"Total loss: {total_loss}")

                if episode % 10 == 0:
                    agent.old_q_network = agent.q_network
                    win_per_temp = test_agent(agent, num_episodes=10, train_epoch=episode, path="test_data.csv", repeat=rep)

                # # Early stop
                # if total_loss <= 1:
                #     print("Trigger early stop!")
                #     generate_q_plot(agent, episode)
                #     print(f"Total loss: {total_loss}")
                #     break

        for round_nb in round_nbs:
            # # Test the agent
            print(f"Testing round_nb={round_nb}")
            current_rwd_perc += test_agent(agent)

        if current_rwd_perc > max_rwd_perc:
            max_rwd_perc = current_rwd_perc
            print(f"Current best reward perc: {max_rwd_perc}")
            agent.export_model("best_agent.pt")
        agent.export_model(f"agent{rep}.pt")


if __name__ == '__main__':
    main()

# q0_values = [q[0] for q in q_plot]
# q1_values = [q[1] for q in q_plot]

# # Plot the line
# plt.plot(range(200), q0_values, 'k.')
# plt.plot(range(200), q1_values, 'g.')
# plt.xlabel('round')
# plt.ylabel('Q')
# plt.show()
