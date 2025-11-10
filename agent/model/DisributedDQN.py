import numpy as np
import torch
import itertools

from tqdm import tqdm
import logging

class DistributedDQN():
    def __init__(self, base_model, base_agent_nb, target_agent_nb, in_dim = None):
        """

        :param base_model:
        :param base_agent_nb: length of the output of base_model should equal to base_agent_nb
        :param target_agent_nb: length of the target output

        One "actor" is considered as one "agent". In the Multi Armed Bandit problem, one slot machine is one agent.
        base_agent_nb is actually the length of output of the base_model.
        """
        self.base_model = base_model.to(self.device)
        self.base_agent_nb = base_agent_nb
        self.target_agent_nb = target_agent_nb
        self.in_dim = in_dim
        self.device = torch.device('mps' if torch.backends.mps.is_available() else "cpu")

    def forward(self, input, size_input_agent=1):
        input_ = torch.squeeze(input)

        # Separate input for different agents
        dict_input = {i: input_[i*size_input_agent:(i+1)*size_input_agent] for i in range(self.target_agent_nb)}
        # List of all possible combination between agents
        combinations = [i for i in itertools.combinations([j for j in range(self.target_agent_nb)], self.base_agent_nb)]
        output = np.zeros(self.target_agent_nb)

        for combination in combinations:
            # Create input for each combination of agent
            combination_input_ = np.concatenate([dict_input[agent] for agent in combination])
            combination_input = torch.from_numpy(combination_input_).float().to(self.device)
            # Get output for each combination of agent
            combination_output = self.base_model.pe_model(combination_input)
            for i in range(len(combination)):
                # Accumulate output for each agent
                output[combination[i]] = output[combination[i]] + combination_output[i]

        return output

    def test_model(self, env, max_moves=50, display=True, noise=True, size_input_agent=1):
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
            qval = self.forward(state, size_input_agent)
            # qval_ = qval.data.numpy()
            action_ = np.argmax(qval.cpu())

            state_, reward, done, *_ = env.step(action_, display)
            moves_count += 1
            state_ = state_.reshape(1, self.in_dim)
            #             print(f"Q_values: {qval_}, Action: {action_}")
            #             print(f"Step: {moves_count}, Reward: {reward}")

            if noise:
                state_ = state_ + np.random.rand(1, self.in_dim) / 100.0
            state = torch.from_numpy(state_).float().to(self.device)
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

    def test(self, env, max_games=1000, max_moves=50, display=True, noise=True, size_input_agent=1):
        wins = 0
        total_rewards = 0
        for i in tqdm(range(max_games)):
            if display:
                print(f"Game # {i}:")
            test_result = self.test_model(env, max_moves, display, noise, size_input_agent)
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


