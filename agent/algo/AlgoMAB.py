import random
import math
import numpy as np
import logging


class AlgoMAB:
    """
    This class is not adapted to the new framework
    """

    def __init__(self, load_path=None, in_dim=1, out_dim=1, machine_nb=1, alpha=1, **kwargs):
        self.load_path = load_path
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.machine_nb = machine_nb
        self.alpha = alpha

    def train(self, env, epsilon=1, epochs=1000, memory_size=1000,
              max_moves=50, display=True, noise=True):
        losses = []

        for i in range(epochs):
            # TODO: Make sure reset returns the state directly
            state1 = env.reset().tolist()
            finish = False
            moves_count = 0
            rewards = []

            if display:
                print(f"Epoch # {i}:")

            while not finish:
                possible_reward = \
                    [float(state1[3 * j + 3] + float(self.alpha) / float(state1[3 * j + 2] + 1)) / float(
                        state1[3 * j + 2] + 1)
                     for j in range(self.machine_nb)]
                moves_count += 1

                action_ = possible_reward.index(max(possible_reward))

                state2_, reward, done, *_ = env.step(action_, display)
                print(f"Step: {moves_count}, Reward: {reward}")

                state2_ = state2_.tolist()

                state1 = state2_

                if done or moves_count >= max_moves:
                    finish = True
                    moves_count = 0

                # Can be added to comments
                if done:
                    print("DONE!")

            total_reward = sum(rewards)
            if display:
                print(f"Total rewards: {total_reward}")

        return np.array(losses)

    def test_model(self, env, max_moves=50, display=True, noise=True):
        i = 0
        state = env.reset().tolist()
        finish = False
        moves_count = 0
        rewards = []
        done = False
        # if display:
        #     env.render()

        while not finish:
            possible_reward = \
                [float(state[3 * j + 3] + float(self.alpha) / float(state[3 * j + 2] + 1)) / float(state[3 * j + 2] + 1)
                 for j in range(self.machine_nb)]
            # t = float(state[0])
            # possible_reward = [(float(state[0 * j + 4])) + math.sqrt(0 * math.log(t) / float(state[0 * j + 0] + 1))
            #                    for j in range(self.machine_nb)]
            action_ = possible_reward.index(max(possible_reward))

            state_, reward, done, *_ = env.step(action_, display)
            moves_count += 1
            state = state_.tolist()

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
        for i in range(max_games):
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
        pass

    def load_model(self, path="agent.pt"):
        pass
