import os
import logging
import csv

from agent.abstract.AbstractResultRecorder import AbstractResultRecorder
from agent.result_recorder.PlayHistory import PlayHistory


class TestResult(AbstractResultRecorder):
    def __init__(self, agent_name=None):
        self.agent_name = agent_name
        self.wins = 0
        self.total_rewards = 0
        self.nb_games = 0

        # Key: epoch
        self.play_history = {}
        self.q_values = {}
        self.states = {}
        self.env_snaps = {}

    def update(self, win=False, total_reward=0, play_history=None, q_values=None,
               env_snap=None, epoch=None, states=None):
        if win:
            self.wins += 1
        self.total_rewards += total_reward
        self.nb_games += 1
        self.play_history[epoch] = play_history
        self.q_values[epoch] = q_values
        self.env_snaps[epoch] = env_snap
        self.states[epoch] = states

    def get_env_snaps(self):
        return self.env_snaps

    def get_q_values(self):
        return self.q_values

    def update_from_single_result(self, single_result, epoch=None):
        self.update(**single_result.get_result_dict(), epoch=epoch)

    def get_summary(self):
        win_perc = float(self.wins) / float(self.nb_games)
        average_reward = float(self.total_rewards) / float(self.nb_games)

        return win_perc, average_reward

    def get_summary_dict(self):
        win_perc, average_reward = self.get_summary()
        output_dict = {
            "agent_name": self.agent_name,
            "win_perc": win_perc,
            "average_reward": average_reward,
            "nb_games": self.nb_games,
        }
        return output_dict

    # def export_to_csv(self, path, *args, **kwargs):
    #     fieldnames = [
    #         "agent_name",
    #         # "win_perc",
    #         "average_reward",
    #         "nb_games",
    #     ]
    #     row = [self.get_summary_dict()]
    #
    #     file_exist = os.path.isfile(path)
    #     if not file_exist:
    #         f = open(path, 'w', newline='')
    #     else:
    #         f = open(path, 'a', newline='')
    #
    #     writer = csv.DictWriter(f, fieldnames=fieldnames)
    #     if not file_exist:
    #         writer.writeheader()
    #     writer.writerow(row)
    #
    #     f.close()

    def get_history(self):
        return self.play_history

    def log(self, display=False):
        win_perc, average_reward = self.get_summary()
        logging.info(f"Agent name:{self.agent_name}")
        logging.info(f"Games played: {self.nb_games}, # of wins: {self.wins}")
        logging.info(f"Win percentage: {100.0 * win_perc}%")
        logging.info(f"Average reward :{average_reward}")
        if display:
            print(f"Agent name:{self.agent_name}")
            # print(f"Games played: {self.nb_games}, # of wins: {self.wins}")
            # print(f"Win percentage: {100.0 * win_perc}%")
            print(f"Average reward :{average_reward}")
