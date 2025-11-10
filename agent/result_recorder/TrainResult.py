import numpy as np
from utils.utils import plot_losses
import copy

from agent.abstract.AbstractResultRecorder import AbstractResultRecorder
from agent.result_recorder.PlayHistory import PlayHistory


class TrainResult(AbstractResultRecorder):
    def __init__(self, agent_name=None):
        self.agent_name = agent_name
        self.losses = []
        self.total_reward = None
        self.play_history = {}
        self.q_values = {}
        self.env_snaps = {}
        self.states = {}

    def update(self, new_loss, step):
        # self.looses[step] = new_result
        self.losses.append((step, new_loss))

    def update_total_reward(self, total_reward):
        self.total_reward = total_reward

    # def get_summary(self):
    #     return np.array([i[-1] for i in self.looses])

    def get_losses(self):
        return copy.deepcopy(self.losses)

    def get_summary_dict(self):
        output_dict = {
            "agent_name": self.agent_name,
            "total_reward": self.total_reward,
        }
        return output_dict

    def update_play_history(self, epochs, play_history):
        self.play_history[epochs] = play_history

    def update_q_values(self, epochs, q_values):
        self.q_values[epochs] = q_values

    def update_env_snaps(self, epochs, env_snap):
        self.env_snaps[epochs] = env_snap

    def update_states(self, epochs, states):
        self.states[epochs] = states

    def get_history(self):
        return self.play_history

    def get_q_values(self):
        return self.q_values

    def get_env_snaps(self):
        return self.env_snaps

    # def plot(self, path):
    #     plot_losses(self.get_summary(), path)
