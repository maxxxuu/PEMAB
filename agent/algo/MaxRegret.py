import numpy as np
from agent.algo.base_algo.BaseAlgo import BaseAlgo


class MaxRegret(BaseAlgo):

    def choose_action(self, state, env, *args, **kwargs):
        potential_action = env.get_max_reward_distris_keys(reverse=True)
        # TODO: what if play_nb > 1?
        # action = np.array(potential_action)[:env.play_nb]
        action = np.array(potential_action)[0]
        return action
