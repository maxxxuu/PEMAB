from abc import ABC, abstractmethod


class AbstractEnvironment(ABC):
    def env_snap(self):
        return vars(self)

    @abstractmethod
    def step(self, action, display=False):
        """
        Take action and return feedback from env:
        :return: new state (state), reward, flag of done (done), and other information
        """
        raise NotImplementedError("Subclass of Environment should implement step methode")

    @abstractmethod
    def reset(self):
        """
        Reset the env

        :return: The initial state
        """
        raise NotImplementedError("Subclass of Environment should implement reset methode")

    # @abstractmethod
    # def get_state_space(self):
    #     """
    #     Get the size of state space
    #     Attention: different from get_state_size
    #
    #     :return: The size of state space
    #     """
    #     raise NotImplementedError("Subclass of Environment should implement this methode")

    @abstractmethod
    def get_state(self):
        """

        :return: Current state (or observation for Gym package)
        """
        raise NotImplementedError("Subclass of Environment should implement get_state methode")

    @abstractmethod
    def get_state_size(self):
        """
        Get the size of one state, which is generally the size of the
        input layer of the  DQN


        :return: The size of state space
        """
        raise NotImplementedError("Subclass of Environment should implement this methode")

    @abstractmethod
    def get_action_space(self):
        """
        Get the size of action space, which is generally the size of the
        output layer of the  DQN


        :return: The size of action space
        """
        raise NotImplementedError("Subclass of Environment should implement this methode")

    @abstractmethod
    def render(self):
        """
        Show a visio of the environment

        """
        raise NotImplementedError("Subclass of Environment should implement this methode")

    def judge_win(self, done, total_reward):
        """
        Judge if a game is win or not

        Generally, if the game is done, it is a win (e.g. for maze or taxi), but in some case it should achieve a
        certain level of reward before done to win (e.g. cartpole)

        :param done:
        :param total_reward:
        :return: win: whether the game is won
        """
        win = done

        return win

