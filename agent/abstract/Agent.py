from abc import ABC, abstractmethod


class AbstractAgent(ABC):
    def __init__(self, name=None):
        if name is None:
            self.name = type(self).__name__
        else:
            assert type(name) == str, f"Name of an agent should be a str, received {type(name)}"
            self.name = name

    @abstractmethod
    def choose_action(self, state, *args, **kwargs):
        """
        Choose the action that the agent is going to take
        :param state: state on which the action is chosen
        :return: the action
        """

        raise NotImplementedError("Subclass of Agent should implement choose_action methode")

    @abstractmethod
    def single_test(self, env, init_state):
        """
        Run one test on the environment
        :param env: the env on which the agent runs the test
        :param init_state: the initial state of the single test
        :return: the result (and the history?) of the single test
        """
        raise NotImplementedError("Subclass of Agent should implement single_test methode")

    @abstractmethod
    def multi_test(self, env, test_nb):
        """
        Run several tests on the given environment
        :param env: the env on which the agent runs the test
        :param test_nb: hoe many times the test will run
        :return: the result (and the history?) of the multi test
        """
        raise NotImplementedError("Subclass of Agent should implement multi_test methode")
