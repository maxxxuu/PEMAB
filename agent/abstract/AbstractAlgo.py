from abc import ABC, abstractmethod

from agent.abstract.Agent import AbstractAgent


class AbstractAlgo(AbstractAgent, ABC):

    @abstractmethod
    def fit(self, env):
        """
        Fit algo to the env. (?This methode may not need to be implemented. But I can't find another methode
        necessary for all algo. To keep this abstract class I keep this methode here
        :param env: the environment on which the algo get fitted
        :return: numpy array of looses
        """
        raise NotImplementedError("Subclass of AbstractAlgo should implement fit methode")

