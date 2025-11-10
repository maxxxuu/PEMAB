from abc import ABC, abstractmethod
from typing import Type

from agent.abstract.Agent import AbstractAgent
from env.abstract.Environment import AbstractEnvironment
from agent.result_recorder.TrainResult import TrainResult


class AbstractModel(AbstractAgent, ABC):
    @abstractmethod
    def learn(self, env: Type[AbstractEnvironment]) -> TrainResult:
        """
        Train the model in the environment
        :param env: the environment on which the model get trained
        :return: a numpy array of looses
        """
        raise NotImplementedError("Subclass of AbstractModel should implement train methode")

    @abstractmethod
    def export_model(self, path):
        """
        Export the trained model
        :param path: the path for exported model
        :return: None
        """
        raise NotImplementedError("Subclass of AbstractModel should implement export_model methode")

    @abstractmethod
    def import_model(self, path):
        """
        Import the model from a file
        :param path: path of the file to be imported
        :return: None
        """
        raise NotImplementedError("Subclass of AbstractModel should implement import_model methode")