from abc import ABC, abstractmethod


class AbstractResultRecorder(ABC):

    @abstractmethod
    def update(self, *args, **kwargs) -> None:

        """
        Update the result (counters etc. ) with new test outcome
        :return: None
        """
        raise NotImplementedError("Subclass of Result should implement update methode")

    def get_summary(self, *args, **kwargs) -> dict:
        """
        Get summary of the result
        :return:
        """
        raise NotImplementedError("Subclass of Result should implement get summary methode")