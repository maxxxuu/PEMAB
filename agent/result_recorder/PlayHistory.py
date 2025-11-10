from typing_extensions import override

from agent.abstract.AbstractResultRecorder import AbstractResultRecorder


class PlayHistory(AbstractResultRecorder):
    """
    !! Currently, PlayHistory is implemented in the class of environment
    The class here is not in use
    """
    def __init__(self):
        """
        action_history: A dict whose key: epoch, value: list of action of history
        env_setting: A dict whose key: epoch, value: dict of setting of env
        """
        self.action_history = {}
        self.env_setting = {}

    @override
    def update(self, epoch: int, action_history, env_setting) -> None:
        self.action_history[epoch] = action_history
        self.env_setting[epoch] = env_setting

    @override
    def get_summary(self) -> dict[str, int]:
        return {"epoch_nb": len(self.action_history)}
