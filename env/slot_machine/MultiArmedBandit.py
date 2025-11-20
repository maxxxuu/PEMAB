import random
from typing import Callable, Iterable, Union, Collection, Any, Optional

from env.slot_machine.BinarySlotMachine import BinarySlotMachine


class MultiArmedBandit:
    def __init__(self, machine_nb: int=1, reward_distris: Any=None,
                 reward: Union[int, float, None]=None, evol_co: float=0, evol_diff: float=0) -> None:
        # TODO: Include more types of machine
        if reward_distris is None:
            assert machine_nb > 0, f"machine_nb should be > 0. Current value:{machine_nb}"
            assert type(machine_nb) == int, f"machine_nb should be int. Current type:{type(machine_nb)}"

            self.machine_nb = machine_nb
            self.slot_machines = [BinarySlotMachine(index=i, reward=reward) for i in range(machine_nb)]

        # If the reward_distris is a random number generator function:
        elif hasattr(reward_distris, '__call__'):
            self.machine_nb = machine_nb
            self.slot_machines = [BinarySlotMachine(
                index=i, reward_distri=reward_distris, reward=reward) for i in range(machine_nb)]
        # If the reward_distris is a list of reward distri (for binary machine):
        else:
            assert hasattr(reward_distris, '__iter__')
            self.machine_nb = len(reward_distris)
            # Make the reward distribution random
            # See if the training influence the start
            if random.random() <= 1:
                random.shuffle(reward_distris)
            self.slot_machines = [
                BinarySlotMachine(
                    index=i, reward_distri=reward_distris[i], reward=reward) for i in range(len(reward_distris))
            ]

        assert 0 <= evol_co < 1
        self.evol_co = evol_co

        assert 0 <= evol_diff < 1
        self.evol_diff = evol_diff
        # return self.get_reward_distris()

    def play(self, machine_indexs: Collection[int]=set()) -> dict[int, Union[int, float, None]]:
        assert hasattr(machine_indexs, '__iter__')
        rewards = {}
        for i in range(len(self.slot_machines)):
            if i in machine_indexs:
                rewards[i] = self.slot_machines[i].play()
            else:
                self.slot_machines[i].not_played()
        # for machine_index in machine_indexs:
        #     rewards[machine_index] = self.slot_machines[machine_index].play()

        # Evolve reward distri:
        if self.evol_co != 0:
            for i in range(len(self.slot_machines)):
                if i in machine_indexs:
                    self.slot_machines[i].evolve_reward_distri(1-self.evol_co)
                else:
                    self.slot_machines[i].evolve_reward_distri(1+self.evol_co)

        if self.evol_diff != 0:
            # Simple version
            total_diff = self.evol_diff * len(machine_indexs)
            ind_diff = total_diff / (self.machine_nb - len(machine_indexs))

            for i in range(len(self.slot_machines)):
                if i in machine_indexs:
                    self.slot_machines[i].evolve_reward_distri(diff=-self.evol_diff)
                else:
                    self.slot_machines[i].evolve_reward_distri(diff=ind_diff)

            # New version
            # total_diff = sum(
            #     [self.evol_diff if self.slot_machines[i].get_reward_distri() >= self.evol_diff
            #      else self.slot_machines[i].get_reward_distri() for i in machine_indexs])
            #
            # ind_diff = total_diff/(self.machine_nb - len(machine_indexs))
            # done = False
            # to_process = {i for i in range(len(self.slot_machines))}
            #
            # while not done:
            #     saturate = set()
            #     for i in to_process:
            #         if i in machine_indexs:
            #             if self.slot_machines[i].get_reward_distri() >= self.evol_diff:
            #                 self.slot_machines[i].evolve_reward_distri(diff=-self.evol_diff)
            #             else:
            #                 self.slot_machines[i].evolve_reward_distri(diff=-self.slot_machines[i].get_reward_distri())
            #             saturate.add(i)
            #         else:
            #             if 1 >= self.slot_machines[i].get_reward_distri() + ind_diff:
            #                 self.slot_machines[i].evolve_reward_distri(diff=ind_diff)
            #                 total_diff -= ind_diff
            #             else:
            #                 ind_diff_special = 1 - self.slot_machines[i].get_reward_distri()
            #                 self.slot_machines[i].evolve_reward_distri(diff=ind_diff_special)
            #                 total_diff -= ind_diff_special
            #                 saturate.add(i)
            #
            #     to_process = to_process.difference(saturate)
            #     if total_diff <= 1e-10:
            #         done = True
            #     elif len(to_process) > 0:
            #         ind_diff = total_diff / len(to_process)
            #     else:
            #         # total_diff > 0 and to_process is empty
            #         for i in machine_indexs:
            #             self.slot_machines[i].evolve_reward_distri(diff=total_diff / len(machine_indexs))
            #         done = True

        return rewards

    def get_state(self) -> list[dict]:
        states = [slot_machine.get_state() for slot_machine in self.slot_machines]
        return states

    def display(self) -> None:
        for slot_machine in self.slot_machines:
            slot_machine.display()

    def get_reward_distris(self) -> list[float]:
        return [slot_machine.get_reward_distri() for slot_machine in self.slot_machines]

    def get_ind_reward_distri(self, machine_no: Collection[int]) -> Union[list[float], float]:
        if len(machine_no) > 1:
            return [self.get_reward_distris()[machine] for machine in machine_no]
        else:
            return [self.get_reward_distris()[machine] for machine in machine_no][0]

    def get_max_ind_reward_distri(self) -> float:
        return max(self.get_reward_distris())
