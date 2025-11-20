import torch
from matplotlib import pylab as plt # type: ignore
from matplotlib.lines import Line2D # type: ignore
import numpy as np
from functools import wraps
import time
import itertools
import os
import sys
from typing import Union, Callable, Optional, TypeVar, Any, Iterator
import torch.nn as nn

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = TypeVar('T')

class RandomGeneratorError(Exception):
    def __init__(self, inf: Union[int, float, None], sup: Union[int, float, None], trial: int) -> None:
        error_message = f"The random generator can't generate a number between ({inf}, {sup}) within {trial} trials."
        super().__init__(error_message)


def random_verifier(rand_func: Callable, low: Optional[Union[int, float]]=None,
                    up: Optional[Union[int, float]]=None, trial: int=100) -> float:
    '''
    Use a given generator to generate random number, if the random number do not fall in the defined interval,
    retry until correct number generated or max trial reached.

    :param rand_func: Random generator function
    :param low: Lower bound of target interval
    :param up: Upper bound of target interval
    :param trial: Max trial number
    :return: Generated random number
    '''
    for i in range(trial):
        output = rand_func()
        inf_valid = True if (low is None) or (output >= low) else False
        sup_valid = True if (up is None) or (output <= up) else False
        if inf_valid and sup_valid:
            return output
    raise RandomGeneratorError(low, up, trial)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def func_timer(func: Callable[..., T]) -> Callable[..., T]:
    @wraps(func)
    def func_timer_wrapper(*args, **kwargs) -> T:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args}{kwargs} took {total_time:.4f} seconds')
        return result

    return func_timer_wrapper


def combination_dict(dicts: dict[Any, list]={}) -> list[dict]:
    """
    When dicts contains more than one option for certain value, and options for one key are presented as a
    list, this function unpack the dicts and return a list of dictionary with all possible combination, which
    only have one option of each key.

    For example, if dicts = {x: [a, b], y: [c, d], z: [e]}, then return would be [{x: a, y: c, z: e},
    {x: b, y: c, z: e}, {x: a, y: d, z: e}, {x: b, y: d, z: e},]

    All value should be in form of list.
    """
    keys, values = zip(*dicts.items())
    combinations = [dict(zip(keys, combined_value)) for combined_value in itertools.product(*values)]
    return combinations


def get_root_dir() -> str:
    return ROOT_DIR


def add_project_dir() -> None:
    sys.path.insert(0, ROOT_DIR)


def plot_losses(losses: Union[list[float], np.ndarray], path: str='results/default.png', title: Optional[str]=None) -> None:
    plt.figure(figsize=(10, 7))
    plt.plot(losses)
    if title is not None:
        plt.title(title)
    plt.xlabel("Epochs", fontsize=22)
    plt.ylabel("Loss", fontsize=22)
    plt.savefig(path)
    plt.show()


def plot_win_percs(win_percs: Union[list[float], np.ndarray], path: str='results/win_percs.png', title: Optional[str]=None):
    plt.figure(figsize=(10, 7))
    plt.plot(win_percs)
    if title is not None:
        plt.title(title)
    plt.xlabel("run", fontsize=22)
    plt.ylabel("win_perc (%)", fontsize=22)
    plt.savefig(path)
    plt.show()


def plot_general(data: Union[list[float], np.ndarray], path: str='results/win_percs.png', xlabel: str="epoch", ylabel: str="loss", title: Optional[str]=None):
    plt.figure(figsize=(10, 7))
    plt.plot(data)
    if title is not None:
        plt.title(title)
    plt.xlabel(xlabel, fontsize=22)
    plt.ylabel(ylabel, fontsize=22)
    plt.savefig(path)
    plt.show()


def plot_grad_flow(named_parameters: Iterator[tuple[str, torch.nn.parameter.Parameter]]) -> None:
    """
    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
    Ref: https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10
    """
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            assert p.grad is not None
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])