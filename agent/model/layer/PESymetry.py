import torch
import torch.nn as nn


class PESymetryMean(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super(PESymetryMean, self).__init__()
        self.diagonal = nn.Linear(in_dim, out_dim)
        self.rest = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x_mean = x.mean(0, keepdim=True)
        x_mean = x.mean(-2, keepdim=True)
        x_mean = self.rest(x_mean)
        x = self.diagonal(x)
        x = x + x_mean
        return x


class PESymetryMax(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super(PESymetryMax, self).__init__()
        self.diagonal = nn.Linear(in_dim, out_dim)
        self.rest = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_max, _ = x.max(0, keepdim=True)
        x_max = self.rest(x_max)
        x = self.diagonal(x)
        x = x + x_max
        return x

class PESymetryMeanAct(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super(PESymetryMeanAct, self).__init__()
        self.diagonal = nn.Linear(in_dim, out_dim)
        self.rest = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x_mean = x.mean(0, keepdim=True)
        x_mean = x.mean(-2, keepdim=True)
        x_mean = self.rest(x_mean)
        x_mean = nn.ELU()(x_mean)
        x = self.diagonal(x)
        # x = nn.ELU()(x)
        x = x + x_mean
        return x