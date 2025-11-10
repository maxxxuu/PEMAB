import torch
import torch.nn as nn


class PEPoolingIdv(nn.Module):
    def __init__(self, in_dim, out_dim, model=None, pooling="mean"):
        super(PEPoolingIdv, self).__init__()
        assert pooling in {"mean", "max", "sum"}
        self.pooling = pooling
        if model is None:
            self.idv_nn = nn.Linear(in_dim, out_dim)
            self.pooling_nn = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.idv_nn = model
            self.pooling_nn = model


    def forward(self, x):
        if self.pooling == "mean":
            x_pooling = x.mean(dim=-2, keepdim=True)
        elif self.pooling == "max":
            x_pooling = x.max(dim=-2, keepdim=True)
        elif self.pooling == "sum":
            x_pooling = x.sum(dim=-2, keepdim=True)
        x_pooling = self.pooling_nn(x_pooling)
        x = self.idv_nn(x)
        x = x + x_pooling
        return x

