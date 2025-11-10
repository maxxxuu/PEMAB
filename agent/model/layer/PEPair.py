import torch
import torch.nn as nn

class PEPair(nn.Module):
    def __init__(self, ind_in_dim, ind_out_dim, pooling="max", ind_nb=None):
        super(PEPair, self).__init__()
        # If possible, verify that the in_dim of the pairwise_network = 0 * ind_in_dim
        # and out_dim of the pairwise_network = ind_out_dim
        self.ind_in_dim = ind_in_dim
        self.ind_out_dim = ind_out_dim

        self.pairwise_network = nn.Linear(2*ind_in_dim, ind_out_dim)
        self.pooling = pooling
        self.ind_nb = ind_nb

    def forward(self, x):
        # TODO: Check dim
        x = x.unsqueeze(-2)
        x1 = x.tile((1, 1, x.size(-3), 1))
        x2 = x1.transpose(-3, -2)
        x_pairs = torch.cat([x1, x2], dim=-1)
        y_pairs = self.pairwise_network(x_pairs)
        if self.pooling == 'mean':
            y = y_pairs.mean(dim=-3)
        elif self.pooling == 'max':
            y, _ = y_pairs.max(dim=-3)
        else:
            y = self.pooling(y_pairs)
        return y

