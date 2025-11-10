import torch
import torch.nn as nn
import torch.nn.functional as F
from agent.model.layer.PESymetry import PESymetryMean, PESymetryMax
from agent.model.layer.ScaledSelfAttention import ScaledSelfAttention


class PEA(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PEA, self).__init__()
        self.PE = PESymetryMean(in_dim=in_dim, out_dim=out_dim)
        self.attention = ScaledSelfAttention(emb_dim=out_dim, q_dim=out_dim, v_dim=out_dim)

    def forward(self, x):
        output_pe = self.PE(x)
        return output_pe + self.attention(output_pe)


class PEA_Gated(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PEA_Gated, self).__init__()
        self.diagonal = nn.Linear(in_dim, out_dim)
        self.rest = nn.Linear(in_dim, out_dim, bias=False)
        self.gate_pooling = nn.Sequential(nn.Linear(out_dim, out_dim, bias=False), nn.Sigmoid())
        self.attention = ScaledSelfAttention(emb_dim=out_dim, q_dim=out_dim, v_dim=out_dim)
        self.gate_attention = nn.Sequential(nn.Linear(out_dim, out_dim, bias=False), nn.Sigmoid())

    def forward(self, x):
        x_mean = x.mean(0, keepdim=True)
        x_mean = self.rest(x_mean)
        x = self.diagonal(x)
        output_pe = x + (1 - self.gate_pooling(x_mean)) * x_mean
        output_attention = self.attention(output_pe)

        return output_pe + (1 - self.gate_attention(output_pe)) * output_attention

class PEA_Gated2(nn.Module):
    # Change the order of calculation: Att -> PE
    def __init__(self, in_dim, out_dim):
        super(PEA_Gated2, self).__init__()
        self.diagonal = nn.Linear(in_dim, out_dim)
        self.rest = nn.Linear(in_dim, out_dim, bias=False)
        self.gate_pooling = nn.Sequential(nn.Linear(out_dim, out_dim, bias=False), nn.Sigmoid())
        self.attention = ScaledSelfAttention(emb_dim=in_dim, q_dim=in_dim, v_dim=in_dim)
        self.gate_attention = nn.Sequential(nn.Linear(in_dim, in_dim, bias=False), nn.Sigmoid())

    def forward(self, x):

        gate_attention = self.gate_attention(x)
        output_attention = x + (1 - gate_attention) * self.attention(x)
        x_mean = output_attention.mean(0, keepdim=True)
        x_mean = self.rest(x_mean)
        x = self.diagonal(output_attention)
        output_pe = x + (1 - self.gate_pooling(x_mean)) * x_mean

        return output_pe

