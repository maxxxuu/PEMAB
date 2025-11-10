import numpy as np
import torch
import random
import copy
import logging
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim

from collections import deque
from IPython.display import clear_output


class OneStepLstmRNN(nn.Module):
    def __init__(self, load_path=None, in_dim=1, out_dim=1, hidden_dim=1, rnn_layers=2, model=None, learning_rate=1e-3,
                 gamma=0.9):
        super(OneStepLstmRNN, self).__init__()
        # self.load_path = load_path
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.rnn_layers = rnn_layers
        # self.learning_rate = learning_rate
        # self.gamma = gamma

        # Defining RNN layer
        # TODO: make sure input and out put are unbatched
        self.rnn = nn.LSTM(in_dim, hidden_dim, rnn_layers, batch_first=True)
        # self.rnn = nn.RNN(in_dim, hidden_dim, rnn_layers)
        # Defining fully connected layer
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 10)
        self.fc2 = nn.Linear(hidden_dim * 10, out_dim * 10)
        self.fc3 = nn.Linear(out_dim * 10, out_dim)

        self.loss_fn = nn.MSELoss()

    def forward(self, x, hidden_state=None, cell_state=None):
        batch_size = x.size(0)
        if hidden_state is None:
            hidden_state = self.init_hidden_state(batch_size=batch_size)
        if cell_state is None:
            cell_state = self.init_hidden_state(batch_size=batch_size)
        # hidden_state contains final hidden state of each layer
        output, (hidden_state, cell_state) = self.rnn(x, (hidden_state, cell_state))

        # Passing fully connected layer
        output = output[-1, ...].contiguous().view(-1, self.hidden_dim)
        output = self.fc1(output)
        # output = nn.LeakyReLU(output)
        output = self.fc2(output)
        # output = nn.LeakyReLU(output)
        output = self.fc3(output)

        return output, hidden_state, cell_state

    def init_hidden_state(self, batch_size):
        if batch_size == 1:
            hidden = torch.zeros(self.rnn_layers, self.hidden_dim, requires_grad=False)
        else:
            hidden = torch.zeros(self.rnn_layers, batch_size, self.hidden_dim, requires_grad=False)
        return hidden
