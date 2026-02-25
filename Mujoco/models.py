import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim: list[int]):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim[0]),
            nn.ReLU(),
        )
        for i in range(len(hidden_dim) - 1):
            self.network.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
            self.network.append(nn.ReLU())
        self.network.append(nn.Linear(hidden_dim[-1], action_dim * 2))
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        mu = self.network(x)
        std = torch.exp(self.log_std)
        return mu, std


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim: list[int]):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim[0]),
            nn.ReLU(),
        )
        for i in range(len(hidden_dim) - 1):
            self.network.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
            self.network.append(nn.ReLU()),
        self.network.append(nn.Linear(hidden_dim[-1], 1))

    def forward(self, x):
        return self.network(x).squeeze(-1)
