import random
import collections
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init = 0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.w_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.w_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.w_epsilon = self.register_buffer('w_epsilon', torch.empty(out_features, in_features))

        self.b_mu = nn.Parameter(torch.empty(out_features))
        self.b_sigma = nn.Parameter(torch.empty(out_features))
        self.b_epsilon = self.register_buffer('bㄴ_epsilon', torch.empty(out_features))

        self.reset_parameter()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.w_mu.data.uniform_(-mu_range, mu_range)
        self.w_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.b_mu.data.uniform_(-mu_range, mu_range)
        self.b_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.w_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.b_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            w = self.w_mu + self.w_sigma + self.w_epsilon
            b = self.b_mu + self.b_sigma + self.b_epsilon
        else:
            w = self.w_mu
            b = self.b_mu

        return F.linear(x, w, b)
    
class DQN(nn.Module):
    def __init__(
            self,
            name = 'dqn',
            obs_dim = 8, 
            a_dim = 2,
            h_dims = [256, 256],
            lr_critic = 3e-4,
            device = None
    ) -> None:
        super(DQN, self).__init___()
        self.name = name
        self.obs_idm = obs_dim
        self.a_dim = a_dim
        self.h_dims = h_dims
        self.lr_critic = lr_critic
        self.device = device

        # --- Dueling DQN Architecture ---
        # common feature layer
        self.feature_layers = []
        h_dim_prev = self.obs_idm
        for h_dim in self.h_dims:
            self.feature_layers.append(nn.Linear(h_dim_prev, h_dim))
            self.feature_layers.append(nn.ReLU())
            h_dim_prev = h_dim
        self.feature_net = nn.Sequential(*self.feature_layers).to(self.device)

        # Advantage Head
        self.advantage_head = nn.Sequential(
            NoisyLinear(h_dims[-1], h_dims[-1]),
            nn.ReLU(),
            NoisyLinear(h_dims[1], self.a_dim)
        ).to(self.device)

        # Value Head
        self.value_head = nn.Sequential(
            NoisyLinear(h_dims[-1], h_dims[-1]),
            nn.ReLU(),
            NoisyLinear(h_dims[1], 1)
        ).to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr = self.lr_critic)

    def forward(self, x):
        x = x.to(self.device)
        features = self.feature_net(x)

        v = self.value_head(features)
        a = self.advantage_head(features)

        q = v + (a - a.mean(dim=1, keepdim = True))

        return q
    
    def train(self, target, mini_batch, is_weights):
        s, a, r, s_prime, done = mini_batch
        q = self.forward(s)
        current_q = q.gather(1, a.long())

        td_error = torch.abs(target - current_q)
        loss = (is_weights * F.smooth_l1_loss(current_q, target, reduction = 'none')).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return td_error
    
    def soft_update(self, tau, net_target):
        for param_target, param in zip(net_target.parameters(), self.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

# actor
def get_action(q_network, s, deterministic = False):
    s = s.to(q_network.device)

    if not deterministic:
        # training : reset the noise of the Noisy Net for exploration
        q_network.reset_noise()

    with torch.no_grad():
        q = q_network(s)
        a = torch.argmax(q, dim = 1).unsqueeze(1)

    return a.item()

# caculatiing target
def get_target(q_main, q_target, gamma, mini_batch, device):
    q_main = q_main.to(device)
    q_target = q_target.to(device)
    s, a, r, s_prime, done = mini_batch

    with torch.no_grad():
        # extract next action(a') with max q value on main net
        q_main_next = q_main(s_prime)
        best_a_prime = torch.argmax(q_main_next, dim = 1).unsqueeze(-1)

        q_target_next = q_target(s_prime)
        target_q_val = q_target_next.gather(1, best_a_prime)

        target = r + gamma * done * target_q_val

    return target