import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

#Actor (discrete)
class ActorClass(nn.Module):
    def __init__(self,
                name = 'actor',
                obs_dim = 8,    # need to modify to suit the project 
                h_dims = [256, 256],
                a_dim = 2,    # recommend or not
                init_alpha = 0.1,
                lr_actor = 0.0003,
                lr_alpha = 0.0003,
                device = None) -> None:
        super(ActorClass, self).__init__()
        #initialize
        self.name = name
        self.obs_dim = obs_dim
        self.h_dims = h_dims
        self.a_dim = a_dim
        self.init_alpha = init_alpha
        self.lr_actor = lr_actor
        self.lr_alpha = lr_alpha
        self.device = device
        self.init_layers()
        self.init_params()
        # set optimizer
        self.actor_optimizer = optim.Adam(self.parameters(), lr = self.lr_actor)
        self.log_alpha = torch.tensor(np.log(self.init_alpha), requires_grad=True, dtype = torch.float32, device = self.device)
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr = self.lr_alpha)

    def init_layers(self):
        """
        initialize layers
        """
        self.layers = {}
        h_dim_prev = self.obs_dim
        for h_idx, h_dim in enumerate(self.h_dims):
            self.layers['mlp_{}'.format(h_idx)] = nn.Linear(h_dim_prev, h_dim)
            self.layers['relu_{}'.format(h_idx)] = nn.ReLU()
            h_dim_prev = h_dim        
        self.layers['logits'] = nn.Linear(h_dim_prev, self.a_dim)

        self.param_dict = {}
        for key in self.layers.keys():
            layer = self.layers[key]
            if isinstance(layer,nn.Linear):
                self.param_dict[key+'_w'] = layer.weight
                self.param_dict[key+'_b'] = layer.bias
        self.parameters = nn.ParameterDict(self.param_dict)

    def init_params(self):
        """
            Initialize parameters
        """
        for key in self.layers.keys():
            layer = self.layers[key]
            if isinstance(layer,nn.Linear):
                nn.init.normal_(layer.weight,mean=0.0,std=0.01)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer,nn.BatchNorm2d):
                nn.init.constant_(layer.weight,1.0)
                nn.init.constant_(layer.bias,0.0)
            elif isinstance(layer,nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self,x,SAMPLE_ACTION=True):
        """
            Forward
        """
        x = x.to(self.device)
        for h_idx, _ in enumerate(self.h_dims):
            x = self.layers['relu_{}'.format(h_idx)](self.layers['mlp_{}'.format(h_idx)](x))
        
        # calculate
        logits = self.layers['logits'](x)
        probs = F.softmax(logits, dim = -1)
        log_probs = F.log_softmax(logits, dim = -1)

        return probs, log_probs     # probabilities for [recommend or not]
    
    def get_action_prob(self, s, deterministic = False):
        """
        sampling the real actions and return log probabilities
        """
        probs, log_probs = self.forward(s)
        dist = torch.distributions.Categorical(probs)

        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            action = dist.sample()

        # log prob for specific actions
        # .gathers acts same as dist.log_prob(action)
        action_log_probs = log_probs.gather(1, action.unsqueeze(-1))

        return action, action_log_probs

    def train(self,
                q_1,
                q_2,
                target_entropy,
                s_batch):
        """
            Train
        """
        probs, log_probs = self.forward(s_batch)

        #calculate Q-value
        q_1_value = q_1(s_batch)
        q_2_value = q_2(s_batch)
        mini_q_value = torch.min(q_1_value, q_2_value)

        alpha = self.log_alpha.exp().detach()

        actor_loss = (probs * (alpha * log_probs - mini_q_value.detach())).sum(dim=1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        entropy = -(probs * log_probs).sum(dim=-1)
        alpha_loss = -(self.log_alpha.exp() * (entropy.detach() + target_entropy)).mean()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()



# Critic
class CriticClass(nn.Module):
    def __init__(self,
                name      = "critic",
                obs_dim   = 8,     # state dimension / need to modify to suit the project 
                a_dim     = 2,     # recommend or not
                h_dims    = [256,256],
                out_dim   = 1,
                lr_critic = 0.0003,
                device    = None) -> None:
        """
            Initialize Critic
        """
        super(CriticClass, self).__init__()
        # Initialize
        self.name      = name
        self.obs_dim   = obs_dim
        self.a_dim     = a_dim
        self.h_dims    = h_dims
        self.out_dim   = self.a_dim
        self.lr_critic = lr_critic
        self.device    = device
        self.init_layers()
        self.init_params()
        # Set optimizer
        self.critic_optimizer = optim.Adam(self.parameters(),lr=self.lr_critic)

    def init_layers(self):
        """
            Initialize layers
        """
        self.layers = {}
        h_dim_prev = self.obs_dim
        for h_idx, h_dim in enumerate(self.h_dims):
            self.layers[f'mlp_{h_idx}'] = nn.Linear(h_dim_prev, h_dim)
            self.layers[f'relu_{h_idx}'] = nn.ReLU()
            h_dim_prev = h_dim
        self.layers['out'] = nn.Linear(h_dim_prev, self.out_dim)     # output : [Q(s, a_0), Q(s, a_1)]

        # Accumulate layers weights
        self.param_dict = {}
        for key in self.layers.keys():
            layer = self.layers[key]
            if isinstance(layer,nn.Linear):
                self.param_dict[key+'_w'] = layer.weight
                self.param_dict[key+'_b'] = layer.bias
        self.parameters = nn.ParameterDict(self.param_dict)

    def init_params(self):
        """
            Initialize parameters
        """
        for key in self.layers.keys():
            layer = self.layers[key]
            if isinstance(layer,nn.Linear):
                nn.init.normal_(layer.weight,mean=0.0,std=0.01)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer,nn.BatchNorm2d):
                nn.init.constant_(layer.weight,1.0)
                nn.init.constant_(layer.bias,0.0)
            elif isinstance(layer,nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
                
    def forward(self, x):
        x = x.to(self.device)
        for h_idx, _ in enumerate(self.h_dims):
            x = self.layers[f'relu_{h_idx}'](self.layers[f'mlp_{h_idx}'](x))
        q = self.layers['out'](x)     # output : [batch_size, a_dim] 
        return q
    
    def train(self,
            target,
            mini_batch,
            is_weights):
        """
            Train
        """
        s, a, r, s_prime, done = mini_batch
        
        # return Q_value only using state(s)
        # gather real action(a)'s Q_value
        q_values = self.forward(s)
        current_q = q_values.gather(1, a.long())     # a need to be [N, 1]

        td_error = torch.abs(target - current_q)
        loss = (is_weights * F.smooth_l1_loss(current_q, target, reduction='none')).mean()

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return td_error
        
    def soft_update(self, tau, net_target):
        """
            Soft update of Critic
        """
        for param_target, param in zip(net_target.parameters(), self.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
            
# Bellman backup operator
def get_target(pi, q1, q2, gamma, mini_batch, device):
    q1 = q1.to(device)
    q2 = q2.to(device)
    pi = pi.to(device)
    s, a, r, s_prime, done = mini_batch
    with torch.no_grad():
        next_probs, next_log_probs = pi(s_prime)
        alpha = pi.log_alpha.exp()
        entropy = -alpha * next_log_probs
        
        q1_val = q1(s_prime)
        q2_val = q2(s_prime)
        min_q = torch.min(q1_val, q2_val)

        # V(s') = Σ [ probs(a'|s') * ( Q_target(s',a') + entropy(a'|s') ) ]
        soft_v_prime = (next_probs * (min_q + entropy)).sum(dim=1, keemdim=True)

        target = r + gamma * done * soft_v_prime

    return target 