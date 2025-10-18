import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

def list2torch(x, device):
    """
    List to torch
    """
    return torch.tensor(np.array(x), dtype = torch.float).to(device)

#PER(Prioritized Experience Replay)
class Sumtree:
    """
    algorithm for PER to sample faster, more efficient
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype = object)
        self.n_entries = 0
        self.write = 0

    def _propagate(self, idx, change):
        parent = (idx - 1)//2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
        
    def total(self):
        return self.tree[0]
    
    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)
    
    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])
    
class PERBufferClass:
    """
    Prioritized Experience Replay (PER) Buffer for efficient sampling
    """
    def __init__(self, 
                buffer_limit = 100000, 
                device = 'cpu', 
                alpha = 0.6, 
                beta = 0.4, 
                beta_increment = 0.001):
        self.tree = Sumtree(buffer_limit)
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.capacity = buffer_limit
        self.max_priority = 1.0
        self.epsilon = 1e-6 # to prevent the priority becomes 0
    
    def put(self, item):
        """
        save the item in the buffer (in max priority)
        """
        self.tree.add(self.max_priority, item)

    def size(self):
        return self.tree.n_entries
    
    def sample(self, n):
        """
        sample based on priority
        """
        mini_batch = []
        idxs = []
        is_weights = np.empty(n, dtype = np.float32)
        segment = self.tree.total() / n
        self.beta = np.min([1., self.beta + self.beta_increment]) # beta anyling

        for i in range(n):
            s = random.uniform(segment * i, segment * (i+1))
            (idx, p, data) = self.tree.get(s)

            prob = p / self.tree.total()
            is_weights[i] = np.power(self.size() * prob, -self.beta) # calculate priority sampling weight

            mini_batch.append(data)
            idxs.append(idx)
    
        s_list, a_list, r_list, s_prime_list, done_mask_list = [], [], [], [], []
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_list.append(s)
            a_list.append([a]) #as action is scalar(0 or 1)
            r_list.append([r])
            s_prime_list.append(s_prime)
            done_mask = 0.0 if done_mask else 1.0
            done_mask_list.append(done_mask)

        is_weights /= is_weights.max() # normalize is_weight

        return (list2torch(s_list, self.device),
                list2torch(a_list, self.device),
                list2torch(r_list, self.device),
                list2torch(s_prime_list, self.device),
                list2torch(done_mask_list, self.device),
                torch.tensor(is_weights, dtype=torch.float).to(self.device).reshape(-1, 1), idx)
    
    def update_priorities(self, batch_indices, td_errors):
        """
        update the priority of sampling batch (based on TD-Error)
        """
        td_errors = td_errors.detach().cpu().numpy()
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha

        for idx, p in zip(batch_indices, priorities):
            self.tree.update(idx, p[0])
            self.max_priority = max(self.max_priority, p[0])

    def clear(self):
        self.__init__(self.capacity, self.device)

#Actor (discrete)
class ActorClass(nn.Module):
    def __init__(self,
                name = 'actor',
                obs_dim = 8,    # need to modificate to suit the project 
                h_dims = [256, 256],
                a_dim = 2,    #recommend or not
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
                obs_dim   = 75,
                a_dim     = 8,
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
        self.out_dim   = out_dim
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
        h_dim_prev = self.h_dims[0]
        for h_idx, h_dim in enumerate(self.h_dims):
            if h_idx == 0:
                self.layers['obs']      = nn.Linear(self.obs_dim, int(self.h_dims[0]/2))
                self.layers['obs_relu'] = nn.ReLU()
                self.layers['act']      = nn.Linear(self.a_dim, int(self.h_dims[0]/2))
                self.layers['act_relu'] = nn.ReLU()
            else:
                self.layers['mlp_{}'.format(h_idx)] = nn.Linear(h_dim_prev, h_dim)
                self.layers['relu_{}'.format(h_idx)] = nn.ReLU()
            h_dim_prev = h_dim
        self.layers['out'] = nn.Linear(h_dim_prev, self.out_dim)

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
                
    def forward(self,
                x,
                a):
        x = x.to(self.device)
        a = a.to(self.device)
        for h_idx, _ in enumerate(self.h_dims):
            if h_idx == 0:
                x = self.layers['obs_relu'](self.layers['obs'](x))
                a = self.layers['act_relu'](self.layers['act'](a))
                cat = torch.cat([x,a], dim=1)
            else:
                q = self.layers['relu_{}'.format(h_idx)](self.layers['mlp_{}'.format(h_idx)](cat))
        q = self.layers['out'](q)
        return q
    
    def train(self,
              target,
              mini_batch):
        """
            Train
        """
        s, a, r, s_prime, done = mini_batch
        critic_loss = F.smooth_l1_loss(self.forward(s,a), target)
        self.critic_optimizer.zero_grad()
        critic_loss.mean().backward()
        self.critic_optimizer.step()
        
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
        # We use the action the current 'pi' making SAC off-policy
        a_prime, log_prob = pi(s_prime)
        entropy = -pi.log_alpha.exp() * log_prob # incentivize exploration
        
        # Use the minimum Q among q1 and q2 (to avoid over-confident issue)
        q1_val, q2_val = q1(s_prime,a_prime), q2(s_prime,a_prime)
        q = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q, 1, keepdim=True)[0]
        
        # Get target using Bellman backup operator
        target = r + gamma * done * (min_q + entropy.mean()) # Eg. (3)
    return target 