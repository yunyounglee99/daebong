import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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