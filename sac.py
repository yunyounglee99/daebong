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

#Replay Buffer
class ReplayBufferClass():
  def __init__(self, buffer_limit=100000, device='cpu') -> None:
    """
    Initialize Buffer
    """
    self.buffer = collections.deque(maxlen=buffer_limit)
    self.device = device

  def size(self):
    """
    Get buffer size
    """
    return len(self.buffer)
  
  def clear(self):
    """
    Clear buffer
    """
    self.buffer.clear()

  def put(self, item):
    """"
    Put item in buffer
    """
    self.buffer.append(item)

  def put_mini_batch(self, mini_batch):
    """
    Put mini batch in buffer
    """
    for transition in mini_batch:
      self.put(transition)

  def sample(self, n):
    """
    Sampling 
    """
    mini_batch = random.sample(self.buffer, n)
    s_list, a_list, r_list, s_prime_list, done_mask_list = [], [], [], [], []
    for transition in mini_batch:
      s, a, r, s_prime, done_mask = transition
      s_list.append(s)
      a_list.append(a)
      r_list.append([r])
      s_prime_list.append(s_prime)
      done_mask = 0.0 if done_mask else 1.0
      done_mask_list.append([done_mask])
    
    return list2torch(s_list, self.device), list2torch(a_list, self.device), list2torch(r_list, self.device), list2torch(s_prime_list, self.device), list2torch(done_mask_list, self.device)
  


