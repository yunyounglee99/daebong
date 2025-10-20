# excution exsample : python training/train_rl.py

import torch
import numpy as np
import os
import time
import src.config as config
from src.utils.replay_buffer import PERBufferClass
from src.models.rl_model import ActorClass, CriticClass, get_target

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device : {DEVICE}')

TARGET_ENTROPY = -np.log(1.0 / config.ACTION_DIM) * 0.98

def train():
  print('Initializing models and buffer...')

  buffer = PERBufferClass(
    buffer_limit=config.BUFFER_LIMIT,
    device = DEVICE,
    alpha = config.PER_ALPHA,
    beta = config.PER_BETA_START,
    beta_increment = config.PER_BETA_INCREMENT
  )

  actor = ActorClass(
    obs_dim = config.OBS_DIM,
    a_dim = config.ACTION_DIM,
    h_dims = config.HIDDEN_DIMS,
    device = DEVICE
  ).to(DEVICE)

  critic1 = CriticClass(
    obs_dim = config.OBS_DIM,
    a_dim = config.ACTION_DIM,
    h_dims = config.HIDDEN_DIMS,
    device = DEVICE
  ).to(DEVICE)

  critic2 = CriticClass(
    obs_dim = config.OBS_DIM,
    a_dim = config.ACTION_DIM,
    h_dims = config.HIDDEN_DIMS,
    device = DEVICE
  ).to(DEVICE)

  critic1_target = CriticClass(
    obs_dim = config.OBS_DIM,
    a_dim = config.ACTION_DIM,
    h_dims = config.HIDDEN_DIMS,
    device = DEVICE
  ).to(DEVICE)

  critic2_target = CriticClass(
    obs_dim = config.OBS_DIM,
    a_dim = config.ACTION_DIM,
    h_dims = config.HIDDEN_DIMS,
    device = DEVICE
  ).to(DEVICE)

  critic1_target.load_state_dict(critic1.state_dict())
  critic2_target.load_state_dict(critic2.state_dict())

  os.makedirs(config.RL_MODEL_PATH, exist_ok = True)

  print("Training started...")

  # Main Loop

  total_steps = 0
  for episode in range(config.NUM_EPISODES):
    """
    now using dummy data please change codes below after you get data
    """ 
    s = np.random.rand(config.OBS_DIM).astype(np.float32) # this is dummy state you must change here to real state
    episode_reward = 0

    for step in range(config.MAX_STEPS_PER_EPISODE):
      total_steps += 1

