# excution exsample : python training/train_rl.py

import torch
import json
import numpy as np
import os
import time
import src.config as config
from src.utils.replay_buffer import PERBufferClass
from src.models.rl_model_SAC import ActorClass, CriticClass, get_target as get_target_sac
from src.models.rl_model_DQN import (
  DQN,
  get_action as get_action_dqn,
  get_target as get_target_dqn
)
from training.utils import flatten_state_to_vector, calculate_reward

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device : {DEVICE}')
print(f'Running Model Type : {config.MODEL_TYPE}')

if config.MODEL_TYPE == 'SAC':
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
  # --- SAC Training ---
  if config.MODEL_TYPE == 'SAC':
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

  elif config.MODEL_TYPE == 'DQN':
    q_main = DQN(
      obs_dim = config.OBS_DIM,
      a_dim = config.ACTION_DIM,
      h_dims = config.HIDDEN_DIMS,
      lr_critic = config.LR_CRITIC,
      device = DEVICE
    )

    q_target = DQN(
      obs_dim = config.OBS_DIM,
      a_dim = config.ACTION_DIM,
      h_dims = config.HIDDEN_DIMS,
      lr_critic = config.LR_CRITIC,
      device = DEVICE
    )

    q_target.load_state_dict(q_main.state_dict())

  # --- load checkpoint ---
  start_episode = 0
  total_steps_start = 0

  if config.LOAD_CHECKPOINT:
    checkpoint_path = os.path.join(config.RL_MODEL_PATH, config.CHECKPOINT_FILE_TO_LOAD)
    if os.path.exists(checkpoint_path):
      print(f'Loading checkpoint from : {checkpoint_path}')
      checkpoint = torch.load(checkpoint_path, map_location = DEVICE)

      if config.MODEL_TYPE == 'SAC':
        actor.load_state_dict(checkpoint['actor_state_dict'])
        critic1.load_state_dict(checkpoint['critic1_state_dict'])
        critic2.load_state_dict(checkpoint['critic2_state_dict'])

        critic1_target.load_state_dict(checkpoint['critic1_state_dict'])
        critic2_target.load_state_dict(checkpoint['critic2_state_dict'])

        actor.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        actor.log_alpha_optimizer.load_state_dict(checkpoint['log_alpha_optimizer_state_dict'])
        critic1.critic_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        critic2.critic_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])

        actor.log_alpha = checkpoint['log_alpha']

      elif config.MODEL_TYPE == 'DQN':
        q_main.load_state_dict(checkpoint['q_main_state_dict'])
        q_target.load_state_dict(checkpoint['q_main_state_dict'])
        q_main.optimizer.load_state_dict(checkpoint['q_main_optimizer_state_dict'])

      start_episode = checkpoint['episode'] + 1
      total_steps_start = checkpoint['total_steps']

      print(f'Checkpoint loaded. Resuming from episode {start_episode}')
    else:
      print(f'Checkpoint file not found at : {checkpoint_path}. Starting from scratch')

    # ---------------

  os.makedirs(config.RL_MODEL_PATH, exist_ok = True)

  print("Training started...")

  # Main Loop
  DATA_FILE = 'data/raw/dummy_data_100.json'
  print(f'Loading transitions from {DATA_FILE}...')
  loaded_count = 0
  try:
    with open(DATA_FILE, 'r', encoding = 'utf-8') as f:
      transitions = json.load(f)

      for trans in transitions:
        s = flatten_state_to_vector(trans['state_t'])
        a = trans['action_t']
        r = calculate_reward(trans['reward_ingredients'])
        s_prime = flatten_state_to_vector(trans['next_state_t_plus_1'])
        done = trans['done']

        if s is not None and s_prime is not None:
          buffer.put((s, a, r, s_prime, done))
          loaded_count += 1

    print(f'Successfully loaded and parsed {loaded_count} transitions into buffer.')

  except FileNotFoundError:
    print(f'Error: {DATA_FILE} not found. Starting with an empth buffer')
  except json.JSONDecodeError:
    print(f'Error: Could not decode {DATA_FILE}. Is it a valid JSON list?')
  except Exception as e:
    print(f'An error occured while loading data: {e}')
  
  if buffer.size() < config.BATCH_SIZE:
    print(f'Warning: Buffer size ({buffer.size()}) is smaller than BATCH_SIZE ({config.BATCH_SIZE})')
    print('Training will not start unless BATCH_SIZE in config.py is lowered.')

  os.makedirs(config.RL_MODEL_PATH, exist_ok = True)
  print('Traing started...')

  total_steps = total_steps_start

  for total_steps in range(total_steps_start, config.NUM_TRAINING_STEPS):
    if buffer.size() < config.BATCH_SIZE:
      print("Buffer has insufficient data to contine training. Stopping.")
      break

    s_b, a_b, r_b, s_p_b, done_mask_b, is_weights_b, idxs_b = buffer.sample(config.BATCH_SIZE)
    mini_batch = (s_b, a_b, r_b, s_p_b, done_mask_b)

    if config.MODEL_TYPE == 'SAC':
      # caculate target Q-value
      target = get_target_sac(
        actor, critic1_target, critic2_target, config.GAMMA, mini_batch, DEVICE
      )

      # training critic
      td_error1 = critic1.train(target, mini_batch, is_weights_b)
      td_error2 = critic2.train(target, mini_batch, is_weights_b)

      # PER update
      avg_td_error = (td_error1 + td_error2) / 2
      buffer.update_priorities(idxs_b, avg_td_error)

      # training actor
      if total_steps % config.ACTOR_UPDATE_DELAY == 0:
        actor.train(critic1, critic2, TARGET_ENTROPY, s_b)

        # target net soft update
        critic1.soft_update(config.TAU, critic1_target)
        critic2.soft_update(config.TAU, critic2_target)
      
    elif config.MODEL_TYPE == 'DQN':
      # calculate the target
      target = get_target_dqn(
        q_main, q_target, config.GAMMA, mini_batch, DEVICE
      )

      #training Q-Net
      td_error = q_main.train(target, mini_batch, is_weights_b)

      # PER update
      buffer.update_priorities(idxs_b, td_error)

      # Target Net update
      if total_steps % config.TARGET_UPDATE_INTERVAL == 0:
        q_main.soft_update(config.TAU, q_target)

    if done:
      break

    # logging training process
    if total_steps % config.LOG_INTERVAL == 0:
      print(f"Epi : {total_steps}, Total Steps : {total_steps}, TD-Error (mean): {td_error.mean().item():.4f}")
    # save the model periodically
    if total_steps % config.SAVE_INTERVAL == 0 and total_steps > 0:

      if config.MODEL_TYPE == 'SAC':
        checkpoint = {
          'episode' : total_steps,
          'total_steps' : total_steps,
          'actor_state_dict' : actor.state_dict(),
          'critic1_state_dict' : critic1.state_dict(),
          'critic2_state_dict' : critic2.state_dict(),
          'actor_optimizer_state_dict' : actor.actor_optimizer.state_dict(),
          'log_alpha_optimizer_state_dict' : actor.log_alpha_optimizer.state_dict(),
          'critic1_optimizer_state_dict' : critic1.critic_optimizer.state_dict(),
          'critic2_optimizer_state_dict' : critic2.critic_optimizer.state_dict(),
          'log_alpha' : actor.log_alpha
        }

      elif config.MODEL_TYPE == 'DQN':
        checkpoint = {
          'episode' : total_steps,
          'total_steps' : total_steps,
          'q_main_state_dict' : q_main.state_dict(),
          'q_main_optimizer_state_dict' : q_main.optimizer.state_dict()
        }

      checkpoint_path = os.path.join(config.RL_MODEL_PATH, f'checkpoint_ep{total_steps}.pth')
      torch.save(checkpoint, checkpoint_path)

      print(f'\n--- Full checkpoint saved at : {checkpoint_path} ---')

if __name__ == '__main__':
  train()