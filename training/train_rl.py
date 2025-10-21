# excution exsample : python training/train_rl.py

import torch
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

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device : {DEVICE}')
print(f'Running Model Type : {config.MODEL_TYPE}')

if config.MODEL_TYPE == 'SAC':
  TARGET_ENTROPY = -np.log(1.0 / config.ACTION_DIM) * 0.98

def train():
  print('Initializing models and buffer...')

  # --- SAC Training ---
  if config.MODEL_TYPE == 'SAC':
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

  total_steps = 0
  for episode in range(config.NUM_EPISODES):
    """
    now using dummy data please change codes below after you get data
    """ 
    s = np.random.rand(config.OBS_DIM).astype(np.float32) # this is dummy state you must change here to real state
    episode_reward = 0

    for step in range(config.MAX_STEPS_PER_EPISODE):
      total_steps += 1

      # generate dummy data (s, a, r, s_prime, done)
      # 이 부분은 나중에 'reward_collector.py'가 데이터 받아와서 buffer.put을 호출하는 로직으로 대체
      # ===================================================

      with torch.no_grad():
        s_tensor = torch.from_numpy(s).float().to(DEVICE).unsqueeze(0)

        if config.MODEL_TYPE == 'SAC':
          a, _ = actor.get_action_logprob(s_tensor, deterministic = False)
          a = a.item()

        elif config.MODEL_TYPE == 'DQN':
          a = get_action_dqn(q_main, s_tensor, deterministic = False)

      s_prime = np.random.rand(config.OBS_DIM).astype(np.float32)

      # sparse reward as in real world, clicking recommended products is rare
      # if success(click), reward = +1, or not 0
      r = 1.0 if np.random.rand() < 0.05 else 0.0
      done = True if step == config.MAX_STEPS_PER_EPISODE -1 else False

      # save (s, a, r, s_prime, done) in buffer
      buffer.put((s, a, r, s_prime, done))

      episode_reward += r
      s = s_prime

      # End simulation
      # ===================================================

      # begin training when sufficient data accumulates in the buffer
      if buffer.size() < config.BATCH_SIZE:
        continue

      # sample in the PER buffer
      s_b, a_b, r_b, s_p_b, done_b, is_weights_b, idxs_b = buffer.sample(config.BATCH_SIZE)
      mini_batch = (s_b, a_b, r_b, s_p_b, done_b)

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
    if episode % config.LOG_INTERVAL == 0:
      print(f"Epi : {episode}, Total Steps : {total_steps}, Avg Reward : {episode_reward/config.MAX_STEPS_PER_EPISODE:.4f}")

    # save the model periodically
    if episode % config.SAVE_INTERVAL == 0 and episode > 0:

      if config.MODEL_TYPE == 'SAC':
        checkpoint = {
          'episode' : episode,
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
          'episode' : episode,
          'total_steps' : total_steps,
          'q_main_state_dict' : q_main.state_dict(),
          'q_main_optimizer_state_dict' : q_main.optimizer.state_dict()
        }

      checkpoint_path = os.path.join(config.RL_MODEL_PATH, f'checkpoint_ep{episode}.pth')
      torch.save(checkpoint, checkpoint_path)

      print(f'\n--- Full checkpoint saved at : {checkpoint_path} ---')

if __name__ == '__main__':
  train()