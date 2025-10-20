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

      # generate dummy data (s, a, r, s_prime, done)
      # 이 부분은 나중에 'reward_collector.py'가 데이터 받아와서 buffer.put을 호출하는 로직으로 대체
      # ===================================================

      with torch.no_grad():
        s_tensor = torch.from_numpy(s).float().to(DEVICE).unsqueeze(0)
        a, _ = actor.get_action_logprob(s_tensor, deterministic = False)
        a = a.item()

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

      # caculate target Q-value
      target = get_target(
        actor, critic1_target, critic2_target, config.GAMMA, mini_batch, device
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

      if done:
        break

    # logging training process
    if episode % config.LOG_INTERVAL == 0:
      print(f"Epi : {episode}, Total Steps : {total_steps}, Avg Reward : {episode_reward/config.MAX_STEPS_PER_EPISODE:.4f}")

    # save the model periodically
    if episode % config.SAVE_INTERVAL == 0 and episode > 0:
      model_path = os.path.join(config.RL_MODEL_PATH, f'sac_actor_ep{episode}.pth')
      torch.save(actor.state_dict(), model_path)
      print(f'Model saved at : {model_path}')

if __name__ == '__main__':
  train()