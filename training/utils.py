import torch
import numpy as np
import os
import time
import json
import src.config as config
from src.utils.replay_buffer import PERBufferClass

# This function is for parsing the state, reward -> convert the json datum to 1d vector(state) and scalar(reward)
def flatten_state_to_vector(state_obj):
  """
  convert the object to 1d numpy vector
  should be same with the config's OBS_DIM !
  """
  try:
    s_profile = state_obj['user_profile']['user_group_vector']

    s_history = state_obj['session_history']['last_clicked_item_vector'] \
                    + [state_obj['session_history']['session_click_count']] \
                    + [state_obj['session_history']['session_total_view_time']] \
                    + state_obj['session_history']['category_affinity_vector']
    
    s_candidate = [
            state_obj['candidate_item_info']['predicted_price'],
            state_obj['candidate_item_info']['lead_time'],
            state_obj['candidate_item_info']['predicted_quality'],
            state_obj['candidate_item_info']['promotion']
    ]

    flat_vector = np.concatenate([s_profile, s_history, s_candidate]).astype(np.float32)

    # check the dimension whether it is same with the OBS_DIM
    if len(flat_vector) != config.OBS_DIM:
      raise ValueError(f'Error : Flattened vector dim ({len(flat_vector)}) != config.OBS_DIM ({config.OBS_DIM})')
  
    return flat_vector
  
  except KeyError as e:
    print(f'State JSON parsing error : cannot find key {e} ')
    return None
  except TypeError as e:
    print(f'State JSON parsing error : {e}')
    return None
  
def calculate_reward(sources):
  clicked = sources.get('clicked', 0)
  view_time = sources.get('page_view_time_sec', 0)
  purchased = sources.get('purchased', 0)

  reward = 0.0

  if clicked == 0:
    reward = 0.0

  elif view_time < 5:
    reward = -0.5
  else:
    reward = 1.0 + (view_time / 60.0)

  if purchased == 1:
    reward += 5.0

  return reward