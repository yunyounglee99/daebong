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
    state_t 또는 next_state_t_plus_1 객체를 1D numpy 벡터로 변환합니다.
    ★ nan 방어 및 차원 체크 강화 ★
    """
    flat_vector_list = []
    
    try:
        # User Profile (기본값: [0, 0], 0.0)
        user_profile = state_obj.get('user_profile', {})
        flat_vector_list.extend(user_profile.get('user_group_vector', [0.0, 0.0]))
        # flat_vector_list.append(user_profile.get('ltv_score', 0.0)) # LTV 사용 시
        
        session_history = state_obj.get('session_history', {})
        
        vec_dim_item = 32 
        vec_dim_cat = 10 
        flat_vector_list.extend(session_history.get('last_clicked_item_vector', [0.0] * vec_dim_item))
        flat_vector_list.append(session_history.get('session_click_count', 0.0))
        flat_vector_list.append(session_history.get('session_total_view_time', 0.0))
        flat_vector_list.extend(session_history.get('category_affinity_vector', [0.0] * vec_dim_cat))
        
        # Candidate Item (기본값: 0)
        candidate_item = state_obj.get('candidate_item_info', {})
        flat_vector_list.append(candidate_item.get('predicted_price', 0.0))
        flat_vector_list.append(candidate_item.get('lead_time', 0.0))
        flat_vector_list.append(candidate_item.get('predicted_quality', 0.0))
        flat_vector_list.append(candidate_item.get('promotion', 0.0))
        # flat_vector_list.append(candidate_item.get('stock_level', 0.0)) # 재고 사용 시

        flat_vector = np.array(flat_vector_list).astype(np.float32)
        
        if np.isnan(flat_vector).any() or np.isinf(flat_vector).any():
            print(f"!!! Warning: Invalid value (nan/inf) detected in state vector! State object was: {state_obj}")
            flat_vector = np.nan_to_num(flat_vector, nan=0.0, posinf=0.0, neginf=0.0)
            print("!!! Replaced nan/inf with 0.")

        if len(flat_vector) != config.OBS_DIM:
            print(f"!!! Error: Flattened vector dim ({len(flat_vector)}) != config.OBS_DIM ({config.OBS_DIM}). Returning None.")
            print(f"Problematic state object: {state_obj}")
            return None # 에러 발생 시 None 반환

        return flat_vector
        
    except Exception as e:
        # 예상치 못한 오류 발생 시 (예: 리스트가 아닌 값이 들어옴)
        print(f"!!! Error during state flattening: {e}. Returning None.")
        print(f"Problematic state object: {state_obj}")
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