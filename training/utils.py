"""
강화학습 데이터 전처리 유틸리티

RL 모델 학습을 위한 데이터 변환 및 보상 계산 함수들을 제공합니다.
- State JSON을 벡터로 변환
- 사용자 행동 기반 보상(Reward) 계산
- 데이터 검증 및 오류 처리
"""
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

    Args:
        state_obj (dict): RL 상태 정보를 담은 딕셔너리
            - user_profile: 사용자 그룹 벡터
            - session_history: 세션 히스토리 정보 (클릭, 조회시간, 아이템 임베딩 등)
            - candidate_item_info: 후보 아이템 정보 (가격, 품질, 리드타임 등)

    Returns:
        np.ndarray or None: OBS_DIM 크기의 1D numpy 배열 (float32), 오류 시 None

    로직:
        1. user_profile에서 사용자 그룹 벡터 추출
        2. session_history에서 아이템 임베딩, 클릭/조회 통계 추출
        3. candidate_item_info에서 예측 가격, 품질, 리드타임 추출
        4. 모든 벡터를 concatenate하여 1D 벡터 생성
        5. nan/inf 값 검증 및 제거
        6. 벡터 차원이 OBS_DIM과 일치하는지 검증
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
  """
  사용자 행동에 따른 보상(Reward) 계산

  Args:
      sources (dict): 보상 계산에 필요한 사용자 행동 정보
          - clicked (int): 클릭 여부 (0 또는 1)
          - page_view_time_sec (int): 페이지 조회 시간 (초)
          - purchased (int): 구매 여부 (0 또는 1)

  Returns:
      float: 계산된 보상 값
          - 클릭 안함: 0.0
          - 클릭했지만 5초 미만 조회: -0.5 (부정적 피드백)
          - 5초 이상 조회: 1.0 + (조회시간/60) (긍정적 피드백)
          - 구매 완료: 추가 +5.0

  로직:
      1. 클릭하지 않은 경우 보상 0
      2. 클릭했으나 짧은 조회(5초 미만)는 부정적 피드백 (-0.5)
      3. 충분한 조회 시간은 긍정적 피드백 (1.0 + 조회시간 보너스)
      4. 구매 완료 시 추가 보상 +5.0
  """
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