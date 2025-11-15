"""
RL 학습 데이터 로더

Raw 데이터와 ML 모델 예측을 결합하여 RL 학습 데이터 생성
- CSV 데이터 로드
- ML 모델로 가격/품질 예측
- State 벡터 생성
- Reward 계산
- (State, Action, Reward, Next_State, Done) transition 생성
"""
import pandas as pd
import numpy as np
import os
import sys
import joblib
from datetime import timedelta
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from serving.inference import ModelInferenceEngine
from src.config import OBS_DIM, ML_MODEL_PATH

class RLDataLoader:
    """
    Raw 데이터(CSV)와 학습된 ML 모델(InferenceEngine)을 결합하여 
    RL 학습에 필요한 (State, Action, Reward, Next_State) 데이터를 생성합니다.
    """
    def __init__(self, data_dir=os.path.join(PROJECT_ROOT, 'data', 'raw')):
        
        # --- Loading ML inference model
        self.data_dir = data_dir
        self.inference_engine = ModelInferenceEngine()
        
        # --- Loading raw data for training RL model ---
        print("Loading Raw Data for RL Environment Construction...")
        try:
            self.sales_df = pd.read_csv(os.path.join(data_dir, '초창패개발_데이터_판매데이터.csv'))
            self.cs_df = pd.read_csv(os.path.join(data_dir, '초창패개발_데이터_CS데이터.csv'))
            lead_time_df = pd.read_csv(os.path.join(data_dir, '초창패개발_데이터_평균출고소요일.csv'))
            
            # CS 발생 여부 딕셔너리 (빠른 조회를 위해)
            self.cs_order_set = set(self.cs_df['발주번호'].unique())
            
            # 리드타임 맵 딕셔너리 (빠른 조회를 위해)
            self.lead_time_map = pd.Series(
                lead_time_df.출고소요일.values, 
                index=lead_time_df.업체명
            ).to_dict()
            self.default_lead_time = lead_time_df['출고소요일'].median()

            # processing dates
            self.sales_df['발주날짜'] = pd.to_datetime(self.sales_df['발주날짜'], errors='coerce')
            self.sales_df = self.sales_df.sort_values('발주날짜').dropna(subset=['발주날짜'])
            print("✓ Raw Data (Sales, CS, LeadTime) Loaded.")
            
        except FileNotFoundError as e:
            print(f"!!! FATAL: Raw data file not found. {e}")
            raise

    def _construct_state_vector(self, row, pred_price, pred_quality_rate):
        """
        용도: 
            (내부 함수) Raw 데이터 한 행과 ML 예측값들을 결합하여
            RL 모델(DQN)의 입력으로 사용할 1차원 상태(State) 벡터를 생성합니다.
        Args:
            row (pd.Series): '판매데이터.csv'의 단일 행(거래 내역).
            pred_price (float): ML 모델이 예측한 가격.
            pred_quality_rate (float): ML 모델이 예측한 품질(하자) 확률.
        Returns:
            np.ndarray: 
                `config.OBS_DIM` (예: 50) 차원에 맞춰진 1차원 float32 벡터.
        로직:
            1. (User Profile) '업체명'을 기반으로 임시 사용자 그룹 벡터를 생성합니다.
            2. (Session History) '수량', '공급가격' 등을 기반으로 임시 세션 이력 벡터를 생성합니다.
            3. (Candidate Item) ML 예측값(`pred_price`, `pred_quality_rate`)과 
                `lead_time_map`에서 조회한 리드타임을 `candidate_vec`로 만듭니다.
            4. 모든 벡터를 `np.concatenate`로 결합합니다.
            5. `np.pad` 또는 슬라이싱(`[:OBS_DIM]`)을 사용해 벡터 크기를 `OBS_DIM`에 정확히 맞춥니다.
        """
        #  ====== 제가 현재 대봉 백엔드에 직접 접근할 수 없기에 여기서는 일단 임의로 샘플로 진행했습니다 =====
        # 1. User Profile (가정: 업체명을 그룹으로 매핑)
        # 실제로는 업체별 메타데이터가 필요하지만, 여기서는 해시값으로 임의 생성
        user_group_vec = [1, 0] if hash(row['업체명']) % 2 == 0 else [0, 1]
        
        # 2. Session History (과거 이력 시뮬레이션)
        # (실제 로그가 없으므로 판매량 등을 기반으로 근사치 생성)
        history_vec = [0.0] * 32 # Item Embedding (Placeholder)
        click_count = np.log1p(row['수량'])
        view_time = row['공급가격'] / 100 # (Placeholder)
        affinity_vec = [0.0] * 10
        
        # 3. 리드타임 조회 
        lead_time = self.lead_time_map.get(row['업체명'], self.default_lead_time)
        
        # 4. Candidate Item Info 
        candidate_vec = [
            pred_price,          # ML이 예측한 가격
            lead_time,           # 조회한 리드타임
            pred_quality_rate,   # ML이 예측한 품질(하자율)
            0                    # 프로모션 여부 (Placeholder)
        ]
        
        # concatenating vectors (User + History + Candidate)
        flat_vector = np.concatenate([
            user_group_vec, 
            history_vec, 
            [click_count, view_time], 
            affinity_vec, 
            candidate_vec
        ]).astype(np.float32)
        
        # 차원 맞추기 (부족하면 0 padding, 넘치면 자름)
        if len(flat_vector) < OBS_DIM:
            flat_vector = np.pad(flat_vector, (0, OBS_DIM - len(flat_vector)))
        elif len(flat_vector) > OBS_DIM:
            flat_vector = flat_vector[:OBS_DIM]
            
        return flat_vector

    def _calculate_reward(self, row):
        """
        용도: 
            (내부 함수) Raw 데이터 행(row)을 기반으로 RL 보상(Reward) 점수를 계산합니다.
            '판매데이터'는 모두 성공한 추천(Action=1)이라고 가정합니다.
        Args:
            row (pd.Series): '판매데이터.csv'의 단일 행(거래 내역).
        Returns:
            float: 계산된 최종 보상 점수.
        로직:
            1. 기본 보상으로 1.0 (판매 성공)을 부여합니다.
            2. 해당 `row`의 '주문코드'가 `self.cs_order_set`(CS 데이터)에 존재하는지 확인합니다.
            3. CS가 발생한 주문이면 -5.0의 큰 페널티를 차감합니다.
        """
        reward = 1.0 # initial reward: 판매 발생
        
        is_cs = row['주문코드'] in self.cs_order_set
        
        if is_cs:
            reward -= 5.0 # CS 발생 시 큰 페널티
        
        return reward

    def _prepare_ml_input_features(self, row_series, feature_names):
        """
        용도: 
            (내부 함수) 단일 Raw 데이터 행(row_series)을 ML 모델 추론에 필요한
            DataFrame 형태로 가공합니다.
        Args:
            row_series (pd.Series): '판매데이터.csv'의 단일 행.
            feature_names (list): 
                ML 모델이 학습할 때 사용했던 피처 이름 목록 (예: `self.inference_engine.feature_names_price`)
        Returns:
            pd.DataFrame: 
                ML 모델의 `predict()` 메서드에 바로 입력할 수 있는,
                1개 행과 `feature_names`의 순서를 따르는 DataFrame.
        로직:
            1. `row_series`에 존재하는 피처는 그대로 사용합니다.
            2. '발주날짜'로부터 'year', 'month' 등 시간 피처를 생성합니다.
            3. `feature_names` 목록에는 있지만 `row_series`나 시간 피처에 없는 
                (예: `price_lag_7d` 같은 Lag 피처) 값들은 0.0으로 채웁니다.
            4. 최종적으로 `feature_names` 순서에 맞는 DataFrame을 생성하여 반환합니다.
        """
        ml_input_dict = {}
        
        # 1. row_series에서 직접 가져올 수 있는 값 채우기
        for col in row_series.index:
            if col in feature_names:
                ml_input_dict[col] = row_series[col]
        
        # 2. 날짜 피처 생성 (train_ml.py와 동일하게)
        if '발주날짜' in row_series:
            date = pd.to_datetime(row_series['발주날짜'])
            if 'year' in feature_names: ml_input_dict['year'] = date.year
            if 'month' in feature_names: ml_input_dict['month'] = date.month
            if 'day' in feature_names: ml_input_dict['day'] = date.day
            if 'dayofweek' in feature_names: ml_input_dict['dayofweek'] = date.dayofweek
        
        # 3. 그 외 (Lag, Rolling 등) 피처들은 0으로 채움
        for col in feature_names:
            if col not in ml_input_dict:
                ml_input_dict[col] = 0.0 # Lag/Rolling 피처 등은 0으로 초기화
        
        # 4. DataFrame 생성 (피처 순서 보장)
        return pd.DataFrame([ml_input_dict])[feature_names]


    def generate_rl_training_data(self, sample_size=1000):
        """
        용도: 
            `train_rl.py`가 호출하는 메인 함수.
            ML 모델 추론을 기반으로 현실적인 RL 학습 데이터(transitions)를 생성합니다.
        Args:
            sample_size (int): 
                생성할 총 transition(경험)의 수. `sales_df`에서 이만큼 샘플링합니다.
        Returns:
            list[tuple]: 
                RL 리플레이 버퍼에 저장될 `(state, action, reward, next_state, done_mask)` 
                튜플의 리스트.
        로직:
            1. ML 모델이 로드되었는지 확인합니다.
            2. ML 모델이 요구하는 피처 목록(`feature_names_price`, `feature_names_quality`)을 가져옵니다.
            3. `sales_df`에서 `sample_size`만큼 무작위 샘플링합니다.
            4. 각 샘플(row)에 대해 `tqdm` (진행바)을 실행합니다.
            5. `_prepare_ml_input_features`로 ML 추론용 입력을 만듭니다.
            6. `self.inference_engine.predict_price`와 `predict_quality_rate`를 호출하여
                `pred_price`와 `pred_quality`를 얻습니다.
            7. `_construct_state_vector`로 (ML 예측값이 포함된) `state` 벡터를 생성합니다.
            8. `action=1`, `done=True` (Contextual Bandit)로 설정하고, `_calculate_reward`로 `reward`를 계산합니다.
            9. `(state, action, reward, next_state, not done)` 튜플을 `transitions` 리스트에 추가합니다.
        """
        transitions = []
        
        # ML 모델이 로드되었는지 확인
        if not self.inference_engine.price_model or not self.inference_engine.quality_model:
            print("!!! Error: ML Models are not loaded. Cannot generate RL data.")
            return []
            
        print(f"Generating {sample_size} RL transitions using ML models...")
        
        # ML 모델이 요구하는 피처 이름 목록 가져오기
        price_features_list = self.inference_engine.feature_names_price
        quality_features_list = self.inference_engine.feature_names_quality
        
        # 데이터를 샘플링하여 순회
        samples = self.sales_df.sample(n=min(sample_size, len(self.sales_df)))
        
        for idx, row in tqdm(samples.iterrows(), total=len(samples)):
            try:
                # 1. ML 모델 추론을 위한 입력 피처 준비
                # (1.1) 가격 예측용 피처
                price_ml_input = self._prepare_ml_input_features(row, price_features_list)
                # (1.2) 품질 예측용 피처
                quality_ml_input = self._prepare_ml_input_features(row, quality_features_list)
                
                # 2. ML 모델 추론 수행
                pred_price = self.inference_engine.predict_price(price_ml_input)[0]
                pred_quality = self.inference_engine.predict_quality_rate(quality_ml_input)[0]
                
                # 3. RL State 생성
                state = self._construct_state_vector(row, pred_price, pred_quality)
                
                # 4. Action (과거 판매 데이터는 '추천 성공'으로 간주)
                action = 1 # (1: 추천함)
                
                # 5. Reward 계산
                reward = self._calculate_reward(row)
                
                # 6. Next State (Contextual Bandit: 다음 상태 없음)
                # (순차적 학습을 위해선 여기서 다음 row를 조회해야 함)
                next_state = np.zeros_like(state) 
                done = True # (즉시 에피소드 종료)
                
                # (state, action, reward, next_state, done_mask)
                # done=True -> done_mask=False
                transitions.append((state, action, reward, next_state, not done)) 
                
            except Exception as e:
                print(f"Warning: Skipping one transition due to error: {e}")
                continue # 데이터 문제로 인한 스킵

        print(f"✓ Generated {len(transitions)} valid transitions.")
        return transitions

# 테스트 실행
if __name__ == "__main__":
    loader = RLDataLoader()
    if loader.inference_engine.price_model and loader.inference_engine.quality_model:
        data = loader.generate_rl_training_data(10)
        if data:
            print(f"Sample Data shape (State): {data[0][0].shape}")
            # (S, A, R, S', Done_Mask)
            print(f"Sample Transition (A, R, Done_Mask): {data[0][1], data[0][2], data[0][4]}")
    else:
        print("Test skipped: ML models not found in 'model_register/ml_model'.")
        print("Please run 'python -m training.train_ml --model all' first.")