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
        단일 거래 행(row)과 ML 예측값을 사용하여 RL State 벡터를 생성합니다.
        (config.py의 OBS_DIM=50 차원에 맞춤)
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
        비즈니스 목표에 따른 보상 계산
        (매출 발생 +1, CS 발생 시 페널티)
        """
        reward = 1.0 # initial reward: 판매 발생
        
        is_cs = row['주문코드'] in self.cs_order_set
        
        if is_cs:
            reward -= 5.0 # CS 발생 시 큰 페널티
        
        return reward

    def _prepare_ml_input_features(self, row_series, feature_names):
        """
        ML 모델이 요구하는 모든 피처(feature_names)를 가진 DataFrame을 생성합니다.
        현재 행(row) 정보로 만들 수 있는 피처만 채우고, 나머지는 0으로 둡니다.
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