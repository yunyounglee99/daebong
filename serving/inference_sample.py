"""
통합 모델 추론 엔진

ML 모델(가격/품질)과 RL 모델(추천)을 로드하여 추론을 수행하는 통합 엔진
- 최신 모델 자동 로드
- 가격 예측 (회귀)
- 품질/하자율 예측 (분류)
- Q-value 예측 (강화학습)
- FastAPI 서버에서 사용
"""
import os
import sys
import glob
import joblib
import torch
import pandas as pd
import numpy as np
from typing import Optional

# --- 1. 프로젝트 루트 경로 추가 ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Inference: Running on device: {DEVICE}")

# --- 2. 필요한 모든 모듈 임포트 ---
from src.config import (
    ML_MODEL_PATH, RL_MODEL_PATH,
    OBS_DIM, ACTION_DIM, HIDDEN_DIMS, LR_CRITIC
)

# Importing ML model class
try:
    from src.models.ml_price_model import EnsemblePriceModel
    from src.models.ml_quality_model import EnsembleQualityModel
    print("Inference: ML Model classes loaded.")
except ImportError as e:
    print(f"Warning: Could not import ML model classes from 'src.models'. {e}")
    EnsemblePriceModel = None
    EnsembleQualityModel = None

# Importing RL model class
try:
    from src.models.rl_model_DQN import DQN
    from training.utils import flatten_state_to_vector
    print("Inference: RL Model classes (DQN, utils) loaded.")
except ImportError as e:
    print(f"Warning: Could not import RL model classes. {e}")
    DQN = None
    flatten_state_to_vector = None

class ModelInferenceEngine:
    """
    통합 모델 추론 엔진 클래스

    ML과 RL 모델을 모두 로드하고 예측 API를 제공
    """
    def __init__(self, ml_dir = ML_MODEL_PATH, rl_dir = RL_MODEL_PATH):
        print("\n" + "="*60)
        print("Initializing Unified Model Inference Engine...")
        print("="*60)

        self.ml_model_dir = ml_dir
        self.rl_model_dir = rl_dir
        self.device = DEVICE

        # ML models config
        self.price_model = None
        self.quality_model = None
        self.feature_names_price = None
        self.feature_names_quality = None
        
        # RL models config
        self.q_network = None

        self.latest_price_prefix_loaded: Optional[str] = None
        self.latest_quality_prefix_loaded: Optional[str] = None
        self.latest_rl_path_loaded: Optional[str] = None

        if EnsemblePriceModel and EnsembleQualityModel:
            self._load_latest_ml_models()
        else:
            print("ERROR: ML Model classes not loaded. Skipping ML model loading.")
        
        if DQN and flatten_state_to_vector:
            self._initialize_rl_network()
            self._load_latest_rl_checkpoint()
        else:
            print("ERROR: RL Model classes not loaded. Skipping RL model loading.")

        print("="*60)
        print("Inference Engine Initialized.")
        print("="*60)


    # --- 1. Loading ML Models Logic ---
    def _find_latest_ml_model(self, prefix: str):
        """
        용도: ML 모델 디렉토리에서 가장 최신 모델 파일의 접두사(타임스탬프 포함)를 검색합니다.
        Args:
            prefix (str): 찾을 모델 접두사 (예: 'price_ensemble', 'quality_ensemble')
        Returns:
            str: 가장 최신 모델의 전체 경로를 포함한 접두사 (예: '.../ml_model/price_ensemble_20251112_133540')
            None: 파일을 찾지 못한 경우
        로직:
            1. `glob`을 사용하여 `{prefix}_*.pkl` 패턴의 모든 파일을 찾습니다.
            2. 파일 목록을 정렬(sorted)하여 가장 최신 파일을 찾습니다.
            3. 파일명에서 `_lgb.pkl` 같은 마지막 부분을 제외한 접두사(prefix)만 파싱하여 반환합니다.
        """
        search_path = os.path.join(self.ml_model_dir, f'{prefix}_*.pkl')
        model_files = glob.glob(search_path)
        if not model_files:
            return None
        latest_file = sorted(model_files, reverse = True)[0]
        base_name = os.path.basename(latest_file)
        latest_prefix = '_'.join(base_name.split('_')[:-1])

        return os.path.join(self.ml_model_dir, latest_prefix)
    
    def _load_latest_ml_models(self):
        """
        용도: (내부 함수) 최신 Price 및 Quality ML 모델과 피처 목록을 로드하여 클래스 속성에 저장합니다.
        Args:
            None
        Returns:
            None
        로직:
            1. `_find_latest_ml_model`을 호출하여 'price_ensemble'과 'quality_ensemble'의 최신 접두사를 찾습니다.
            2. `joblib.load`를 사용해 `_config.pkl`을 읽어 앙상블 설정을 로드합니다.
            3. `EnsemblePriceModel`/`EnsembleQualityModel` 클래스를 인스턴스화하고 `load_models()` 메서드를 호출합니다.
            4. `_features.pkl`을 로드하여 `self.feature_names_price`/`self.feature_names_quality`에 저장합니다.
            5. 로드된 경로를 `self.latest_..._loaded` 속성에 저장합니다.
        """
        print('Loading ML Models (Price & Quality)...')
        # 1.1 Loading price prediction model (EnsemblePriceModel)
        latest_price_prefix = self._find_latest_ml_model("price_ensemble")
        if latest_price_prefix:
            try:
                print(f"Loading Price Model from: {latest_price_prefix}_*.pkl")
                config = joblib.load(f"{latest_price_prefix}_config.pkl")
                self.price_model = EnsemblePriceModel(ensemble_method=config.get('ensemble_method', 'voting'))
                self.price_model.load_models(latest_price_prefix)
                self.feature_names_price = joblib.load(f"{latest_price_prefix}_features.pkl")
                print(f"✓ Price Model (Regression) loaded with {len(self.feature_names_price)} features.")

                self.latest_price_prefix_loaded = latest_price_prefix
            except Exception as e:
                print(f"!!! Error loading Price Model: {e}")

        # 1.2 Loading Quality classification model (EnsembleQualityModel)
        latest_quality_prefix = self._find_latest_ml_model("quality_ensemble")
        if latest_quality_prefix:
            try:
                print(f"Loading Quality Model from: {latest_quality_prefix}_*.pkl")
                config = joblib.load(f"{latest_quality_prefix}_config.pkl")
                self.quality_model = EnsembleQualityModel(ensemble_method=config.get('ensemble_method', 'voting'))
                self.quality_model.load_models(latest_quality_prefix)
                self.feature_names_quality = joblib.load(f"{latest_quality_prefix}_features.pkl")
                print(f"✓ Quality Model (Classification) loaded with {len(self.feature_names_quality)} features.")

                self.latest_quality_prefix_loaded = latest_quality_prefix
            except Exception as e:
                print(f"!!! Error loading Quality Model: {e}")


    # --- 2. Loading RL Model Logic ---
    def _initialize_rl_network(self):
        """
        용도: (내부 함수) RL DQN 네트워크의 '껍데기'(아키텍처)를 `src/config.py` 값으로 초기화합니다.
        Args:
            None
        Returns:
            None
        로직:
            `DQN` 클래스를 `config`의 `OBS_DIM`, `ACTION_DIM` 등으로 인스턴스화하여 `self.q_network`에 할당합니다.
        """
        self.q_network = DQN(
            obs_dim = OBS_DIM,
            a_dim = ACTION_DIM,
            h_dims = HIDDEN_DIMS,
            lr_critic = LR_CRITIC,
            device = self.device
        )

    def _load_latest_rl_checkpoint(self):
        """
        용도: (내부 함수) RL 모델 디렉토리에서 가장 최신 `.pth` 체크포인트를 찾아 `self.q_network`에 가중치를 로드합니다.
        Args:
            None
        Returns:
            None
        로직:
            1. `glob`으로 `dqn_checkpoint_*.pth` 패턴의 모든 파일을 찾습니다.
            2. `os.path.getctime` 기준으로 가장 최신 파일을 찾습니다.
            3. `torch.load`로 체크포인트를 로드하고 `q_main_state_dict` 키를 읽어옵니다.
            4. `self.q_network.load_state_dict()`로 가중치를 로드합니다.
            5. `self.q_network.eval()`을 호출하여 평가(추론) 모드로 설정합니다.
            6. 로드된 경로를 `self.latest_rl_path_loaded` 속성에 저장합니다.
        """
        print(f'Searching for latest DQN checkpoint in {self.rl_model_dir}')
        search_path = os.path.join(self.rl_model_dir, 'dqn_checkpoint_*.pth')
        checkpoints = glob.glob(search_path)
        if not checkpoints:
            print(f'!!! Warning: No DQN checkpoints found matching : {search_path}.')
            self.q_network = None
            return
        
        latest_checkpoint_path = max(checkpoints, key=os.path.getctime)
        print(f'Loading RL weights from {latest_checkpoint_path}')
        try:
            checkpoint = torch.load(latest_checkpoint_path, map_location = self.device)
            self.q_network.load_state_dict(checkpoint['q_main_state_dict'])
            self.q_network.eval()
            print('RL Model (DQN) loaded and set to eval mode')

            self.latest_rl_path_loaded = latest_checkpoint_path
        except Exception as e:
            print(f'!!! Error loading RL checkpoint file: {e}')
            self.q_network = None

    # --- 3. prediction API method ---
    def predict_price(self, input_features: pd.DataFrame):
        """
        용도: ML 가격 모델을 사용하여 1일 후 가격(회귀)을 예측합니다.
        Args:
            input_features (pd.DataFrame): 
                예측에 필요한 피처가 포함된 DataFrame.
                (반드시 `self.feature_names_price` 목록의 컬럼을 모두 포함해야 함)
        Returns:
            np.array: 예측된 가격 값 (예: [25000.0, 31000.0])
        로직:
            1. 모델 로드 여부를 확인합니다.
            2. `self.feature_names_price` 리스트를 사용해 입력 DataFrame의 컬럼 순서를 학습 당시와 동일하게 맞춥니다.
            3. `self.price_model.predict()` (회귀) 메서드를 호출하여 예측값을 반환합니다.
        """
        if self.price_model is None or self.feature_names_price is None:
            print('Error: Price model or features not loaded.')
            return np.array([0.0] * len(input_features))
        try:
            X = input_features[self.feature_names_price]
            return self.price_model.predict(X)
        except Exception as e:
            print(f"Error during price prediction: {e}")
            return np.array([0.0] * len(input_features))

    def predict_quality_rate(self, input_features: pd.DataFrame):
        """
        용도: ML 품질 모델을 사용하여 1일 후 하자비율(분류 확률)을 예측합니다.
        Args:
            input_features (pd.DataFrame): 
                예측에 필요한 피처가 포함된 DataFrame.
                (반드시 `self.feature_names_quality` 목록의 컬럼을 모두 포함해야 함)
        Returns:
            np.array: 예측된 하자비율 (0.0 ~ 1.0 사이의 CS=1 확률 값, 예: [0.02, 0.05])
        로직:
            1. 모델 로드 여부를 확인합니다.
            2. `self.feature_names_quality` 리스트를 사용해 입력 DataFrame의 컬럼 순서를 맞춥니다.
            3. `self.quality_model.predict_proba()` (분류 확률) 메서드를 호출하여 예측값을 반환합니다.
        """
        if self.quality_model is None or self.feature_names_quality is None:
            print("Error: Quality model or features not loaded.")
            return np.array([0.0] * len(input_features))
        try:
            X = input_features[self.feature_names_quality]
            return self.quality_model.predict_proba(X)
        except Exception as e:
            print(f"Error during quality prediction: {e}")
            return np.array([0.0] * len(input_features))
        
    def predict_q_values(self, state_t_json: dict):
        """
        용도: RL DQN 모델을 사용하여 현재 State의 Q-value(예상 만족도)를 예측합니다.
        Args:
            state_t_json (dict): 
                `data_structure.md`에 정의된 'state_t' JSON 객체.
        Returns:
            list: [Q(미추천 점수), Q(추천 점수)] (예: [0.1, 2.5])
        로직:
            1. 모델 로드 여부를 확인합니다.
            2. `flatten_state_to_vector` 유틸 함수로 JSON 객체를 1D Numpy 벡터로 변환합니다.
            3. `torch.from_numpy`로 벡터를 텐서로 변환하고 배치 차원([1, obs_dim])을 추가합니다.
            4. `torch.no_grad()` 컨텍스트 안에서 `self.q_network(s_tensor)`를 호출하여 Q-value를 추론합니다.
            5. 결과를 Python 리스트로 변환하여 반환합니다.
        """
        if self.q_network is None:
            print('Cannot predict: RL Model is not loaded.')
            return [0.0, 0.0]
        
        state_vector = flatten_state_to_vector(state_t_json)
        if state_vector is None:
            print('Error: Could not flatten state JSON.')
            return [0.0, 0.0]
        
        s_tensor = torch.from_numpy(state_vector).float().to(self.device).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_network(s_tensor)

        return q_values.squeeze(0).cpu().tolist()