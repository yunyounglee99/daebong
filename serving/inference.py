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
import requests
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Inference: Running on device: {DEVICE}")

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
        """
        용도: 클래스 초기화. ML/RL 모델 디렉토리 설정 및 모델 로딩 프로세스 시작.
        Args:
            ml_dir (str): ML 모델 파일(.pkl)이 저장된 경로.
            rl_dir (str): RL 모델 파일(.pth)이 저장된 경로.
        Returns:
            None
        로직:
            1. 각종 속성(모델, 피처명, 로드된 경로)을 None으로 초기화합니다.
            2. ML 모델 클래스가 임포트되었는지 확인 후 `_load_latest_ml_models()`를 호출합니다.
            3. RL 모델 클래스가 임포트되었는지 확인 후 `_initialize_rl_network()` 및 `_load_latest_rl_checkpoint()`를 호출합니다.
        """
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
        # 1.1 Loading Price Model
        latest_price_prefix = self._find_latest_ml_model("price_ensemble")
        if latest_price_prefix:
            try:
                print(f"Loading Price Model from: {latest_price_prefix}_*.pkl")
                config = joblib.load(f"{latest_price_prefix}_config.pkl")
                
                self.price_model = EnsemblePriceModel(ensemble_method=config.get('ensemble_method', 'voting'))
                self.price_model.load_models(latest_price_prefix)
                
                # LightGBM 모델이 포함되어 있다고 가정하고 feature_name_ 속성을 사용
                if 'lgb' in self.price_model.models:
                    self.feature_names_price = self.price_model.models['lgb'].feature_name_
                    print(f"✓ Price Model loaded with {len(self.feature_names_price)} features (extracted from model).")
                else:
                    # LGBM이 없는 경우 (드문 경우), 다른 방식이나 _features.pkl 시도
                    print("Warning: LightGBM model not found in ensemble. Trying fallback to _features.pkl...")
                    self.feature_names_price = joblib.load(f"{latest_price_prefix}_features.pkl")

                self.latest_price_prefix_loaded = latest_price_prefix
            except Exception as e:
                print(f"!!! Error loading Price Model: {e}")

        # 1.2 Loading Quality Model
        latest_quality_prefix = self._find_latest_ml_model("quality_ensemble")
        if latest_quality_prefix:
            try:
                print(f"Loading Quality Model from: {latest_quality_prefix}_*.pkl")
                config = joblib.load(f"{latest_quality_prefix}_config.pkl")
                
                self.quality_model = EnsembleQualityModel(ensemble_method=config.get('ensemble_method', 'voting'))
                self.quality_model.load_models(latest_quality_prefix)
                
                if 'lgb' in self.quality_model.models:
                    self.feature_names_quality = self.quality_model.models['lgb'].feature_name_
                    print(f"✓ Quality Model loaded with {len(self.feature_names_quality)} features (extracted from model).")
                else:
                    print("Warning: LightGBM model not found in ensemble. Trying fallback to _features.pkl...")
                    self.feature_names_quality = joblib.load(f"{latest_quality_prefix}_features.pkl")

                self.latest_quality_prefix_loaded = latest_quality_prefix
            except Exception as e:
                print(f"!!! Error loading Quality Model: {e}")

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
        print(f'\nSearching for latest DQN checkpoint in {self.rl_model_dir}\n')
        search_path = os.path.join(self.rl_model_dir, 'checkpoint_*.pth')
        checkpoints = glob.glob(search_path)
        if not checkpoints:
            print(f'!!! Warning: No DQN checkpoints found matching : {search_path}.')
            self.q_network = None
            return
        
        latest_checkpoint_path = max(checkpoints, key=os.path.getctime)
        print(f'\nLoading RL weights from {latest_checkpoint_path}\n')
        try:
            checkpoint = torch.load(latest_checkpoint_path, map_location = self.device)
            self.q_network.load_state_dict(checkpoint['q_main_state_dict'])
            self.q_network.training = False
            for module in self.q_network.modules():
                module.training = False

            print('RL Model (DQN) loaded and set to eval mode')

            self.latest_rl_path_loaded = latest_checkpoint_path
        except Exception as e:
            print(f'!!! Error loading RL checkpoint file: {e}')
            self.q_network = None

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

class DailyBatchProcessor:
    """
    용도:
        매일 실행되는 일일 배치(Daily Batch) 작업을 관리하는 클래스입니다.
        최신 Raw 데이터와 기상청 API를 사용하여 ML 모델의 입력 피처를 구성하고,
        모든 대상 품목에 대해 '가격'과 '품질(하자율)'을 예측하여
        'data/inference_data/daily_predictions.csv' 파일로 저장합니다.
        이 파일은 RL 모델이 실시간 추론 시 ML 예측값을 참조(Lookup)하는 캐시로 사용됩니다.
    """

    def __init__(self, inference_engine: ModelInferenceEngine):
        """
        용도: 배치 처리기 초기화.
        Args:
            inference_engine (ModelInferenceEngine): 
                이미 로드된 ML/RL 모델을 가지고 있는 추론 엔진 인스턴스.
        Returns:
            None
        로직:
            1. 추론 엔진을 인스턴스 변수로 저장합니다.
            2. 데이터 디렉토리(INFERENCE_DATA_DIR)가 존재하는지 확인하고 없으면 생성합니다.
        """
        self.engine = inference_engine
        self.data_dir = INFERENCE_DATA_DIR
        os.makedirs(self.data_dir, exist_ok=True)

    def _fetch_yesterday_weather(self):
        """
        용도: 
            (내부 함수) 기상청 API를 호출하여 '어제'의 기상 데이터를 가져옵니다.
            예측 시점(오늘)에서 가장 확실하게 확보 가능한 최신 기상 정보이기 때문입니다.
        Args:
            None
        Returns:
            dict: 
                {'avgTa': ..., 'sumRn': ...} 형태의 기상 데이터 딕셔너리.
                API 호출 실패 시 0.0으로 채워진 기본 딕셔너리 반환.
        로직:
            1. `datetime.now() - timedelta(days=1)`로 어제 날짜를 구합니다.
            2. 기상청 '종관기상관측(ASOS) 일자료 조회' API를 호출합니다. (안동: 136 기준)
            3. 응답 데이터를 파싱하여 기온, 강수량, 풍속, 습도 등을 추출합니다.
            4. 데이터가 없으면 '그제' 데이터로 재시도하거나 기본값(0.0)을 반환합니다.
        """
        print("\nFetching weather data...\n")
        try:
            today = datetime.now()
            target_date = today - timedelta(days=1)
            date_str = target_date.strftime('%Y%m%d')
            
            url = 'http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList'
            # (주의) 실제 운영 시에는 환경 변수나 config에서 키를 가져와야 합니다.
            api_key = '17f766b554e3e4bd767ecb2913a2450783b7e243995f928e2d28532c672550f4'
            
            params = {
                'serviceKey': api_key,
                'pageNo': '1', 'numOfRows': '10', 'dataType': 'JSON',
                'dataCd': 'ASOS', 'dateCd': 'DAY', 
                'startDt': date_str, 'endDt': date_str, 'stnIds': '136'
            }
            
            response = requests.get(url, params=params)
            
            # 응답 처리 로직 (RLDataLoader와 동일)
            if response.status_code == 200:
                data = response.json()
                items = data.get('response', {}).get('body', {}).get('items', {}).get('item', [])
                if items:
                    item = items[0]
                    weather_data = {
                        'avgTa': float(item.get('avgTa', 0) or 0),
                        'minTa': float(item.get('minTa', 0) or 0),
                        'maxTa': float(item.get('maxTa', 0) or 0),
                        'sumRn': float(item.get('sumRn', 0) or 0),
                        'avgWs': float(item.get('avgWs', 0) or 0),
                        'avgRhm': float(item.get('avgRhm', 0) or 0)
                    }
                    # ★★★ 확인용 프린트문 추가 ★★★
                    print(f"\nWeather Data Fetched ({date_str}): {weather_data}\n")
                    return weather_data
                    
        except Exception as e:
            print(f"Weather API Error: {e}")
            
        return {'avgTa': 0.0, 'minTa': 0.0, 'maxTa': 0.0, 'sumRn': 0.0, 'avgWs': 0.0, 'avgRhm': 0.0}

    def _prepare_input_df(self, feature_names, weather_data, item_info):
        """
        용도: 
            (내부 함수) ML 모델 추론을 위한 단일 행(1-row) DataFrame을 생성합니다.
        Args:
            feature_names (list): 모델이 요구하는 피처 이름 목록.
            weather_data (dict): API로 가져온 기상 데이터.
            item_info (dict): 품목 정보 (예: 업체명, 품종 등 - 여기선 활용 예시).
        Returns:
            pd.DataFrame: 모델 입력용 DataFrame.
        로직:
            1. `feature_names` 컬럼을 가진 0으로 초기화된 딕셔너리를 만듭니다.
            2. 현재 날짜(오늘)를 기준으로 'year', 'month', 'day', 'dayofweek'를 채웁니다.
            3. `weather_data`의 값들을 해당 피처(avgTa 등)에 채웁니다.
            4. (고급) `item_info`를 이용해 과거 데이터에서 Lag Feature를 가져와 채울 수 있습니다.
              (현재 코드는 단순화를 위해 Lag 피처는 0으로 둡니다.)
            5. DataFrame으로 변환하여 반환합니다.
        """
        input_dict = {col: 0.0 for col in feature_names}
        
        # 1. 날짜 피처 (오늘 기준 - 내일 가격 예측을 위함)
        today = datetime.now()
        print(f'\ntoday is {today}  \n')
        if 'year' in input_dict: input_dict['year'] = today.year
        if 'month' in input_dict: input_dict['month'] = today.month
        if 'day' in input_dict: input_dict['day'] = today.day
        if 'dayofweek' in input_dict: input_dict['dayofweek'] = today.dayofweek
        
        # 2. 날씨 피처
        for k, v in weather_data.items():
            if k in input_dict:
                input_dict[k] = v
                
        # 3. DataFrame 생성
        return pd.DataFrame([input_dict])

    def run_daily_update(self):
        """
        용도: 
            일일 배치 작업의 메인 실행 함수입니다.
            외부에서 `processor.run_daily_update()` 형태로 호출합니다.
        Args:
            None
        Returns:
            None
        로직:
            1. ML 모델(가격, 품질)이 로드되었는지 확인합니다.
            2. `data/inference_data` 폴더에서 최신 Raw 데이터(판매, 평균출고소요일)를 로드합니다.
            3. `_fetch_yesterday_weather`를 호출해 기상 데이터를 준비합니다.
            4. 판매 데이터에서 예측 대상이 될 '유니크한 품목(품종+업체명)' 목록을 추출합니다.
            5. 각 품목에 대해 반복문(Loop)을 돕니다:
              - `_prepare_input_df`로 가격/품질 모델용 입력 데이터를 각각 만듭니다.
              - `engine.predict_price`와 `predict_quality_rate`로 예측을 수행합니다.
              - 리드타임 데이터에서 해당 업체의 출고소요일을 조회합니다.
              - 예측 결과(가격, 품질, 리드타임)를 리스트에 모읍니다.
            6. 결과 리스트를 `pd.DataFrame`으로 변환하고 `daily_predictions.csv`로 저장합니다.
            7. `engine.load_daily_cache()`를 호출하여 추론 엔진이 새 데이터를 즉시 반영하도록 합니다.
        """
        print("\n=== Starting Daily Batch Update ===")
        
        # 1. 모델 확인
        if not self.engine.price_model or not self.engine.quality_model:
            print("!!! Error: ML models not loaded. Cannot run batch.")
            return
        
        # 2. 기상청 데이터 로드
        weather_data = self._fetch_yesterday_weather()

        # 3. 데이터 로드
        try:
            # (참고: 대봉 측이 이 폴더에 매일 최신 파일을 덮어쓴다고 가정)
            sales_raw = pd.read_csv(os.path.join(self.data_dir, '초창패개발_데이터_판매데이터.csv'))
            lead_time_raw = pd.read_csv(os.path.join(self.data_dir, '초창패개발_데이터_평균출고소요일.csv'))
            
            # 리드타임 맵 생성
            lead_time_map = pd.Series(
                lead_time_raw.출고소요일.values, 
                index=lead_time_raw.업체명
            ).to_dict()
            default_lead_time = lead_time_raw['출고소요일'].median()
            
        except FileNotFoundError as e:
            print(f"!!! Error: Raw data file not found in {self.data_dir}. {e}")
            return

        # 4. 타겟 아이템 추출 (판매 이력이 있는 모든 품종-업체 조합)
        # (실제로는 '활성화된 상품 마스터' 테이블을 쓰는 게 좋음)
        if '품종' in sales_raw.columns and '업체명' in sales_raw.columns:
            target_items = sales_raw[['품종', '업체명']].drop_duplicates()
        else:
            print("!!! Error: Sales data missing '품종' or '업체명' columns.")
            return
        
        results = []
        print(f"Predicting for {len(target_items)} items...")
        
        for _, item in target_items.iterrows():
            item_name = item['품종']
            company = item['업체명']
            
            # (1) ML 모델 입력 구성
            price_input = self._prepare_input_df(
                self.engine.feature_names_price, weather_data, item
            )
            quality_input = self._prepare_input_df(
                self.engine.feature_names_quality, weather_data, item
            )
            
            # (2) 예측 수행
            try:
                pred_price = self.engine.price_model.predict(price_input)[0]
                pred_quality = self.engine.quality_model.predict_proba(quality_input)[0] # (확률 1차원)
            except Exception as e:
                print(f"Prediction failed for {item_name}: {e}")
                pred_price, pred_quality = 0, 0

            # (3) 리드타임 조회
            lead_time = lead_time_map.get(company, default_lead_time)
            
            # (4) 결과 수집 (인덱스로 쓸 키: 품종)
            # 실제로는 product_id가 고유 키여야 함
            results.append({
                '품종': item_name, 
                '업체명': company,
                'pred_price': pred_price,
                'pred_quality': pred_quality,
                'lead_time': lead_time
            })
            
        # 5. 결과 저장
        if results:
            result_df = pd.DataFrame(results)
            # '품종'을 인덱스로 저장하지 않고 컬럼으로 저장 (load 시 set_index 함)
            result_df.to_csv(DAILY_PREDICTION_FILE, index=False)
            print(f"✓ Daily predictions saved to {DAILY_PREDICTION_FILE}")
            
            # 6. 엔진 캐시 리로드 (즉시 반영)
            self.engine.load_daily_cache()
        else:
            print("Warning: No predictions generated.")

# --- (테스트 실행용) ---
if __name__ == "__main__":
    # 필요한 경로 설정 (단독 실행 시)
    from src.config import INFERENCE_DATA_DIR, DAILY_PREDICTION_FILE
    
    engine = ModelInferenceEngine()
    if engine.price_model and engine.quality_model:
        processor = DailyBatchProcessor(engine)
        processor.run_daily_update()
    else:
        print("Skipping batch test: Models not loaded.")