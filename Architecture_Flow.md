---

# 대봉 AI 프로젝트 아키텍처 흐름도 (Architecture Flow)

이 문서는 `daebong` 프로젝트의 전체 데이터 흐름과 코드 실행 순서를 설명합니다.
이 시스템은 3단계의 파이프라인으로 구성됩니다.

1.  **Phase 1: ML 모델 학습** (가격/품질 예측 모델 생성)
2.  **Phase 2: RL 모델 학습** (ML 모델을 활용한 RL 데이터 생성 및 학습)
3.  **Phase 3: 통합 서빙** (API를 통한 예측 제공)

---

## ## 1. Phase 1: ML 모델 학습 (가격 & 품질)

- **목표:** `model_register/ml_model/` 폴더에 `price_ensemble_...pkl`과 `quality_ensemble_...pkl` 모델 파일을 생성합니다.
- **실행 스크립트:** `training/train_ml.py`
- **실행 명령어:** `python -m training.train_ml --model all --ensemble voting`

### 1.1. `main()` 함수 (in `train_ml.py`)

1.  **초기화:** `DataPreprocessor` 클래스를 초기화합니다.
2.  **데이터 로드:** `preprocessor.load_raw_data()`
    - **입력:** `data/raw/` 폴더의 원본 CSV 4개 (`농넷_시장별_사과가격.csv`, `초창패개발_데이터_판매데이터.csv`, `...CS데이터.csv`, `weather_data.json`)
    - **로직:** Pandas DataFrame으로 파일을 읽어들입니다.
    - **출력:** `price_df`, `sales_df`, `cs_df`, `weather_df`
3.  **데이터 전처리:** `preprocessor.preprocess_..._data()` (각 DF별로 실행)
    - **입력:** 원본 DataFrames
    - **로직:** 날짜/시간 변환, 쉼표/원화 기호 제거, 결측치 처리 등을 수행합니다.
    - **출력:** 정리된 DataFrames

### 1.2. `PriceModelTrainer` (가격 모델)

1.  **피처 생성:** `preprocessor.create_price_features()`
    - **입력:** `price_df`, `weather_df`
    - **로직:**
      - 날짜 피처 생성 (year, month, dayofweek)
      - 날씨 데이터 병합 (날짜 기준)
      - 범주형 피처 인코딩 (`LabelEncoder`)
      - Lag/Rolling(이동 평균) 피처 생성 (예: `price_lag_7d`)
    - **출력:** 가격 예측용 `price_features` DataFrame
2.  **데이터 준비:** `price_trainer.prepare_data()`
    - **입력:** `price_features` DataFrame
    - **로직:**
      - `Target`을 `평균가격`으로 설정.
      - **데이터 누수(Leakage) 방지**: `총거래금액`, `총거래물량`을 `exclude_cols`로 지정하여 피처에서 제외.
      - 데이터를 시간순으로 70%/15%/15% (Train/Validation/Test) 분할.
    - **출력:** `X_train`, `y_train`, `X_val`, `y_val`, `X_test`, `y_test`, `feature_cols`
3.  **학습:** `price_trainer.train()`
    - **입력:** `X_train`, `y_train`, `X_val`, `y_val`
    - **로직:** `src.models.ml_price_model.EnsemblePriceModel`을 호출합니다.
      - `EnsemblePriceModel.train()`은 `LGBMRegressor`, `XGBRegressor` 등 회귀 모델들을 `X_train`으로 학습시키고 `X_val`로 검증합니다.
    - **출력:** 학습된 `price_model` 객체
4.  **평가:** `price_trainer.evaluate()`
    - **입력:** `price_model`, `X_test`, `y_test`
    - **로직:** `model.predict(X_test)`를 호출하고, `r2_score`, `mean_absolute_error` (MAE) 등을 계산하여 출력합니다.
5.  **저장:** `price_trainer.save()`
    - **입력:** 학습된 `price_model`
    - **로직:** `model.save_models()`를 호출하여 `model_register/ml_model/` 경로에 앙상블 모델 파일들(`..._lgb.pkl`, `..._config.pkl` 등)과 **`..._features.pkl` (피처 목록)**을 저장합니다.

### 1.3. `QualityModelTrainer` (품질 모델)

- **피처 생성:** `preprocessor.create_quality_features()`
  - **입력:** `sales_df`, `cs_df`, `weather_df`
  - **로직:** `sales_df`와 `cs_df`를 병합하여 `defect_rate` (타겟, 0 또는 1) 생성, `seller_cs_rate` 등 피처 생성.
  - **출력:** 품질 예측용 `quality_features` DataFrame
- **데이터 준비:** `quality_trainer.prepare_data()` (가격 모델과 유사)
- **학습:** `quality_trainer.train()`
  - **로직:** `src.models.ml_quality_model.EnsembleQualityModel`을 호출합니다.
    - `EnsembleQualityModel.train()`은 **`LGBMClassifier`** 등 **분류 모델**을 `is_unbalanced=True` 또는 `scale_pos_weight` 옵션(데이터 불균형 처리)과 함께 학습시킵니다.
- **평가:** `quality_trainer.evaluate()`
  - **로직:** `model.predict()`(0/1 예측) 및 **`model.predict_proba()`(확률 예측)**를 호출합니다.
  - **출력:** `Precision`, `Recall`, `F1-Score`, **`AUC-ROC`** 등 분류 지표를 출력합니다.
- **저장:** `quality_trainer.save()` (가격 모델과 동일하게 모델 및 피처 목록 저장)

---

## ## 2. Phase 2: RL 모델 학습 (DQN)

- **목표:** `model_register/rl_model/` 폴더에 `dqn_checkpoint_..._stepXXXX.pth` 모델 파일을 생성합니다.
- **실행 스크립트:** `training/train_rl.py`
- **실행 명령어:** `python -m training.train_rl`

### 2.1. `train()` 함수 (in `train_rl.py`)

1.  **버퍼 초기화:** `src.utils.replay_buffer.PERBufferClass`를 초기화합니다.
2.  **모델 초기화:** `src.config.MODEL_TYPE` ('DQN')에 따라 `src.models.rl_model_DQN.DQN` 클래스를 `q_main`과 `q_target`으로 초기화합니다.
3.  **체크포인트 로드:** `config.LOAD_CHECKPOINT`가 True면 `model_register/rl_model/`에서 `.pth` 파일을 로드합니다.

### 2.2. `RLDataLoader` (데이터 생성)

1.  **`train_rl.py`가 `src.data.rl_dataloader.RLDataLoader`를 초기화합니다.**
2.  **`RLDataLoader.__init__()`**
    - **로직:** **`serving.inference.ModelInferenceEngine`을 초기화합니다.**
    - **결과:** `InferenceEngine`은 **Phase 1에서 학습된 ML 모델**(`..._price_...pkl`, `..._quality_...pkl`)과 **피처 목록**(`..._features.pkl`)을 메모리로 로드합니다.
    - **로직:** `data/raw/`의 `판매데이터.csv`, `CS데이터.csv`, `평균출고소요일.csv`를 로드합니다.
3.  **`train_rl.py`가 `data_loader.generate_rl_training_data()`를 호출합니다.**
    - **입력:** `config.RL_DATA_SAMPLE_SIZE` (예: 50000)
    - **로직 (루프):**
      1.  `판매데이터.csv`에서 과거 판매 내역(row)을 하나씩 샘플링합니다.
      2.  `_prepare_ml_input_features()`: `row` 데이터를 ML 모델이 요구하는 피처 (`feature_names_price`, `feature_names_quality` 목록)에 맞게 `pd.DataFrame`으로 변환합니다. (Lag 피처 등은 0으로 채워짐)
      3.  `self.inference_engine.predict_price()`와 `self.inference_engine.predict_quality_rate()`를 호출하여 **ML 모델의 예측값**(`pred_price`, `pred_quality`)을 얻습니다.
      4.  `_construct_state_vector()`: `row` 데이터(수량, 공급가 등)와 **ML 예측값 2개**, `lead_time` 등을 결합하여 `config.OBS_DIM`(예: 50) 차원의 `State` 벡터를 생성합니다.
      5.  `_calculate_reward()`: `row`가 CS 데이터에 있는지 확인하여 `Reward`를 계산합니다 (예: 판매 `+1`, CS `-5`).
      6.  `Action=1` (판매되었으므로), `Done=True` (컨텍스추얼 밴딧)로 설정하고 `Next_State`는 0 벡터로 채웁니다.
    - **출력:** `(State, Action, Reward, Next_State, Done)` 튜플 리스트 (`transitions`)
4.  **버퍼 채우기:** `train_rl.py`가 `transitions` 리스트를 `buffer.put()`을 통해 PER 버퍼에 모두 저장합니다.

### 2.3. 메인 학습 루프 (in `train_rl.py`)

1.  `for total_steps in range(config.NUM_TRAINING_STEPS):` 루프가 실행됩니다.
2.  **`buffer.sample()`**: `BATCH_SIZE`만큼 `(s_b, a_b, r_b, s_p_b, done_mask_b, ...)` 미니배치를 샘플링합니다.
3.  **`get_target_dqn()`**
    - **입력:** `q_main`, `q_target`, `mini_batch`
    - **로직:** 더블 DQN(Double DQN) 공식을 사용해 타겟 Q-value를 계산합니다.
      - `target = r + gamma * done_mask * Q_target(s_prime, argmax Q_main(s_prime))`
    - **출력:** `target` 텐서
4.  **`q_main.train()`**
    - **입력:** `target`, `mini_batch`, `is_weights` (PER 가중치)
    - **로직:** `q_main.forward(s_b)`로 `current_q`를 계산하고, `F.smooth_l1_loss(current_q, target)`로 손실(Loss)을 계산합니다. PER 가중치를 적용하고 `loss.backward()` 및 `optimizer.step()`을 실행합니다.
    - **출력:** `td_error`
5.  **`buffer.update_priorities()`**: `td_error`를 이용해 PER 버퍼의 우선순위를 업데이트합니다.
6.  **`q_main.soft_update()`**: 주기적으로 `q_target` 네트워크의 가중치를 `q_main` 쪽으로 서서히 업데이트합니다.
7.  **`torch.save()`**: 주기적으로 `model_register/rl_model/` 경로에 `dqn_checkpoint_step...pth` 체크포인트 파일을 저장합니다.

---

## ## 3. Phase 3: 통합 서빙 (API)

- **목표:** 학습된 모든 모델(ML+RL)을 사용하여 실시간 예측 API를 제공합니다.
- **실행 스크립트:** `serving/main.py`
- **실행 명령어:** `python -m serving.main` (또는 `uvicorn ...`)

### 3.1. `serving/model_loader.py` (백그라운드)

1.  **`ModelLoader` (글로벌 인스턴스):** API 서버가 시작될 때 `ModelLoader`가 **단 하나** 생성됩니다.
2.  **`ModelLoader.__init__()`**
    - **`_update_models()`**를 1회 호출하여 **최초 모델 로드**를 시도합니다.
    - **`_run_update_checker()`** 백그라운드 스레드를 실행합니다.
3.  **`ModelLoader._update_models()`**
    - **`_find_latest_model_path()`**를 사용해 `model_register/ml_model/` 및 `model_register/rl_model/` 디렉토리를 스캔하여 가장 최신 모델 파일의 경로/접두사를 찾습니다.
    - 현재 로드된 모델(`self._latest_price_prefix` 등)과 비교하여 새 파일이 감지되면, `ModelInferenceEngine()`을 **새로 생성**하여 `self._engine`을 **교체(Hot-swap)**합니다.
    - `self._engine`이 교체되면 `self.latest_..._loaded` 속성도 새 경로로 업데이트합니다.
4.  **`ModelLoader._run_update_checker()`**
    - `time.sleep(600)` (10분)마다 `_update_models()`를 호출하여 새 모델을 감시합니다.

### 3.2. `serving/inference.py` (설계도)

- **`ModelInferenceEngine.__init__()`**:

  - **`_load_latest_ml_models()`**: 최신 `...features.pkl`과 `..._lgb.pkl` 등을 `joblib.load`로 로드하여 `self.price_model`, `self.quality_model`에 저장합니다.
  - **`_load_latest_rl_checkpoint()`**: 최신 `dqn_checkpoint_...pth` 파일을 `torch.load`하여 `self.q_network`에 로드하고, **`self.q_network.eval()` (평가 모드)**로 설정합니다.

- **`ModelInferenceEngine.predict_q_values()`**:

  - **입력:** `state_t` (JSON/Dict)
  - **로직:**
    1.  `flatten_state_to_vector()`를 호출하여 `state_t`를 1D Numpy 벡터로 변환합니다.
    2.  `torch.from_numpy(...).unsqueeze(0)`로 [1, 50] 크기의 텐서를 만듭니다.
    3.  `with torch.no_grad():` (추론 모드)
    4.  **`self.q_network(s_tensor)`**를 호출하여 `[Q(미추천), Q(추천)]` (예: `[1, 2]`) 텐서를 얻습니다.
  - **출력:** `[float, float]` (예: `[0.12, 0.95]`)

- **`ModelInferenceEngine.predict_price()`**:

  - **입력:** `pd.DataFrame` (ML 피처)
  - **로직:** `self.price_model.predict(X)` 호출 (회귀 예측)
  - **출력:** `float` (예측 가격)

- **`ModelInferenceEngine.predict_quality_rate()`**:
  - **입력:** `pd.DataFrame` (ML 피처)
  - **로직:** **`self.quality_model.predict_proba(X)`** 호출 (분류 확률 예측)
  - **출력:** `float` (하자비율 0.0 ~ 1.0)

### 3.3. `serving/main.py` (API 서버)

1.  **의존성 주입 (Dependency):** `get_engine()` 함수가 정의됩니다.
2.  **API 요청 (예: `/predict_q_values`):**
    - 클라이언트가 `RLState` JSON을 `POST` 요청으로 보냅니다.
    - FastAPI가 `get_engine()`을 호출하여 `model_loader.py`의 **`global_model_loader`**로부터 현재 활성화된 `ModelInferenceEngine` 인스턴스(`engine`)를 가져옵니다.
    - `engine.predict_q_values(state.dict())`를 호출합니다.
    - 결과(Q-value)를 JSON 응답으로 반환합니다.
