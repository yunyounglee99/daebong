"""
앙상블 가격 예측 모델

LightGBM, XGBoost, RandomForest를 결합한 앙상블 회귀 모델
- Voting: 가중 평균 앙상블
- Stacking: 메타 모델 기반 앙상블
- Blending: 별도 검증 세트 기반 앙상블
- 시계열 교차 검증 지원
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')

class EnsemblePriceModel:
    """
    앙상블 가격 예측 모델 클래스

    여러 트리 기반 모델을 결합하여 사과 가격을 예측
    """
    def __init__(self, ensemble_method = 'voting'):
      """"
      Args:
      ensemble_method: voting, stacking, bleding 중 선택
      """
      self. ensemble_method = ensemble_method
      self.models = {}
      self.meta_model = None
      self.weights = None

      self._init_base_models()

    def _init_base_models(self):
      """
      용도: (내부 함수) 앙상블에 사용될 기본(Base) 회귀 모델들을 초기화합니다.
      Args:
          None
      Returns:
          None
      로직:
          1. LGBMRegressor, XGBRegressor, RandomForestRegressor 모델을 
              미리 설정된 하이퍼파라미터로 초기화합니다.
          2. `self.models` 딕셔너리에 'lgb', 'xgb', 'rf' 키로 각 모델을 저장합니다.
      """

      # 1. LightGBM (Main Model)
      self.models['lgb'] = lgb.LGBMRegressor(
        objective = 'regression',
        n_estimators = 1000,
        learning_rate = 0.05,
        num_leaves = 31,
        feature_fraction = 0.8,
        bagging_fraction = 0.8,
        bagging_freq = 5,
        verbose = -1
      )

      # 2. XGBoost
      self.models['xgb'] = xgb.XGBRegressor(
        n_estimators = 1000,
        learning_rate = 0.05,
        max_depth = 6,
        subsample = 0.8,
        colsample_bytree = 0.8,
        objective = 'reg:squarederror',
        eval_metric = 'mae',
        verbosity = 0
      )

      # 3. Random Forest
      self.models['rf'] = RandomForestRegressor(
        n_estimators = 200,
        max_depth = 15,
        min_samples_split = 5,
        min_samples_leaf = 2,
        random_state = 42,
        n_jobs = -1
      )

    def _calculate_weights(self, val_predictions, y_val):
      """
      용도: (내부 함수) Voting 앙상블에 사용될 모델별 가중치를 계산합니다.
      Args:
          val_predictions (dict): {'lgb': [pred1, ...], 'xgb': [pred2, ...]} 형태의 검증 세트 예측 결과.
          y_val (pd.Series): 검증 세트의 실제 정답 값.
      Returns:
          None
      로직:
          1. 각 모델의 예측(pred)과 실제(y_val) 간의 MAE(Mean Absolute Error)를 계산합니다.
          2. MAE가 낮을수록 성능이 좋으므로, MAE의 역수(1 / MAE)를 기본 가중치로 사용합니다.
          3. 이 역수 값들을 정규화(총합이 1이 되도록)하여 `self.weights` 딕셔너리에 저장합니다.
      """
      mae_scores = {}
      for name, pred in val_predictions.items():
        mae_scores[name] = mean_absolute_error(y_val, pred)

      inverse_mae = {name: 1.0 / mae for name, mae in mae_scores.items()}
      total = sum(inverse_mae.values())
      self.weights = {name: w / total for name, w in inverse_mae.items()}

    def train_voting_ensemble(self, X_train, y_train, X_val = None, y_val = None):
      """
      용도: Voting 앙상블 방식으로 모델을 학습합니다.
      Args:
          X_train (pd.DataFrame): 훈련용 피처 데이터.
          y_train (pd.Series): 훈련용 타겟 데이터.
          X_val (pd.DataFrame, optional): 검증용 피처 데이터.
          y_val (pd.Series, optional): 검증용 타겟 데이터.
      Returns:
          None
      로직:
          1. `self.models` 딕셔너리를 순회하며 'lgb', 'xgb', 'rf' 모델을 각각 `fit()` 메서드로 학습시킵니다.
          2. (LGB/XGB) `X_val`이 제공되면, Early Stopping을 사용하여 최적의 성능으로 학습을 조기 종료합니다.
          3. (RF) `X_train`, `y_train` 전체로 학습합니다.
          4. `X_val`이 제공되면, 각 모델의 검증 성능(MAE, RMSE)을 출력하고, `_calculate_weights`를 호출하여 
              성능 기반의 가중치(`self.weights`)를 계산합니다.
          5. `X_val`이 없으면, 모든 모델에 동일한 가중치(1/N)를 할당합니다.
      """
      print("=" * 20)
      print("Training Voting Ensemble Models...")
      print("=" * 20)

      val_predictions = {}

      for name, model in self.models.items():
        print(f'\n[{name.upper()}] Training...')

        if name == 'lgb':
          model.fit(
            X_train, y_train,
            eval_set = [(X_val, y_val)] if X_val is not None else None,
            callbacks = [lgb.early_stopping(stopping_rounds = 50)] if X_val is not None else None
          )
        elif name == 'xgb':
          model.fit(
            X_train, y_train,
            eval_set = [(X_val, y_val)] if X_val is not None else None,
            verbose = False
          )
        else:
          model.fit(X_train, y_train)

        # Validation
        if X_val is not None:
          y_pred = model.predict(X_val)
          mae = mean_absolute_error(y_val, y_pred)
          rmse = np.sqrt(mean_squared_error(y_val, y_pred))
          print(f'Validation MAE: {mae:.2f}. RMSE: {rmse:.2f}')

          val_predictions[name] = y_pred

        #caculating the weights
        if X_val is not None:
          self._calculate_weights(val_predictions, y_val)
        else:
          # same weight
          self.weights = {name: 1.0 / len(self.models) for name in self.models.keys()}

        print('\n[Ensemble Weights]')
        for name, weight in self.weights.items():
          print(f'{name}: {weight:.4f}')

    def train_stacking_ensemble(self, X_train, y_train, n_splits = 5):
      """
      용도: Stacking 앙상블 방식으로 모델을 학습합니다.
      Args:
          X_train (pd.DataFrame): 전체 훈련용 피처 데이터.
          y_train (pd.Series): 전체 훈련용 타겟 데이터.
          n_splits (int): 시계열 교차 검증(TimeSeriesSplit)을 위한 Fold 수.
      Returns:
          None
      로직:
          1. (Base Model 학습) `TimeSeriesSplit`으로 교차 검증(CV)을 수행합니다.
              - 각 Fold에서, Base 모델(lgb, xgb, rf)을 Fold의 훈련 데이터로 학습시킵니다.
              - 학습된 모델로 Fold의 검증 데이터를 예측하여 'Out-of-Fold(OOF)' 예측값을 생성합니다.
              - 이 OOF 예측값은 Meta-Model을 학습시키기 위한 입력(피처)이 됩니다.
          2. (Meta-Model 학습) `oof_predictions` (Base 모델들의 예측값)를 피처로, 
              `y_train` (원본 타겟)을 타겟으로 하여 `Ridge` 회귀 모델(Meta-Model)을 학습시킵니다.
          3. (Base Model 재학습) CV가 끝난 후, 모든 Base 모델을 `X_train` **전체** 데이터로 다시 학습시켜 
              나중에 `predict()`를 호출할 때 사용할 최종 모델로 저장합니다.
      """
      print("=" * 20)
      print("Training stacking Ensemble Models...")
      print("=" * 20)

      # 1. generate out-of-fold prediction by time series cross-validation
      tscv = TimeSeriesSplit(n_splits = n_splits)
      oof_predictions = np.zeros((len(X_train), len(self.models)))

      for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        print(f'\n--- Fold {fold + 1}/{n_splits} ---')

        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        for i, (name, model) in enumerate(self.models.items()):
          #training models
          if name == 'lgb':
            model.fit(X_tr, y_tr, eval_set = [(X_val, y_val)],
                      callbacks = [lgb.early_stopping(stopping_rounds = 50)])
          elif name == 'xgb':
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
          else:
            model.fit(X_tr, y_tr)

          oof_predictions[val_idx, i] = model.predit(X_val)

      # 2. training meta model
      print('\n[Training Meta-Model (Ridge)]')
      self.metal_model =Ridge(alpha=1.0)
      self.meta_model.fit(oof_predictions, y_train)

      print('\n[Meta-Model Coefficients]')
      for name, coef in zip(self.models.keys(), self.meta_model.coef_):
        print(f'{name}: {coef:.4f}')

      # 3. Re-training by Full data
      print('\n[Retraining Base Model on Full data]')
      for name, model in self.models.items():
        model.fit(X_train, y_train)

    def train_blending_ensemble(self, X_train, y_train, X_blend, y_blend):
      """
      용도: Blending 앙상블 방식으로 모델을 학습합니다. (train/val 분할이 외부에서 이루어져야 함)
      Args:
          X_train (pd.DataFrame): Base 모델 학습용 피처 데이터.
          y_train (pd.Series): Base 모델 학습용 타겟 데이터.
          X_blend (pd.DataFrame): Meta-Model 학습용 피처 데이터 (Validation Set).
          y_blend (pd.Series): Meta-Model 학습용 타겟 데이터 (Validation Set).
      Returns:
          None
      로직:
          1. (Base Model 학습) `X_train`, `y_train`으로 Base 모델(lgb, xgb, rf)을 학습시킵니다.
          2. (Meta-Model 피처 생성) 학습된 Base 모델로 `X_blend`를 예측하여 `blend_predictions`를 생성합니다.
          3. (Meta-Model 학습) `blend_predictions`를 피처로, `y_blend`를 타겟으로 하여 
              `Ridge` 회귀 모델(Meta-Model)을 학습시킵니다.
      """
      print("=" * 20)
      print("Training blending Ensemble Models...")
      print("=" * 20)

      # 1. training blend models
      blend_predictions = np.zeros((len(X_blend), len(self.models)))

      for i, (name, model) in enumerate(self.models.items()):
        print(f'\n[{name.upper()}] Traning...')
        model.fit(X_train, y_train)

        # predict blending set
        blend_predictions[:, i] = model.predict(X_blend)

        mae = mean_absolute_error(y_blend, blend_predictions[:, i])
        print(f' Blending MAE: {mae:.2f}')

      # 2. training meta-model
      print('\n[training Meta-Model]')
      self.meta_model = Ridge(alpha = 1.0)
      self.meta_model.fit(blend_predictions, y_blend)

      print('\n[Meta-Model Coefficients]')
      for name, coef in zip(self.models.keys(), self.meta_model.coef_):
        print(f'{name}: {coef:.4f}')

    def train(self, X_train, y_train, X_val = None, y_val = None):
      """
      용도: `train_ml.py`에서 호출하는 메인 학습 함수입니다.
      Args:
          X_train (pd.DataFrame): 훈련용 피처 데이터.
          y_train (pd.Series): 훈련용 타겟 데이터.
          X_val (pd.DataFrame, optional): 검증용 피처 데이터. (Voting/Blending 시 사용)
          y_val (pd.Series, optional): 검증용 타겟 데이터. (Voting/Blending 시 사용)
      Returns:
          None
      로직:
          `self.ensemble_method` ('voting', 'stacking', 'blending') 설정값에 따라
          적절한 train_..._ensemble 내부 함수를 호출합니다.
      """
      if self.ensemble_method == 'voting':
        self.train_voting_ensemble(X_train, y_train, X_val, y_val)

      elif self.ensemble_method == 'stacking':
        self.train_stacking_ensemble(X_train, y_train, n_splits = 5)

      elif self.ensemble_method == 'blending':
        if X_val is None or y_val is None:
          raise ValueError('Blending requires seperate validation set!')
        self.train_blending_ensemble(X_train, y_train, X_val, y_val)

    def predict(self, X):
      """
      용도: 학습된 앙상블 모델을 사용하여 새로운 데이터(X)의 가격을 예측합니다.
      Args:
          X (pd.DataFrame): 예측할 피처 데이터.
      Returns:
          np.array: 예측된 가격 값.
      로직:
          - (Voting) `self.weights`를 사용하여 각 Base 모델의 예측값을 가중 평균합니다.
          - (Stacking/Blending) `X`를 모든 Base 모델로 예측한 후, 그 결과를 다시
            `self.meta_model`에 입력하여 최종 예측값을 받습니다.
      """
      if self.ensemble_method == 'voting':
        predictions = np.zeros(len(X))
        for name, model in self.models.items():
          predictions += self.weights[name] * model.predict(X)
        return predictions
      
      elif self.ensemble_method in ['stacking', 'blending']:
        base_predictions = np.column_stack([
          model.predict(X) for model in self.models.values()
        ])

        return self.meta_model.predict(base_predictions)
      
    def get_feature_importance(self):
      """
      용도: (Voting 방식일 때 유용) 학습된 Base 모델들의 피처 중요도를 추출하여 평균냅니다.
      Args:
          None
      Returns:
          pd.DataFrame: 피처 이름과 모델별 중요도, 평균 중요도가 포함된 DataFrame.
      로직:
          1. `feature_importances_` 속성을 가진 모델(lgb, xgb, rf)을 순회합니다.
          2. 각 모델의 피처 중요도를 DataFrame에 저장합니다.
          3. 모델별 중요도의 평균을 계산하고, 평균값 기준으로 정렬하여 반환합니다.
      """
      importance_df = pd.DataFrame()

      for name, model in self.models.items():
        if hasattr(model, 'feature_importances_'):
          importance_df[name] = model.feature_importances_

      importance_df['mean'] = importance_df.mean(axis=1)
      
      return importance_df.sort_values('mean', ascending = False)
    
    def save_models(self, filepath_prefix: str):
      for name, model in self.models.items():
        joblib.dump(model, f'{filepath_prefix}_{name}.pkl')

      if self.meta_model:
        joblib.dump(self.meta_model, f'{filepath_prefix}_meta.pkl')

      joblib.dump(self.weights, f"{filepath_prefix}_weights.pkl")
      print(f'Ensemble models saved to {filepath_prefix}_*.pkl')

    def load_models(self, filepath_prefix: str):
      for name in self.models.keys():
        self.models[name] = joblib.load(f'{filepath_prefix}_{name}.pkl')

      if self.ensemble_method in ['stacking', 'blending']:
        self.meta_model = joblib.load(f'{filepath_prefix}_meta.pkl')

      self.weights = joblib.load(f'{filepath_prefix}_weights.pkl')
      print(f'Ensemble models loaded from {filepath_prefix}_*.pkl')