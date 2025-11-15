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
    initialize the basic models
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
    mae_scores = {}
    for name, pred in val_predictions.items():
      mae_scores[name] = mean_absolute_error(y_val, pred)

    inverse_mae = {name: 1.0 / mae for name, mae in mae_scores.items()}
    total = sum(inverse_mae.values())
    self.weights = {name: w / total for name, w in inverse_mae.items()}

  def train_voting_ensemble(self, X_train, y_train, X_val = None, y_val = None):
    """"
    voting ensemble : train the model seperately(?) and caculate the avg of the prediction
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
    stacking ensemble: first, base model training and using meta model for final prediction
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
    blending : train base model by other validation set
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
    if self.ensemble_method == 'voting':
      self.train_voting_ensemble(X_train, y_train, X_val, y_val)

    elif self.ensemble_method == 'stacking':
      self.train_stacking_ensemble(X_train, y_train, n_splits = 5)

    elif self.ensemble_method == 'blending':
      if X_val is None or y_val is None:
        raise ValueError('Blending requires seperate validation set!')
      self.train_blending_ensemble(X_train, y_train, X_val, y_val)

  def predict(self, X):
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
    extract each model's feature importance
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