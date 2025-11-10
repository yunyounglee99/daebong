"""
LightGBM, XGBoost, CatBoost, RandomForest 앙상블 품질(하자율) 예측 모델
하자율은 0~1 범위의 연속값 (회귀 문제)
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class EnsembleQualityModel:
    def __init__(self, ensemble_method='voting'):
        """
        Args:
            ensemble_method: 'voting', 'stacking', 'blending' 중 선택
        """
        self.ensemble_method = ensemble_method
        self.models = {}
        self.meta_model = None
        self.weights = None
        
        # Base Models 초기화
        self._init_base_models()
    
    def _init_base_models(self):
        """
        기본 모델들 초기화 (하자율 예측에 최적화된 파라미터)
        """
        # 1. LightGBM (주력 모델)
        self.models['lgb'] = lgb.LGBMRegressor(
            objective='regression',
            n_estimators=800,
            learning_rate=0.03,
            num_leaves=20,  # 과적합 방지
            feature_fraction=0.7,
            bagging_fraction=0.7,
            bagging_freq=5,
            min_child_samples=20,  # 안정성 증가
            verbose=-1
        )
        
        # 2. XGBoost
        self.models['xgb'] = xgb.XGBRegressor(
            n_estimators=800,
            learning_rate=0.03,
            max_depth=4,  # 얕은 트리로 과적합 방지
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=5,
            objective='reg:squarederror',
            eval_metric='mae',
            verbosity=0
        )
        
        # 4. Random Forest
        self.models['rf'] = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        # 5. Gradient Boosting (추가 모델)
        self.models['gb'] = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.7,
            random_state=42
        )
    
    def train_voting_ensemble(self, X_train, y_train, X_val=None, y_val=None):
        """
        Voting Ensemble: 각 모델을 독립적으로 학습 후 예측값 가중 평균
        """
        print("=" * 60)
        print("Training Voting Ensemble for Quality Prediction")
        print("=" * 60)
        
        val_predictions = {}
        
        for name, model in self.models.items():
            print(f"\n[{name.upper()}] Training...")
            
            if name == 'lgb':
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)] if X_val is not None else None,
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)] if X_val is not None else None
                )
            elif name == 'xgb':
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)] if X_val is not None else None,
                    verbose=False
                )
            else:
                model.fit(X_train, y_train)
            
            # Validation 성능 평가
            if X_val is not None:
                y_pred = self._clip_predictions(model.predict(X_val))
                mae = mean_absolute_error(y_val, y_pred)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                r2 = r2_score(y_val, y_pred)
                
                print(f"  Validation MAE:  {mae:.4f}")
                print(f"  Validation RMSE: {rmse:.4f}")
                print(f"  Validation R²:   {r2:.4f}")
                
                val_predictions[name] = y_pred
        
        # 가중치 계산 (성능 역수 기반)
        if X_val is not None:
            self._calculate_weights(val_predictions, y_val)
        else:
            # 동일 가중치
            self.weights = {name: 1.0 / len(self.models) for name in self.models.keys()}
        
        print("\n[Ensemble Weights]")
        for name, weight in self.weights.items():
            print(f"  {name}: {weight:.4f}")
    
    def train_stacking_ensemble(self, X_train, y_train, n_splits=5):
        """
        Stacking Ensemble: CV로 Base Model 학습 → Meta-Model로 최종 예측
        """
        print("=" * 60)
        print("Training Stacking Ensemble for Quality Prediction")
        print("=" * 60)
        
        # Step 1: Time Series Cross-Validation으로 Out-of-Fold 예측값 생성
        tscv = TimeSeriesSplit(n_splits=n_splits)
        oof_predictions = np.zeros((len(X_train), len(self.models)))
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            print(f"\n--- Fold {fold + 1}/{n_splits} ---")
            
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            fold_mae_scores = []
            
            for i, (name, model) in enumerate(self.models.items()):
                # 각 모델 학습
                if name == 'lgb':
                    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], 
                              callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
                elif name == 'xgb':
                    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                else:
                    model.fit(X_tr, y_tr)
                
                # Out-of-Fold 예측
                oof_pred = self._clip_predictions(model.predict(X_val))
                oof_predictions[val_idx, i] = oof_pred
                
                mae = mean_absolute_error(y_val, oof_pred)
                fold_mae_scores.append(mae)
            
            print(f"  Fold {fold+1} Average MAE: {np.mean(fold_mae_scores):.4f}")
        
        # Step 2: Meta-Model 학습 (Ridge Regression)
        print("\n[Training Meta-Model (Ridge)]")
        self.meta_model = Ridge(alpha=1.0)
        self.meta_model.fit(oof_predictions, y_train)
        
        # Meta-Model 계수 출력
        print("\n[Meta-Model Coefficients]")
        for name, coef in zip(self.models.keys(), self.meta_model.coef_):
            print(f"  {name}: {coef:.4f}")
        
        # Step 3: 전체 데이터로 Base Model 재학습
        print("\n[Retraining Base Models on Full Data]")
        for name, model in self.models.items():
            if name == 'lgb':
                model.fit(X_train, y_train)
            elif name == 'xgb':
                model.fit(X_train, y_train, verbose=False)
            elif name == 'catboost':
                model.fit(X_train, y_train, verbose=False)
            else:
                model.fit(X_train, y_train)
    
    def train_blending_ensemble(self, X_train, y_train, X_blend, y_blend):
        """
        Blending: 별도 Validation Set으로 Base Model 학습 → Meta-Model 학습
        """
        print("=" * 60)
        print("Training Blending Ensemble for Quality Prediction")
        print("=" * 60)
        
        # Step 1: Base Models 학습
        blend_predictions = np.zeros((len(X_blend), len(self.models)))
        
        for i, (name, model) in enumerate(self.models.items()):
            print(f"\n[{name.upper()}] Training...")
            
            if name == 'lgb':
                model.fit(X_train, y_train)
            elif name == 'xgb':
                model.fit(X_train, y_train, verbose=False)
            elif name == 'catboost':
                model.fit(X_train, y_train, verbose=False)
            else:
                model.fit(X_train, y_train)
            
            # Blending Set 예측
            blend_pred = self._clip_predictions(model.predict(X_blend))
            blend_predictions[:, i] = blend_pred
            
            mae = mean_absolute_error(y_blend, blend_pred)
            rmse = np.sqrt(mean_squared_error(y_blend, blend_pred))
            print(f"  Blending MAE:  {mae:.4f}")
            print(f"  Blending RMSE: {rmse:.4f}")
        
        # Step 2: Meta-Model 학습
        print("\n[Training Meta-Model]")
        self.meta_model = Ridge(alpha=1.0)
        self.meta_model.fit(blend_predictions, y_blend)
        
        print("\n[Meta-Model Coefficients]")
        for name, coef in zip(self.models.keys(), self.meta_model.coef_):
            print(f"  {name}: {coef:.4f}")
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        선택된 앙상블 방법으로 학습
        """
        if self.ensemble_method == 'voting':
            self.train_voting_ensemble(X_train, y_train, X_val, y_val)
        
        elif self.ensemble_method == 'stacking':
            self.train_stacking_ensemble(X_train, y_train, n_splits=5)
        
        elif self.ensemble_method == 'blending':
            if X_val is None or y_val is None:
                raise ValueError("Blending requires separate validation set!")
            self.train_blending_ensemble(X_train, y_train, X_val, y_val)
    
    def predict(self, X):
        """
        앙상블 예측 (0~1 범위로 클리핑)
        """
        if self.ensemble_method == 'voting':
            # 가중 평균
            predictions = np.zeros(len(X))
            for name, model in self.models.items():
                predictions += self.weights[name] * model.predict(X)
            return self._clip_predictions(predictions)
        
        elif self.ensemble_method in ['stacking', 'blending']:
            # Base Models 예측 → Meta-Model 최종 예측
            base_predictions = np.column_stack([
                model.predict(X) for model in self.models.values()
            ])
            predictions = self.meta_model.predict(base_predictions)
            return self._clip_predictions(predictions)
    
    def _clip_predictions(self, predictions):
        """
        하자율을 0~1 범위로 제한
        """
        return np.clip(predictions, 0, 1)
    
    def _calculate_weights(self, val_predictions, y_val):
        """
        Validation 성능 기반 가중치 계산 (MAE 역수)
        """
        mae_scores = {}
        for name, pred in val_predictions.items():
            mae_scores[name] = mean_absolute_error(y_val, pred)
        
        # 가중치 = 1 / MAE (정규화)
        inverse_mae = {name: 1.0 / mae for name, mae in mae_scores.items()}
        total = sum(inverse_mae.values())
        self.weights = {name: w / total for name, w in inverse_mae.items()}
    
    def get_feature_importance(self, feature_names):
        """
        각 모델의 Feature Importance 추출
        """
        importance_df = pd.DataFrame()
        importance_df['feature'] = feature_names
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_df[name] = model.feature_importances_
        
        importance_df['mean'] = importance_df.drop('feature', axis=1).mean(axis=1)
        return importance_df.sort_values('mean', ascending=False)
    
    def evaluate(self, X_test, y_test):
        """
        테스트 데이터 평가
        """
        y_pred = self.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # MAPE (Mean Absolute Percentage Error) - 0으로 나누기 방지
        mask = y_test != 0
        mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100 if mask.any() else np.nan
        
        print("\n[Test Set Evaluation]")
        print(f"  MAE:   {mae:.4f}")
        print(f"  RMSE:  {rmse:.4f}")
        print(f"  R²:    {r2:.4f}")
        if not np.isnan(mape):
            print(f"  MAPE:  {mape:.2f}%")
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
    
    def save_models(self, filepath_prefix: str):
        """
        앙상블 모델 저장
        """
        for name, model in self.models.items():
            joblib.dump(model, f"{filepath_prefix}_{name}.pkl")
        
        if self.meta_model:
            joblib.dump(self.meta_model, f"{filepath_prefix}_meta.pkl")
        
        if self.weights:
            joblib.dump(self.weights, f"{filepath_prefix}_weights.pkl")
        
        # 앙상블 설정 저장
        config = {'ensemble_method': self.ensemble_method}
        joblib.dump(config, f"{filepath_prefix}_config.pkl")
        
        print(f"Ensemble models saved to {filepath_prefix}_*.pkl")
    
    def load_models(self, filepath_prefix: str):
        """
        앙상블 모델 로드
        """
        # 설정 로드
        config = joblib.load(f"{filepath_prefix}_config.pkl")
        self.ensemble_method = config['ensemble_method']
        
        # Base Models 로드
        for name in self.models.keys():
            self.models[name] = joblib.load(f"{filepath_prefix}_{name}.pkl")
        
        # Meta-Model & Weights 로드
        if self.ensemble_method in ['stacking', 'blending']:
            self.meta_model = joblib.load(f"{filepath_prefix}_meta.pkl")
        
        if self.ensemble_method == 'voting':
            self.weights = joblib.load(f"{filepath_prefix}_weights.pkl")
        
        print(f"Ensemble models loaded from {filepath_prefix}_*.pkl")
