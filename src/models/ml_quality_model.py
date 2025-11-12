# -*- coding: utf-8 -*-
"""
LightGBM, XGBoost, CatBoost, RandomForest 앙상블 품질(하자율) 예측 모델
★ 수정: 0/1 분류(Classification) 문제로 접근 ★
★ 예측값: CS 발생 확률 (0~1 사이의 하자비율) ★
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier # ★ 수정
from sklearn.linear_model import LogisticRegression # ★ 수정 (Meta-model)
from sklearn.model_selection import TimeSeriesSplit
# ★ 수정: 분류 평가 지표 임포트
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    log_loss
)
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
        
        # Base Models 초기화 (분류 모델로 변경)
        self._init_base_models()
    
    def _init_base_models(self):
        """
        기본 모델들 초기화 (불균형 분류 문제에 최적화)
        """
        
        # ★★★ 불균형 데이터 가중치 계산 ★★★
        # CS Rate: 0.0021 (0.21%) 기준
        # scale_pos_weight = (전체 샘플 - CS 샘플) / (CS 샘플) = 0.9979 / 0.0021
        scale_pos_weight = 475 # (약 475배 페널티)
        
        # 1. LightGBM (주력 모델)
        self.models['lgb'] = lgb.LGBMClassifier( # ★ 수정: Classifier
            objective='binary', # ★ 수정: 이진 분류
            n_estimators=800,
            learning_rate=0.03,
            num_leaves=20,
            feature_fraction=0.7,
            bagging_fraction=0.7,
            bagging_freq=5,
            min_child_samples=20,
            verbose=-1,
            is_unbalanced=True # ★★★ 불균형 데이터 처리 옵션 (또는 scale_pos_weight 사용)
            # scale_pos_weight=scale_pos_weight 
        )
        
        # 2. XGBoost
        self.models['xgb'] = xgb.XGBClassifier( # ★ 수정: Classifier
            n_estimators=800,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=5,
            objective='binary:logistic', # ★ 수정: 이진 분류
            eval_metric='logloss', # ★ 수정: 평가 지표
            verbosity=0,
            scale_pos_weight=scale_pos_weight # ★★★ 불균형 데이터 처리
        )
        
        # 4. Random Forest
        self.models['rf'] = RandomForestClassifier( # ★ 수정: Classifier
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced' # ★★★ 불균형 데이터 처리
        )
        
        # 5. Gradient Boosting (추가 모델)
        self.models['gb'] = GradientBoostingClassifier( # ★ 수정: Classifier
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.7,
            random_state=42
            # GBC는 불균형 옵션이 약하므로, 샘플링(SMOTE)이 더 좋을 수 있음
        )
    
    def train_voting_ensemble(self, X_train, y_train, X_val=None, y_val=None):
        """
        Voting Ensemble: 각 모델을 독립적으로 학습 (Soft Voting을 위해 확률 예측)
        """
        print("=" * 60)
        print("Training Voting Ensemble for Quality Classification") # ★ 수정
        print("=" * 60)
        
        val_predictions_proba = {} # ★ 수정: 확률(score) 저장
        
        for name, model in self.models.items():
            print(f"\n[{name.upper()}] Training...")
            
            # (학습 로직은 회귀와 거의 동일)
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
            
            # Validation 성능 평가 (분류 지표 사용)
            if X_val is not None:
                # ★ 수정: predict_proba()로 확률(하자비율) 예측
                y_pred_proba = model.predict_proba(X_val)[:, 1] # (CS=1일 확률)
                y_pred_binary = (y_pred_proba > 0.5).astype(int) # (0.5 임계값 기준)
                
                auc = roc_auc_score(y_val, y_pred_proba)
                f1 = f1_score(y_val, y_pred_binary, zero_division=0)
                recall = recall_score(y_val, y_pred_binary, zero_division=0)
                
                print(f"  Validation AUC-ROC: {auc:.4f}")
                print(f"  Validation F1 (CS=1): {f1:.4f}")
                print(f"  Validation Recall (CS=1): {recall:.4f}")
                
                val_predictions_proba[name] = y_pred_proba
        
        # 가중치 계산 (AUC-ROC 점수 기반)
        if X_val is not None:
            self._calculate_weights_classification(val_predictions_proba, y_val)
        else:
            self.weights = {name: 1.0 / len(self.models) for name in self.models.keys()}
        
        print("\n[Ensemble Weights]")
        for name, weight in self.weights.items():
            print(f"  {name}: {weight:.4f}")
    
    def train_stacking_ensemble(self, X_train, y_train, n_splits=5):
        """
        Stacking Ensemble: CV로 Base Model 학습 → Meta-Model로 최종 예측
        """
        print("=" * 60)
        print("Training Stacking Ensemble for Quality Classification") # ★ 수정
        print("=" * 60)
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        # ★ 수정: 확률(score)을 피처로 사용
        oof_predictions_proba = np.zeros((len(X_train), len(self.models)))
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            print(f"\n--- Fold {fold + 1}/{n_splits} ---")
            
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            fold_auc_scores = []
            
            for i, (name, model) in enumerate(self.models.items()):
                # (학습 로직 동일)
                if name == 'lgb':
                    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], 
                              callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
                elif name == 'xgb':
                    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                else:
                    model.fit(X_tr, y_tr)
                
                # ★ 수정: Out-of-Fold 확률(score) 예측
                oof_pred_proba = model.predict_proba(X_val)[:, 1]
                oof_predictions_proba[val_idx, i] = oof_pred_proba
                
                auc = roc_auc_score(y_val, oof_pred_proba)
                fold_auc_scores.append(auc)
            
            print(f"  Fold {fold+1} Average AUC: {np.mean(fold_auc_scores):.4f}")
        
        # Step 2: Meta-Model 학습 (★ 수정: Logistic Regression)
        print("\n[Training Meta-Model (LogisticRegression)]")
        self.meta_model = LogisticRegression(class_weight='balanced') # 불균형 고려
        self.meta_model.fit(oof_predictions_proba, y_train)
        
        # Meta-Model 계수 출력
        print("\n[Meta-Model Coefficients]")
        for name, coef in zip(self.models.keys(), self.meta_model.coef_[0]):
            print(f"  {name}: {coef:.4f}")
        
        # Step 3: 전체 데이터로 Base Model 재학습 (동일)
        print("\n[Retraining Base Models on Full Data]")
        for name, model in self.models.items():
            model.fit(X_train, y_train) # XGB/LGB eval_set 없이 전체 학습
    
    # (train_blending_ensemble은 Stacking과 유사하게 수정 가능, 여기서는 생략)
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        선택된 앙상블 방법으로 학습
        """
        if self.ensemble_method == 'voting':
            self.train_voting_ensemble(X_train, y_train, X_val, y_val)
        
        elif self.ensemble_method == 'stacking':
            self.train_stacking_ensemble(X_train, y_train, n_splits=5)
        
        elif self.ensemble_method == 'blending':
             raise NotImplementedError("Blending (Classification)은 이 예제에서 구현되지 않았습니다.")
    
    
    # ★★★ 핵심: predict_proba (확률/하자비율 예측) ★★★
    def predict_proba(self, X):
        """
        앙상블 모델로 CS 발생 확률 (0~1 하자비율) 예측
        """
        if self.ensemble_method == 'voting':
            # 가중 평균 (Soft Voting)
            predictions_proba = np.zeros(len(X))
            for name, model in self.models.items():
                # 각 모델의 CS=1 확률에 가중치를 곱하여 더함
                predictions_proba += self.weights[name] * model.predict_proba(X)[:, 1]
            # 최종 확률이 0~1 범위를 벗어날 경우를 대비해 클리핑
            return np.clip(predictions_proba, 0, 1)
        
        elif self.ensemble_method in ['stacking', 'blending']:
            # Base Models 확률 예측 → Meta-Model 최종 확률 예측
            base_predictions_proba = np.column_stack([
                model.predict_proba(X)[:, 1] for model in self.models.values()
            ])
            # Meta-Model의 CS=1 확률 반환
            predictions_proba = self.meta_model.predict_proba(base_predictions_proba)[:, 1]
            return np.clip(predictions_proba, 0, 1)

    # ★★★ 핵심: predict (0/1 분류 예측) ★★★
    def predict(self, X, threshold=0.5):
        """
        앙상블 모델로 0/1 분류 예측 (임계값 기준)
        """
        # 먼저 확률(하자비율)을 계산
        probabilities = self.predict_proba(X)
        # 임계값(threshold)을 넘으면 1(CS), 아니면 0(정상)
        return (probabilities >= threshold).astype(int)

    def _clip_predictions(self, predictions):
        """
        (회귀용이었으나 분류에서는 predict_proba 내부에서 처리)
        """
        return np.clip(predictions, 0, 1)
    
    def _calculate_weights_classification(self, val_predictions_proba, y_val):
        """
        Validation 성능 기반 가중치 계산 (AUC-ROC 점수 기반)
        """
        auc_scores = {}
        for name, proba in val_predictions_proba.items():
            auc_scores[name] = roc_auc_score(y_val, proba)
        
        # 가중치 = AUC^2 (AUC가 0.5 근처(랜덤)면 가중치 낮춤, 1에 가까우면 높임)
        # (MAE 역수보다 AUC 제곱이 더 안정적일 수 있음)
        weights_raw = {name: (max(0.5, auc) - 0.5)**2 for name, auc in auc_scores.items()}
        total = sum(weights_raw.values())
        if total == 0: # 모든 모델이 0.5 이하일 경우
            self.weights = {name: 1.0 / len(self.models) for name in self.models.keys()}
        else:
            self.weights = {name: w / total for name, w in weights_raw.items()}

    def get_feature_importance(self, feature_names):
        """
        각 모델의 Feature Importance 추출 (분류 모델도 동일하게 작동)
        """
        importance_df = pd.DataFrame()
        importance_df['feature'] = feature_names
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_df[name] = model.feature_importances_
        
        importance_df['mean'] = importance_df.drop('feature', axis=1).mean(axis=1)
        return importance_df.sort_values('mean', ascending=False)
    
    # ★★★ 수정: evaluate 메서드는 이제 train_ml.py로 이동함 ★★★
    # (이 파일의 evaluate는 train_ml.py의 QualityModelTrainer.evaluate가 덮어쓰므로)
    # (코드의 일관성을 위해 여기서는 삭제하거나 주석 처리)
    
    # def evaluate(self, X_test, y_test): ... (삭제 또는 주석 처리)

    def save_models(self, filepath_prefix: str):
        """
        앙상블 모델 저장 (동일)
        """
        for name, model in self.models.items():
            joblib.dump(model, f"{filepath_prefix}_{name}.pkl")
        
        if self.meta_model:
            joblib.dump(self.meta_model, f"{filepath_prefix}_meta.pkl")
        
        if self.weights:
            joblib.dump(self.weights, f"{filepath_prefix}_weights.pkl")
        
        config = {'ensemble_method': self.ensemble_method}
        joblib.dump(config, f"{filepath_prefix}_config.pkl")
        
        print(f"Ensemble models saved to {filepath_prefix}_*.pkl")
    
    def load_models(self, filepath_prefix: str):
        """
        앙상블 모델 로드 (동일)
        """
        config = joblib.load(f"{filepath_prefix}_config.pkl")
        self.ensemble_method = config['ensemble_method']
        
        for name in self.models.keys():
            self.models[name] = joblib.load(f"{filepath_prefix}_{name}.pkl")
        
        if self.ensemble_method in ['stacking', 'blending']:
            self.meta_model = joblib.load(f"{filepath_prefix}_meta.pkl")
        
        if self.ensemble_method == 'voting':
            self.weights = joblib.load(f"{filepath_prefix}_weights.pkl")
        
        print(f"Ensemble models loaded from {filepath_prefix}_*.pkl")