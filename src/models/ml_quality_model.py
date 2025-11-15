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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
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
        
        self._init_base_models()
    
    def _init_base_models(self):
        
        scale_pos_weight = 475 # (약 475배 페널티)
        
        # 1. LightGBM 
        self.models['lgb'] = lgb.LGBMClassifier( # Classifier
            objective='binary', # binary classification
            n_estimators=800,
            learning_rate=0.03,
            num_leaves=20,
            feature_fraction=0.7,
            bagging_fraction=0.7,
            bagging_freq=5,
            min_child_samples=20,
            verbose=-1,
            is_unbalanced=True # 불균형 데이터 처리 옵션 (또는 scale_pos_weight 사용)
            # scale_pos_weight=scale_pos_weight 
        )
        
        # 2. XGBoost
        self.models['xgb'] = xgb.XGBClassifier(
            n_estimators=800,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=5,
            objective='binary:logistic',
            eval_metric='logloss',
            verbosity=0,
            scale_pos_weight=scale_pos_weight
        )
        
        # 4. Random Forest
        self.models['rf'] = RandomForestClassifier( 
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced' 
        )
        
        # 5. Gradient Boosting 
        self.models['gb'] = GradientBoostingClassifier( 
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
        print("Training Voting Ensemble for Quality Classification") 
        print("=" * 60)
        
        val_predictions_proba = {} # saving score (as model is now classifying 0, 1 and we need rate)
        
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
            
            # Validation
            if X_val is not None:
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                y_pred_binary = (y_pred_proba > 0.5).astype(int) # (threshold = 0.5)
                
                auc = roc_auc_score(y_val, y_pred_proba)
                f1 = f1_score(y_val, y_pred_binary, zero_division=0)
                recall = recall_score(y_val, y_pred_binary, zero_division=0)
                
                print(f"  Validation AUC-ROC: {auc:.4f}")
                print(f"  Validation F1 (CS=1): {f1:.4f}")
                print(f"  Validation Recall (CS=1): {recall:.4f}")
                
                val_predictions_proba[name] = y_pred_proba
        
        # calculating weight (AUC-ROC 점수 기반)
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
        print("Training Stacking Ensemble for Quality Classification")
        print("=" * 60)
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        oof_predictions_proba = np.zeros((len(X_train), len(self.models)))
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            print(f"\n--- Fold {fold + 1}/{n_splits} ---")
            
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            fold_auc_scores = []
            
            for i, (name, model) in enumerate(self.models.items()):
                if name == 'lgb':
                    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], 
                              callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
                elif name == 'xgb':
                    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                else:
                    model.fit(X_tr, y_tr)
                
                oof_pred_proba = model.predict_proba(X_val)[:, 1]
                oof_predictions_proba[val_idx, i] = oof_pred_proba
                
                auc = roc_auc_score(y_val, oof_pred_proba)
                fold_auc_scores.append(auc)
            
            print(f"  Fold {fold+1} Average AUC: {np.mean(fold_auc_scores):.4f}")
        
        print("\n[Training Meta-Model (LogisticRegression)]")
        self.meta_model = LogisticRegression(class_weight='balanced') 
        self.meta_model.fit(oof_predictions_proba, y_train)
        
        print("\n[Meta-Model Coefficients]")
        for name, coef in zip(self.models.keys(), self.meta_model.coef_[0]):
            print(f"  {name}: {coef:.4f}")
        
        print("\n[Retraining Base Models on Full Data]")
        for name, model in self.models.items():
            model.fit(X_train, y_train) 
    
    def train(self, X_train, y_train, X_val=None, y_val=None):

        if self.ensemble_method == 'voting':
            self.train_voting_ensemble(X_train, y_train, X_val, y_val)
        
        elif self.ensemble_method == 'stacking':
            self.train_stacking_ensemble(X_train, y_train, n_splits=5)
        
        elif self.ensemble_method == 'blending':
            raise NotImplementedError("Blending (Classification)은 현재 코드에서 구현되지 않았습니다.")
    
    
    def predict_proba(self, X):
        if self.ensemble_method == 'voting':
            predictions_proba = np.zeros(len(X))
            for name, model in self.models.items():
                predictions_proba += self.weights[name] * model.predict_proba(X)[:, 1]
            return np.clip(predictions_proba, 0, 1)
        
        elif self.ensemble_method in ['stacking', 'blending']:
            base_predictions_proba = np.column_stack([
                model.predict_proba(X)[:, 1] for model in self.models.values()
            ])
            predictions_proba = self.meta_model.predict_proba(base_predictions_proba)[:, 1]
            return np.clip(predictions_proba, 0, 1)

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        # 임계값(threshold)을 넘으면 1(CS), 아니면 0(정상)
        return (probabilities >= threshold).astype(int)

    def _clip_predictions(self, predictions):
        return np.clip(predictions, 0, 1)
    
    def _calculate_weights_classification(self, val_predictions_proba, y_val):
        auc_scores = {}
        for name, proba in val_predictions_proba.items():
            auc_scores[name] = roc_auc_score(y_val, proba)
        
        # 가중치 = AUC^2 (AUC가 0.5 근처(랜덤)면 가중치 낮춤, 1에 가까우면 높임)
        weights_raw = {name: (max(0.5, auc) - 0.5)**2 for name, auc in auc_scores.items()}
        total = sum(weights_raw.values())
        if total == 0: # 모든 모델이 0.5 이하일 경우
            self.weights = {name: 1.0 / len(self.models) for name in self.models.keys()}
        else:
            self.weights = {name: w / total for name, w in weights_raw.items()}

    def get_feature_importance(self, feature_names):
        
        importance_df = pd.DataFrame()
        importance_df['feature'] = feature_names
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_df[name] = model.feature_importances_
        
        importance_df['mean'] = importance_df.drop('feature', axis=1).mean(axis=1)
        return importance_df.sort_values('mean', ascending=False)

    def save_models(self, filepath_prefix: str):
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
        config = joblib.load(f"{filepath_prefix}_config.pkl")
        self.ensemble_method = config['ensemble_method']
        
        for name in self.models.keys():
            self.models[name] = joblib.load(f"{filepath_prefix}_{name}.pkl")
        
        if self.ensemble_method in ['stacking', 'blending']:
            self.meta_model = joblib.load(f"{filepath_prefix}_meta.pkl")
        
        if self.ensemble_method == 'voting':
            self.weights = joblib.load(f"{filepath_prefix}_weights.pkl")
        
        print(f"Ensemble models loaded from {filepath_prefix}_*.pkl")