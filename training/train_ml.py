# -*- coding: utf-8 -*-
"""
ML Model Training Script
- Price prediction model (ml_price_model.py)
- Quality/defect rate prediction model (ml_quality_model.py)

Usage:
    python train_ml.py --model price --ensemble voting
    python train_ml.py --model quality --ensemble stacking
    python train_ml.py --model all --ensemble voting
"""

import os
import sys
import joblib
import argparse
import pandas as pd
import numpy as np
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ★★★ (추가) 분류 평가 지표 임포트 ★★★
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    precision_score, recall_score, f1_score, roc_auc_score, 
    confusion_matrix, classification_report
)

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.models.ml_price_model import EnsemblePriceModel
from src.models.ml_quality_model import EnsembleQualityModel
from src.config import ML_MODEL_PATH


class DataPreprocessor:
    """Data preprocessing and feature engineering class"""

    def __init__(self, data_path):
        self.data_path = data_path
        self.label_encoders = {}

    def load_raw_data(self):
        """Load raw data"""
        print("\n" + "="*60)
        print("Loading Raw Data...")
        print("="*60)

        # 1. Apple price data
        price_df = pd.read_csv(
            os.path.join(self.data_path, '농넷_시장별_사과가격.csv'),
            encoding='utf-8-sig'
        )
        print(f"✓ Apple Price Data: {len(price_df)} rows")

        # 2. Sales data
        sales_df = pd.read_csv(
            os.path.join(self.data_path, '초창패개발_데이터_판매데이터.csv'),
            encoding='utf-8-sig'
        )
        print(f"✓ Sales Data: {len(sales_df)} rows")

        # 3. CS data
        cs_df = pd.read_csv(
            os.path.join(self.data_path, '초창패개발_데이터_CS데이터.csv'),
            encoding='utf-8-sig'
        )
        print(f"✓ CS Data: {len(cs_df)} rows")

        # 4. Weather data
        with open(os.path.join(self.data_path, 'weather_data.json'), 'r', encoding='utf-8') as f:
            weather_data = json.load(f)
        weather_df = pd.DataFrame(weather_data)
        print(f"✓ Weather Data: {len(weather_df)} rows")

        return price_df, sales_df, cs_df, weather_df

    def preprocess_price_data(self, price_df):
        """Preprocess price data"""
        df = price_df.copy()

        # Parse dates
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')

        # Clean numeric fields (remove commas)
        df['평균가격'] = df['평균가격'].astype(str).str.replace(',', '').astype(float)
        df['총거래물량'] = df['총거래물량'].astype(str).str.replace(',', '').astype(float)
        df['총거래금액'] = df['총거래금액'].astype(str).str.replace(',', '').astype(float)

        # Remove NaN
        df = df.dropna(subset=['DATE', '평균가격'])

        # Remove outliers
        df = df[(df['평균가격'] > 0) & (df['평균가격'] < 500000)]

        return df

    def preprocess_sales_data(self, sales_df):
        """Preprocess sales data"""
        df = sales_df.copy()

        # Parse dates
        df['발주날짜'] = pd.to_datetime(df['발주날짜'], errors='coerce')

        # Clean prices
        df['공급가격'] = df['공급가격'].astype(str).str.replace('₩', '').str.replace(',', '').astype(float)

        # Convert CS flag to binary
        df['CS여부_binary'] = (df['CS여부'] == 'Y').astype(int)

        # Remove NaN
        df = df.dropna(subset=['발주날짜', '공급가격'])

        return df

    def preprocess_cs_data(self, cs_df):
        """Preprocess CS data"""
        df = cs_df.copy()

        # Parse dates
        df['발주일자'] = pd.to_datetime(df['발주일자'], errors='coerce')

        # Clean prices
        for col in ['매출원가', '상품공급가']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Clean percentages
        for col in ['요청퍼센트', '셀러확정퍼센트', '업체확정퍼센트']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def preprocess_weather_data(self, weather_df):
        """Preprocess weather data"""
        df = weather_df.copy()

        # Parse dates
        df['tm'] = pd.to_datetime(df['tm'], errors='coerce')

        # Convert numeric fields
        numeric_cols = ['avgTs', 'avgRhm', 'minTa', 'maxTa', 'avgTa', 'sumRn', 'avgWs']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Fill rainfall NaN with 0
        if 'sumRn' in df.columns:
            df['sumRn'] = df['sumRn'].fillna(0)

        return df

    def create_price_features(self, price_df, weather_df):
        """Create features for price prediction"""
        print("\n" + "="*60)
        print("Creating Features for Price Model...")
        print("="*60)

        df = price_df.copy()

        # 1. Time features
        df['year'] = df['DATE'].dt.year
        df['month'] = df['DATE'].dt.month
        df['day'] = df['DATE'].dt.day
        df['dayofweek'] = df['DATE'].dt.dayofweek
        df['quarter'] = df['DATE'].dt.quarter
        df['weekofyear'] = df['DATE'].dt.isocalendar().week

        # 2. Season features
        df['is_spring'] = df['month'].isin([3, 4, 5]).astype(int)
        df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
        df['is_fall'] = df['month'].isin([9, 10, 11]).astype(int)
        df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)

        # 3. Categorical encoding
        categorical_cols = ['도매시장', '품종', '산지-광역시도', '산지-시군구', '등급']
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col].astype(str))

        # 4. Merge weather data
        weather_df = weather_df.rename(columns={'tm': 'DATE'})
        df = df.merge(weather_df, on='DATE', how='left')

        # Fill weather feature NaNs
        weather_features = ['avgTa', 'minTa', 'maxTa', 'avgRhm', 'sumRn', 'avgWs']
        for col in weather_features:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        # 5. Trade volume/amount features
        df['price_per_kg'] = df['평균가격'] / 20  # 20kg box
        df['total_trade_volume_log'] = np.log1p(df['총거래물량'])
        df['total_trade_amount_log'] = np.log1p(df['총거래금액'])

        # 6. Time-based statistical features (Lag features)
        df = df.sort_values(['품종', '등급', 'DATE'])

        for window in [7, 14, 30]:
            df[f'price_lag_{window}d'] = df.groupby(['품종', '등급'])['평균가격'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df[f'price_std_{window}d'] = df.groupby(['품종', '등급'])['평균가격'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )

        # 7. Price change rate
        df['price_change_7d'] = df.groupby(['품종', '등급'])['평균가격'].transform(
            lambda x: x.pct_change(periods=7)
        )

        # Fill NaNs
        df = df.fillna(0)

        print(f"✓ Total Features: {len(df.columns)}")
        print(f"✓ Total Samples: {len(df)}")

        return df

    def create_quality_features(self, sales_df, cs_df, weather_df):
        """Create features for quality (defect rate) prediction"""
        print("\n" + "="*60)
        print("Creating Features for Quality Model...")
        print("="*60)

        # 1. Merge CS info with sales data
        df = sales_df.copy()

        # Check CS occurrence by order code
        cs_orders = cs_df['발주번호'].unique()
        df['has_cs'] = df['주문코드'].isin(cs_orders).astype(int)

        # 2. Calculate CS rate by seller
        seller_cs_rate = df.groupby('셀러코드')['CS여부_binary'].mean().reset_index()
        seller_cs_rate.columns = ['셀러코드', 'seller_cs_rate']
        df = df.merge(seller_cs_rate, on='셀러코드', how='left')

        # 3. CS rate by company
        company_cs_rate = df.groupby('업체명')['CS여부_binary'].mean().reset_index()
        company_cs_rate.columns = ['업체명', 'company_cs_rate']
        df = df.merge(company_cs_rate, on='업체명', how='left')

        # 4. Time features
        df['year'] = df['발주날짜'].dt.year
        df['month'] = df['발주날짜'].dt.month
        df['day'] = df['발주날짜'].dt.day
        df['dayofweek'] = df['발주날짜'].dt.dayofweek
        df['quarter'] = df['발주날짜'].dt.quarter

        # 5. Season features
        df['is_spring'] = df['month'].isin([3, 4, 5]).astype(int)
        df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
        df['is_fall'] = df['month'].isin([9, 10, 11]).astype(int)
        df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)

        # 6. Merge weather data
        weather_df = weather_df.rename(columns={'tm': '발주날짜'})
        df = df.merge(weather_df, on='발주날짜', how='left')

        # Fill weather feature NaNs
        weather_features = ['avgTa', 'minTa', 'maxTa', 'avgRhm', 'sumRn', 'avgWs']
        for col in weather_features:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        # 7. Product-related features
        # Extract weight from product name
        df['product_weight'] = df['판매상품명'].str.extract(r'(\d+)kg').astype(float)
        df['product_weight'] = df['product_weight'].fillna(df['product_weight'].median())

        # 8. Price features
        df['price_log'] = np.log1p(df['공급가격'])
        df['price_per_kg'] = df['공급가격'] / df['product_weight']

        # 9. Categorical encoding
        categorical_cols = ['셀러코드', '업체명']
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                # Handle new categories as -1
                le = self.label_encoders[col]
                df[f'{col}_encoded'] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

        # 10. Time-based CS rate (Rolling window)
        df = df.sort_values('발주날짜')

        for window in [7, 14, 30]:
            df[f'cs_rate_{window}d'] = df['CS여부_binary'].rolling(
                window=window, min_periods=1
            ).mean()

        # 11. Target: CS flag (defect rate)
        df['defect_rate'] = df['CS여부_binary']

        # Fill NaNs
        df = df.fillna(0)

        print(f"✓ Total Features: {len(df.columns)}")
        print(f"✓ Total Samples: {len(df)}")
        print(f"✓ CS Rate: {df['defect_rate'].mean():.4f}")

        return df


class PriceModelTrainer:
    """Price prediction model trainer"""

    def __init__(self, ensemble_method='voting'):
        self.ensemble_method = ensemble_method
        self.model = None

    def prepare_data(self, df):
        """Prepare training data"""
        print("\n" + "="*60)
        print("Preparing Price Prediction Data...")
        print("="*60)

        # Target
        target_col = '평균가격'

        # Select features (numeric only)
        # (데이터 누수 방지: '총거래금액', '총거래물량', 'price_per_kg' 제외)
        exclude_cols = [
            'DATE', '평균가격', '거래단위', '도매시장', '도매법인',
            '품목', '품종', '산지-광역시도', '산지-시군구', '등급',
            'stnId', 'stnNm', 'avgTs', '총거래금액', '총거래물량', 'price_per_kg'
        ]

        feature_cols = [col for col in df.columns if col not in exclude_cols]

        X = df[feature_cols]
        y = df[target_col]

        # Time-series split
        split_idx = int(len(df) * 0.7)
        val_idx = int(len(df) * 0.85)

        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]

        X_val = X.iloc[split_idx:val_idx]
        y_val = y.iloc[split_idx:val_idx]

        X_test = X.iloc[val_idx:]
        y_test = y.iloc[val_idx:]

        print(f"✓ Training Set:   {len(X_train)} samples")
        print(f"✓ Validation Set: {len(X_val)} samples")
        print(f"✓ Test Set:       {len(X_test)} samples")
        print(f"✓ Features:       {len(feature_cols)}")

        return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols

    def train(self, X_train, y_train, X_val, y_val):
        """Train model"""
        print("\n" + "="*60)
        print(f"Training Price Model (Ensemble: {self.ensemble_method})")
        print("="*60)

        # Initialize model
        self.model = EnsemblePriceModel(ensemble_method=self.ensemble_method)

        # Train
        self.model.train(X_train, y_train, X_val, y_val)

        return self.model

    def evaluate(self, model, X_test, y_test):
        """Evaluate model"""
        print("\n" + "="*60)
        print("Evaluating Price Model...")
        print("="*60)

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        print(f"✓ MAE:   {mae:.2f} KRW")
        print(f"✓ RMSE:  {rmse:.2f} KRW")
        print(f"✓ R²:    {r2:.4f}")
        print(f"✓ MAPE:  {mape:.2f}%")

        return {'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape}

    def save(self, model, save_path):
        """Save model"""
        os.makedirs(save_path, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(save_path, f"price_ensemble_{timestamp}")

        model.save_models(filepath)

        try:
            # LightGBM 모델에서 피처 이름을 직접 가져옴
            feature_names = model.models['lgb'].feature_name_
            joblib.dump(feature_names, f"{filepath}_features.pkl")
            print(f"\n✓ Price model features saved.")
        except Exception as e:
            print(f"\n!!! Warning: Could not save price model features. {e}")

        print(f"\n✓ Price model saved to: {filepath}")

        return filepath


class QualityModelTrainer:
    """Quality (defect rate) prediction model trainer"""

    def __init__(self, ensemble_method='voting'):
        self.ensemble_method = ensemble_method
        self.model = None

    def prepare_data(self, df):
        """Prepare training data"""
        print("\n" + "="*60)
        print("Preparing Quality Prediction Data...")
        print("="*60)

        # Target
        target_col = 'defect_rate'

        # Select features
        exclude_cols = [
            '발주날짜', '판매상품명', '주문코드', 'CS여부',
            'CS여부_binary', 'defect_rate', 'has_cs',
            'stnId', 'stnNm', 'avgTs', '셀러코드', '업체명'
        ]

        feature_cols = [col for col in df.columns if col not in exclude_cols]

        X = df[feature_cols]
        y = df[target_col]

        # Time-series split
        split_idx = int(len(df) * 0.7)
        val_idx = int(len(df) * 0.85)

        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]

        X_val = X.iloc[split_idx:val_idx]
        y_val = y.iloc[split_idx:val_idx]

        X_test = X.iloc[val_idx:]
        y_test = y.iloc[val_idx:]

        print(f"✓ Training Set:   {len(X_train)} samples")
        print(f"✓ Validation Set: {len(X_val)} samples")
        print(f"✓ Test Set:       {len(X_test)} samples")
        print(f"✓ Features:       {len(feature_cols)}")

        return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols

    def train(self, X_train, y_train, X_val, y_val):
        """Train model"""
        print("\n" + "="*60)
        print(f"Training Quality Model (Ensemble: {self.ensemble_method})")
        print("="*60)

        # Initialize model
        # ★★★ 중요 ★★★
        # src/models/ml_quality_model.py의 EnsembleQualityModel이
        # 회귀(Regression)가 아닌 '분류(Classification)' 모델(예: LGBMClassifier)을
        # 사용하도록 수정되어야 합니다.
        # 또한, 불균형 데이터 처리를 위해 is_unbalanced=True 또는
        # scale_pos_weight 파라미터를 사용하는 것이 좋습니다.
        #
        # 예: self.model = EnsembleQualityModel(
        #           ensemble_method=self.ensemble_method,
        #           is_unbalanced=True 
        #      )
        # ★★★★★★★★★★★★★
        self.model = EnsembleQualityModel(ensemble_method=self.ensemble_method)

        # Train
        self.model.train(X_train, y_train, X_val, y_val)

        return self.model

    # ★★★ (수정된 부분) 분류 지표로 평가 ★★★
    def evaluate(self, model, X_test, y_test):
        """Evaluate classification model"""
        print("\n" + "="*60)
        print("Evaluating Quality Model (as Classification)...")
        print("="*60)
        
        # ---
        # 중요: 이 코드가 작동하려면 src/models/ml_quality_model.py의
        # EnsembleQualityModel이 .predict() (0/1 반환) 및 
        # .predict_proba() (확률 반환) 메서드를 구현하고 있어야 합니다.
        # ---
        
        metrics = {}
        try:
            # 1. Binary Predictions (0 or 1)
            # (임계값 0.5 기준)
            y_pred = model.predict(X_test)
            
            # 2. Probability Predictions (for AUC-ROC)
            # (positive class, 1)의 확률을 가져옵니다.
            y_pred_proba = model.predict_proba(X_test)

            # 3. Calculate Metrics
            # zero_division=0: CS=1 (positive) 클래스를 하나도 못 맞출 경우 경고 대신 0.0 반환
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            print("\n[Test Set Classification Evaluation]")
            print(f"✓ Precision (CS=1): {precision:.4f}")
            print(f"✓ Recall (CS=1):    {recall:.4f}")
            print(f"✓ F1-Score (CS=1):  {f1:.4f}")
            print(f"✓ AUC-ROC:          {auc:.4f}")

            print("\n[Classification Report]")
            print(classification_report(y_test, y_pred, zero_division=0, target_names=['Normal (0)', 'CS (1)']))

            print("\n[Confusion Matrix]")
            print("         (Pred 0) (Pred 1)")
            cm = confusion_matrix(y_test, y_pred)
            print(f"Actual 0: {cm[0][0]:<8} {cm[0][1]:<8}")
            print(f"Actual 1: {cm[1][0]:<8} {cm[1][1]:<8}")

            metrics = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_roc': auc,
                'confusion_matrix': cm.tolist()
            }
            
        except AttributeError as e:
            print(f"!!! 모델 평가 오류: {e}")
            print("!!! 'EnsembleQualityModel'이 .predict() 또는 .predict_proba()를 지원하지 않는 것 같습니다.")
            print("!!! src/models/ml_quality_model.py를 회귀 모델 대신 '분류' 모델로 수정해야 합니다.")
        except Exception as e:
            print(f"!!! 알 수 없는 평가 오류: {e}")

        return metrics

    def save(self, model, save_path):
        """Save model"""
        os.makedirs(save_path, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(save_path, f"quality_ensemble_{timestamp}")

        model.save_models(filepath)

        try:
            feature_names = model.models['lgb'].feature_name_
            joblib.dump(feature_names, f"{filepath}_features.pkl")
            print(f"\n✓ Quality model features saved.")
        except Exception as e:
            print(f"\n!!! Warning: Could not save quality model features. {e}")

        print(f"\n✓ Quality model saved to: {filepath}")

        return filepath


def main():
    parser = argparse.ArgumentParser(description='ML Model Training Script')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['price', 'quality', 'all'],
        help='Model to train: price, quality, or all'
    )
    parser.add_argument(
        '--ensemble',
        type=str,
        default='voting',
        choices=['voting', 'stacking', 'blending'],
        help='Ensemble method: voting, stacking, or blending'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default=os.path.join(PROJECT_ROOT, 'data', 'raw'),
        help='Path to raw data directory'
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default=ML_MODEL_PATH,
        help='Path to save trained models'
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("ML Model Training Script")
    print("="*60)
    print(f"Model Type:      {args.model}")
    print(f"Ensemble Method: {args.ensemble}")
    print(f"Data Path:       {args.data_path}")
    print(f"Save Path:       {args.save_path}")

    # Initialize data preprocessor
    preprocessor = DataPreprocessor(args.data_path)

    # Load raw data
    price_df, sales_df, cs_df, weather_df = preprocessor.load_raw_data()

    # Preprocess data
    price_df = preprocessor.preprocess_price_data(price_df)
    sales_df = preprocessor.preprocess_sales_data(sales_df)
    cs_df = preprocessor.preprocess_cs_data(cs_df)
    weather_df = preprocessor.preprocess_weather_data(weather_df)

    # Train price model
    if args.model in ['price', 'all']:
        print("\n" + "="*60)
        print("TRAINING PRICE MODEL")
        print("="*60)

        # Create features
        price_features = preprocessor.create_price_features(price_df, weather_df)

        # Initialize trainer
        price_trainer = PriceModelTrainer(ensemble_method=args.ensemble)

        # Prepare data
        X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = \
            price_trainer.prepare_data(price_features)

        # Train model
        price_model = price_trainer.train(X_train, y_train, X_val, y_val)

        # Evaluate model
        price_metrics = price_trainer.evaluate(price_model, X_test, y_test)

        # Save model
        price_trainer.save(price_model, args.save_path)

    # Train quality model
    if args.model in ['quality', 'all']:
        print("\n" + "="*60)
        print("TRAINING QUALITY MODEL")
        print("="*60)

        # Create features
        quality_features = preprocessor.create_quality_features(
            sales_df, cs_df, weather_df
        )

        # Initialize trainer
        quality_trainer = QualityModelTrainer(ensemble_method=args.ensemble)

        # Prepare data
        X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = \
            quality_trainer.prepare_data(quality_features)

        # Train model
        quality_model = quality_trainer.train(X_train, y_train, X_val, y_val)

        # Evaluate model
        quality_metrics = quality_trainer.evaluate(quality_model, X_test, y_test)

        # Save model
        quality_trainer.save(quality_model, args.save_path)

    print("\n" + "="*60)
    print("Training Completed Successfully!")
    print("="*60)


if __name__ == '__main__':
    main()