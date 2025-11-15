"""
예측 가격 및 품질 데이터셋 생성 스크립트

이 스크립트는 raw 폴더의 CSV 데이터를 사용하여
학습된 ML 모델로 예측을 수행하고 결과를 CSV로 저장합니다.

Usage:
    python data/processed/create_predictions.py
    python data/processed/create_predictions.py --output_dir custom_dir
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from training.train_ml import DataPreprocessor
from serving.inference import ModelInferenceEngine


class PredictionDatasetCreator:
    """
    용도:
        Raw 데이터와 학습된 ML 모델(InferenceEngine)을 결합하여,
        과거 데이터에 대한 '예측값'과 '실제값'을 포함하는 CSV 데이터셋을 생성합니다.
        (예: 가격 예측 결과, 품질(하자율) 예측 결과)
        이 스크립트는 'train_ml.py'가 성공적으로 실행되어 
        'model_register/ml_model/'에 모델 파일이 저장된 이후에 사용됩니다.
    """
    def __init__(self, raw_data_path, output_path):

        self.raw_data_path = raw_data_path
        self.output_path = output_path

        # 데이터 전처리기 초기화
        self.preprocessor = DataPreprocessor(raw_data_path)

        # 추론 엔진 초기화
        print("\n" + "="*60)
        print("추론 엔진 초기화 중...")
        print("="*60)
        self.inference_engine = ModelInferenceEngine()

    def create_price_predictions(self):
        """
        용도: 
            '농넷_시장별_사과가격.csv' 원본 데이터를 기반으로 가격 예측을 수행하고,
            실제 가격과 예측 가격을 비교하는 CSV 파일을 생성합니다.
        Args:
            None
        Returns:
            (pd.DataFrame, str): 
                결과 DataFrame과 저장된 파일 경로. 
                오류 발생 시 (None, None).
        로직:
            1. 'load_raw_data'로 원본 데이터를 로드합니다.
            2. 'preprocess_price_data', 'preprocess_weather_data'로 데이터를 전처리합니다.
            3. 'create_price_features'로 ML 모델에 입력할 피처를 생성합니다.
            4. 'inference_engine.predict_price()'를 호출하여 예측 가격을 얻습니다.
            5. 원본 데이터(날짜, 품종 등)와 실제 가격, 예측 가격을 결합하여 DataFrame을 만듭니다.
            6. 가격 차이, 오차율 등 통계 컬럼을 추가합니다.
            7. 'data/processed/' 경로에 'price_predictions_...csv' 파일로 저장하고 통계를 출력합니다.
        """
        """가격 예측 데이터셋 생성"""
        print("\n" + "="*60)
        print("가격 예측 데이터셋 생성 중...")
        print("="*60)

        # 1. Raw 데이터 로드
        price_df, sales_df, cs_df, weather_df = self.preprocessor.load_raw_data()

        # 2. 데이터 전처리
        price_df = self.preprocessor.preprocess_price_data(price_df)
        weather_df = self.preprocessor.preprocess_weather_data(weather_df)

        # 3. Feature 생성
        price_features = self.preprocessor.create_price_features(price_df, weather_df)

        # 4. 원본 데이터 보존 (예측에 사용되지 않는 컬럼들)
        original_columns = ['DATE', '도매시장', '품종', '산지-광역시도', '산지-시군구', '등급', '평균가격']
        original_data = price_features[original_columns].copy()

        # 5. 예측 수행
        print("\n가격 예측 수행 중...")
        try:
            predicted_prices = self.inference_engine.predict_price(price_features)

            # 6. 결과 데이터프레임 생성
            result_df = original_data.copy()
            result_df['실제가격'] = price_features['평균가격']
            result_df['예측가격'] = predicted_prices
            result_df['가격차이'] = result_df['예측가격'] - result_df['실제가격']
            result_df['가격차이율(%)'] = (result_df['가격차이'] / result_df['실제가격']) * 100

            # 7. 저장
            os.makedirs(self.output_path, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_path, f"price_predictions_{timestamp}.csv")

            result_df.to_csv(output_file, index=False, encoding='utf-8-sig')

            print(f"\n✓ 가격 예측 데이터셋 생성 완료!")
            print(f"✓ 총 {len(result_df)} 개 샘플")
            print(f"✓ 저장 위치: {output_file}")

            # 통계 정보 출력
            print("\n[가격 예측 통계]")
            print(f"  평균 실제가격: {result_df['실제가격'].mean():,.0f} 원")
            print(f"  평균 예측가격: {result_df['예측가격'].mean():,.0f} 원")
            print(f"  평균 절대 오차: {abs(result_df['가격차이']).mean():,.0f} 원")
            print(f"  평균 절대 오차율: {abs(result_df['가격차이율(%)']).mean():.2f}%")

            return result_df, output_file

        except Exception as e:
            print(f"\n!!! 가격 예측 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def create_quality_predictions(self):
        """
        용도: 
            '판매데이터.csv', 'CS데이터.csv' 원본 데이터를 기반으로 품질(하자율) 예측을 수행하고,
            실제 CS 여부와 예측 하자 확률을 비교하는 CSV 파일을 생성합니다.
        Args:
            None
        Returns:
            (pd.DataFrame, str): 
                결과 DataFrame과 저장된 파일 경로. 
                오류 발생 시 (None, None).
        로직:
            1. 'load_raw_data'로 원본 데이터를 로드합니다.
            2. 'preprocess_sales_data', 'preprocess_cs_data' 등으로 데이터를 전처리합니다.
            3. 'create_quality_features'로 ML 모델에 입력할 피처를 생성합니다. (타겟: defect_rate=0/1)
            4. 'inference_engine.predict_quality_rate()'를 호출하여 예측 하자 확률(0.0~1.0)을 얻습니다.
            5. 원본 데이터(상품명, CS여부 등)와 실제 CS(0/1), 예측 확률을 결합하여 DataFrame을 만듭니다.
            6. 예측 정확도 등 통계 컬럼을 추가합니다.
            7. 'data/processed/' 경로에 'quality_predictions_...csv' 파일로 저장하고 통계를 출력합니다.
        """
        """품질 예측 데이터셋 생성"""
        print("\n" + "="*60)
        print("품질 예측 데이터셋 생성 중...")
        print("="*60)

        # 1. Raw 데이터 로드
        price_df, sales_df, cs_df, weather_df = self.preprocessor.load_raw_data()

        # 2. 데이터 전처리
        sales_df = self.preprocessor.preprocess_sales_data(sales_df)
        cs_df = self.preprocessor.preprocess_cs_data(cs_df)
        weather_df = self.preprocessor.preprocess_weather_data(weather_df)

        # 3. Feature 생성
        quality_features = self.preprocessor.create_quality_features(
            sales_df, cs_df, weather_df
        )

        # 4. 원본 데이터 보존
        original_columns = ['발주날짜', '셀러코드', '업체명', '판매상품명', '공급가격', 'CS여부']
        original_data = quality_features[original_columns].copy()

        # 5. 예측 수행
        print("\n품질(하자율) 예측 수행 중...")
        try:
            predicted_quality_proba = self.inference_engine.predict_quality_rate(quality_features)

            # 6. 결과 데이터프레임 생성
            result_df = original_data.copy()
            result_df['실제_CS여부'] = quality_features['defect_rate']
            result_df['예측_하자확률'] = predicted_quality_proba
            result_df['예측_CS여부'] = (predicted_quality_proba >= 0.5).astype(int)

            # 정확도 계산
            result_df['예측정확'] = (result_df['실제_CS여부'] == result_df['예측_CS여부']).astype(int)

            # 7. 저장
            os.makedirs(self.output_path, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_path, f"quality_predictions_{timestamp}.csv")

            result_df.to_csv(output_file, index=False, encoding='utf-8-sig')

            print(f"\n✓ 품질 예측 데이터셋 생성 완료!")
            print(f"✓ 총 {len(result_df)} 개 샘플")
            print(f"✓ 저장 위치: {output_file}")

            # 통계 정보 출력
            print("\n[품질 예측 통계]")
            print(f"  실제 CS 발생 건수: {result_df['실제_CS여부'].sum()}")
            print(f"  실제 CS 발생률: {result_df['실제_CS여부'].mean():.4f} ({result_df['실제_CS여부'].mean()*100:.2f}%)")
            print(f"  예측 CS 발생 건수: {result_df['예측_CS여부'].sum()}")
            print(f"  예측 정확도: {result_df['예측정확'].mean():.4f} ({result_df['예측정확'].mean()*100:.2f}%)")
            print(f"  평균 예측 하자확률: {result_df['예측_하자확률'].mean():.4f}")

            # CS 발생 샘플에 대한 통계
            cs_samples = result_df[result_df['실제_CS여부'] == 1]
            if len(cs_samples) > 0:
                print(f"\n[CS 발생 샘플 분석]")
                print(f"  CS 샘플 수: {len(cs_samples)}")
                print(f"  CS 샘플 평균 예측 확률: {cs_samples['예측_하자확률'].mean():.4f}")
                print(f"  CS 샘플 예측 정확도: {cs_samples['예측정확'].mean():.4f}")

            return result_df, output_file

        except Exception as e:
            print(f"\n!!! 품질 예측 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def create_combined_predictions(self):
        """
        용도: 
            가격 예측과 품질 예측을 모두 수행하고 결과를 요약합니다.
        Args:
            None
        Returns:
            dict: 
                'price'와 'quality' 키를 포함하며, 각 키는 
                {'df': DataFrame, 'file': str} 값을 가집니다.
        로직:
            1. `create_price_predictions()`를 호출합니다.
            2. `create_quality_predictions()`를 호출합니다.
            3. 두 함수의 반환값을 딕셔너리로 묶어 반환합니다.
        """
        """가격 + 품질 통합 예측 데이터셋 생성"""
        print("\n" + "="*60)
        print("통합 예측 데이터셋 생성 중...")
        print("="*60)

        # 두 예측 수행
        price_df, price_file = self.create_price_predictions()
        quality_df, quality_file = self.create_quality_predictions()

        # 결과 요약
        print("\n" + "="*60)
        print("데이터셋 생성 완료!")
        print("="*60)

        if price_file:
            print(f"\n📊 가격 예측 데이터: {price_file}")
        if quality_file:
            print(f"📊 품질 예측 데이터: {quality_file}")

        return {
            'price': {'df': price_df, 'file': price_file},
            'quality': {'df': quality_df, 'file': quality_file}
        }


def main():
    """
    용도: 
        스크립트의 메인 진입점(Entrypoint)입니다.
        커맨드 라인 인자(--model, --output_dir 등)를 파싱하여 
        `PredictionDatasetCreator`를 실행합니다.
    Args:
        None (sys.argv에서 인자를 받음)
    Returns:
        None
    로직:
        1. `argparse`로 커맨드 라인 인자를 파싱합니다.
        2. `PredictionDatasetCreator` 클래스를 인스턴스화합니다.
        3. `--model_type` 인자에 따라 `create_price_predictions`, 
            `create_quality_predictions`, `create_combined_predictions` 중 
            적절한 함수를 호출합니다.
    """
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description='ML 모델을 사용하여 가격 및 품질 예측 데이터셋 생성'
    )
    parser.add_argument(
        '--raw_data_path',
        type=str,
        default=os.path.join(PROJECT_ROOT, 'data', 'raw'),
        help='Raw 데이터 경로'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=os.path.join(PROJECT_ROOT, 'data', 'processed'),
        help='출력 디렉토리 경로'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='all',
        choices=['price', 'quality', 'all'],
        help='생성할 예측 데이터셋 타입 (price, quality, all)'
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("예측 데이터셋 생성 스크립트")
    print("="*60)
    print(f"Raw 데이터 경로: {args.raw_data_path}")
    print(f"출력 디렉토리: {args.output_dir}")
    print(f"모델 타입: {args.model_type}")

    # 예측 데이터셋 생성기 초기화
    creator = PredictionDatasetCreator(
        raw_data_path=args.raw_data_path,
        output_path=args.output_dir
    )

    # 선택된 모델로 예측 수행
    if args.model_type == 'all':
        results = creator.create_combined_predictions()
    elif args.model_type == 'price':
        price_df, price_file = creator.create_price_predictions()
        results = {'price': {'df': price_df, 'file': price_file}}
    elif args.model_type == 'quality':
        quality_df, quality_file = creator.create_quality_predictions()
        results = {'quality': {'df': quality_df, 'file': quality_file}}

    print("\n" + "="*60)
    print("✓ 모든 작업 완료!")
    print("="*60)


if __name__ == '__main__':
    main()
