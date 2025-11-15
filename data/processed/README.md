# 예측 데이터셋 생성 가이드

이 디렉토리는 ML 모델을 사용하여 가격 및 품질 예측 데이터셋을 생성하는 스크립트를 포함하고 있습니다.

## 📁 파일 구조

```
data/
├── raw/                    # 원본 데이터 (CSV, JSON)
│   ├── 농넷_시장별_사과가격.csv
│   ├── 초창패개발_데이터_판매데이터.csv
│   ├── 초창패개발_데이터_CS데이터.csv
│   └── weather_data.json
└── processed/              # 예측 결과 데이터
    ├── create_predictions.py  # 예측 데이터셋 생성 스크립트
    ├── price_predictions_*.csv    # 가격 예측 결과
    └── quality_predictions_*.csv  # 품질 예측 결과
```

## 🚀 사용 방법

### 1. 필수 패키지 설치

```bash
pip install pandas numpy scikit-learn joblib lightgbm xgboost torch
```

### 2. 스크립트 실행

#### 모든 예측 수행 (가격 + 품질)
```bash
python data/processed/create_predictions.py --model_type all
```

#### 가격 예측만 수행
```bash
python data/processed/create_predictions.py --model_type price
```

#### 품질 예측만 수행
```bash
python data/processed/create_predictions.py --model_type quality
```

#### 사용자 정의 경로 지정
```bash
python data/processed/create_predictions.py \
    --model_type all \
    --raw_data_path /path/to/raw \
    --output_dir /path/to/output
```

## 📊 출력 데이터 형식

### 가격 예측 결과 (price_predictions_*.csv)

| 컬럼명 | 설명 |
|--------|------|
| DATE | 거래 날짜 |
| 도매시장 | 도매시장명 |
| 품종 | 사과 품종 |
| 산지-광역시도 | 산지 광역시도 |
| 산지-시군구 | 산지 시군구 |
| 등급 | 품질 등급 |
| 평균가격 | 원본 데이터의 평균가격 |
| 실제가격 | 실제 거래 가격 |
| **예측가격** | **ML 모델이 예측한 가격** |
| 가격차이 | 예측가격 - 실제가격 |
| 가격차이율(%) | (가격차이 / 실제가격) × 100 |

**예시:**
```csv
DATE,도매시장,품종,산지-광역시도,산지-시군구,등급,평균가격,실제가격,예측가격,가격차이,가격차이율(%)
2025-06-02,안동도매시장,감홍,경북,문경시,등외,20000.0,20000.0,20395.41,395.41,1.98
```

### 품질 예측 결과 (quality_predictions_*.csv)

| 컬럼명 | 설명 |
|--------|------|
| 발주날짜 | 주문 날짜 |
| 셀러코드 | 판매자 코드 |
| 업체명 | 업체명 |
| 판매상품명 | 상품명 |
| 공급가격 | 공급 가격 |
| CS여부 | 실제 CS 발생 여부 (Y/N) |
| 실제_CS여부 | CS 여부를 숫자로 변환 (1/0) |
| **예측_하자확률** | **ML 모델이 예측한 하자 발생 확률 (0~1)** |
| **예측_CS여부** | **예측 하자 여부 (0.5 임계값 기준)** |
| 예측정확 | 예측이 실제와 일치하는지 여부 |

**예시:**
```csv
발주날짜,셀러코드,업체명,판매상품명,공급가격,CS여부,실제_CS여부,예측_하자확률,예측_CS여부,예측정확
2025-05-26,2124,도매GB,(행사용)(도매GB)부사쥬스용/6kg,15000.0,N,0,0.0001,0,1
```

## 🎯 주요 특징

### 가격 예측 모델
- **모델**: LightGBM, XGBoost, Random Forest 앙상블
- **예측 대상**: 사과 도매 가격
- **특징**:
  - 시간 특성 (연, 월, 일, 요일, 분기)
  - 계절 특성 (봄, 여름, 가을, 겨울)
  - 날씨 정보 (온도, 습도, 강수량, 풍속)
  - 시계열 특성 (7일/14일/30일 이동평균)

### 품질 예측 모델
- **모델**: LightGBM, XGBoost, Random Forest, Gradient Boosting 앙상블
- **예측 대상**: CS(Customer Service) 발생 확률
- **특징**:
  - 셀러/업체별 과거 CS 발생률
  - 시간 특성 및 계절 특성
  - 날씨 정보
  - 상품 가격 및 중량 정보
  - 불균형 데이터 처리 적용

## 📈 성능 지표

최근 실행 결과:

### 가격 예측 성능
- 총 샘플 수: 32,208개
- 평균 절대 오차: 2,726원
- 평균 절대 오차율: **3.61%**

### 품질 예측 성능
- 총 샘플 수: 51,936개
- 실제 CS 발생률: 0.21%
- 예측 정확도: **99.74%**
- CS 샘플 예측 정확도: **92.79%**
- CS 샘플 평균 예측 확률: 0.7574

## 🔧 고급 사용법

### Python에서 직접 사용

```python
import sys
sys.path.append('/home/user/daebong')

from data.processed.create_predictions import PredictionDatasetCreator

# 예측 데이터셋 생성기 초기화
creator = PredictionDatasetCreator(
    raw_data_path='data/raw',
    output_path='data/processed'
)

# 가격 예측만 수행
price_df, price_file = creator.create_price_predictions()

# 품질 예측만 수행
quality_df, quality_file = creator.create_quality_predictions()

# 모든 예측 수행
results = creator.create_combined_predictions()
```

### 생성된 데이터 분석

```python
import pandas as pd

# 가격 예측 결과 로드
price_df = pd.read_csv('data/processed/price_predictions_20251115_122915.csv')

# 통계 확인
print(f"평균 예측 오차: {price_df['가격차이'].abs().mean():.2f}원")
print(f"중앙값 예측 오차: {price_df['가격차이'].abs().median():.2f}원")

# 품질 예측 결과 로드
quality_df = pd.read_csv('data/processed/quality_predictions_20251115_122916.csv')

# CS 발생 예측 분석
high_risk = quality_df[quality_df['예측_하자확률'] > 0.3]
print(f"고위험 주문 수: {len(high_risk)}")
```

## 📝 참고사항

1. **모델 경로**: 학습된 모델은 `model_register/ml_model/` 디렉토리에 저장되어 있어야 합니다.
2. **데이터 전처리**: raw 데이터는 자동으로 전처리되며, training/train_ml.py의 DataPreprocessor를 사용합니다.
3. **타임스탬프**: 생성된 파일명에는 실행 시각이 포함됩니다 (예: `price_predictions_20251115_122915.csv`).
4. **추론 엔진**: serving/inference.py의 ModelInferenceEngine을 사용하여 예측을 수행합니다.

## 🐛 문제 해결

### 모델을 찾을 수 없는 경우
```bash
# ML 모델 학습 먼저 수행
python training/train_ml.py --model all --ensemble voting
```

### 패키지 오류가 발생하는 경우
```bash
# 필수 패키지 재설치
pip install -r requirements.txt  # requirements.txt가 있는 경우
# 또는
pip install pandas numpy scikit-learn joblib lightgbm xgboost torch
```

### 메모리 부족 오류
- 데이터가 너무 큰 경우, 배치 단위로 처리하도록 스크립트를 수정할 수 있습니다.
- 또는 한 번에 하나의 모델만 실행하세요 (`--model_type price` 또는 `--model_type quality`).

## 📧 문의

문제가 발생하거나 개선 사항이 있으면 프로젝트 담당자에게 문의하세요.
