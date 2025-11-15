"""
FastAPI 모델 서빙 서버

ML/RL 모델을 REST API로 제공하는 FastAPI 애플리케이션
- /predict_price: 가격 예측 API
- /predict_quality_rate: 품질/하자율 예측 API
- /predict_q_values: RL Q-value 예측 API
- 핫 스와핑(Hot-swapping) 지원

사용법:
    uvicorn serving.main:app --reload --host 127.0.0.1 --port 8000
"""
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from serving.model_loader import global_model_loader
from serving.inference import ModelInferenceEngine

class MLFeatures(BaseModel):
    """
    ML 모델(가격/품질) 예측을 위한 입력 피처
    model_register에 저장된 _features.pkl의 모든 키와 값을 포함해야 합니다.
    """
    features: Dict[str, Any] = Field(
        ..., 
        example={"year": 2025, "month": 11, "dayofweek": 5, "price_lag_7d": 20000, "seller_cs_rate": 0.01}
    )

class PricePredictionResponse(BaseModel):
    predicted_price: float

class QualityRateResponse(BaseModel):
    predicted_quality_rate: float # 하자비율 (0.0 ~ 1.0)

class RLState(BaseModel):
    """
    RL 모델(DQN) Q-value 예측을 위한 'state_t' 입력
    (dummy_data_100.json의 state_t 구조와 일치)
    """
    user_profile: dict
    session_history: dict
    candidate_item_info: dict

class QValueResponse(BaseModel):
    q_not_recommend: float
    q_recommend: float

# --- FastAPI 애플리케이션 ---

app = FastAPI(
    title="대봉 유통 AI 모델 서빙 API",
    description="가격/품질 예측 ML 모델 및 추천 RL 모델의 예측 API를 제공합니다.",
    version="1.0.0"
)

# --- Dependency ---

def get_engine() -> ModelInferenceEngine:
    """
    API 엔드포인트가 호출될 때마다, model_loader로부터
    현재 활성화된 'ModelInferenceEngine' 인스턴스를 가져옵니다.
    """
    engine = global_model_loader.get_inference_engine()
    if engine is None:
        raise HTTPException(
            status_code=503, # Service Unavailable
            detail="ModelInferenceEngine is not loaded or still initializing. Please try again later."
        )
    return engine

# --- API Endpoints ---

@app.get("/", summary="API 상태 확인")
def read_root():
    """API 서버가 정상적으로 실행 중인지 확인합니다."""
    return {"message": "Daebong Inference API is running."}


@app.post("/predict_price", 
          response_model=PricePredictionResponse, 
          summary="ML 가격 예측")
def predict_price(
    data: MLFeatures, 
    engine: ModelInferenceEngine = Depends(get_engine)
):
    """
    ML 모델을 사용하여 1일 후 가격을 예측합니다.
    'features' 필드에는 price_ensemble_..._features.pkl에 정의된 모든 피처가 포함되어야 합니다.
    """
    if engine.price_model is None or engine.feature_names_price is None:
        raise HTTPException(status_code=503, detail="Price Model is not loaded.")
        
    try:
        # 1. Pydantic 딕셔너리를 단일 행 DataFrame으로 변환
        input_df = pd.DataFrame([data.features])
        
        # 2. 예측 수행 (inference.py가 피처 순서 재정렬)
        price = engine.predict_price(input_df)
        
        return {"predicted_price": price[0]}
    
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing feature in input data: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during price prediction: {e}")


@app.post("/predict_quality_rate", 
          response_model=QualityRateResponse, 
          summary="ML 품질(하자비율) 예측")
def predict_quality_rate(
    data: MLFeatures, 
    engine: ModelInferenceEngine = Depends(get_engine)
):
    """
    ML 모델을 사용하여 1일 후 품질(CS 발생 확률/하자비율)을 0.0 ~ 1.0 사이 값으로 예측합니다.
    'features' 필드에는 quality_ensemble_..._features.pkl에 정의된 모든 피처가 포함되어야 합니다.
    """
    if engine.quality_model is None or engine.feature_names_quality is None:
        raise HTTPException(status_code=503, detail="Quality Model is not loaded.")
        
    try:
        # 1. Pydantic 딕셔너리를 단일 행 DataFrame으로 변환
        input_df = pd.DataFrame([data.features])
        
        # 2. 예측 수행 (inference.py가 피처 순서 재정렬)
        rate = engine.predict_quality_rate(input_df)
        
        return {"predicted_quality_rate": rate[0]}
        
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing feature in input data: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during quality prediction: {e}")


@app.post("/predict_q_values", 
          response_model=QValueResponse, 
          summary="RL 추천 만족도(Q-value) 예측")
def predict_rl_q_values(
    state: RLState, 
    engine: ModelInferenceEngine = Depends(get_engine)
):
    """
    RL 모델(DQN)을 사용하여 현재 상태(State)에서 
    '추천' 행동의 예상 만족도 점수(Q-value)를 반환합니다.
    """
    if engine.q_network is None:
        raise HTTPException(status_code=503, detail="RL Model is not loaded.")
        
    try:
        # Pydantic 모델을 Python 딕셔너리로 변환
        state_dict = state.dict()
        
        # 예측 수행
        q_values = engine.predict_q_values(state_dict) # [Q(미추천), Q(추천)]
        
        return {
            "q_not_recommend": q_values[0],
            "q_recommend": q_values[1] # <- 이 점수를 랭킹에 사용
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during Q-value prediction: {e}")

# --- (선택 사항) 서버 실행 (터미널에서 uvicorn을 사용하는 것을 권장) ---
if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server locally...")
    print("Access docs at http://127.0.0.1:8000/docs")
    # uvicorn main:app --reload --host 127.0.0.1 --port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)