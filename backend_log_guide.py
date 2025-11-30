"""
[강화학습(RL) 데이터 수집 가이드]

백엔드(서버)에서 사용자가 상품 추천을 '볼 때'와 '클릭할 때'
다음과 같은 구조로 로그를 DB나 파일(.jsonl)에 저장하시면 실제 사용자 데이터를 수집할 수 있을 것 같습니다.
아래 예시 코드를 작성해두겠습니다.

로그가 수집되면, 이 데이터를 파싱하여
RL 모델의 State와 Reward를 구성하는 데 사용합니다.
"""

from datetime import datetime
import json

# --- 예시 함수 (실제 백엔드 로직에 맞게 수정 필요) ---

def log_recommendation_impression(session_id: str, user_id: str, 
                                recommendation_list: list, context: dict):
    """
    [로그 1: 추천 노출 (Action=1)]
    사용자에게 추천 목록이 '보여지는' 순간 호출합니다.
    RL의 Action과 State의 일부가 됩니다.
    """
    log_data = {
        "event_type": "rl_impression",
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "user_id": user_id,
        "context": context, # 예: {'page_type': 'search_results', 'keyword': '사과'}
        
        # 이 시점에 RL 에이전트가 추천한 상품 목록 (State의 일부)
        "recommendations": [
            {
                "product_id": item.get("id"),
                "rank": idx + 1,
                # 이 추천을 결정할 때 사용된 ML 모델의 예측값 (RL State 구성용)
                "predicted_price": item.get("pred_price"),
                "predicted_quality_rate": item.get("pred_quality"),
                "lead_time": item.get("lead_time")
            }
            for idx, item in enumerate(recommendation_list)
        ]
    }
    
    # (실제로는 Kafka, LogStash 또는 DB에 저장)
    print(f"[BACKEND LOG] Impression: {json.dumps(log_data, ensure_ascii=False)}")


def log_user_reaction(session_id: str, product_id: str, 
                        action_type: str, view_time_sec: int = 0):
    """
    [로그 2: 사용자 반응 (Reward)]
    사용자가 노출된 상품을 '클릭'하거나 '구매'하는 순간 호출합니다.
    RL의 Reward가 됩니다.
    """
    log_data = {
        "event_type": "rl_reaction",
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "product_id": product_id, # 어떤 상품에 반응했는지
        
        # 'click', 'purchase', 'add_to_cart'
        "action": action_type, 
        
        # Reward 계산을 위한 핵심 지표
        "page_view_time_sec": view_time_sec, # (클릭 시) 상세페이지 체류 시간
    }
    
    # (실제로는 Kafka, LogStash 또는 DB에 저장)
    print(f"[BACKEND LOG] Reaction: {json.dumps(log_data, ensure_ascii=False)}")

# --- 예시 시나리오 ---
if __name__ == "__main__":
    
    print("--- 대봉 백엔드 로그 수집 예시 ---")
    
    # 1. 사용자가 접속, AI가 상품 3개를 추천함
    SESSION_ID = "sess_abc_999"
    USER_ID = "user_123"
    RECOMMENDATIONS = [
        {"id": "P-1001", "pred_price": 25000, "pred_quality": 0.02, "lead_time": 1},
        {"id": "P-2005", "pred_price": 8900, "pred_quality": 0.01, "lead_time": 2},
        {"id": "P-3030", "pred_price": 19000, "pred_quality": 0.05, "lead_time": 1},
    ]
    log_recommendation_impression(SESSION_ID, USER_ID, RECOMMENDATIONS, {"page_type": "main"})
    
    # 2. 잠시 후, 사용자가 P-1001을 클릭하고 45초간 봄
    # (프론트엔드/클라이언트가 이 이벤트를 백엔드로 전송)
    log_user_reaction(
        session_id=SESSION_ID,
        product_id="P-1001",
        action_type="click",
        view_time_sec=45
    )
    
    # 3. P-2005와 P-3030은 클릭되지 않음
    # ('실패(Reward=0)' 데이터가 됨)
    # -> 나중에 데이터 분석 시 'impression' 로그에는 있지만 'reaction' 로그에는
    #    없는 상품들을 찾아 '실패' 케이스로 가공합니다.
    
    print("\n[데이터 활용]")
    print("수집된 'Impression'과 'Reaction' 로그를 조인하여")
    print("RL 모델 학습용 (S, A=1, R=+1.5, S') 튜플과 (S, A=1, R=0, S') 튜플을 생성합니다.")