{
"session_id": "dummy_session_001",
"step": 1,
"timestamp": "2025-10-24T14:00:00Z",
"state_t": {
"user_profile": {
"user_group_vector": [1, 0],
"\_description": "[high_quality, low_price]"
},
"session_history": {
"\_description": "(세션 첫 단계이므로 비어있음)",
"last_clicked_item_vector": [0.0, 0.0, 0.0],
"session_click_count": 0,
"session_total_view_time": 0,
"category_affinity_vector": [0.0, 0.0]
},
"candidate_item_info": {
"item_id": "P-1001",
"predicted_price": 25000,
"lead_time": 1,
"predicted_quality": 0.98,
"promotion": 1
}
},
"action_t": 1,
"reward_ingredients": {
"\_description": "train_rl.py가 이 객체를 보고 최종 보상 점수를 계산",
"clicked": 1,
"page_view_time_sec": 30,
"purchased": 0
},
"next_state_t_plus_1": {
"user_profile": {
"user_group_vector": [1, 0]
},
"session_history": {
"\_description": "(S_t의 행동/보상이 여기에 'update'됨)",
"last_clicked_item_vector": [0.1, 0.9, 0.2],
"session_click_count": 1,
"session_total_view_time": 30,
"category_affinity_vector": [0.8, 0.1]
},
"candidate_item_info": {
"\_description": "(이것은 t+1 시점의 '다음' 추천 후보 정보임)",
"item_id": "P-2005",
"predicted_price": 8900,
"lead_time": 2,
"predicted_quality": 0.92,
"promotion": 0
}
},
"done": 0
}

네, 이 JSON 구조는 `(S, A, R, S', Done)` 튜플을 나타내며, DQN 모델을 학습시키는 핵심 단위입니다.

각 키(key)의 의미와, 특히 차원을 가진 값들이 왜 그 차원을 가지며 무엇을 의미하는지 구체적으로 설명해 드리겠습니다.

---

## ## 1. 최상위 키 (Transition)

이 키들은 '하나의 경험 조각(transition)'을 정의합니다.

- **`session_id`** (문자열)
  - **설명:** 하나의 사용자 방문(세션)을 식별하는 고유 ID입니다. DQN의 **'에피소드(Episode)'**를 구분하는 단위입니다.
- **`step`** (정수)
  - **설명:** 해당 세션(에피소드) 내에서 몇 번째 행동(t)인지를 나타냅니다. (1, 2, 3, ...)
- **`timestamp`** (문자열)
  - **설명:** 이 행동이 발생한 실제 시간입니다. (데이터 정렬 및 분석용)
- **`state_t`** (객체)
  - **설명:** **현재 상태(S)**입니다. DQN 네트워크에 입력될 **모든 정보의 스냅샷**입니다. (아래 2번에서 상세 설명)
- **`action_t`** (정수: 0 또는 1)
  - **설명:** **행동(A)**입니다. `state_t` 상태에서 `candidate_item`을 추천했는지(`1`), 안 했는지(`0`)를 나타냅니다.
  - **왜 2차원인가?** `a_dim=2`로, DQN이 예측해야 할 Q-value가 `Q(S, 0)`과 `Q(S, 1)` 두 개이기 때문입니다.
- **`reward_ingredients`** (객체)
  - **설명:** **보상(R)의 재료**입니다. `train_rl.py` 스크립트는 이 객체의 값들을 조합하여 최종적인 '숫자 보상(scalar reward)'을 계산합니다. (예: `R = (clicked * 1.0) + (page_view_time_sec * 0.01)`)
- **`next_state_t_plus_1`** (객체)
  - **설명:** **다음 상태(S')**입니다. `action_t`를 수행한 결과로 도달한 '미래'의 상태입니다. (아래 3번에서 상세 설명)
- **`done`** (정수: 0 또는 1)
  - **설명:** 이 행동으로 인해 세션(에피소드)이 **종료**되었는지(`1`) 아닌지(`0`)를 나타냅니다. 이 값이 `1`이면, DQN은 `Target = Reward`로 계산합니다 (미래 가치 없음).

---

## ## 2. `state_t` (Current State, S) 상세

DQN의 `obs_dim`(관측 차원)을 구성하는 **입력 벡터(Input Vector)**입니다. `train_rl.py`는 `state_t` 내부의 모든 숫자/벡터를 **하나의 긴 벡터로 '펼쳐서(flatten)'** 모델에 입력합니다.

- **`user_profile` (사용자 프로필)**

  - **`user_group_vector`** (배열, **2차원**)
    - **설명:** 사용자가 어떤 그룹인지 나타내는 **'원-핫 인코딩(One-Hot Encoding)'** 벡터입니다.
    - **왜 2차원인가?** "high_quality"와 "low_price"라는 2개의 상호 배타적인 그룹이 있기 때문입니다.
    - **의미:**
      - `[1, 0]` = "high_quality" 그룹
      - `[0, 1]` = "low_price" 그룹

- **`session_history` (세션 이력)**: **이것이 `S`를 '순차적'으로 만드는 핵심입니다.**

  - **`last_clicked_item_vector`** (배열, **32차원**)
    - **설명:** 이 세션에서 '바로 직전에' 클릭했던 상품을 나타내는 **'상품 임베딩(Item Embedding)'** 벡터입니다.
    - **왜 32차원인가?** 'P-1001' 같은 ID 대신, 상품의 복잡한 특징(과일인지, 비싼지, 신맛인지...)을 32개의 숫자로 압축하여 표현하기 위함입니다. 차원 수는 임의로 정할 수 있으나(e.g., 32, 64, 128), 모델이 상품 간의 미묘한 차이를 학습하기에 충분한 크기여야 합니다.
    - **의미:** 각 차원(예: `[0.1, 0.9, ...]`)이 개별적으로 특정 의미(예: '당도')를 갖기보다는, **32개 숫자의 '조합'** 자체가 수학적 공간에서 해당 상품의 '개념'을 나타냅니다. (세션 첫 단계라면 `[0.0, ...]`으로 비어있습니다.)
  - **`session_click_count`** (숫자, 1차원): 이번 세션에서 지금까지 총 몇 번 클릭했는지 (누적값)
  - **`session_total_view_time`** (숫자, 1차원): 이번 세션에서 지금까지 총 몇 초 머물렀는지 (누적값)
  - **`category_affinity_vector`** (배열, **10차원**)
    - **설명:** 이번 세션에서 '과일', '채소' 등 각 카테고리를 얼마나 많이 봤는지 나타내는 **'선호도 특징 벡터'**입니다.
    - **왜 10차원인가?** '대봉 유통'이 총 10개의 대분류 카테고리를 가지고 있다고 가정한 것입니다.
    - **의미:**
      - `[0.8, 0.1, 0.0, ...]` = "이번 세션에서 '과일'(1번 차원)을 80% 봤고, '채소'(2번 차원)를 10% 봤음"

- **`candidate_item_info` (추천 후보 아이템 정보)**
  - **설명:** 지금 `action_t`로 "추천할까 말까" **고민 중인 바로 그 대상 상품**의 정보입니다.
  - `item_id`: 상품 ID (학습 입력에는 사용되지 않음, 분석용)
  - `predicted_price`, `lead_time`, `predicted_quality`, `promotion`: (각 1차원) ML 모델이 예측한 값 또는 실시간 비즈니스 정보입니다.

---

## ## 3. `next_state_t_plus_1` (Next State, S') 상세

`state_t`와 구조가 100% 동일하지만, 내용물이 **'업데이트'**된 상태입니다.

- **`user_profile`**: 불변입니다.
- **`session_history`**: **핵심적인 차이점입니다.**
  - `state_t`에서 발생한 `reward_ingredients`의 결과(클릭, 체류 시간)가 **'누적 반영'**되어 업데이트된 값입니다.
  - **예시:** `state_t`의 `session_click_count`가 0이었고 `reward_ingredients.clicked`가 1이라면, `next_state_t_plus_1`의 `session_click_count`는 1이 됩니다.
- **`candidate_item_info`**: `t+1` 시점에 새로 추천할 (또 다른) 후보 상품의 정보로 교체됩니다.
