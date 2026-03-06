# Daebong AI Recommendation System

An end-to-end **ML + Reinforcement Learning recommendation system** for apple wholesale distribution.  
The system predicts product price and quality via ensemble ML models, then uses a DQN-based RL agent to decide whether to recommend a candidate item — optimizing long-term user engagement and conversion.

---

## Table of Contents

- [Project Pipeline](#project-pipeline)
- [Model Architecture](#model-architecture)
  - [Price Prediction — EnsemblePriceModel](#price-prediction--ensemblepricemodel)
  - [Quality Prediction — EnsembleQualityModel](#quality-prediction--ensemblequalitymodel)
  - [RL Agent — DQN](#rl-agent--dqn)
  - [RL Agent — SAC (Alternative)](#rl-agent--sac-alternative)
  - [Serving Layer](#serving-layer)
- [State Design](#state-design)
- [Action Design](#action-design)
- [Reward Design](#reward-design)
- [Project Structure](#project-structure)
- [Setup & Execution](#setup--execution)
- [Configuration](#configuration)
- [Performance](#performance)

---

## Project Pipeline

The system is divided into three sequential phases. Each phase depends on the artifacts produced by the previous one.

---

### Phase 1 — ML Training

**Entry point:** `python -m training.train_ml --model all --ensemble voting`

`DataPreprocessor` loads raw CSV and JSON files from `data/raw/` and runs two separate feature engineering pipelines. For price prediction, it calls `preprocess_price_data()` followed by `create_price_features()`, which produces a time-enriched DataFrame with lag features, weather joins, and encoded categoricals. For quality prediction, it calls `preprocess_sales_data()`, `preprocess_cs_data()`, and `create_quality_features()`, which merges CS event data onto sales records and computes seller-level rolling CS rates.

The resulting DataFrames are passed to `EnsemblePriceModel.train()` and `EnsembleQualityModel.train()` respectively. Both trainers split data in time-ordered 70 / 15 / 15 fashion and save trained models plus feature name lists under `model_register/ml_model/` as a set of `.pkl` files.

**Outputs:** `price_ensemble_<timestamp>_*.pkl`, `quality_ensemble_<timestamp>_*.pkl`

---

### Phase 2 — RL Training

**Entry point:** `python -m training.train_rl`

`RLDataLoader` initializes a `ModelInferenceEngine` that loads the Phase 1 `.pkl` files into memory. It then iterates over rows in `sales_df`, calls `predict_price()` and `predict_quality_rate()` for each row, and feeds those ML predictions into `_construct_state_vector()` to produce a 50-dimensional state vector. `_calculate_reward()` assigns a scalar reward based on whether the order appears in the CS dataset.

Each `(State, Action=1, Reward, NextState, Done)` tuple is inserted into a `PERBufferClass` SumTree buffer via `buffer.put()`. Once the buffer is populated, the main training loop runs for `NUM_TRAINING_STEPS` iterations. At each step, it samples a prioritized mini-batch, computes a Double DQN target Q-value via `get_target_dqn()`, updates the main network with IS-weighted Smooth L1 loss, refreshes SumTree priorities using the resulting TD-error, and applies a soft update to the target network. Checkpoints are saved to `model_register/rl_model/` every `SAVE_INTERVAL` steps.

**Outputs:** `checkpoint_ep<step>.pth`

---

### Phase 3 — Serving

**Entry point:** `uvicorn serving.main:app --host 127.0.0.1 --port 8000`

At startup, a `ModelLoader` singleton is created. It immediately calls `_update_models()`, which scans `model_register/` for the latest `.pkl` and `.pth` files and instantiates a `ModelInferenceEngine` that loads all three models (price, quality, DQN) into memory. A background daemon thread then repeats this scan every 10 minutes. If a newer model file is detected, a new `ModelInferenceEngine` is instantiated and swapped in under a `threading.Lock`, achieving zero-downtime hot-swapping without restarting the server.

The FastAPI app exposes three prediction endpoints. `/predict_price` calls `EnsemblePriceModel.predict()`, `/predict_quality_rate` calls `EnsembleQualityModel.predict_proba()`, and `/predict_q_values` passes the incoming `state_t` JSON through `flatten_state_to_vector()` and then through `DQN.forward()` under `torch.no_grad()`, returning `[Q(a=0), Q(a=1)]` as the recommendation scores.

---

## Model Architecture

<img width="4400" height="2475" alt="daebong_model_architecture" src="https://github.com/user-attachments/assets/4c6d5817-637c-40fa-a087-380544980d0c" />


### Price Prediction — EnsemblePriceModel

`EnsemblePriceModel` is a **regression ensemble** trained to predict the next-day wholesale apple price in KRW.

#### Input Features

| Group | Features |
|-------|----------|
| Date | `year`, `month`, `day`, `dayofweek`, `quarter`, `weekofyear` |
| Season | `is_spring`, `is_summer`, `is_fall`, `is_winter` |
| Weather | `avgTa`, `minTa`, `maxTa`, `avgRhm`, `sumRn`, `avgWs` |
| Categorical | `도매시장`, `품종`, `산지-광역시도`, `산지-시군구`, `등급` (LabelEncoded) |
| Lag / Rolling | `price_lag_7d/14d/30d`, `price_std_7d/14d/30d`, `price_change_7d` |

> **Data leakage prevention:** `총거래금액`, `총거래물량`, and `price_per_kg` are excluded at training time because they are derived from or directly correlated with the target variable `평균가격`.

#### Base Models

| Model | Key Parameters |
|-------|---------------|
| `LGBMRegressor` | `n_estimators=1000`, `learning_rate=0.05`, `num_leaves=31`, bagging & feature fraction 0.8 |
| `XGBRegressor` | `n_estimators=1000`, `learning_rate=0.05`, `max_depth=6`, `subsample=0.8` |
| `RandomForestRegressor` | `n_estimators=200`, `max_depth=15`, `min_samples_leaf=2` |

#### Ensemble Methods

Three ensemble strategies are supported, selected via `--ensemble`:

- **Voting** — Each base model predicts independently on the validation set. Weights are computed as the normalized inverse of each model's MAE: `weight ∝ 1 / MAE`. Final prediction is the weighted average of all base model outputs.
- **Stacking** — `TimeSeriesSplit` cross-validation (5 folds) generates out-of-fold predictions from each base model. A `Ridge` meta-model is then trained on these OOF predictions. Base models are retrained on the full dataset after CV.
- **Blending** — Base models are trained on the train split only. Their predictions on a held-out blend set are used to fit the `Ridge` meta-model.

---

### Quality Prediction — EnsembleQualityModel

`EnsembleQualityModel` is a **binary classification ensemble** that predicts the probability of a CS (customer complaint / defect) event for a given order. Its output `predict_proba()` serves as the `predicted_quality` feature in the RL state vector.

#### Input Features

| Group | Features |
|-------|----------|
| Seller stats | `seller_cs_rate`, `company_cs_rate` |
| Date | `year`, `month`, `day`, `dayofweek`, `quarter` |
| Season | `is_spring`, `is_summer`, `is_fall`, `is_winter` |
| Weather | `avgTa`, `minTa`, `maxTa`, `avgRhm`, `sumRn`, `avgWs` |
| Product | `product_weight`, `price_log`, `price_per_kg` |
| Categorical | `셀러코드`, `업체명` (LabelEncoded) |
| Rolling CS | `cs_rate_7d`, `cs_rate_14d`, `cs_rate_30d` |

**Target:** `defect_rate` — binary flag (1 if the order's `주문코드` appears in the CS dataset, else 0).

#### Base Models

| Model | Key Parameters |
|-------|---------------|
| `LGBMClassifier` | `objective='binary'`, `is_unbalanced=True`, `n_estimators=800`, `lr=0.03` |
| `XGBClassifier` | `objective='binary:logistic'`, `scale_pos_weight=475`, `n_estimators=800`, `lr=0.03` |
| `RandomForestClassifier` | `class_weight='balanced'`, `n_estimators=200`, `max_depth=10` |
| `GradientBoostingClassifier` | `n_estimators=200`, `lr=0.05`, `max_depth=4`, `subsample=0.7` |

#### Ensemble Methods

- **Voting** — Weights are computed as `weight ∝ (max(0.5, AUC) - 0.5)²`, which suppresses near-random models (AUC ≈ 0.5) and amplifies high-performing ones. Final output is the weighted average of `predict_proba()` outputs from all base models.
- **Stacking** — OOF probabilities from 5-fold `TimeSeriesSplit` CV are used to train a `LogisticRegression` meta-model with `class_weight='balanced'`.

> **Class imbalance:** The CS rate is approximately 0.21%. All four base models apply class-imbalance correction. Evaluation focuses on AUC-ROC, F1, and Recall rather than accuracy.

---

### RL Agent — DQN

<img width="4400" height="2484" alt="daebong s,a,r" src="https://github.com/user-attachments/assets/eeeb28ea-7872-43b7-b996-590498b78099" />


The recommendation decision is framed as a **contextual bandit** (single-step MDP): given the current state `s`, the agent outputs Q-values for both actions and selects `a* = argmax Q(s, a)`.

#### Network Architecture

The DQN uses a **Dueling Network** architecture with **NoisyLinear** layers.

The input state vector `s ∈ ℝ⁵⁰` first passes through a shared **Feature Network** consisting of two fully connected layers — `Linear(50 → 64) + ReLU` and `Linear(64 → 64) + ReLU` — which extract a common latent representation.

The output of the Feature Network is then fed into two parallel heads:

The **Value Head** estimates the scalar state value `V(s) ∈ ℝ¹`. It consists of a `NoisyLinear(64 → 64)` layer, a ReLU activation, and a final `NoisyLinear(64 → 1)` projection.

The **Advantage Head** estimates the relative advantage of each action `A(s, a) ∈ ℝ²`. It has the same structure as the Value Head but outputs a 2-dimensional vector via `NoisyLinear(64 → 2)`.

The two streams are merged using the Dueling DQN formula:

```
Q(s, a) = V(s) + A(s, a) − mean_a[ A(s, a) ]
```

This decomposition allows the network to learn which states are inherently valuable independent of the action taken, improving sample efficiency in states where action choice has little effect.

#### Key Design Decisions

**NoisyLinear (exploration):** All `NoisyLinear` layers replace the weight matrix `W` with a learnable distribution `W = μ + σ ⊙ ε`, where `ε` is sampled from a factorised Gaussian. This enables parametric exploration without ε-greedy schedules. During inference, `model.eval()` mode sets `W = μ`, eliminating noise for deterministic action selection.

**Double DQN (stability):** Action selection and value evaluation are decoupled across two networks. The main network `Q_main` selects the best next action `a'* = argmax Q_main(s', a)`; the target network `Q_target` evaluates its value `Q_target(s', a'*)`. This prevents the overestimation bias inherent in vanilla DQN.

**Soft Target Update:** The target network is updated every step via `θ_target ← (1 − τ) θ_target + τ θ_main` with `τ = 0.01`, providing stable regression targets without hard periodic copying.

**PER — Prioritized Experience Replay:** The replay buffer uses a SumTree for O(log N) priority sampling. Priority is defined as `p ∝ (|δ| + ε)^α`, where `δ` is the TD-error. Importance Sampling (IS) weights correct for the non-uniform sampling distribution, normalized and annealed toward 1 over training.

#### One Training Step

```
s, a, r, s', done_mask, is_weights, idxs = buffer.sample(BATCH_SIZE)

# Double DQN target
a'_best = argmax_a  Q_main(s', a)
target  = r + γ · done_mask · Q_target(s', a'_best)

# Loss and backprop
current_q = Q_main(s).gather(dim=1, index=a)
td_error  = |target − current_q|
loss      = mean(is_weights · SmoothL1(current_q, target))
loss.backward() → optimizer.step()

# Priority update and target sync
buffer.update_priorities(idxs, td_error)
Q_target ← (1 − τ) · Q_target + τ · Q_main
```

---

### RL Agent — SAC (Alternative)

A **Soft Actor-Critic** implementation for discrete action spaces is also provided in `src/models/rl_model_SAC.py` but is currently on hold. SAC maintains an explicit stochastic policy network (Actor) and two Q-networks (Critics), optimizing a maximum-entropy objective that balances reward and policy entropy via a learned temperature parameter `α`. It is better suited than DQN when the state space is dense and strong exploratory pressure is needed from the start.

| | DQN | SAC |
|---|---|---|
| Exploration | NoisyNet (parametric noise in weights) | Max-entropy (temperature `α`, auto-tuned) |
| Actor | Implicit (argmax over Q) | Explicit stochastic policy network |
| Networks | `Q_main` + `Q_target` | Actor + `Critic₁` + `Critic₂` + `TargetCritic₁` + `TargetCritic₂` |
| Status | **Active** | On hold |

---

### Serving Layer

When the FastAPI server starts, a `ModelLoader` singleton is created globally in `serving/model_loader.py`. It immediately loads the latest model artifacts from `model_register/` by instantiating a `ModelInferenceEngine`, which internally loads both ML `.pkl` files and the RL `.pth` checkpoint, then sets `q_network.eval()`.

A background `daemon` thread runs `_update_models()` every 10 minutes. If a newer `.pkl` or `.pth` file is detected in `model_register/`, a new `ModelInferenceEngine` is constructed and swapped in under a `threading.Lock`. The old engine is dereferenced and garbage-collected. This hot-swap mechanism allows model updates without restarting the server.

Every API request calls `get_engine()`, which acquires the lock and returns the currently active `ModelInferenceEngine` instance. The inference flow for `/predict_q_values` is:

1. Receive `state_t` as a JSON body.
2. Call `flatten_state_to_vector(state_t)` → `np.ndarray` of shape `(50,)`.
3. Convert to a `torch.Tensor` of shape `(1, 50)`.
4. Call `q_network(s_tensor)` under `torch.no_grad()` → tensor of shape `(1, 2)`.
5. Return `[Q(a=0), Q(a=1)]` as the JSON response.

---

## State Design

The state vector `s ∈ ℝ⁵⁰` is constructed by `flatten_state_to_vector()` from three semantic groups.

### Group 1 — User Profile (dims 0–1)

A 2-dimensional one-hot vector encoding the user's segment: `[1, 0]` for the *high-quality* group and `[0, 1]` for the *low-price* group. This value is fixed for the duration of a session.

### Group 2 — Session History (dims 2–45)

Captures the user's accumulated in-session behavior. This group is updated at every state transition `s_t → s_{t+1}` to reflect the outcome of the previous action.

| Field | Dims | Description |
|-------|------|-------------|
| `last_clicked_item_vector` | 2–33 (32) | 32-dimensional embedding of the most recently clicked product. Zero-vector at session start. The combination of all 32 values encodes the product's semantic identity in a learned embedding space. |
| `session_click_count` | 34 (1) | Cumulative number of clicks within the current session. |
| `session_total_view_time` | 35 (1) | Cumulative page-view dwell time in seconds within the current session. |
| `category_affinity_vector` | 36–45 (10) | Per-category share of page views in the current session, across 10 product categories. e.g., `[0.8, 0.1, 0.0, ...]` means 80% of views were in category 0. |

### Group 3 — Candidate Item Info (dims 46–49)

Attributes of the specific product being evaluated at timestep `t`. This group is replaced with the next candidate's features when transitioning to `s_{t+1}`.

| Field | Dim | Description |
|-------|-----|-------------|
| `predicted_price` | 46 | Next-day price predicted by `EnsemblePriceModel`. |
| `lead_time` | 47 | Average shipping days for the candidate's seller, looked up from `lead_time_map`. |
| `predicted_quality` | 48 | CS (defect) probability in `[0, 1]` predicted by `EnsembleQualityModel`. |
| `promotion` | 49 | Binary flag indicating whether the item is currently on promotion. |

### State Transition

The user profile is unchanged across steps. The session history accumulates the result of the previous action — a click increments `session_click_count` and `session_total_view_time` and updates the `last_clicked_item_vector` and `category_affinity_vector`. The candidate item info is replaced with the next candidate's features, making `s_{t+1}` distinct from `s_t` even when the user profile does not change.

---

## Action Design

The action space is binary and discrete: `A = {0, 1}`.

- `a = 1` — Recommend the candidate item to the user.
- `a = 0` — Do not recommend; skip or surface a different item.

The DQN outputs a 2-dimensional Q-value vector `[Q(s, a=0), Q(s, a=1)]` for each state. At inference, the recommended action is `a* = argmax Q(s, a)`. During training with NoisyNet, `reset_noise()` is called before each forward pass, so the argmax is taken over a stochastically perturbed Q-surface, which drives exploration without an explicit ε schedule.

---

## Reward Design

Reward is a scalar signal computed from user reaction events logged by the backend after each recommendation impression.

### Reward Function

```python
def calculate_reward(sources: dict) -> float:
    clicked   = sources.get("clicked", 0)
    view_time = sources.get("page_view_time_sec", 0)
    purchased = sources.get("purchased", 0)

    if clicked == 0:
        reward = 0.0                        # No engagement
    elif view_time < 5:
        reward = -0.5                       # Bounce: poor relevance signal
    else:
        reward = 1.0 + (view_time / 60.0)  # Sustained engagement, scaled by dwell time

    if purchased:
        reward += 5.0                       # Conversion bonus

    return reward
```

### Reward Signal Summary

| User Behavior | Reward | Rationale |
|---------------|--------|-----------|
| No click | `0.0` | Null signal — no gradient pressure |
| Click + view < 5s | `−0.5` | Penalizes recommendations that attract a click but fail to hold attention (bounce) |
| Click + view ≥ 5s | `1.0 + t/60` | Rewards genuine engagement; the dwell-time bonus incentivizes quality over clickbait |
| Purchase | `+5.0` (additive) | Strong conversion signal, layered on top of the engagement reward |

### Offline RL Data Construction

Since real-time log collection is not yet active, training transitions are constructed from historical sales records. Every row in `sales_df` represents a past successful sale and is treated as a completed `action=1` episode. The base reward is `+1.0` for each row; if the order's `주문코드` appears in the CS dataset, a penalty of `−5.0` is subtracted. The next state is set to a zero vector and `done=True`, framing each sample as an independent contextual bandit episode. This eliminates the need for sequential next-state modeling until real backend logs are available.

---

## Project Structure

```
daebong/
├── data/
│   ├── collectors/
│   │   └── weather_api_check.py     # KMA ASOS API data collection
│   ├── raw/                         # Source data (CSV + JSON)
│   └── processed/
│       └── create_predictions.py    # Batch ML prediction export script
│
├── model_register/
│   ├── ml_model/                    # Trained ML model artifacts (.pkl)
│   └── rl_model/                    # RL checkpoints (.pth)
│
├── src/
│   ├── config.py                    # All hyperparameters and paths
│   ├── data/
│   │   └── rl_dataloader.py         # Offline RL transition generator
│   ├── models/
│   │   ├── ml_price_model.py        # EnsemblePriceModel (regression)
│   │   ├── ml_quality_model.py      # EnsembleQualityModel (classification)
│   │   ├── rl_model_DQN.py          # DQN: NoisyLinear, Dueling, train/soft-update ops
│   │   └── rl_model_SAC.py          # SAC: Actor, Critic (on hold)
│   └── utils/
│       └── replay_buffer.py         # PERBufferClass + SumTree
│
├── serving/
│   ├── main.py                      # FastAPI app and endpoint definitions
│   ├── inference.py                 # ModelInferenceEngine (unified ML + RL)
│   └── model_loader.py              # Hot-swap model manager
│
├── training/
│   ├── train_ml.py                  # ML training entrypoint
│   ├── train_rl.py                  # RL training entrypoint
│   └── utils.py                     # flatten_state_to_vector, calculate_reward
│
├── backend_log_guide.py             # Backend logging spec for live data collection
└── data_structure.md                # (S, A, R, S', done) JSON schema definition
```

---

## Setup & Execution

```bash
# Install dependencies
pip install -r requirements.txt

# Phase 1 — Train ML models
python -m training.train_ml --model all --ensemble voting

# Phase 2 — Train RL model (requires Phase 1 outputs)
python -m training.train_rl

# Phase 3 — Launch serving API
uvicorn serving.main:app --reload --host 127.0.0.1 --port 8000
```

Swagger UI is available at `http://127.0.0.1:8000/docs`.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict_price` | POST | Returns next-day predicted price given ML feature dict |
| `/predict_quality_rate` | POST | Returns CS probability in `[0.0, 1.0]` given ML feature dict |
| `/predict_q_values` | POST | Returns `[Q(not recommend), Q(recommend)]` given `state_t` JSON |

---

## Configuration

All hyperparameters are centralized in `src/config.py`.

```python
# State / Action
OBS_DIM     = 50          # State vector dimension
ACTION_DIM  = 2           # {0: not recommend, 1: recommend}
HIDDEN_DIMS = [64, 64]    # Shared feature network hidden layer sizes

# RL Training
GAMMA      = 0.99         # Discount factor
TAU        = 0.01         # Soft target update coefficient
LR_CRITIC  = 1e-5         # DQN Adam optimizer learning rate
BATCH_SIZE = 32
BUFFER_LIMIT        = 1_000_000
NUM_TRAINING_STEPS  = 10_000
SAVE_INTERVAL       = 1_000   # Checkpoint frequency (steps)
TARGET_UPDATE_INTERVAL = 1    # Target network update frequency (steps)

# PER
PER_ALPHA         = 0.6   # Priority exponent (0 = uniform, 1 = full priority)
PER_BETA_START    = 0.4   # IS weight initial value (annealed toward 1.0)
PER_BETA_INCREMENT = 1e-5 # Beta increment per training step

# Model
MODEL_TYPE = 'DQN'        # 'DQN' or 'SAC'
```

---

## Performance

Results from the most recent training run:

| Model | Metric | Value |
|-------|--------|-------|
| Price Prediction | Mean Absolute Error | 2,726 KRW |
| Price Prediction | Mean Absolute Percentage Error | **3.61%** |
| Quality Prediction | Overall Accuracy | **99.74%** |
| Quality Prediction | CS-sample Recall | **92.79%** |
| Quality Prediction | CS-sample Mean Predicted Probability | **0.7574** |
