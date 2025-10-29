import os

# --- 1. Paths ---

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_REGISTER_PATH = os.path.join(PROJECT_ROOT, "model_register")
RL_MODEL_PATH = os.path.join(MODEL_REGISTER_PATH, "rl_model")
ML_MODEL_PATH = os.path.join(MODEL_REGISTER_PATH, "ml_model")

# --- 2. Model hyperparameters ---

OBS_DIM = 50        # State의 실제 차원 (예: 품목정보 10 + 그룹 2 + 가격 1 + 리드타임 1 + 하자비율 1)
ACTION_DIM = 2      # [rec, not rec]
HIDDEN_DIMS = [64, 64]

# --- 3. Training hyperparameters ---

GAMMA = 0.99
TAU = 0.01         # ratio of target net soft update
LR_ACTOR = 3e-4
LR_CRITIC = 1e-5
LR_ALPHA = 3e-4

BATCH_SIZE = 32
BUFFER_LIMIT = int(1e6) # 1,000,000
NUM_EPISODES = 5000
NUM_TRAINING_STEPS = 10000
MAX_STEPS_PER_EPISODE = 100 # (for simulation)

ACTOR_UPDATE_DELAY = 2  # Actor와 Target Net은 Critic보다 2배 늦게 업데이트
LOG_INTERVAL = 100       # print log for 10 epi
SAVE_INTERVAL = 1000     # save the model for 100 epi

# --- 4. PER hyperparameters ---

PER_ALPHA = 0.6         # 우선순위 강도
PER_BETA_START = 0.4    # IS 가중치 시작 값
PER_BETA_INCREMENT = 0.00001 # 1e-6 (BETA가 1.0에 도달하기까지 걸리는 시간 조절)

# --- 5. checkpoint ---
LOAD_CHECKPOINT = False
CHECKPOINT_FILE_TO_LOAD = 'this_is_for_example.pth'

# --- 6. DQN ---
TARGET_UPDATE_INTERVAL  = 1

# --- 7. Model Change ---
MODEL_TYPE = 'DQN'