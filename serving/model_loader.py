"""
모델 로더 및 핫 스와핑 관리자

백그라운드에서 주기적으로 새 모델을 확인하고 자동으로 교체 (Hot-swapping)
- 무중단 모델 업데이트
- 스레드 안전 모델 교체
- 최신 모델 자동 감지
- FastAPI 서버와 통합
"""
import os
import glob
import time
import threading
from typing import Optional
from serving.inference import ModelInferenceEngine, ML_MODEL_PATH, RL_MODEL_PATH

class ModelLoader:
    """
    ModelInferenceEngine의 글로벌 인스턴스를 관리하고,
    백그라운드에서 주기적으로 모델 업데이트를 확인하여 핫 스와핑(Hot-swapping)을 수행합니다.
    """
    def __init__(self, check_interval_sec: int = 86400): # 86400초(하루)마다 체크
        print("Initializing Model Loader...")
        self._engine: Optional[ModelInferenceEngine] = None
        self._lock = threading.Lock() # 스레드 충돌 방지용
        self.check_interval = check_interval_sec
        
        # 현재 로드된 ML/RL 모델의 경로(또는 접두사)를 저장합니다.
        self._latest_price_prefix: Optional[str] = ""
        self._latest_quality_prefix: Optional[str] = ""
        self._latest_rl_path: Optional[str] = ""

        # 초기 모델 로드 (첫 실행)
        self._update_models()
        
        # 백그라운드 업데이트 스레드 시작
        self.daemon_thread = threading.Thread(target=self._run_update_checker, daemon=True, name="ModelUpdateChecker")
        self.daemon_thread.start()
        print(f"Background model checker started. (Interval: {self.check_interval}s)")

    def _find_latest_model_path(self, model_dir, search_prefix):
        """
        가장 최신 모델 파일의 접두사(ML) 또는 경로(RL)를 찾습니다.
        """
        search_path = os.path.join(model_dir, f"{search_prefix}_*.pkl") # ML 모델 기준
        
        if search_prefix == "dqn_checkpoint": # RL 모델 기준
            search_path = os.path.join(model_dir, "dqn_checkpoint_*.pth")

        files = glob.glob(search_path)
        if not files:
            return None # 해당 경로에 파일이 없음
            
        # 수정 시간(ctime) 기준 가장 최신 파일
        latest_file_path = max(files, key=os.path.getctime)
        
        if search_prefix == "dqn_checkpoint":
            return latest_file_path # RL은 전체 경로 반환
        else:
            # ML은 접두사 반환 (예: price_ensemble_20251112_133540)
            base_name = os.path.basename(latest_file_path)
            prefix_name = "_".join(base_name.split('_')[:-1])
            return os.path.join(model_dir, prefix_name) # 경로를 포함한 접두사

    def _update_models(self):
        """
        새 모델이 있는지 확인하고, 있으면 ModelInferenceEngine을 다시 로드합니다.
        """
        print(f"[{threading.current_thread().name}] Checking for new models...")
        
        # 1. 디스크에서 가장 최신 모델 정보 확인
        latest_price_on_disk = self._find_latest_model_path(ML_MODEL_PATH, "price_ensemble")
        latest_quality_on_disk = self._find_latest_model_path(ML_MODEL_PATH, "quality_ensemble")
        latest_rl_on_disk = self._find_latest_model_path(RL_MODEL_PATH, "dqn_checkpoint")

        # 2. 현재 로드된 모델과 비교하여 업데이트가 필요한지 확인
        needs_update = False
        if self._engine is None:
            needs_update = True
            print("Initial model load required.")
        elif latest_price_on_disk and latest_price_on_disk != self._latest_price_prefix:
            needs_update = True
            print(f"New Price Model detected: {os.path.basename(latest_price_on_disk)}")
        elif latest_quality_on_disk and latest_quality_on_disk != self._latest_quality_prefix:
            needs_update = True
            print(f"New Quality Model detected: {os.path.basename(latest_quality_on_disk)}")
        elif latest_rl_on_disk and latest_rl_on_disk != self._latest_rl_path:
            needs_update = True
            print(f"New RL Model detected: {os.path.basename(latest_rl_on_disk)}")

        # 3. 업데이트 수행
        if needs_update:
            print("Reloading ModelInferenceEngine...")
            try:
                # ModelInferenceEngine()은 자체적으로 최신 모델을 찾아 로드함
                new_engine = ModelInferenceEngine() 
                
                # 새 엔진이 필수 모델들을 로드했는지 확인
                if (new_engine.price_model is None or 
                    new_engine.quality_model is None or 
                    new_engine.q_network is None):
                    
                    print("!!! Failed to load new models: One or more models were not loaded successfully by the engine.")
                    if self._engine is None:
                        print("!!! Initial model load failed. Engine remains None.")
                    else:
                        print("!!! Hot-swap aborted. Continuing with old models.")
                    return # 함수 종료 (기존 엔진 유지)

                # 스레드 안전하게 글로벌 인스턴스 교체 (Hot-swap)
                with self._lock:
                    self._engine = new_engine
                    # 현재 로드된 모델의 경로/접두사를 최신으로 업데이트
                    # (inference.py가 로드한 실제 경로를 가져와야 함)
                    self._latest_price_prefix = new_engine.latest_price_prefix_loaded
                    self._latest_quality_prefix = new_engine.latest_quality_prefix_loaded
                    self._latest_rl_path = new_engine.latest_rl_path_loaded
                
                print("Models successfully hot-swapped.")
                
            except Exception as e:
                print(f"!!! Failed to reload ModelInferenceEngine: {e}")
        else:
            print("No new models detected. Current models are up-to-date.")

    def _run_update_checker(self):
        """
        백그라운드 스레드에서 주기적으로 _update_models를 호출하는 루프
        """
        while True:
            time.sleep(self.check_interval)
            try:
                self._update_models()
            except Exception as e:
                print(f"Error in background model checker: {e}")

    def get_inference_engine(self) -> Optional[ModelInferenceEngine]:
        """
        FastAPI 서버(main.py)가 호출할 메서드.
        현재 활성화된 모델 엔진을 스레드 안전하게 반환합니다.
        """
        with self._lock:
            return self._engine

# --- 애플리케이션 전체에서 사용할 단일 인스턴스 생성 ---
# main.py가 이 'global_model_loader'를 임포트해서 사용합니다.
global_model_loader = ModelLoader(check_interval_sec=600)