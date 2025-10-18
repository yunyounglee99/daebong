# excution exsample : python training/train_rl.py

import torch
import numpy as np
import os
import time
import src.config as config
from src.utils.replay_buffer import PERBufferClass
from src.models.rl_model import ActorClass, CriticClass, get_target

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device : {DEVICE}')

TARGET_ENTROPY = -np.log(1.0 / config.ACTION_DIM) * 0.98