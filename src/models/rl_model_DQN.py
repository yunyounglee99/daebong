"""
DQN (Deep Q-Network) 강화학습 모델 구현

Double DQN, Dueling Network, Noisy Net을 결합한 고급 DQN 구현
- Dueling Architecture: Value와 Advantage 분리 학습
- Noisy Linear: 파라메트릭 노이즈 기반 탐색
- Double DQN: Q-value 과대평가 방지
- PER (Prioritized Experience Replay) 지원
"""
import random
import collections
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

class NoisyLinear(nn.Module):
    """
    용도:
        nn.Linear를 대체하는 'Noisy Net' 레이어입니다.
        e-greedy 탐험 대신, 가중치 자체에 학습 가능한 노이즈를 추가하여
        모델이 스스로 효율적인 탐험(Exploration)을 학습하도록 합니다. (DQN에 기본적으로 필요)
    """
    def __init__(self, in_features, out_features, std_init = 0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.w_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.w_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('w_epsilon', torch.empty(out_features, in_features))

        self.b_mu = nn.Parameter(torch.empty(out_features))
        self.b_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('b_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """
        용도: (내부 함수) 가중치의 평균(mu)과 노이즈 표준편차(sigma)를 초기화합니다.
        Args:
            None
        Returns:
            None
        로직:
            1. 'He/Kaiming' 초기화와 유사하게 입력 피처 수에 기반한 범위(mu_range)를 설정합니다.
            2. 가중치(w_mu)와 편향(b_mu)의 평균 파라미터를 Uniform 분포로 초기화합니다.
            3. 가중치(w_sigma)와 편향(b_sigma)의 노이즈 파라미터를 `std_init` 기반의 작은 상수로 초기화합니다.
        """
        mu_range = 1 / math.sqrt(self.in_features)
        self.w_mu.data.uniform_(-mu_range, mu_range)
        self.w_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.b_mu.data.uniform_(-mu_range, mu_range)
        self.b_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        """
        용도: (내부 함수) Factorised Gaussian noise를 생성하기 위한 팩터(factor)를 계산합니다.
        Args:
            size (int): 노이즈 벡터의 크기.
        Returns:
            torch.Tensor: 부호(sign)와 절댓값의 제곱근(sqrt)이 곱해진 팩터.
        로직:
            `x = torch.randn(size)`로 표준 정규분포 노이즈를 샘플링한 뒤,
            `sign(x) * sqrt(|x|)` 연산을 적용합니다.
        """
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())
    
    def reset_noise(self):
        """
        용도: 
            (외부 호출) `get_action` 함수에서 행동을 선택(탐험)하기 직전에 호출됩니다.
            레이어의 노이즈 텐서(w_epsilon, b_epsilon)를 새로운 랜덤 값으로 교체합니다.
        Args:
            None
        Returns:
            None
        로직:
            1. `_scale_noise`를 사용해 입력/출력 차원에 맞는 노이즈 팩터를 생성합니다.
            2. `epsilon_out.ger(epsilon_in)` (외적)을 통해 가중치 노이즈(`w_epsilon`)를 생성합니다.
            3. `epsilon_out`을 편향 노이즈(`b_epsilon`)로 사용합니다.
            4. `.copy_()`를 사용해 버퍼의 값을 덮어씁니다.
        """
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.w_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.b_epsilon.copy_(epsilon_out)

    def forward(self, x):
        """
        용도: (PyTorch) 레이어의 순전파를 수행합니다.
        Args:
            x (torch.Tensor): 입력 텐서.
        Returns:
            torch.Tensor: 선형 변환(Linear)이 적용된 출력 텐서.
        로직:
            1. `self.training` (학습) 모드일 경우:
               - `w = w_mu + w_sigma * w_epsilon` (노이즈가 적용된 가중치)
               - `b = b_mu + b_sigma * b_epsilon` (노이즈가 적용된 편향)
            2. `self.eval()` (평가) 모드일 경우:
                - `w = w_mu` (노이즈가 제거된 평균 가중치)
                - `b = b_mu` (노이즈가 제거된 평균 편향)
            3. `F.linear(x, w, b)`를 통해 최종 선형 변환을 수행합니다.
        """
        if self.training:
            w = self.w_mu + self.w_sigma * self.w_epsilon
            b = self.b_mu + self.b_sigma * self.b_epsilon
        else:
            w = self.w_mu
            b = self.b_mu

        return F.linear(x, w, b)
    
class DQN(nn.Module):
    """
    용도:
        Dueling + Double + Noisy DQN 네트워크의 아키텍처를 정의합니다.
        이 클래스는 Q-value를 예측하는 신경망('q_main', 'q_target')을 생성합니다.
    """
    def __init__(
            self,
            name = 'dqn',
            obs_dim = 8, 
            a_dim = 2,
            h_dims = [256, 256],
            lr_critic = 3e-4,
            device = None
    ) -> None:
        super(DQN, self).__init__()
        self.name = name
        self.obs_idm = obs_dim
        self.a_dim = a_dim
        self.h_dims = h_dims
        self.lr_critic = lr_critic
        self.device = device

        # --- Dueling DQN Architecture ---
        # common feature layer
        self.feature_layers = []
        h_dim_prev = self.obs_idm
        for h_dim in self.h_dims:
            self.feature_layers.append(nn.Linear(h_dim_prev, h_dim))
            self.feature_layers.append(nn.ReLU())
            h_dim_prev = h_dim
        self.feature_net = nn.Sequential(*self.feature_layers).to(self.device)

        # Advantage Head
        self.advantage_head = nn.Sequential(
            NoisyLinear(h_dims[-1], h_dims[-1]),
            nn.ReLU(),
            NoisyLinear(h_dims[1], self.a_dim)
        ).to(self.device)

        # Value Head
        self.value_head = nn.Sequential(
            NoisyLinear(h_dims[-1], h_dims[-1]),
            nn.ReLU(),
            NoisyLinear(h_dims[1], 1)
        ).to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr = self.lr_critic)

    def forward(self, x):
        """
        용도: (PyTorch) 네트워크의 순전파를 수행하여 Q-value를 계산합니다. (Dueling 방식)
        Args:
            x (torch.Tensor): 상태(State) 배치. (크기: [batch_size, obs_dim])
        Returns:
            torch.Tensor: 각 행동에 대한 Q-value 배치. (크기: [batch_size, a_dim])
        로직:
            1. 입력 `x`를 공통 피처 네트워크(`self.feature_net`)에 통과시켜 `features`를 추출합니다.
            2. `features`를 Value Head에 통과시켜 상태 가치 `v` (크기: [batch_size, 1])를 계산합니다.
            3. `features`를 Advantage Head에 통과시켜 행동 우위 `a` (크기: [batch_size, a_dim])를 계산합니다.
            4. Dueling DQN 공식 `q = v + (a - a.mean(dim=1))`을 사용하여
                두 스트림을 결합한 최종 Q-value를 반환합니다.
        """
        x = x.to(self.device)
        features = self.feature_net(x)

        v = self.value_head(features)
        a = self.advantage_head(features)

        q = v + (a - a.mean(dim=1, keepdim = True))

        return q
    
    def train(self, target, mini_batch, is_weights):
        """
        용도: `train_rl.py`가 호출하는 메인 학습 함수. DQN 네트워크를 1 스텝 업데이트합니다.
        Args:
            target (torch.Tensor): 
                `get_target` 함수로 계산된 타겟 Q-value (크기: [batch_size, 1])
            mini_batch (tuple): 
                (s, a, r, s_prime, done_mask) 튜플.
            is_weights (torch.Tensor): 
                PER 버퍼에서 샘플링된 중요도 샘플링(IS) 가중치 (크기: [batch_size, 1])
        Returns:
            torch.Tensor: 
                우선순위(Priority) 업데이트에 사용할 TD-Error (크기: [batch_size, 1])
        로직:
            1. `self.forward(s)`를 호출하여 현재 상태(s)에 대한 모든 Q-value를 예측합니다.
            2. `q.gather(1, a.long())`를 사용해 실제 취했던 행동(a)의 Q-value(`current_q`)만 추출합니다.
            3. `td_error = torch.abs(target - current_q)`를 계산합니다.
            4. `loss = (is_weights * F.smooth_l1_loss(current_q, target, ...))`를 계산하여 
                PER의 IS 가중치가 적용된 손실(Loss)을 구합니다.
            5. `loss.backward()` 및 `optimizer.step()`으로 네트워크 가중치를 업데이트합니다.
            6. `td_error`를 PER 버퍼 업데이트를 위해 반환합니다.
        """
        s, a, r, s_prime, done = mini_batch
        q = self.forward(s)
        current_q = q.gather(1, a.long())

        td_error = torch.abs(target - current_q)
        loss = (is_weights * F.smooth_l1_loss(current_q, target, reduction = 'none')).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm = 0.5)
        self.optimizer.step()

        return td_error
    
    def soft_update(self, tau, net_target):
        """
        용도: `train_rl.py`가 주기적으로 호출. 타겟 네트워크(q_target)의 가중치를 
                메인 네트워크(q_main) 쪽으로 서서히 업데이트합니다.
        Args:
            tau (float): 소프트 업데이트 비율 (예: 0.01). (1.0이면 하드 업데이트)
            net_target (DQN): 타겟 네트워크(`q_target`)의 인스턴스.
        Returns:
            None
        로직:
            `param_target = (1.0 - tau) * param_target + tau * param` 공식을 사용하여
            `net_target`의 모든 파라미터를 현재 네트워크(`self`)의 파라미터 쪽으로 조금씩 이동시킵니다.
        """
        for param_target, param in zip(net_target.parameters(), self.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

    def reset_noise(self):
        """
        용도: (외부 호출) `get_action` 함수가 탐험을 위해 호출합니다.
                네트워크 내의 모든 `NoisyLinear` 레이어의 노이즈를 리셋합니다.
        Args:
            None
        Returns:
            None
        로직:
            `self.modules()`를 순회하며 `NoisyLinear` 타입의 레이어를 찾아
            해당 레이어의 `reset_noise()` 메서드를 호출합니다.
        """
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

# actor
def get_action(q_network, s, deterministic = False):
    """
    용도: 
        현재 상태(s)에서 DQN 네트워크(q_network)를 기반으로 행동(Action)을 결정합니다.
        (e-greedy 대신 Noisy Net을 사용한 탐험)
    Args:
        q_network (DQN): Q-value를 예측할 DQN 네트워크.
        s (torch.Tensor): 현재 상태(State) 텐서 (크기: [1, obs_dim])
        deterministic (bool): 
            - (학습 시) False: `q_network.reset_noise()`를 호출하여 탐험(Noisy) 행동을 유도.
            - (평가 시) True: 노이즈를 리셋하지 않아 결정론적(Greedy) 행동 수행.
    Returns:
        int: 결정된 행동 (예: 0 또는 1).
    로직:
        1. `deterministic=False`이면 `q_network.reset_noise()`를 호출합니다.
        2. `torch.no_grad()` 컨텍스트에서 `q_network(s)`를 호출하여 모든 행동의 Q-value를 얻습니다.
        3. `torch.argmax(q)`를 사용해 Q-value가 가장 높은 행동(a)을 선택합니다.
        4. `a.item()`을 통해 텐서를 파이썬 정수(int)로 변환하여 반환합니다.
    """
    s = s.to(q_network.device)

    if not deterministic:
        # training : reset the noise of the Noisy Net for exploration
        q_network.reset_noise()

    with torch.no_grad():
        q = q_network(s)
        a = torch.argmax(q, dim = 1).unsqueeze(1)

    return a.item()

# caculatiing target
def get_target(q_main, q_target, gamma, mini_batch, device):
    """
    용도: 
        DQN의 학습 목표가 되는 타겟(Target) Q-value를 계산합니다. (Double DQN 방식)
    Args:
        q_main (DQN): 메인 Q-네트워크.
        q_target (DQN): 타겟 Q-네트워크.
        gamma (float): 할인율 (Discount factor, 예: 0.99).
        mini_batch (tuple): (s, a, r, s_prime, done_mask) 튜플.
        device (str): 'cuda' 또는 'cpu'.
    Returns:
        torch.Tensor: 학습에 사용할 타겟 Q-value (크기: [batch_size, 1])
    로직: (Double DQN)
        1. `torch.no_grad()` 컨텍스트에서 계산을 수행합니다.
        2. (행동 선택) 메인 네트워크(`q_main`)로 다음 상태(`s_prime`)의 Q-value를 예측하고,
           가장 높은 Q-value를 가진 행동(`best_a_prime`)을 *선택*합니다.
        3. (가치 평가) 타겟 네트워크(`q_target`)로 다음 상태(`s_prime`)의 Q-value를 예측하고,
           `best_a_prime`에 해당하는 Q-value(`target_q_val`)만 *평가*합니다.
        4. Bellman Equation `target = r + gamma * done_mask * target_q_val`을 계산하여 반환합니다.
            (done_mask가 0이면 미래 가치가 0이 됨)
    """
    q_main = q_main.to(device)
    q_target = q_target.to(device)
    s, a, r, s_prime, done = mini_batch

    with torch.no_grad():
        # extract next action(a') with max q value on main net
        q_main_next = q_main(s_prime)
        best_a_prime = torch.argmax(q_main_next, dim = 1).unsqueeze(-1)

        q_target_next = q_target(s_prime)
        target_q_val = q_target_next.gather(1, best_a_prime)

        target = r + gamma * done * target_q_val

    return target