"""
Prioritized Experience Replay (PER) 버퍼 구현

강화학습에서 경험 재생을 효율적으로 수행하기 위한 우선순위 기반 샘플링 버퍼입니다.
- Sumtree 자료구조를 활용한 O(log N) 샘플링
- TD-Error 기반 우선순위 업데이트
- Importance Sampling 가중치 계산
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

def list2torch(x, device):
    """
    List를 PyTorch Tensor로 변환

    Args:
        x (list): 변환할 리스트
        device (str): 텐서를 할당할 디바이스 ('cpu' 또는 'cuda')

    Returns:
        torch.Tensor: float 타입의 PyTorch 텐서

    로직:
        1. 리스트를 numpy 배열로 변환
        2. numpy 배열을 PyTorch 텐서로 변환 (float 타입)
        3. 지정된 디바이스로 텐서 이동
    """
    return torch.tensor(np.array(x), dtype = torch.float).to(device)

#PER(Prioritized Experience Replay)
class Sumtree:
    """
    algorithm for PER to sample faster, more efficient
    """
    def __init__(self, capacity):
        """
        Sumtree 초기화

        Args:
            capacity (int): 트리에 저장할 최대 데이터 개수

        로직:
            1. capacity 크기만큼의 이진 트리 구조 생성 (2*capacity-1 노드)
            2. 데이터 저장 배열 초기화
            3. 엔트리 카운터와 쓰기 포인터 초기화
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype = object)
        self.n_entries = 0
        self.write = 0

    def _propagate(self, idx, change):
        """
        우선순위 변경사항을 부모 노드로 전파

        Args:
            idx (int): 변경된 노드의 인덱스
            change (float): 변경된 우선순위 값

        로직:
            1. 부모 노드 인덱스 계산
            2. 부모 노드의 우선순위에 변경값 반영
            3. 루트 노드에 도달할 때까지 재귀적으로 전파
        """
        parent = (idx - 1)//2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx, s):
        """
        주어진 값에 해당하는 리프 노드를 재귀적으로 탐색

        Args:
            idx (int): 현재 탐색 중인 노드 인덱스
            s (float): 탐색할 누적 우선순위 값

        Returns:
            int: 찾은 리프 노드의 인덱스

        로직:
            1. 왼쪽/오른쪽 자식 노드 인덱스 계산
            2. 리프 노드에 도달하면 해당 인덱스 반환
            3. 값이 왼쪽 서브트리 합보다 작으면 왼쪽으로 탐색
            4. 그렇지 않으면 오른쪽으로 탐색 (값에서 왼쪽 서브트리 합 차감)
        """
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        """
        트리의 전체 우선순위 합 반환

        Returns:
            float: 루트 노드에 저장된 전체 우선순위 합

        로직:
            루트 노드(인덱스 0)의 값을 반환 (모든 리프 노드의 우선순위 합)
        """
        return self.tree[0]

    def update(self, idx, p):
        """
        특정 노드의 우선순위 업데이트

        Args:
            idx (int): 업데이트할 노드의 인덱스
            p (float): 새로운 우선순위 값

        로직:
            1. 기존 값과 새 값의 차이 계산
            2. 노드에 새 우선순위 값 저장
            3. 변경사항을 부모 노드들로 전파
        """
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)
    
    def add(self, p, data):
        """
        새로운 데이터를 트리에 추가

        Args:
            p (float): 데이터의 우선순위 값
            data (tuple): 저장할 transition 데이터 (s, a, r, s', done)

        로직:
            1. 현재 쓰기 위치에 해당하는 트리 인덱스 계산
            2. 데이터 배열에 transition 저장
            3. 트리에 우선순위 값 업데이트 (부모 노드까지 전파)
            4. 쓰기 포인터를 다음 위치로 이동 (순환 버퍼)
            5. 엔트리 수 업데이트 (최대 capacity까지)
        """
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def get(self, s):
        """
        주어진 우선순위 값에 해당하는 데이터 조회

        Args:
            s (float): 조회할 누적 우선순위 값

        Returns:
            tuple: (트리 인덱스, 우선순위 값, transition 데이터)

        로직:
            1. 루트 노드부터 시작하여 값에 해당하는 리프 노드 탐색
            2. 리프 노드의 트리 인덱스를 데이터 배열 인덱스로 변환
            3. (트리 인덱스, 우선순위, 데이터) 튜플 반환
        """
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])
    
class PERBufferClass:
    """
    Prioritized Experience Replay (PER) Buffer for efficient sampling
    """
    def __init__(self,
                buffer_limit = 100000,
                device = 'cpu',
                alpha = 0.6,
                beta = 0.4,
                beta_increment = 0.001):
        """
        PER 버퍼 초기화

        Args:
            buffer_limit (int): 버퍼의 최대 크기 (기본값: 100000)
            device (str): PyTorch 디바이스 ('cpu' 또는 'cuda') (기본값: 'cpu')
            alpha (float): 우선순위 강도 (0=균등 샘플링, 1=완전 우선순위) (기본값: 0.6)
            beta (float): Importance Sampling 가중치 시작값 (기본값: 0.4)
            beta_increment (float): beta 증가 비율 (기본값: 0.001)

        로직:
            1. Sumtree 자료구조 초기화
            2. alpha, beta 등 PER 하이퍼파라미터 설정
            3. 최대 우선순위 및 엡실론 값 초기화
        """
        self.tree = Sumtree(buffer_limit)
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.capacity = buffer_limit
        self.max_priority = 1.0
        self.epsilon = 1e-5 # to prevent the priority becomes 0
    
    def put(self, item):
        """
        save the item in the buffer (in max priority)

        Args:
            item (tuple): 저장할 transition 데이터 (s, a, r, s', done)

        로직:
            새로운 경험을 최대 우선순위로 버퍼에 추가 (새 데이터는 중요하다고 가정)
        """
        self.tree.add(self.max_priority, item)

    def size(self):
        """
        버퍼에 저장된 데이터 개수 반환

        Returns:
            int: 현재 버퍼에 저장된 transition 개수

        로직:
            Sumtree의 엔트리 수 반환
        """
        return self.tree.n_entries

    def sample(self, n):
        """
        sample based on priority

        Args:
            n (int): 샘플링할 미니배치 크기

        Returns:
            tuple: (states, actions, rewards, next_states, done_masks, is_weights, indices)
                - states: 상태 텐서 (batch_size, state_dim)
                - actions: 행동 텐서 (batch_size, 1)
                - rewards: 보상 텐서 (batch_size, 1)
                - next_states: 다음 상태 텐서 (batch_size, state_dim)
                - done_masks: 종료 플래그 텐서 (batch_size, 1)
                - is_weights: Importance Sampling 가중치 (batch_size, 1)
                - indices: 샘플링된 데이터의 인덱스 리스트

        로직:
            1. 전체 우선순위 합을 n개 구간으로 분할
            2. 각 구간에서 uniform하게 값을 샘플링
            3. 샘플링된 값에 해당하는 transition 조회
            4. Importance Sampling 가중치 계산 및 정규화
            5. transition을 PyTorch 텐서로 변환하여 반환
            6. beta annealing (점진적으로 1.0으로 증가)
        """
        mini_batch = []
        idxs = []
        is_weights = np.empty(n, dtype = np.float32)
        segment = self.tree.total() / n
        self.beta = np.min([1., self.beta + self.beta_increment]) # beta anyling

        for i in range(n):
            s = random.uniform(segment * i, segment * (i+1))
            (idx, p, data) = self.tree.get(s)

            prob = p / self.tree.total()
            is_weights[i] = np.power(self.size() * prob, -self.beta) # calculate priority sampling weight

            mini_batch.append(data)
            idxs.append(idx)
    
        s_list, a_list, r_list, s_prime_list, done_mask_list = [], [], [], [], []
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_list.append(s)
            a_list.append([a]) #as action is scalar(0 or 1)
            r_list.append([r])
            s_prime_list.append(s_prime)
            done_mask = 0.0 if done_mask else 1.0
            done_mask_list.append([done_mask])

        is_weights /= is_weights.max() # normalize is_weight

        return (list2torch(s_list, self.device),
                list2torch(a_list, self.device),
                list2torch(r_list, self.device),
                list2torch(s_prime_list, self.device),
                list2torch(done_mask_list, self.device),
                torch.tensor(is_weights, dtype=torch.float).to(self.device).reshape(-1, 1), idxs)
    
    def update_priorities(self, batch_indices, td_errors):
        """
        update the priority of sampling batch (based on TD-Error)

        Args:
            batch_indices (list): 업데이트할 transition의 트리 인덱스 리스트
            td_errors (torch.Tensor): TD-Error 값들 (batch_size, 1)

        로직:
            1. TD-Error를 numpy 배열로 변환
            2. |TD-Error| + epsilon을 alpha 제곱하여 우선순위 계산
            3. 각 인덱스의 우선순위를 트리에 업데이트
            4. 최대 우선순위 값 갱신 (새 데이터 추가시 사용)
        """
        td_errors = td_errors.detach().cpu().numpy()
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha

        for idx, p in zip(batch_indices, priorities):
            self.tree.update(idx, p[0])
            self.max_priority = max(self.max_priority, p[0])

    def clear(self):
        """
        버퍼 초기화

        로직:
            현재 capacity와 device 설정을 유지하면서 버퍼를 재초기화
        """
        self.__init__(self.capacity, self.device)