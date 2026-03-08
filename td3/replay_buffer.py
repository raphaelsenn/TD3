from typing import Tuple

import numpy as np
import torch


class ReplayBuffer:
    """Simple replay buffer as described in the nature DQN paper or the DDPG paper.""" 
    def __init__(
            self, 
            obs_shape: Tuple[int, ...], 
            action_dim: int,
            buffer_capacity: int,
            batch_size: int,
            device: torch.device
    ) -> None:
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size 
        self.device = device
        
        self.states = np.empty(shape=(buffer_capacity, *obs_shape), dtype=np.float32) 
        self.states_nxt = np.empty(shape=(buffer_capacity, *obs_shape), dtype=np.float32) 
        self.actions = np.empty(shape=(buffer_capacity, action_dim), dtype=np.float32) 
        self.rewards = np.empty(shape=(buffer_capacity,), dtype=np.float32)
        self.dones = np.empty(shape=(buffer_capacity,), dtype=np.float32)

        self.position = 0
        self.size = 0


    def push(self, s: np.ndarray, a: np.ndarray, r: float, s_nxt: np.ndarray, done: bool) -> None:
        """Pushes experience s_{i}, a_{i}, r_{i}, s_{i+1}, d_{i} into ReplayBuffer.""" 
        i = self.position
        capacity = self.buffer_capacity

        self.states[i] = s.astype(np.float32)
        self.actions[i] = a.astype(np.float32)
        self.rewards[i] = float(r)
        self.states_nxt[i] = s_nxt.astype(np.float32)
        self.dones[i] = float(done)

        self.position = (i + 1) % capacity
        self.size = min(self.size + 1, self.buffer_capacity)

    def sample(self) -> Tuple:
        """Samples a random minibatch (s_{i}, a_{i}, r_{i}, s_{i+1}, d_{i}) ~ ReplayBuffer.""" 
        idx = np.random.randint(0, self.size, size=self.batch_size)

        s = torch.as_tensor(self.states[idx], dtype=torch.float32, device=self.device)
        a = torch.as_tensor(self.actions[idx], dtype=torch.float32, device=self.device)
        r = torch.as_tensor(self.rewards[idx], dtype=torch.float32, device=self.device)
        s_nxt = torch.as_tensor(self.states_nxt[idx], dtype=torch.float32, device=self.device)
        d = torch.as_tensor(self.dones[idx], dtype=torch.float32, device=self.device)

        return (s, a, r, s_nxt, d)