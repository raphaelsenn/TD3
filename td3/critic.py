import copy
from typing import Tuple
from abc import ABC, abstractmethod

import numpy as np

import torch
import torch.nn as nn


class Critic(nn.Module, ABC):
    """Critic interface for an action-value function.""" 
    def __init__(
            self, 
            obs_shape: Tuple[int, ...], 
            action_dim: int
    ) -> None:
        super().__init__()
        if action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got: {action_dim}") 
        if len(obs_shape) == 0:
            raise ValueError(f"obs_shape must be non-empty, got: {obs_shape}")

        self.obs_shape = tuple(int(element) for element in obs_shape)
        self.action_dim = action_dim

    @abstractmethod
    def forward(
        self, 
        s: torch.Tensor, 
        a: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @torch.inference_mode()
    def predict(
        self, 
        s: np.ndarray | torch.Tensor, 
        a: np.ndarray | torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        device = next(self.parameters()).device 
        if isinstance(s, np.ndarray):
            s_t = torch.as_tensor(s, dtype=torch.float32, device=device)
        else:
            s_t = s.to(device)
        if isinstance(a, np.ndarray):
            a_t = torch.as_tensor(a, dtype=torch.float32, device=device)
        else:
            a_t = a.to(device)

        if s_t.dim() == len(self.obs_shape):
            s_t.unsqueeze_(0)
        
        if a_t.dim() == 1:
            a_t.unsqueeze_(0)

        q1, q2 = self(s_t, a_t)
        return q1.detach().cpu().numpy(), q2.detach().cpu().numpy()

    def copy(self) -> 'Critic':
        critic = copy.deepcopy(self)
        return critic


class CriticMLP(Critic):
        def __init__(
                self, 
                state_dim: int, 
                h1_dim: int,
                h2_dim: int,
                action_dim: int,

        ) -> None:
            super().__init__((state_dim,), action_dim)
            self.state_dim = state_dim
            self.h1_dim = h1_dim
            self.h2_dim = h2_dim

            self.q1 = nn.Sequential(
                # [B, state_dim + action_dim] -> [B, h1_dim] 
                nn.Linear(state_dim + action_dim, h1_dim),
                nn.ReLU(True),
                
                # [B, h1_dim] -> [B, h2_dim] 
                nn.Linear(h1_dim, h2_dim),
                nn.ReLU(True),
                
                # [B, h2_dim] -> [B, 1] 
                nn.Linear(h2_dim, 1),
            
            )

            self.q2 = nn.Sequential(
                # [B, state_dim + action_dim] -> [B, h1_dim] 
                nn.Linear(state_dim + action_dim, h1_dim),
                nn.ReLU(True),
                
                # [B, h1_dim] -> [B, h2_dim] 
                nn.Linear(h1_dim, h2_dim),
                nn.ReLU(True),
                
                # [B, h2_dim] -> [B, 1]
                nn.Linear(h2_dim, 1),
            )
            
        def forward(
                self, 
                s: torch.Tensor, 
                a: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            x = torch.cat([s, a], dim=-1)
            q1 = self.q1(x)
            q2 = self.q2(x)
            return q1, q2