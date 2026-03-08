import pytest

import torch
import numpy as np

from td3 import CriticMLP


@pytest.fixture
def critic() -> CriticMLP:
    return CriticMLP(784, 400, 300, 10)


@pytest.fixture
def state_batch() -> torch.Tensor:
    return torch.randn(64, 784, dtype=torch.float32)


@pytest.fixture
def action_batch() -> torch.Tensor:
    return torch.randn(64, 10, dtype=torch.float32)


@pytest.fixture
def state() -> torch.Tensor:
    return torch.randn(784, dtype=torch.float32)


@pytest.fixture
def action() -> torch.Tensor:
    return torch.randn(10, dtype=torch.float32)


@pytest.fixture
def state_np() -> torch.Tensor:
    return np.random.randn(784).astype(np.float32)


@pytest.fixture
def action_np() -> torch.Tensor:
    return np.random.randn(10).astype(np.float32)


class TestCriticMLP:
    def test_dims(self, critic: CriticMLP) -> None:
        assert critic.state_dim == 784
        assert critic.h1_dim == 400
        assert critic.h2_dim == 300
        assert critic.action_dim == 10

    def test_forward_shape(self, critic: CriticMLP, state_batch: torch.Tensor, action_batch: torch.Tensor) -> None:
        q1, q2 = critic(state_batch, action_batch)
        assert q1.shape == (64, 1)
        assert q2.shape == (64, 1)

    def test_predict_numpy(self, critic: CriticMLP, state: torch.Tensor, action: np.ndarray) -> None:
        q1, q2 = critic.predict(state, action)
        assert isinstance(q1, np.ndarray)
        assert isinstance(q2, np.ndarray)
        assert q1.shape == (1, 1)
        assert q2.shape == (1, 1)

    def test_predict_numpy_2(self, critic: CriticMLP, state_np: np.ndarray, action_np: np.ndarray) -> None:
        q1, q2 = critic.predict(state_np, action_np)
        assert isinstance(q1, np.ndarray)
        assert isinstance(q2, np.ndarray)
        assert q1.shape == (1, 1)
        assert q2.shape == (1, 1)