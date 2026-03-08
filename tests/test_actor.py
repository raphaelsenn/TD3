import pytest

import torch
import numpy as np

from td3 import ActorMLP


@pytest.fixture
def actor() -> ActorMLP:
    return ActorMLP(784, 400, 300, 10)


@pytest.fixture
def state_batch() -> torch.Tensor:
    return torch.randn(64, 784, dtype=torch.float32)


@pytest.fixture
def state() -> torch.Tensor:
    return torch.randn(784, dtype=torch.float32)


@pytest.fixture
def state_np() -> torch.Tensor:
    return np.random.randn(784).astype(np.float32)


class TestActorMLP:
    def test_dims(self, actor: ActorMLP) -> None:
        assert actor.state_dim == 784
        assert actor.h1_dim == 400
        assert actor.h2_dim == 300
        assert actor.action_dim == 10

    def test_forward_shape(self, actor: ActorMLP, state_batch: torch.Tensor) -> None:
        out = actor(state_batch)
        assert out.shape == (64, 10)

    def test_predict_numpy_1(self, actor: ActorMLP, state: torch.Tensor) -> None:
        out = actor.predict(state)
        assert isinstance(out, np.ndarray)

    def test_predict_numpy_2(self, actor: ActorMLP, state_np: np.ndarray) -> None:
        out = actor.predict(state_np)
        assert isinstance(out, np.ndarray)