import random

import numpy as np

import torch

import pytest

from td3.replay_buffer import ReplayBuffer


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def buffer(device) -> ReplayBuffer:
    return ReplayBuffer((784,), 10, 10, 4, device)


@pytest.fixture
def s() -> np.ndarray:
    return np.random.randn(784).astype(np.float32)


@pytest.fixture
def s_nxt() -> np.ndarray:
    return np.random.randn(784).astype(np.float32)


@pytest.fixture
def a() -> np.ndarray:
    return np.random.rand(10).astype(np.float32)


@pytest.fixture
def r() -> np.ndarray:
    return random.random()


@pytest.fixture
def d() -> np.ndarray:
    return random.randint(0, 1)


class TestReplayBuffer:
    def test_init(self, buffer: ReplayBuffer) -> None:
        assert buffer.obs_shape == (784,)
        assert buffer.action_dim == 10
        assert buffer.buffer_capacity == 10
        assert buffer.position == 0
        assert buffer.size == 0

    def test_push(
            self, 
            buffer: ReplayBuffer, 
            s: np.ndarray, 
            a: np.ndarray, 
            r: float, 
            s_nxt: np.ndarray, 
            d: bool
    ) -> None:
        buffer.push(s, a, r, s_nxt, d)
        assert buffer.size == 1
        assert buffer.position == 1

        buffer.push(s, a, r, s_nxt, d)
        assert buffer.size == 2
        assert buffer.position == 2

        buffer.push(s, a, r, s_nxt, d)
        assert buffer.size == 3
        assert buffer.position == 3

        buffer.push(s, a, r, s_nxt, d)
        assert buffer.size == 4
        assert buffer.position == 4

        buffer.push(s, a, r, s_nxt, d)
        assert buffer.size == 5
        assert buffer.position == 5
        
        buffer.push(s, a, r, s_nxt, d)
        assert buffer.size == 6
        assert buffer.position == 6

        buffer.push(s, a, r, s_nxt, d)
        assert buffer.size == 7
        assert buffer.position == 7

        buffer.push(s, a, r, s_nxt, d)
        assert buffer.size == 8
        assert buffer.position == 8
        
        buffer.push(s, a, r, s_nxt, d)
        assert buffer.size == 9
        assert buffer.position == 9

        buffer.push(s, a, r, s_nxt, d)
        assert buffer.size == 10
        assert buffer.position == 0

        buffer.push(s, a, r, s_nxt, d)
        assert buffer.size == 10
        assert buffer.position == 1
        
        buffer.push(s, a, r, s_nxt, d)
        assert buffer.size == 10
        assert buffer.position == 2

        buffer.push(s, a, r, s_nxt, d)
        assert buffer.size == 10
        assert buffer.position == 3

    def test_sample(
            self, 
            buffer: ReplayBuffer, 
            s: np.ndarray, 
            a: np.ndarray, 
            r: float, 
            s_nxt: np.ndarray, 
            d: bool
    ) -> None:
        for _ in range(10):
            buffer.push(s, a, r, s_nxt, d)

        s, a, r, s_nxt, d = buffer.sample()
        
        assert isinstance(s, torch.Tensor)
        assert isinstance(a, torch.Tensor)
        assert isinstance(r, torch.Tensor)
        assert isinstance(s_nxt, torch.Tensor)
        assert isinstance(d, torch.Tensor)

        assert s.dtype == torch.float32
        assert a.dtype == torch.float32
        assert r.dtype == torch.float32
        assert s_nxt.dtype == torch.float32
        assert d.dtype == torch.float32

        assert s.shape == (4, 784)
        assert a.shape == (4, 10)
        assert r.shape == (4,)
        assert s_nxt.shape == (4, 784)
        assert d.shape == (4,)