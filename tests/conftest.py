import torch
import pytest
from source.constants import TILE_SIZE, TILE_ELEMS, SCRATCH_ELEMS, HBM_ELEMS, NUM_REGS, DEVICE


@pytest.fixture
def regs():
    return [0] * NUM_REGS


@pytest.fixture
def scratchpad():
    return torch.zeros(SCRATCH_ELEMS, dtype=torch.bfloat16, device=DEVICE)


@pytest.fixture
def hbm():
    return torch.zeros(HBM_ELEMS, dtype=torch.bfloat16, device=DEVICE)
