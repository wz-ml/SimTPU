import torch
import pytest
from source.functional_units import MXU
from source.constants import TILE_SIZE, TILE_ELEMS


@pytest.fixture
def mxu():
    return MXU()


class TestMXULoadWeights:
    def test_loads_tile_from_scratchpad(self, mxu, regs, scratchpad, hbm):
        data = torch.arange(TILE_ELEMS, dtype=torch.bfloat16, device=scratchpad.device)
        scratchpad[0:TILE_ELEMS] = data
        regs[1] = 0
        mxu.load_weights(regs, scratchpad, hbm, 1)
        assert mxu.mxu_weights.shape == (TILE_SIZE, TILE_SIZE)
        assert torch.equal(mxu.mxu_weights, data.reshape(TILE_SIZE, TILE_SIZE))

    def test_loads_from_offset(self, mxu, regs, scratchpad, hbm):
        offset = TILE_ELEMS
        data = torch.ones(TILE_ELEMS, dtype=torch.bfloat16, device=scratchpad.device) * 5
        scratchpad[offset:offset + TILE_ELEMS] = data
        regs[1] = offset
        mxu.load_weights(regs, scratchpad, hbm, 1)
        assert torch.equal(mxu.mxu_weights.flatten(), data)

    def test_load_is_a_copy(self, mxu, regs, scratchpad, hbm):
        scratchpad[0:TILE_ELEMS] = 3.0
        regs[1] = 0
        mxu.load_weights(regs, scratchpad, hbm, 1)
        scratchpad[0:TILE_ELEMS] = 0.0
        assert mxu.mxu_weights[0, 0].item() == 3.0


class TestMXUMatmul:
    def test_basic_matmul(self, mxu, regs, scratchpad, hbm):
        weights = torch.eye(TILE_SIZE, dtype=torch.bfloat16, device=scratchpad.device)
        acts = torch.ones(TILE_SIZE, TILE_SIZE, dtype=torch.bfloat16, device=scratchpad.device) * 2
        scratchpad[0:TILE_ELEMS] = weights.flatten()
        regs[1] = 0
        mxu.load_weights(regs, scratchpad, hbm, 1)

        scratchpad[0:TILE_ELEMS] = acts.flatten()
        regs[2] = 0  # activation addr
        regs[3] = 0  # acc=0 -> overwrite
        mxu.matmul(regs, scratchpad, hbm, 2, 3)
        expected = acts @ weights
        assert torch.allclose(mxu.mxu_acc, expected)

    def test_matmul_overwrite(self, mxu, regs, scratchpad, hbm):
        mxu.mxu_acc = torch.ones(TILE_SIZE, TILE_SIZE, dtype=torch.bfloat16, device=scratchpad.device) * 99
        mxu.mxu_weights = torch.eye(TILE_SIZE, dtype=torch.bfloat16, device=scratchpad.device)
        scratchpad[0:TILE_ELEMS] = 1.0
        regs[1] = 0
        regs[2] = 0  # acc=0 -> overwrite
        mxu.matmul(regs, scratchpad, hbm, 1, 2)
        # should NOT contain the old 99s
        assert mxu.mxu_acc[0, 0].item() != 99

    def test_matmul_accumulate(self, mxu, regs, scratchpad, hbm):
        mxu.mxu_weights = torch.eye(TILE_SIZE, dtype=torch.bfloat16, device=scratchpad.device)
        scratchpad[0:TILE_ELEMS] = 1.0
        regs[1] = 0
        regs[2] = 0  # overwrite first
        mxu.matmul(regs, scratchpad, hbm, 1, 2)
        first = mxu.mxu_acc.clone()

        regs[2] = 1  # acc=1 -> accumulate
        mxu.matmul(regs, scratchpad, hbm, 1, 2)
        assert torch.allclose(mxu.mxu_acc, first * 2)


class TestMXUStore:
    def test_store_acc_to_scratchpad(self, mxu, regs, scratchpad, hbm):
        mxu.mxu_acc = torch.ones(TILE_SIZE, TILE_SIZE, dtype=torch.bfloat16, device=scratchpad.device) * 7
        regs[1] = 0
        mxu.store(regs, scratchpad, hbm, 1)
        assert torch.equal(scratchpad[0:TILE_ELEMS], mxu.mxu_acc.flatten())

    def test_store_to_offset(self, mxu, regs, scratchpad, hbm):
        offset = TILE_ELEMS
        mxu.mxu_acc = torch.ones(TILE_SIZE, TILE_SIZE, dtype=torch.bfloat16, device=scratchpad.device) * 3
        regs[1] = offset
        mxu.store(regs, scratchpad, hbm, 1)
        assert torch.equal(scratchpad[offset:offset + TILE_ELEMS], mxu.mxu_acc.flatten())
