import torch
import pytest
from source.functional_units import DMAUnit
from source.constants import TILE_ELEMS, DEVICE


@pytest.fixture
def dma():
    return DMAUnit()


def _expect(val, n):
    return torch.tensor(val, dtype=torch.bfloat16, device=DEVICE).expand(n)


class TestDMALoad:
    def test_load_from_hbm_to_scratchpad(self, dma, regs, scratchpad, hbm):
        data = torch.arange(128, dtype=torch.bfloat16, device=DEVICE)
        hbm[0:128] = data
        regs[1] = 0    # hbm src
        regs[2] = 0    # scratchpad dest
        regs[3] = 128  # size
        dma.load(regs, scratchpad, hbm, 1, 2, 3)
        assert torch.equal(scratchpad[0:128], data)

    def test_load_with_offset(self, dma, regs, scratchpad, hbm):
        hbm[100:200] = 7.0
        regs[1] = 100   # hbm src
        regs[2] = 500   # scratchpad dest
        regs[3] = 100   # size
        dma.load(regs, scratchpad, hbm, 1, 2, 3)
        assert torch.allclose(scratchpad[500:600], _expect(7.0, 100))

    def test_load_clamped_to_tile_elems(self, dma, regs, scratchpad, hbm):
        hbm[0:TILE_ELEMS] = 1.0
        regs[1] = 0
        regs[2] = 0
        regs[3] = TILE_ELEMS * 2  # exceeds limit
        dma.load(regs, scratchpad, hbm, 1, 2, 3)
        assert torch.allclose(scratchpad[0:TILE_ELEMS], _expect(1.0, TILE_ELEMS))


class TestDMAStore:
    def test_store_from_scratchpad_to_hbm(self, dma, regs, scratchpad, hbm):
        data = torch.arange(128, dtype=torch.bfloat16, device=DEVICE)
        scratchpad[0:128] = data
        regs[1] = 0    # scratchpad src
        regs[2] = 0    # hbm dest
        regs[3] = 128  # size
        dma.store(regs, scratchpad, hbm, 1, 2, 3)
        assert torch.equal(hbm[0:128], data)

    def test_store_with_offset(self, dma, regs, scratchpad, hbm):
        scratchpad[200:300] = 3.0
        regs[1] = 200   # scratchpad src
        regs[2] = 1000  # hbm dest
        regs[3] = 100   # size
        dma.store(regs, scratchpad, hbm, 1, 2, 3)
        assert torch.allclose(hbm[1000:1100], _expect(3.0, 100))

    def test_roundtrip(self, dma, regs, scratchpad, hbm):
        data = torch.randn(TILE_ELEMS).to(torch.bfloat16).to(DEVICE)
        scratchpad[0:TILE_ELEMS] = data
        regs[1] = 0
        regs[2] = 0
        regs[3] = TILE_ELEMS
        dma.store(regs, scratchpad, hbm, 1, 2, 3)
        scratchpad[0:TILE_ELEMS] = 0  # clear
        dma.load(regs, scratchpad, hbm, 1, 2, 3)
        assert torch.equal(scratchpad[0:TILE_ELEMS], data)
