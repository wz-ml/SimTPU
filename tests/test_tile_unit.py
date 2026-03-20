import torch
import pytest
from source.functional_units import TileUnit
from source.constants import TILE_SIZE, TILE_ELEMS


def _eye(device):
    return torch.eye(TILE_SIZE, device=device).to(torch.bfloat16)


@pytest.fixture
def tu():
    return TileUnit()


class TestTileTranspose:
    def test_identity_unaffected(self, tu, regs, scratchpad, hbm):
        mat = _eye(scratchpad.device)
        scratchpad[0:TILE_ELEMS] = mat.flatten()
        regs[1] = 0
        tu.transpose(regs, scratchpad, hbm, 1)
        result = scratchpad[0:TILE_ELEMS].reshape(TILE_SIZE, TILE_SIZE)
        assert torch.equal(result, mat)

    def test_transpose_content(self, tu, regs, scratchpad, hbm):
        mat = torch.arange(TILE_ELEMS, dtype=torch.bfloat16, device=scratchpad.device).reshape(TILE_SIZE, TILE_SIZE)
        scratchpad[0:TILE_ELEMS] = mat.flatten()
        regs[1] = 0
        tu.transpose(regs, scratchpad, hbm, 1)
        result = scratchpad[0:TILE_ELEMS].reshape(TILE_SIZE, TILE_SIZE)
        assert torch.equal(result, mat.T)

    def test_double_transpose_is_identity(self, tu, regs, scratchpad, hbm):
        mat = torch.randn(TILE_SIZE, TILE_SIZE).to(torch.bfloat16).to(scratchpad.device)
        scratchpad[0:TILE_ELEMS] = mat.flatten()
        regs[1] = 0
        tu.transpose(regs, scratchpad, hbm, 1)
        tu.transpose(regs, scratchpad, hbm, 1)
        result = scratchpad[0:TILE_ELEMS].reshape(TILE_SIZE, TILE_SIZE)
        assert torch.equal(result, mat)

    def test_transpose_at_offset(self, tu, regs, scratchpad, hbm):
        offset = TILE_ELEMS
        mat = torch.arange(TILE_ELEMS, dtype=torch.bfloat16, device=scratchpad.device).reshape(TILE_SIZE, TILE_SIZE)
        scratchpad[offset:offset+TILE_ELEMS] = mat.flatten()
        regs[1] = offset
        tu.transpose(regs, scratchpad, hbm, 1)
        result = scratchpad[offset:offset+TILE_ELEMS].reshape(TILE_SIZE, TILE_SIZE)
        assert torch.equal(result, mat.T)
