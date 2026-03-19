import torch
import pytest
from source.functional_units import VectorUnit
from source.constants import TILE_SIZE, DEVICE


@pytest.fixture
def vu():
    return VectorUnit()


def _expect(val, n=TILE_SIZE):
    return torch.tensor(val, dtype=torch.bfloat16, device=DEVICE).expand(n)


class TestSetLength:
    def test_default_length(self, vu):
        assert vu.vector_length == TILE_SIZE

    def test_set_smaller(self, vu, regs, scratchpad, hbm):
        regs[1] = 64
        vu.set_length(regs, scratchpad, hbm, 1)
        assert vu.vector_length == 64

    def test_clamped_to_tile_size(self, vu, regs, scratchpad, hbm):
        regs[1] = TILE_SIZE * 2
        vu.set_length(regs, scratchpad, hbm, 1)
        assert vu.vector_length == TILE_SIZE


class TestVectorElementwise:
    def test_add(self, vu, regs, scratchpad, hbm):
        n = TILE_SIZE
        scratchpad[0:n] = 2.0
        scratchpad[n:2*n] = 3.0
        regs[1], regs[2], regs[3] = 2*n, 0, n
        vu.add(regs, scratchpad, hbm, 1, 2, 3)
        assert torch.allclose(scratchpad[2*n:3*n], _expect(5.0))

    def test_sub(self, vu, regs, scratchpad, hbm):
        n = TILE_SIZE
        scratchpad[0:n] = 10.0
        scratchpad[n:2*n] = 3.0
        regs[1], regs[2], regs[3] = 2*n, 0, n
        vu.sub(regs, scratchpad, hbm, 1, 2, 3)
        assert torch.allclose(scratchpad[2*n:3*n], _expect(7.0))

    def test_mult(self, vu, regs, scratchpad, hbm):
        n = TILE_SIZE
        scratchpad[0:n] = 4.0
        scratchpad[n:2*n] = 3.0
        regs[1], regs[2], regs[3] = 2*n, 0, n
        vu.mult(regs, scratchpad, hbm, 1, 2, 3)
        assert torch.allclose(scratchpad[2*n:3*n], _expect(12.0))

    def test_div(self, vu, regs, scratchpad, hbm):
        n = TILE_SIZE
        scratchpad[0:n] = 12.0
        scratchpad[n:2*n] = 4.0
        regs[1], regs[2], regs[3] = 2*n, 0, n
        vu.div(regs, scratchpad, hbm, 1, 2, 3)
        assert torch.allclose(scratchpad[2*n:3*n], _expect(3.0))

    def test_pow(self, vu, regs, scratchpad, hbm):
        n = TILE_SIZE
        scratchpad[0:n] = 2.0
        scratchpad[n:2*n] = 3.0
        regs[1], regs[2], regs[3] = 2*n, 0, n
        vu.pow(regs, scratchpad, hbm, 1, 2, 3)
        assert torch.allclose(scratchpad[2*n:3*n], _expect(8.0))

    def test_max(self, vu, regs, scratchpad, hbm):
        n = TILE_SIZE
        scratchpad[0:n] = torch.arange(n, dtype=torch.bfloat16, device=DEVICE)
        scratchpad[n:2*n] = 64.0
        regs[1], regs[2], regs[3] = 2*n, 0, n
        vu.max(regs, scratchpad, hbm, 1, 2, 3)
        expected = torch.max(
            torch.arange(n, dtype=torch.bfloat16, device=DEVICE),
            torch.tensor(64.0, dtype=torch.bfloat16, device=DEVICE).expand(n),
        )
        assert torch.equal(scratchpad[2*n:3*n], expected)

    def test_min(self, vu, regs, scratchpad, hbm):
        n = TILE_SIZE
        scratchpad[0:n] = torch.arange(n, dtype=torch.bfloat16, device=DEVICE)
        scratchpad[n:2*n] = 64.0
        regs[1], regs[2], regs[3] = 2*n, 0, n
        vu.min(regs, scratchpad, hbm, 1, 2, 3)
        expected = torch.min(
            torch.arange(n, dtype=torch.bfloat16, device=DEVICE),
            torch.tensor(64.0, dtype=torch.bfloat16, device=DEVICE).expand(n),
        )
        assert torch.equal(scratchpad[2*n:3*n], expected)


class TestVectorTranscendental:
    def test_exp(self, vu, regs, scratchpad, hbm):
        n = TILE_SIZE
        scratchpad[0:n] = 1.0
        regs[1], regs[2] = n, 0
        vu.exp(regs, scratchpad, hbm, 1, 2)
        expected = torch.exp(torch.tensor(1.0, dtype=torch.bfloat16, device=DEVICE))
        assert torch.allclose(scratchpad[n:2*n], expected.expand(n))

    def test_log(self, vu, regs, scratchpad, hbm):
        n = TILE_SIZE
        scratchpad[0:n] = 2.0
        regs[1], regs[2] = n, 0
        vu.log(regs, scratchpad, hbm, 1, 2)
        expected = torch.log(torch.tensor(2.0, dtype=torch.bfloat16, device=DEVICE))
        assert torch.allclose(scratchpad[n:2*n], expected.expand(n))

    def test_sqrt(self, vu, regs, scratchpad, hbm):
        n = TILE_SIZE
        scratchpad[0:n] = 16.0
        regs[1], regs[2] = n, 0
        vu.sqrt(regs, scratchpad, hbm, 1, 2)
        assert torch.allclose(scratchpad[n:2*n], _expect(4.0))


class TestVectorComparisons:
    def test_eq(self, vu, regs, scratchpad, hbm):
        n = TILE_SIZE
        scratchpad[0:n] = 5.0
        scratchpad[n:2*n] = 5.0
        regs[1], regs[2], regs[3] = 2*n, 0, n
        vu.eq(regs, scratchpad, hbm, 1, 2, 3)
        assert scratchpad[2*n:3*n].all()

    def test_gt(self, vu, regs, scratchpad, hbm):
        n = TILE_SIZE
        scratchpad[0:n] = 10.0
        scratchpad[n:2*n] = 5.0
        regs[1], regs[2], regs[3] = 2*n, 0, n
        vu.gt(regs, scratchpad, hbm, 1, 2, 3)
        assert scratchpad[2*n:3*n].all()

    def test_lt_false(self, vu, regs, scratchpad, hbm):
        n = TILE_SIZE
        scratchpad[0:n] = 10.0
        scratchpad[n:2*n] = 5.0
        regs[1], regs[2], regs[3] = 2*n, 0, n
        vu.lt(regs, scratchpad, hbm, 1, 2, 3)
        assert not scratchpad[2*n:3*n].any()

    def test_neq(self, vu, regs, scratchpad, hbm):
        n = TILE_SIZE
        scratchpad[0:n] = 5.0
        scratchpad[n:2*n] = 6.0
        regs[1], regs[2], regs[3] = 2*n, 0, n
        vu.neq(regs, scratchpad, hbm, 1, 2, 3)
        assert scratchpad[2*n:3*n].all()


class TestVectorUnary:
    def test_neg(self, vu, regs, scratchpad, hbm):
        n = TILE_SIZE
        scratchpad[0:n] = 5.0
        regs[1], regs[2] = n, 0
        vu.neg(regs, scratchpad, hbm, 1, 2)
        assert torch.allclose(scratchpad[n:2*n], _expect(-5.0))

    def test_abs(self, vu, regs, scratchpad, hbm):
        n = TILE_SIZE
        scratchpad[0:n] = -7.0
        regs[1], regs[2] = n, 0
        vu.abs(regs, scratchpad, hbm, 1, 2)
        assert torch.allclose(scratchpad[n:2*n], _expect(7.0))

    def test_reciprocal(self, vu, regs, scratchpad, hbm):
        n = TILE_SIZE
        scratchpad[0:n] = 4.0
        regs[1], regs[2] = n, 0
        vu.reciprocal(regs, scratchpad, hbm, 1, 2)
        assert torch.allclose(scratchpad[n:2*n], _expect(0.25))

    def test_copy(self, vu, regs, scratchpad, hbm):
        n = TILE_SIZE
        data = torch.arange(n, dtype=torch.bfloat16, device=DEVICE)
        scratchpad[0:n] = data
        regs[1], regs[2] = n, 0
        vu.copy(regs, scratchpad, hbm, 1, 2)
        assert torch.equal(scratchpad[n:2*n], data)

    def test_logical_not(self, vu, regs, scratchpad, hbm):
        n = TILE_SIZE
        scratchpad[0:n] = 1.0
        regs[1], regs[2] = n, 0
        vu.logical_not(regs, scratchpad, hbm, 1, 2)
        assert not scratchpad[n:2*n].any()


class TestVectorReductions:
    def test_reduce_sum(self, vu, regs, scratchpad, hbm):
        n = TILE_SIZE
        scratchpad[0:n] = 1.0
        regs[1], regs[2] = n, 0
        vu.reduce_sum(regs, scratchpad, hbm, 1, 2)
        assert scratchpad[n].item() == float(n)

    def test_reduce_max(self, vu, regs, scratchpad, hbm):
        n = TILE_SIZE
        scratchpad[0:n] = torch.arange(n, dtype=torch.bfloat16, device=DEVICE)
        regs[1], regs[2] = n, 0
        vu.reduce_max(regs, scratchpad, hbm, 1, 2)
        assert scratchpad[n].item() == float(n - 1)

    def test_reduce_or(self, vu, regs, scratchpad, hbm):
        n = TILE_SIZE
        scratchpad[0:n] = 0.0
        scratchpad[0] = 1.0
        regs[1], regs[2] = n, 0
        vu.reduce_or(regs, scratchpad, hbm, 1, 2)
        assert scratchpad[n].item()

    def test_reduce_and_false(self, vu, regs, scratchpad, hbm):
        n = TILE_SIZE
        scratchpad[0:n] = 1.0
        scratchpad[0] = 0.0
        regs[1], regs[2] = n, 0
        vu.reduce_and(regs, scratchpad, hbm, 1, 2)
        assert not scratchpad[n].item()

    def test_reduce_and_true(self, vu, regs, scratchpad, hbm):
        n = TILE_SIZE
        scratchpad[0:n] = 1.0
        regs[1], regs[2] = n, 0
        vu.reduce_and(regs, scratchpad, hbm, 1, 2)
        assert scratchpad[n].item()


class TestVectorBroadcastWhereDot:
    def test_vbroadcast(self, vu, regs, scratchpad, hbm):
        n = TILE_SIZE
        scratchpad[0] = 42.0
        regs[1], regs[2] = n, 0
        vu.vbroadcast(regs, scratchpad, hbm, 1, 2)
        assert torch.allclose(scratchpad[n:2*n], _expect(42.0))

    def test_where(self, vu, regs, scratchpad, hbm):
        n = TILE_SIZE
        cond_addr, a_addr, b_addr, dst_addr = 0, n, 2*n, 3*n
        scratchpad[cond_addr:cond_addr+n] = 0.0
        scratchpad[cond_addr] = 1.0  # first element true
        scratchpad[a_addr:a_addr+n] = 10.0
        scratchpad[b_addr:b_addr+n] = 20.0
        regs[1], regs[2], regs[3], regs[4] = dst_addr, cond_addr, a_addr, b_addr
        vu.where(regs, scratchpad, hbm, 1, 2, 3, 4)
        assert scratchpad[dst_addr].item() == 10.0
        assert scratchpad[dst_addr + 1].item() == 20.0

    def test_dot(self, vu, regs, scratchpad, hbm):
        n = TILE_SIZE
        scratchpad[0:n] = 2.0
        scratchpad[n:2*n] = 3.0
        regs[1], regs[2], regs[3] = 2*n, 0, n
        vu.dot(regs, scratchpad, hbm, 1, 2, 3)
        assert scratchpad[2*n].item() == float(n * 6)


class TestVectorCustomLength:
    def test_add_with_short_length(self, vu, regs, scratchpad, hbm):
        n = 4
        regs[4] = n
        vu.set_length(regs, scratchpad, hbm, 4)
        scratchpad[0:n] = 1.0
        scratchpad[TILE_SIZE:TILE_SIZE+n] = 2.0
        regs[1], regs[2], regs[3] = 2*TILE_SIZE, 0, TILE_SIZE
        vu.add(regs, scratchpad, hbm, 1, 2, 3)
        assert torch.allclose(scratchpad[2*TILE_SIZE:2*TILE_SIZE+n], _expect(3.0, n))
        # elements beyond length should be untouched (zero)
        assert scratchpad[2*TILE_SIZE+n].item() == 0.0
