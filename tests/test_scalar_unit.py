import torch
import pytest
from source.functional_units import ScalarUnit


@pytest.fixture
def su():
    return ScalarUnit()


class TestLoadImm:
    def test_sets_register(self, su, regs, scratchpad, hbm):
        su.load_imm(regs, scratchpad, hbm, 5, 42)
        assert regs[5] == 42

    def test_r0_stays_zero(self, su, regs, scratchpad, hbm):
        su.load_imm(regs, scratchpad, hbm, 0, 99)
        assert regs[0] == 0

    def test_negative_value(self, su, regs, scratchpad, hbm):
        su.load_imm(regs, scratchpad, hbm, 1, -10)
        assert regs[1] == -10


class TestScalarLoadStore:
    def test_store_and_load(self, su, regs, scratchpad, hbm):
        regs[1] = 0   # addr
        regs[2] = 42  # value
        su.store(regs, scratchpad, hbm, 1, 2)
        assert scratchpad[0].item() == 42

        su.load(regs, scratchpad, hbm, 3, 1)
        assert regs[3] == 42


class TestScalarArithmetic:
    def test_add(self, su, regs, scratchpad, hbm):
        regs[1], regs[2] = 10, 20
        su.add(regs, scratchpad, hbm, 3, 1, 2)
        assert regs[3] == 30

    def test_sub(self, su, regs, scratchpad, hbm):
        regs[1], regs[2] = 20, 7
        su.sub(regs, scratchpad, hbm, 3, 1, 2)
        assert regs[3] == 13

    def test_mult(self, su, regs, scratchpad, hbm):
        regs[1], regs[2] = 6, 7
        su.mult(regs, scratchpad, hbm, 3, 1, 2)
        assert regs[3] == 42

    def test_div(self, su, regs, scratchpad, hbm):
        regs[1], regs[2] = 20, 3
        su.div(regs, scratchpad, hbm, 3, 1, 2)
        assert regs[3] == 6  # integer division

    def test_mod(self, su, regs, scratchpad, hbm):
        regs[1], regs[2] = 20, 3
        su.mod(regs, scratchpad, hbm, 3, 1, 2)
        assert regs[3] == 2


class TestScalarComparisons:
    def test_ge_true(self, su, regs, scratchpad, hbm):
        regs[1], regs[2] = 5, 5
        su.ge(regs, scratchpad, hbm, 3, 1, 2)
        assert regs[3] == 1

    def test_ge_false(self, su, regs, scratchpad, hbm):
        regs[1], regs[2] = 4, 5
        su.ge(regs, scratchpad, hbm, 3, 1, 2)
        assert regs[3] == 0

    def test_gt(self, su, regs, scratchpad, hbm):
        regs[1], regs[2] = 6, 5
        su.gt(regs, scratchpad, hbm, 3, 1, 2)
        assert regs[3] == 1

    def test_eq(self, su, regs, scratchpad, hbm):
        regs[1], regs[2] = 5, 5
        su.eq(regs, scratchpad, hbm, 3, 1, 2)
        assert regs[3] == 1

    def test_lt(self, su, regs, scratchpad, hbm):
        regs[1], regs[2] = 3, 5
        su.lt(regs, scratchpad, hbm, 3, 1, 2)
        assert regs[3] == 1

    def test_le(self, su, regs, scratchpad, hbm):
        regs[1], regs[2] = 5, 5
        su.le(regs, scratchpad, hbm, 3, 1, 2)
        assert regs[3] == 1

    def test_neq(self, su, regs, scratchpad, hbm):
        regs[1], regs[2] = 5, 6
        su.neq(regs, scratchpad, hbm, 3, 1, 2)
        assert regs[3] == 1


class TestScalarLogicalNot:
    def test_not_zero(self, su, regs, scratchpad, hbm):
        regs[2] = 0
        su.logical_not(regs, scratchpad, hbm, 1, 2)
        assert regs[1] == ~0

    def test_not_one(self, su, regs, scratchpad, hbm):
        regs[2] = 1
        su.logical_not(regs, scratchpad, hbm, 1, 2)
        assert regs[1] == ~1
