from numpy._core.numeric import bool_
import torch
from dataclasses import dataclass, field
from collections import Counter, defaultdict

# Main file, constants live here

# consts
TILE_SIZE = 128
TILE_ELEMS = TILE_SIZE * TILE_SIZE
SCRATCH_ELEMS = 128 * 1024 # 256KB, 2 bytes per elem
HBM_ELEMS = 64 * 1024 * 1024 # 128MB, 2 bytes per elem
NUM_REGS = 32

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# cycle budgets per bundle
SLOT_BUDGETS = {
    "mxu": 128,
    "tile": 128,
    "dma": 128,
    "scalar": 128,
    "vector": 128,
}
BUNDLE_CYCLES = max(SLOT_BUDGETS.values())
INST_TYPES = SLOT_BUDGETS.keys()

@dataclass
class Instr:
    unit: str  #  fu type: "mxu", "tile", "dma", "scalar", "vector"
    op: str     #  op type: "add", "sub")
    args: tuple   # (r1, r2, r3)
    cycles: int

@dataclass
class Bundle:
    instructions: list[Instr] = field(default_factory=list)
    id: int = -1 # for debugging
    def verify(self):
        for instr in self.instructions:
            assert instr.unit in INST_TYPES
        cycle_counter = Counter()
        for inst in self.instructions:
            cycle_counter[inst.unit] += inst.cycles
        for unit, cycles in cycle_counter.items():
            assert cycles <= SLOT_BUDGETS[unit], f"Bundle has {cycles} cycles for {unit}, > max allowed{SLOT_BUDGETS[unit]}"
        return True

class Bundler:
    # abstract base class for bundlers
    def __init__(self):
        pass
    def __call__(self, instructions: list[Instr]) -> list[Bundle]:
        raise NotImplementedError()
    def verify(self, bundles: list[Bundle]) -> bool:
        for bundle in bundles:
            bundle.verify()
        return True

class ProgramBuilder:
    def __init__(self):
        self.instructions = []
    
    # MXU
    def mfma_load_weights(self, r1):
        self.instructions.append(Instr("mxu", "load_weights", (r1,), 128))
    def mfma_matmul(self, r2, r3):
        self.instructions.append(Instr("mxu", "matmul", (r2, r3), 128))
    def mfma_store(self, r4):
        self.instructions.append(Instr("mxu", "store", (r4,), 128))

    # Tile
    def tu_transpose(self, r1):
        self.instructions.append(Instr("tile", "transpose", (r1,), 128))

    # DMA
    def dma_load(self, r1, r2, r3):
        self.instructions.append(Instr("dma", "load", (r1, r2, r3), 64))
    def dma_store(self, r1, r2, r3):
        self.instructions.append(Instr("dma", "store", (r1, r2, r3), 64))

    # Scalar
    def s_load_imm(self, r1, val):
        self.instructions.append(Instr("scalar", "load_imm", (r1, val), 1))
    def s_add(self, r1, r2, r3):
        self.instructions.append(Instr("scalar", "add", (r1, r2, r3), 1))
    def s_sub(self, r1, r2, r3):
        self.instructions.append(Instr("scalar", "sub", (r1, r2, r3), 1))
    def s_mult(self, r1, r2, r3):
        self.instructions.append(Instr("scalar", "mult", (r1, r2, r3), 1))
    def s_div(self, r1, r2, r3):
        self.instructions.append(Instr("scalar", "div", (r1, r2, r3), 1))
    def s_mod(self, r1, r2, r3):
        self.instructions.append(Instr("scalar", "mod", (r1, r2, r3), 1))
    # Logic ops
    def s_not(self, r1, r2):
        self.instructions.append(Instr("scalar", "not", (r1, r2), 1))
    def s_ge(self, r1, r2, r3):
        self.instructions.append(Instr("scalar", "ge", (r1, r2, r3), 1))
    def s_gt(self, r1, r2, r3):
        self.instructions.append(Instr("scalar", "gt", (r1, r2, r3), 1))
    def s_eq(self, r1, r2, r3):
        self.instructions.append(Instr("scalar", "eq", (r1, r2, r3), 1))
    def s_lt(self, r1, r2, r3):
        self.instructions.append(Instr("scalar", "lt", (r1, r2, r3), 1))
    def s_le(self, r1, r2, r3):
        self.instructions.append(Instr("scalar", "le", (r1, r2, r3), 1))
    def s_neq(self, r1, r2, r3):
        self.instructions.append(Instr("scalar", "neq", (r1, r2, r3), 1))

    # Vector
    def v_set_length(self, r4):
        self.instructions.append(Instr("vector", "set_length", (r4,), 0))  # free
    # reductions
    def v_reduce_sum(self, r1, r2):
        self.instructions.append(Instr("vector", "reduce_sum", (r1, r2), 8))
    def v_reduce_max(self, r1, r2):
        self.instructions.append(Instr("vector", "reduce_max", (r1, r2), 8))
    def v_reduce_or(self, r1, r2):
        self.instructions.append(Instr("vector", "reduce_or", (r1, r2), 8))
    def v_reduce_and(self, r1, r2):
        self.instructions.append(Instr("vector", "reduce_and", (r1, r2), 8))
    def v_vbroadcast(self, r1, r2):
        self.instructions.append(Instr("vector", "vbroadcast", (r1, r2), 1))
    def v_where(self, r1, r2, r3, r4):
        self.instructions.append(Instr("vector", "where", (r1, r2, r3, r4), 1))
    # elementwise ops
    def v_mult(self, r1, r2, r3):
        self.instructions.append(Instr("vector", "mult", (r1, r2, r3), 1))
    def v_div(self, r1, r2, r3):
        self.instructions.append(Instr("vector", "div", (r1, r2, r3), 1))
    def v_add(self, r1, r2, r3):
        self.instructions.append(Instr("vector", "add", (r1, r2, r3), 1))
    def v_sub(self, r1, r2, r3):
        self.instructions.append(Instr("vector", "sub", (r1, r2, r3), 1))
    def v_pow(self, r1, r2, r3):
        self.instructions.append(Instr("vector", "pow", (r1, r2, r3), 1))
    def v_max(self, r1, r2, r3):
        self.instructions.append(Instr("vector", "max", (r1, r2, r3), 1))
    # comparison
    def v_eq(self, r1, r2, r3):
        self.instructions.append(Instr("vector", "eq", (r1, r2, r3), 1))
    def v_ge(self, r1, r2, r3):
        self.instructions.append(Instr("vector", "ge", (r1, r2, r3), 1))
    def v_gt(self, r1, r2, r3):
        self.instructions.append(Instr("vector", "gt", (r1, r2, r3), 1))
    def v_lt(self, r1, r2, r3):
        self.instructions.append(Instr("vector", "lt", (r1, r2, r3), 1))
    def v_le(self, r1, r2, r3):
        self.instructions.append(Instr("vector", "le", (r1, r2, r3), 1))
    def v_neq(self, r1, r2, r3):
        self.instructions.append(Instr("vector", "neq", (r1, r2, r3), 1))
    # unary ops
    def v_not(self, r1, r2):
        self.instructions.append(Instr("vector", "not", (r1, r2), 1))
    def v_reciprocal(self, r1, r2):
        self.instructions.append(Instr("vector", "reciprocal", (r1, r2), 1))
    def v_neg(self, r1, r2):
        self.instructions.append(Instr("vector", "neg", (r1, r2), 1))
    def v_abs(self, r1, r2):
        self.instructions.append(Instr("vector", "abs", (r1, r2), 1))
    def v_copy(self, r1, r2):
        self.instructions.append(Instr("vector", "copy", (r1, r2), 1))
    # transcendental ops
    def v_exp(self, r1, r2):
        self.instructions.append(Instr("vector", "exp", (r1, r2), 8))
    def v_log(self, r1, r2):
        self.instructions.append(Instr("vector", "log", (r1, r2), 8))
    def v_sqrt(self, r1, r2):
        self.instructions.append(Instr("vector", "sqrt", (r1, r2), 8))
    # 3-register reduction
    def v_dot(self, r1, r2, r3):
        self.instructions.append(Instr("vector", "dot", (r1, r2, r3), 8))

    def build(self, bundler: Bundler):
        bundles = bundler(self.instructions)
        bundler.verify(bundles)
        return bundles

class SimTPU:
    def __init__(self, use_device: bool_ = True):
        self.device = DEVICE if use_device else "cpu"

        # Architectural state (on GPU if possible)
        self.regs = [0] * NUM_REGS
        self.scratchpad = torch.zeros(SCRATCH_ELEMS, dtype=torch.bfloat16, device=self.device)
        self.hbm = torch.zeros(HBM_ELEMS, dtype=torch.bfloat16, device=self.device)
        
        # MXU state
        self.mxu_weights = torch.zeros(TILE_SIZE, TILE_SIZE, dtype=torch.bfloat16, device=self.device)
        self.mxu_acc = torch.zeros(TILE_SIZE, TILE_SIZE, dtype=torch.bfloat16, device=self.device)

        # vector state
        self.vector_length = TILE_SIZE

        # exec state (no branching, so no need for pc -> all linear)
        self.cycle_count = 0

    def run(self, bundles: list[Bundle]):
        for bundle in bundles:
            self.exec_bundle(bundle)
            self.cycle_count += BUNDLE_CYCLES
        return self.cycle_count

    def exec_bundle(self, bundle: Bundle):
        for instr in bundle.instructions:
            