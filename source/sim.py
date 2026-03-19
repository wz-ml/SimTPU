from numpy._core.numeric import bool_
import torch
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from constants import INST_TYPES, DEVICE, SCRATCH_ELEMS, HBM_ELEMS, NUM_REGS, TILE_SIZE, BUNDLE_CYCLES, SLOT_BUDGETS
from functional_units import MXU, ScalarUnit, VectorUnit, TileUnit, DMAUnit

# Main sim file

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

class SimTPU:
    def __init__(self, use_device: bool_ = True):
        self.device = DEVICE if use_device else "cpu"

        # Architectural state (on GPU if possible)
        self.regs = [0] * NUM_REGS
        self.scratchpad = torch.zeros(SCRATCH_ELEMS, dtype=torch.bfloat16, device=self.device)
        self.hbm = torch.zeros(HBM_ELEMS, dtype=torch.bfloat16, device=self.device)
    
        # vector state
        self.vector_length = TILE_SIZE

        # exec state (no branching, so we don't need a pc -> all unrolled)
        self.cycle_count = 0

        # functional units
        self.mxu = MXU()
        self.scalar_unit = ScalarUnit()
        self.vector_unit = VectorUnit()
        self.tile_unit = TileUnit()
        self.dma_unit = DMAUnit()

        self.dispatch_table = {
            "mxu": self.mxu,
            "scalar": self.scalar_unit,
            "vector": self.vector_unit,
            "tile": self.tile_unit,
            "dma": self.dma_unit,
        }

    def run(self, bundles: list[Bundle]):
        for bundle in bundles:
            self.exec_bundle(bundle)
            self.cycle_count += BUNDLE_CYCLES
        return self.cycle_count

    def exec_bundle(self, bundle: Bundle):
        for instr in bundle.instructions:
            self.dispatch(instr)

    def dispatch(self, instr: Instr):
        kwargs = {
            "regs": self.regs,
            "scratchpad": self.scratchpad,
            "hbm": self.hbm,
        }
        kwargs.update({f"r{i}": self.regs[arg] for i, arg in enumerate(instr.args)})
        # this is ugly and bad practice but works
        # sid fix it if you want to
        fu_method = getattr(self.dispatch_table[instr.unit], instr.op)
        fu_method(**kwargs)