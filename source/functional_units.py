import torch
from constants import TILE_SIZE, TILE_ELEMS, DEVICE

class MXU:
    def __init__(self):
        self.device = DEVICE
        self.mxu_weights = torch.zeros(TILE_SIZE, TILE_SIZE, dtype=torch.bfloat16, device=self.device)
        self.mxu_acc = torch.zeros(TILE_SIZE, TILE_SIZE, dtype=torch.bfloat16, device=self.device)

    def load_weights(self, regs, scratchpad, r1):
        addr = regs[r1]
        self.mxu_weights = scratchpad[addr:addr+TILE_ELEMS].reshape(TILE_SIZE, TILE_SIZE).clone()

    def matmul(self, regs, scratchpad, r1, r2):
        """
        mfma.matmul r2, r3 # stream activation tile from scratchpad[r2:r2+2**14] into systolic array and performs matmul.
        r3 is a boolean for if the matmul accumulates onto previously computed results or overwrites them (1 for acc, 0 for overwrite).
        """
        addr, acc = regs[r1], regs[r2]
        acts = scratchpad[addr:addr+TILE_ELEMS].reshape(TILE_SIZE, TILE_SIZE)
        result = acts @ self.mxu_weights
        self.mxu_acc = (self.mxu_acc + result) if acc else result

    def store(self, regs, scratchpad, r1):
        addr = regs[r1]
        scratchpad[addr:addr+TILE_ELEMS] = self.mxu_acc.flatten()

class ScalarUnit:
    def __init__(self):
        self.device = DEVICE
    def load_imm(self, regs, scratchpad, r1, val):
        regs[r1] = val
        regs[0] = 0
    def load(self, regs, scratchpad, r1, r2):
        addr = regs[r2]
        regs[r1] = scratchpad[addr]
    def store(self, regs, scratchpad, r1, r2):
        addr = regs[r1]
        scratchpad[addr] = regs[r2]
    def add(self, regs, scratchpad, r1, r2, r3):
        regs[r1] = regs[r2] + regs[r3]
    def sub(self, regs, scratchpad, r1, r2, r3):
        regs[r1] = regs[r2] - regs[r3]
    def mult(self, regs, scratchpad, r1, r2, r3):
        regs[r1] = regs[r2] * regs[r3]
    def div(self, regs, scratchpad, r1, r2, r3):
        regs[r1] = regs[r2] / regs[r3]
    def mod(self, regs, scratchpad, r1, r2, r3):
        regs[r1] = regs[r2] % regs[r3]
    def logical_not(self, regs, scratchpad, r1, r2):
        regs[r1] = ~regs[r2]
    def ge(self, regs, scratchpad, r1, r2, r3):
        regs[r1] = regs[r2] >= regs[r3]
    def gt(self, regs, scratchpad, r1, r2, r3):
        regs[r1] = regs[r2] > regs[r3]
    def eq(self, regs, scratchpad, r1, r2, r3):
        regs[r1] = regs[r2] == regs[r3]
    def lt(self, regs, scratchpad, r1, r2, r3):
        regs[r1] = regs[r2] < regs[r3]
    def le(self, regs, scratchpad, r1, r2, r3):
        regs[r1] = regs[r2] <= regs[r3]
    def neq(self, regs, scratchpad, r1, r2, r3):
        regs[r1] = regs[r2] != regs[r3]

class VectorUnit:
    def __init__(self):
        self.vector_length = TILE_SIZE
    
    def set_length(self, regs, scratchpad, r1):
        self.vector_length = min(regs[r1], TILE_SIZE)
    # elementwise
    def mult(self, scratchpad, r1, r2, r3):
        scratchpad[r1:r1+self.vector_length] = scratchpad[r2:r2+self.vector_length] * scratchpad[r3:r3+self.vector_length]
    def div(self, scratchpad, r1, r2, r3):
        scratchpad[r1:r1+self.vector_length] = scratchpad[r2:r2+self.vector_length] / scratchpad[r3:r3+self.vector_length]
    def add(self, scratchpad, r1, r2, r3):
        scratchpad[r1:r1+self.vector_length] = scratchpad[r2:r2+self.vector_length] + scratchpad[r3:r3+self.vector_length]
    def sub(self, scratchpad, r1, r2, r3):
        scratchpad[r1:r1+self.vector_length] = scratchpad[r2:r2+self.vector_length] - scratchpad[r3:r3+self.vector_length]
    def pow(self, scratchpad, r1, r2, r3):
        scratchpad[r1:r1+self.vector_length] = scratchpad[r2:r2+self.vector_length] ** scratchpad[r3:r3+self.vector_length]
    # transcendental
    def exp(self, scratchpad, r1, r2):
        scratchpad[r1:r1+self.vector_length] = torch.exp(scratchpad[r2:r2+self.vector_length])
    def log(self, scratchpad, r1, r2):
        scratchpad[r1:r1+self.vector_length] = torch.log(scratchpad[r2:r2+self.vector_length])
    def sqrt(self, scratchpad, r1, r2):
        scratchpad[r1:r1+self.vector_length] = torch.sqrt(scratchpad[r2:r2+self.vector_length])
    def max(self, scratchpad, r1, r2, r3):
        scratchpad[r1:r1+self.vector_length] = torch.max(scratchpad[r2:r2+self.vector_length], scratchpad[r3:r3+self.vector_length])
    def min(self, scratchpad, r1, r2, r3):
        scratchpad[r1:r1+self.vector_length] = torch.min(scratchpad[r2:r2+self.vector_length], scratchpad[r3:r3+self.vector_length])
    # binary mask comparison ops
    def eq(self, scratchpad, r1, r2, r3):
        scratchpad[r1:r1+self.vector_length] = scratchpad[r2:r2+self.vector_length] == scratchpad[r3:r3+self.vector_length]
    def ge(self, scratchpad, r1, r2, r3):
        scratchpad[r1:r1+self.vector_length] = scratchpad[r2:r2+self.vector_length] >= scratchpad[r3:r3+self.vector_length]
    def gt(self, scratchpad, r1, r2, r3):
        scratchpad[r1:r1+self.vector_length] = scratchpad[r2:r2+self.vector_length] > scratchpad[r3:r3+self.vector_length]
    def lt(self, scratchpad, r1, r2, r3):
        scratchpad[r1:r1+self.vector_length] = scratchpad[r2:r2+self.vector_length] < scratchpad[r3:r3+self.vector_length]
    def le(self, scratchpad, r1, r2, r3):
        scratchpad[r1:r1+self.vector_length] = scratchpad[r2:r2+self.vector_length] <= scratchpad[r3:r3+self.vector_length]
    def neq(self, scratchpad, r1, r2, r3):
        scratchpad[r1:r1+self.vector_length] = scratchpad[r2:r2+self.vector_length] != scratchpad[r3:r3+self.vector_length]
    # unary ops
    def logical_not(self, scratchpad, r1, r2):
        scratchpad[r1:r1+self.vector_length] = ~scratchpad[r2:r2+self.vector_length]
    def reciprocal(self, scratchpad, r1, r2):
        scratchpad[r1:r1+self.vector_length] = 1 / scratchpad[r2:r2+self.vector_length]
    def neg(self, scratchpad, r1, r2):
        scratchpad[r1:r1+self.vector_length] = -scratchpad[r2:r2+self.vector_length]
    def abs(self, scratchpad, r1, r2):
        scratchpad[r1:r1+self.vector_length] = torch.abs(scratchpad[r2:r2+self.vector_length])
    def copy(self, scratchpad, r1, r2):
        scratchpad[r1:r1+self.vector_length] = scratchpad[r2:r2+self.vector_length]
    # reduction ops
    def reduce_sum(self, scratchpad, r1, r2):
        scratchpad[r1] = torch.sum(scratchpad[r2:r2+self.vector_length])
    def reduce_max(self, scratchpad, r1, r2):
        scratchpad[r1] = torch.max(scratchpad[r2:r2+self.vector_length])
    def reduce_or(self, scratchpad, r1, r2):
        scratchpad[r1] = torch.logical_or(scratchpad[r2:r2+self.vector_length])
    def reduce_and(self, scratchpad, r1, r2):
        scratchpad[r1] = torch.logical_and(scratchpad[r2:r2+self.vector_length])
    def vbroadcast(self, scratchpad, r1, r2):
        scratchpad[r1:r1+self.vector_length] = scratchpad[r2]
    def where(self, scratchpad, r1, r2, r3, r4):
        scratchpad[r1:r1+self.vector_length] = torch.where(
                scratchpad[r2:r2+self.vector_length], 
                scratchpad[r3:r3+self.vector_length], 
                scratchpad[r4:r4+self.vector_length])

class TileUnit:
    def __init__(self):
        self.device = DEVICE
    def transpose(self, regs, scratchpad, r1):
        addr = regs[r1]
        scratchpad[addr:addr+TILE_ELEMS] = scratchpad[addr:addr+TILE_ELEMS].reshape(TILE_SIZE, TILE_SIZE).T.flatten()

class DMAUnit:
    def __init__(self):
        self.device = DEVICE
    def load(self, regs, scratchpad, hbm, r1, r2, r3):
        addr = regs[r1]
        size = min(regs[r3], TILE_ELEMS)
        scratchpad[r2:r2+size] = hbm[addr:addr+size]
    def store(self, regs, scratchpad, hbm, r1, r2, r3):
        addr = regs[r1]
        size = min(regs[r3], TILE_ELEMS)
        hbm[addr:addr+size] = scratchpad[r2:r2+size]