import torch
from source.constants import TILE_SIZE, TILE_ELEMS, DEVICE

class MXU:
    def __init__(self):
        self.device = DEVICE
        self.mxu_weights = torch.zeros(TILE_SIZE, TILE_SIZE, dtype=torch.bfloat16, device=self.device)
        self.mxu_acc = torch.zeros(TILE_SIZE, TILE_SIZE, dtype=torch.bfloat16, device=self.device)

    def load_weights(self, regs, scratchpad, hbm, r1):
        addr = regs[r1]
        self.mxu_weights = scratchpad[addr:addr+TILE_ELEMS].reshape(TILE_SIZE, TILE_SIZE).clone()

    def matmul(self, regs, scratchpad, hbm, r1, r2):
        """
        mfma.matmul r2, r3 # stream activation tile from scratchpad[r2:r2+2**14] into systolic array and performs matmul.
        r3 is a boolean for if the matmul accumulates onto previously computed results or overwrites them (1 for acc, 0 for overwrite).
        """
        addr, acc = regs[r1], regs[r2]
        acts = scratchpad[addr:addr+TILE_ELEMS].reshape(TILE_SIZE, TILE_SIZE)
        result = acts @ self.mxu_weights
        self.mxu_acc = (self.mxu_acc + result) if acc else result

    def store(self, regs, scratchpad, hbm, r1):
        addr = regs[r1]
        scratchpad[addr:addr+TILE_ELEMS] = self.mxu_acc.flatten()

class ScalarUnit:
    def __init__(self):
        self.device = DEVICE
    def load_imm(self, regs, scratchpad, hbm, r1, val):
        regs[r1] = val
        regs[0] = 0
    # scratchpad ops
    def load(self, regs, scratchpad, hbm, r1, r2):
        addr = regs[r2]
        regs[r1] = int(scratchpad[addr].item())
    def store(self, regs, scratchpad, hbm, r1, r2):
        addr = regs[r1]
        scratchpad[addr] = regs[r2]

    def add(self, regs, scratchpad, hbm, r1, r2, r3):
        regs[r1] = regs[r2] + regs[r3]
    def sub(self, regs, scratchpad, hbm, r1, r2, r3):
        regs[r1] = regs[r2] - regs[r3]
    def mult(self, regs, scratchpad, hbm, r1, r2, r3):
        regs[r1] = regs[r2] * regs[r3]
    def div(self, regs, scratchpad, hbm, r1, r2, r3):
        regs[r1] = regs[r2] // regs[r3]
    def mod(self, regs, scratchpad, hbm, r1, r2, r3):
        regs[r1] = regs[r2] % regs[r3]
    def logical_not(self, regs, scratchpad, hbm, r1, r2):
        regs[r1] = ~regs[r2]
    def ge(self, regs, scratchpad, hbm, r1, r2, r3):
        regs[r1] = int(regs[r2] >= regs[r3])
    def gt(self, regs, scratchpad, hbm, r1, r2, r3):
        regs[r1] = int(regs[r2] > regs[r3])
    def eq(self, regs, scratchpad, hbm, r1, r2, r3):
        regs[r1] = int(regs[r2] == regs[r3])
    def lt(self, regs, scratchpad, hbm, r1, r2, r3):
        regs[r1] = int(regs[r2] < regs[r3])
    def le(self, regs, scratchpad, hbm, r1, r2, r3):
        regs[r1] = int(regs[r2] <= regs[r3])
    def neq(self, regs, scratchpad, hbm, r1, r2, r3):
        regs[r1] = int(regs[r2] != regs[r3])

class VectorUnit:
    def __init__(self):
        self.vector_length = TILE_SIZE
    
    def set_length(self, regs, scratchpad, hbm, r1):
        self.vector_length = min(regs[r1], TILE_SIZE)
    # elementwise
    def mult(self, regs, scratchpad, hbm, r1, r2, r3):
        addr1, addr2, addr3 = regs[r1], regs[r2], regs[r3]
        scratchpad[addr1:addr1+self.vector_length] = scratchpad[addr2:addr2+self.vector_length] * scratchpad[addr3:addr3+self.vector_length]
    def div(self, regs, scratchpad, hbm, r1, r2, r3):
        addr1, addr2, addr3 = regs[r1], regs[r2], regs[r3]
        scratchpad[addr1:addr1+self.vector_length] = scratchpad[addr2:addr2+self.vector_length] / scratchpad[addr3:addr3+self.vector_length]
    def add(self, regs, scratchpad, hbm, r1, r2, r3):
        addr1, addr2, addr3 = regs[r1], regs[r2], regs[r3]
        scratchpad[addr1:addr1+self.vector_length] = scratchpad[addr2:addr2+self.vector_length] + scratchpad[addr3:addr3+self.vector_length]
    def sub(self, regs, scratchpad, hbm, r1, r2, r3):
        addr1, addr2, addr3 = regs[r1], regs[r2], regs[r3]
        scratchpad[addr1:addr1+self.vector_length] = scratchpad[addr2:addr2+self.vector_length] - scratchpad[addr3:addr3+self.vector_length]
    def pow(self, regs, scratchpad, hbm, r1, r2, r3):
        addr1, addr2, addr3 = regs[r1], regs[r2], regs[r3]
        scratchpad[addr1:addr1+self.vector_length] = scratchpad[addr2:addr2+self.vector_length] ** scratchpad[addr3:addr3+self.vector_length]
    # transcendental
    def exp(self, regs, scratchpad, hbm, r1, r2):
        addr1, addr2 = regs[r1], regs[r2]
        scratchpad[addr1:addr1+self.vector_length] = torch.exp(scratchpad[addr2:addr2+self.vector_length])
    def log(self, regs, scratchpad, hbm, r1, r2):
        addr1, addr2 = regs[r1], regs[r2]
        scratchpad[addr1:addr1+self.vector_length] = torch.log(scratchpad[addr2:addr2+self.vector_length])
    def sqrt(self, regs, scratchpad, hbm, r1, r2):
        addr1, addr2 = regs[r1], regs[r2]
        scratchpad[addr1:addr1+self.vector_length] = torch.sqrt(scratchpad[addr2:addr2+self.vector_length])
    def max(self, regs, scratchpad, hbm, r1, r2, r3):
        addr1, addr2, addr3 = regs[r1], regs[r2], regs[r3]
        scratchpad[addr1:addr1+self.vector_length] = torch.max(scratchpad[addr2:addr2+self.vector_length], scratchpad[addr3:addr3+self.vector_length])
    def min(self, regs, scratchpad, hbm, r1, r2, r3):
        addr1, addr2, addr3 = regs[r1], regs[r2], regs[r3]
        scratchpad[addr1:addr1+self.vector_length] = torch.min(scratchpad[addr2:addr2+self.vector_length], scratchpad[addr3:addr3+self.vector_length])
    # binary mask comparison ops
    def eq(self, regs, scratchpad, hbm, r1, r2, r3):
        addr1, addr2, addr3 = regs[r1], regs[r2], regs[r3]
        scratchpad[addr1:addr1+self.vector_length] = scratchpad[addr2:addr2+self.vector_length] == scratchpad[addr3:addr3+self.vector_length]
    def ge(self, regs, scratchpad, hbm, r1, r2, r3):
        addr1, addr2, addr3 = regs[r1], regs[r2], regs[r3]
        scratchpad[addr1:addr1+self.vector_length] = scratchpad[addr2:addr2+self.vector_length] >= scratchpad[addr3:addr3+self.vector_length]
    def gt(self, regs, scratchpad, hbm, r1, r2, r3):
        addr1, addr2, addr3 = regs[r1], regs[r2], regs[r3]
        scratchpad[addr1:addr1+self.vector_length] = scratchpad[addr2:addr2+self.vector_length] > scratchpad[addr3:addr3+self.vector_length]
    def lt(self, regs, scratchpad, hbm, r1, r2, r3):
        addr1, addr2, addr3 = regs[r1], regs[r2], regs[r3]
        scratchpad[addr1:addr1+self.vector_length] = scratchpad[addr2:addr2+self.vector_length] < scratchpad[addr3:addr3+self.vector_length]
    def le(self, regs, scratchpad, hbm, r1, r2, r3):
        addr1, addr2, addr3 = regs[r1], regs[r2], regs[r3]
        scratchpad[addr1:addr1+self.vector_length] = scratchpad[addr2:addr2+self.vector_length] <= scratchpad[addr3:addr3+self.vector_length]
    def neq(self, regs, scratchpad, hbm, r1, r2, r3):
        addr1, addr2, addr3 = regs[r1], regs[r2], regs[r3]
        scratchpad[addr1:addr1+self.vector_length] = scratchpad[addr2:addr2+self.vector_length] != scratchpad[addr3:addr3+self.vector_length]
    # unary ops
    def logical_not(self, regs, scratchpad, hbm, r1, r2):
        addr1, addr2 = regs[r1], regs[r2]
        scratchpad[addr1:addr1+self.vector_length] = torch.logical_not(scratchpad[addr2:addr2+self.vector_length]).to(torch.bfloat16)
    def reciprocal(self, regs, scratchpad, hbm, r1, r2):
        addr1, addr2 = regs[r1], regs[r2]
        scratchpad[addr1:addr1+self.vector_length] = 1 / scratchpad[addr2:addr2+self.vector_length]
    def neg(self, regs, scratchpad, hbm, r1, r2):
        addr1, addr2 = regs[r1], regs[r2]
        scratchpad[addr1:addr1+self.vector_length] = -scratchpad[addr2:addr2+self.vector_length]
    def abs(self, regs, scratchpad, hbm, r1, r2):
        addr1, addr2 = regs[r1], regs[r2]
        scratchpad[addr1:addr1+self.vector_length] = torch.abs(scratchpad[addr2:addr2+self.vector_length])
    def copy(self, regs, scratchpad, hbm, r1, r2):
        addr1, addr2 = regs[r1], regs[r2]
        scratchpad[addr1:addr1+self.vector_length] = scratchpad[addr2:addr2+self.vector_length].clone()
    # reduction ops
    def reduce_sum(self, regs, scratchpad, hbm, r1, r2):
        addr1, addr2 = regs[r1], regs[r2]
        scratchpad[addr1] = torch.sum(scratchpad[addr2:addr2+self.vector_length])
    def reduce_max(self, regs, scratchpad, hbm, r1, r2):
        addr1, addr2 = regs[r1], regs[r2]
        scratchpad[addr1] = torch.max(scratchpad[addr2:addr2+self.vector_length])
    def reduce_or(self, regs, scratchpad, hbm, r1, r2):
        addr1, addr2 = regs[r1], regs[r2]
        scratchpad[addr1] = torch.any(scratchpad[addr2:addr2+self.vector_length])
    def reduce_and(self, regs, scratchpad, hbm, r1, r2):
        addr1, addr2 = regs[r1], regs[r2]
        scratchpad[addr1] = torch.all(scratchpad[addr2:addr2+self.vector_length])
    def vbroadcast(self, regs, scratchpad, hbm, r1, r2):
        addr1, addr2 = regs[r1], regs[r2]
        scratchpad[addr1:addr1+self.vector_length] = scratchpad[addr2]
    def where(self, regs, scratchpad, hbm, r1, r2, r3, r4):
        addr1, addr2, addr3, addr4 = regs[r1], regs[r2], regs[r3], regs[r4]
        scratchpad[addr1:addr1+self.vector_length] = torch.where(
                scratchpad[addr2:addr2+self.vector_length].bool(), 
                scratchpad[addr3:addr3+self.vector_length], 
                scratchpad[addr4:addr4+self.vector_length])
    def dot(self, regs, scratchpad, hbm, r1, r2, r3):
        addr1, addr2, addr3 = regs[r1], regs[r2], regs[r3]
        scratchpad[addr1] = torch.dot(scratchpad[addr2:addr2+self.vector_length], scratchpad[addr3:addr3+self.vector_length])

class TileUnit:
    def __init__(self):
        self.device = DEVICE
    def transpose(self, regs, scratchpad, hbm, r1):
        addr = regs[r1]
        scratchpad[addr:addr+TILE_ELEMS] = scratchpad[addr:addr+TILE_ELEMS].reshape(TILE_SIZE, TILE_SIZE).T.flatten()

class DMAUnit:
    def __init__(self):
        self.device = DEVICE
    def load(self, regs, scratchpad, hbm, r1, r2, r3):
        src, dest = regs[r1], regs[r2]
        size = min(regs[r3], TILE_ELEMS)
        scratchpad[dest:dest+size] = hbm[src:src+size]
    def store(self, regs, scratchpad, hbm, r1, r2, r3):
        src, dest = regs[r1], regs[r2]
        size = min(regs[r3], TILE_ELEMS)
        hbm[dest:dest+size] = scratchpad[src:src+size]