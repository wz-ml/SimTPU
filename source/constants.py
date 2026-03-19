import torch

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