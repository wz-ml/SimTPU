import os
import torch

# consts
TILE_SIZE = int(os.getenv("SIMTPU_TILE_SIZE", 128))
TILE_ELEMS = TILE_SIZE * TILE_SIZE
SCRATCH_ELEMS = 128 * 1024 # 256KB, 2 bytes per elem
HBM_ELEMS = 64 * 1024 * 1024 # 128MB, 2 bytes per elem
NUM_REGS = int(os.getenv("SIMTPU_NUM_REGS", 32))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BUNDLE_CYCLES = int(os.getenv("SIMTPU_BUNDLE_CYCLES", 128))

# cycle budgets per bundle
SLOT_BUDGETS = {
    "mxu": int(os.getenv("SIMTPU_MXU_BUDGET", BUNDLE_CYCLES)),
    "tile": int(os.getenv("SIMTPU_TILE_BUDGET", BUNDLE_CYCLES)),
    "dma": int(os.getenv("SIMTPU_DMA_BUDGET", BUNDLE_CYCLES)),
    "scalar": int(os.getenv("SIMTPU_SCALAR_BUDGET", BUNDLE_CYCLES)),
    "vector": int(os.getenv("SIMTPU_VECTOR_BUDGET", BUNDLE_CYCLES)),
}
INST_TYPES = SLOT_BUDGETS.keys()
