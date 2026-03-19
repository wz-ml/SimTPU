from source.builder import ProgramBuilder
from source.bundlers import OneBundleOneInstructionBundler
from source.constants import TILE_ELEMS, TILE_SIZE
from source.sim import SimTPU
import torch


def _load_tile(builder, hbm_ptr, scratch_ptr, rows, stride):
    for row in range(rows):
        builder.s_load_imm(1, hbm_ptr + row * stride)
        builder.s_load_imm(2, scratch_ptr + row * TILE_SIZE)
        builder.s_load_imm(3, TILE_SIZE)
        builder.dma_load(1, 2, 3)


def _store_tile(builder, scratch_ptr, hbm_ptr, rows, stride):
    for row in range(rows):
        builder.s_load_imm(1, scratch_ptr + row * TILE_SIZE)
        builder.s_load_imm(2, hbm_ptr + row * stride)
        builder.s_load_imm(3, TILE_SIZE)
        builder.dma_store(1, 2, 3)


def setup(tpu, M, N, K):
    A_ptr = 0; A_elems = M * K
    B_ptr = A_elems; B_elems = K * N
    C_ptr = B_ptr + B_elems; C_elems = M * N
    A = torch.randn(M, K, dtype=torch.bfloat16)
    B = torch.randn(K, N, dtype=torch.bfloat16)
    C = torch.zeros(C_elems, dtype=torch.bfloat16)
    tpu.to_hbm(A_ptr, A)
    tpu.to_hbm(B_ptr, B)
    tpu.to_hbm(C_ptr, C)

    torch_result = A @ B
    return A_ptr, B_ptr, C_ptr, torch_result

def get_results(tpu, C_ptr,
    M, N):
    C = tpu.read_hbm(C_ptr, (M, N))
    return C

def matmul_kernel(builder, 
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    ):
    b = builder
    assert M % TILE_SIZE == 0
    assert N % TILE_SIZE == 0
    assert K % TILE_SIZE == 0

    a_tile = 0
    b_tile = TILE_ELEMS
    c_tile = 2 * TILE_ELEMS

    for m in range(0, M, TILE_SIZE):
        for n in range(0, N, TILE_SIZE):
            for k in range(0, K, TILE_SIZE):
                a_hbm = A_ptr + m * K + k
                b_hbm = B_ptr + k * N + n

                _load_tile(b, a_hbm, a_tile, TILE_SIZE, K)
                _load_tile(b, b_hbm, b_tile, TILE_SIZE, N)

                b.s_load_imm(1, b_tile)
                b.mfma_load_weights(1)

                b.s_load_imm(1, a_tile)
                b.s_load_imm(2, int(k != 0))
                b.mfma_matmul(1, 2)

            b.s_load_imm(1, c_tile)
            b.mfma_store(1)

            c_hbm = C_ptr + m * N + n
            _store_tile(b, c_tile, c_hbm, TILE_SIZE, N)

if __name__ == "__main__":
    builder = ProgramBuilder()
    M, N, K = 128, 128, 128
    tpu = SimTPU()
    A_ptr, B_ptr, C_ptr, torch_result = setup(tpu, M, N, K)
    matmul_kernel(builder, A_ptr, B_ptr, C_ptr, M, N, K)
    bundles = builder.build(OneBundleOneInstructionBundler())
    tpu.run(bundles)
    tpu_result = get_results(tpu, C_ptr, M, N)
    assert torch.allclose(torch_result, tpu_result)
