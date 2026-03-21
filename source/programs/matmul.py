from source.builder import ProgramBuilder
from source.bundlers import OneBundleOneInstructionBundler
from source.constants import NUM_REGS, TILE_ELEMS, TILE_SIZE
from source.sim import SimTPU
import torch


DMA_PTR_BANKS = [
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
    (17, 18),
    (19, 20),
    (21, 22),
]
A_TILE_REG = 23
B_TILE_REG = 24
C_TILE_REG = 25
TILE_SIZE_REG = 26
STRIDE_REG = 27
CHUNK_REG = 28
STRIDE_STEP_REG = 29
SCRATCH_STEP_REG = 30
ACC_REG = 31

PIPE_A_BANKS = [(reg, reg + 1) for reg in range(1, 57, 2)]
PIPE_B_BANKS = [(reg, reg + 1) for reg in range(57, 113, 2)]
PIPE_A0_REG = 113
PIPE_A1_REG = 114
PIPE_B0_REG = 115
PIPE_B1_REG = 116
PIPE_C_REG = 117
PIPE_TILE_SIZE_REG = 118
PIPE_CHUNK_REG = 119
PIPE_A_STRIDE_REG = 120
PIPE_B_STRIDE_REG = 121
PIPE_A_STEP_REG = 122
PIPE_B_STEP_REG = 123
PIPE_SCRATCH_STEP_REG = 124
PIPE_ACC_REG = 125

PIPE32_A_BANKS = DMA_PTR_BANKS[:5]
PIPE32_B_BANKS = DMA_PTR_BANKS[5:10]
PIPE32_A0_REG = 23
PIPE32_A1_REG = 24
PIPE32_B0_REG = 25
PIPE32_B1_REG = 26
PIPE32_C_REG = 27
PIPE32_TILE_SIZE_REG = 28
PIPE32_CHUNK_REG = 29
PIPE32_STEP_REG = 30
PIPE32_ACC_REG = 31


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


def _init_banked_ptrs(builder, hbm_ptr, scratch_ptr, stride):
    for lane, (src_reg, dst_reg) in enumerate(DMA_PTR_BANKS):
        builder.s_load_imm(src_reg, hbm_ptr + lane * stride)
        builder.s_load_imm(dst_reg, scratch_ptr + lane * TILE_SIZE)


def _stream_load_tile(builder, hbm_ptr, scratch_ptr, rows, stride):
    _init_banked_ptrs(builder, hbm_ptr, scratch_ptr, stride)
    builder.s_load_imm(STRIDE_REG, stride)
    builder.s_mult(STRIDE_STEP_REG, STRIDE_REG, CHUNK_REG)

    for row_base in range(0, rows, len(DMA_PTR_BANKS)):
        active = min(len(DMA_PTR_BANKS), rows - row_base)
        for lane in range(active):
            src_reg, dst_reg = DMA_PTR_BANKS[lane]
            builder.dma_load(src_reg, dst_reg, TILE_SIZE_REG)
        if row_base + len(DMA_PTR_BANKS) >= rows:
            continue
        for lane in range(active):
            src_reg, dst_reg = DMA_PTR_BANKS[lane]
            builder.s_add(src_reg, src_reg, STRIDE_STEP_REG)
            builder.s_add(dst_reg, dst_reg, SCRATCH_STEP_REG)


def _stream_store_tile(builder, scratch_ptr, hbm_ptr, rows, stride):
    _init_banked_ptrs(builder, scratch_ptr, hbm_ptr, TILE_SIZE)
    for lane, (_, dst_reg) in enumerate(DMA_PTR_BANKS):
        builder.s_load_imm(dst_reg, hbm_ptr + lane * stride)
    builder.s_load_imm(STRIDE_REG, stride)
    builder.s_mult(STRIDE_STEP_REG, STRIDE_REG, CHUNK_REG)

    for row_base in range(0, rows, len(DMA_PTR_BANKS)):
        active = min(len(DMA_PTR_BANKS), rows - row_base)
        for lane in range(active):
            src_reg, dst_reg = DMA_PTR_BANKS[lane]
            builder.dma_store(src_reg, dst_reg, TILE_SIZE_REG)
        if row_base + len(DMA_PTR_BANKS) >= rows:
            continue
        for lane in range(active):
            src_reg, dst_reg = DMA_PTR_BANKS[lane]
            builder.s_add(src_reg, src_reg, SCRATCH_STEP_REG)
            builder.s_add(dst_reg, dst_reg, STRIDE_STEP_REG)


def _init_pipe_banks(builder, banks, hbm_ptr, scratch_ptr, stride):
    for lane, (src_reg, dst_reg) in enumerate(banks):
        builder.s_load_imm(src_reg, hbm_ptr + lane * stride)
        builder.s_load_imm(dst_reg, scratch_ptr + lane * TILE_SIZE)


def _advance_pipe_banks(builder, banks, src_step_reg, dst_step_reg, active):
    for lane in range(active):
        src_reg, dst_reg = banks[lane]
        builder.s_add(src_reg, src_reg, src_step_reg)
        builder.s_add(dst_reg, dst_reg, dst_step_reg)


def _stream_pair_prefetch_generic(
    builder,
    a_banks,
    a_hbm_ptr,
    a_scratch_ptr,
    a_stride,
    b_banks,
    b_hbm_ptr,
    b_scratch_ptr,
    b_stride,
    tile_size_reg,
    a_step_reg,
    b_step_reg,
    scratch_step_reg,
    prefix=None,
):
    prefix = prefix or []
    injected = 0

    _init_pipe_banks(builder, a_banks, a_hbm_ptr, a_scratch_ptr, a_stride)
    _init_pipe_banks(builder, b_banks, b_hbm_ptr, b_scratch_ptr, b_stride)

    chunk = len(a_banks)
    builder.s_load_imm(a_step_reg, a_stride * chunk)
    builder.s_load_imm(b_step_reg, b_stride * chunk)

    for row_base in range(0, TILE_SIZE, len(a_banks)):
        active = min(len(a_banks), TILE_SIZE - row_base)
        for lane in range(active):
            if injected < len(prefix):
                prefix[injected]()
                injected += 1
            a_src_reg, a_dst_reg = a_banks[lane]
            b_src_reg, b_dst_reg = b_banks[lane]
            builder.dma_load(a_src_reg, a_dst_reg, tile_size_reg)
            builder.dma_load(b_src_reg, b_dst_reg, tile_size_reg)
        if row_base + len(a_banks) >= TILE_SIZE:
            continue
        _advance_pipe_banks(builder, a_banks, a_step_reg, scratch_step_reg, active)
        _advance_pipe_banks(builder, b_banks, b_step_reg, scratch_step_reg, active)

    while injected < len(prefix):
        prefix[injected]()
        injected += 1


def _stream_pair_prefetch(builder, a_hbm_ptr, a_scratch_ptr, a_stride, b_hbm_ptr, b_scratch_ptr, b_stride, prefix=None):
    _stream_pair_prefetch_generic(
        builder,
        PIPE_A_BANKS, a_hbm_ptr, a_scratch_ptr, a_stride,
        PIPE_B_BANKS, b_hbm_ptr, b_scratch_ptr, b_stride,
        PIPE_TILE_SIZE_REG,
        PIPE_A_STEP_REG, PIPE_B_STEP_REG, PIPE_SCRATCH_STEP_REG,
        prefix,
    )


def _stream_store_pipe_generic(builder, banks, scratch_ptr, hbm_ptr, stride, tile_size_reg, chunk_reg, stride_reg, step_reg, scratch_step_reg):
    _init_pipe_banks(builder, banks, scratch_ptr, hbm_ptr, TILE_SIZE)
    for lane, (_, dst_reg) in enumerate(banks):
        builder.s_load_imm(dst_reg, hbm_ptr + lane * stride)

    builder.s_load_imm(stride_reg, stride)
    builder.s_mult(step_reg, stride_reg, chunk_reg)

    for row_base in range(0, TILE_SIZE, len(banks)):
        active = min(len(banks), TILE_SIZE - row_base)
        for lane in range(active):
            src_reg, dst_reg = banks[lane]
            builder.dma_store(src_reg, dst_reg, tile_size_reg)
        if row_base + len(banks) >= TILE_SIZE:
            continue
        _advance_pipe_banks(builder, banks, scratch_step_reg, step_reg, active)


def _stream_store_pipe(builder, scratch_ptr, hbm_ptr, stride):
    _stream_store_pipe_generic(
        builder,
        PIPE_A_BANKS,
        scratch_ptr,
        hbm_ptr,
        stride,
        PIPE_TILE_SIZE_REG,
        PIPE_CHUNK_REG,
        PIPE_A_STRIDE_REG,
        PIPE_A_STEP_REG,
        PIPE_SCRATCH_STEP_REG,
    )


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


def matmul_kernel_all_regs(
    builder,
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    ):
    b = builder
    assert M % TILE_SIZE == 0
    assert N % TILE_SIZE == 0
    assert K % TILE_SIZE == 0

    b.s_load_imm(A_TILE_REG, 0)
    b.s_load_imm(B_TILE_REG, TILE_ELEMS)
    b.s_load_imm(C_TILE_REG, 2 * TILE_ELEMS)
    b.s_load_imm(TILE_SIZE_REG, TILE_SIZE)
    b.s_load_imm(CHUNK_REG, len(DMA_PTR_BANKS))
    b.s_mult(SCRATCH_STEP_REG, TILE_SIZE_REG, CHUNK_REG)

    for m in range(0, M, TILE_SIZE):
        for n in range(0, N, TILE_SIZE):
            for k in range(0, K, TILE_SIZE):
                a_hbm = A_ptr + m * K + k
                b_hbm = B_ptr + k * N + n

                _stream_load_tile(b, a_hbm, 0, TILE_SIZE, K)
                _stream_load_tile(b, b_hbm, TILE_ELEMS, TILE_SIZE, N)

                b.mfma_load_weights(B_TILE_REG)

                b.s_load_imm(ACC_REG, int(k != 0))
                b.mfma_matmul(A_TILE_REG, ACC_REG)

            b.mfma_store(C_TILE_REG)

            c_hbm = C_ptr + m * N + n
            _stream_store_tile(b, 2 * TILE_ELEMS, c_hbm, TILE_SIZE, N)


def matmul_kernel_pipelined_128regs(
    builder,
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    ):
    assert NUM_REGS >= 126, f"Need at least 126 registers, found {NUM_REGS}"
    assert M % TILE_SIZE == 0
    assert N % TILE_SIZE == 0
    assert K % TILE_SIZE == 0

    b = builder
    num_k_tiles = K // TILE_SIZE

    b.s_load_imm(PIPE_A0_REG, 0)
    b.s_load_imm(PIPE_A1_REG, TILE_ELEMS)
    b.s_load_imm(PIPE_B0_REG, 2 * TILE_ELEMS)
    b.s_load_imm(PIPE_B1_REG, 3 * TILE_ELEMS)
    b.s_load_imm(PIPE_C_REG, 4 * TILE_ELEMS)
    b.s_load_imm(PIPE_TILE_SIZE_REG, TILE_SIZE)
    b.s_load_imm(PIPE_CHUNK_REG, len(PIPE_A_BANKS))
    b.s_mult(PIPE_SCRATCH_STEP_REG, PIPE_TILE_SIZE_REG, PIPE_CHUNK_REG)

    a_buf_regs = [PIPE_A0_REG, PIPE_A1_REG]
    b_buf_regs = [PIPE_B0_REG, PIPE_B1_REG]
    a_buf_ptrs = [0, TILE_ELEMS]
    b_buf_ptrs = [2 * TILE_ELEMS, 3 * TILE_ELEMS]

    for m in range(0, M, TILE_SIZE):
        for n in range(0, N, TILE_SIZE):
            first_a_hbm = A_ptr + m * K
            first_b_hbm = B_ptr + n
            _stream_pair_prefetch(b, first_a_hbm, a_buf_ptrs[0], K, first_b_hbm, b_buf_ptrs[0], N)

            for k_tile in range(num_k_tiles):
                cur = k_tile % 2
                acc = int(k_tile != 0)

                if k_tile + 1 < num_k_tiles:
                    nxt = 1 - cur
                    next_k = (k_tile + 1) * TILE_SIZE
                    next_a_hbm = A_ptr + m * K + next_k
                    next_b_hbm = B_ptr + next_k * N + n

                    _stream_pair_prefetch(
                        b,
                        next_a_hbm, a_buf_ptrs[nxt], K,
                        next_b_hbm, b_buf_ptrs[nxt], N,
                        prefix=[
                            lambda cur=cur: b.mfma_load_weights(b_buf_regs[cur]),
                            lambda cur=cur, acc=acc: (b.s_load_imm(PIPE_ACC_REG, acc), b.mfma_matmul(a_buf_regs[cur], PIPE_ACC_REG)),
                        ],
                    )
                else:
                    b.mfma_load_weights(b_buf_regs[cur])
                    b.s_load_imm(PIPE_ACC_REG, acc)
                    b.mfma_matmul(a_buf_regs[cur], PIPE_ACC_REG)

            b.mfma_store(PIPE_C_REG)
            c_hbm = C_ptr + m * N + n
            _stream_store_pipe(b, 4 * TILE_ELEMS, c_hbm, N)


def matmul_kernel_pipelined_32regs(
    builder,
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    ):
    assert M % TILE_SIZE == 0
    assert N % TILE_SIZE == 0
    assert K % TILE_SIZE == 0

    b = builder
    num_k_tiles = K // TILE_SIZE

    b.s_load_imm(PIPE32_A0_REG, 0)
    b.s_load_imm(PIPE32_A1_REG, TILE_ELEMS)
    b.s_load_imm(PIPE32_B0_REG, 2 * TILE_ELEMS)
    b.s_load_imm(PIPE32_B1_REG, 3 * TILE_ELEMS)
    b.s_load_imm(PIPE32_C_REG, 4 * TILE_ELEMS)
    b.s_load_imm(PIPE32_TILE_SIZE_REG, TILE_SIZE)
    b.s_load_imm(PIPE32_CHUNK_REG, len(PIPE32_A_BANKS))
    b.s_mult(PIPE32_STEP_REG, PIPE32_TILE_SIZE_REG, PIPE32_CHUNK_REG)

    a_buf_regs = [PIPE32_A0_REG, PIPE32_A1_REG]
    b_buf_regs = [PIPE32_B0_REG, PIPE32_B1_REG]
    a_buf_ptrs = [0, TILE_ELEMS]
    b_buf_ptrs = [2 * TILE_ELEMS, 3 * TILE_ELEMS]

    for m in range(0, M, TILE_SIZE):
        for n in range(0, N, TILE_SIZE):
            first_a_hbm = A_ptr + m * K
            first_b_hbm = B_ptr + n
            _stream_pair_prefetch_generic(
                b,
                PIPE32_A_BANKS, first_a_hbm, a_buf_ptrs[0], K,
                PIPE32_B_BANKS, first_b_hbm, b_buf_ptrs[0], N,
                PIPE32_TILE_SIZE_REG,
                21, 22, PIPE32_STEP_REG,
            )

            for k_tile in range(num_k_tiles):
                cur = k_tile % 2
                acc = int(k_tile != 0)

                if k_tile + 1 < num_k_tiles:
                    nxt = 1 - cur
                    next_k = (k_tile + 1) * TILE_SIZE
                    next_a_hbm = A_ptr + m * K + next_k
                    next_b_hbm = B_ptr + next_k * N + n
                    _stream_pair_prefetch_generic(
                        b,
                        PIPE32_A_BANKS, next_a_hbm, a_buf_ptrs[nxt], K,
                        PIPE32_B_BANKS, next_b_hbm, b_buf_ptrs[nxt], N,
                        PIPE32_TILE_SIZE_REG,
                        21, 22, PIPE32_STEP_REG,
                        prefix=[
                            lambda cur=cur: b.mfma_load_weights(b_buf_regs[cur]),
                            lambda cur=cur, acc=acc: (b.s_load_imm(PIPE32_ACC_REG, acc), b.mfma_matmul(a_buf_regs[cur], PIPE32_ACC_REG)),
                        ],
                    )
                else:
                    b.mfma_load_weights(b_buf_regs[cur])
                    b.s_load_imm(PIPE32_ACC_REG, acc)
                    b.mfma_matmul(a_buf_regs[cur], PIPE32_ACC_REG)

            b.mfma_store(PIPE32_C_REG)
            c_hbm = C_ptr + m * N + n
            _stream_store_pipe_generic(
                b,
                PIPE32_A_BANKS,
                4 * TILE_ELEMS,
                c_hbm,
                N,
                PIPE32_TILE_SIZE_REG,
                PIPE32_CHUNK_REG,
                21,
                22,
                PIPE32_STEP_REG,
            )

def run_benchmark(M, N, K):
    builder = ProgramBuilder()
    tpu = SimTPU()
    A_ptr, B_ptr, C_ptr, torch_result = setup(tpu, M, N, K)
    matmul_kernel(builder, A_ptr, B_ptr, C_ptr, M, N, K)
    bundles = builder.build(OneBundleOneInstructionBundler())
    num_cycles = tpu.run(bundles)
    tpu_result = get_results(tpu, C_ptr, M, N)
    assert torch.allclose(torch_result.to(device=tpu.device), tpu_result.to(device=tpu.device), atol=1e-2, rtol=1e-2)
    return num_cycles

if __name__ == "__main__":
    num_cycles = run_benchmark(128, 128, 128)
    print(f"Runs in {num_cycles} cycles")
