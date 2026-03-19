from source.builder import ProgramBuilder
from source.bundlers import OneBundleOneInstructionBundler
from source.sim import SimTPU
import torch


def setup(tpu, rows, cols):
    x_ptr = 0
    y_ptr = rows * cols
    x = torch.randn(rows, cols, dtype=torch.bfloat16)
    y = torch.zeros(rows, cols, dtype=torch.bfloat16)
    tpu.to_hbm(x_ptr, x)
    tpu.to_hbm(y_ptr, y)
    return x_ptr, y_ptr, torch.softmax(x.float(), dim=-1).to(torch.bfloat16)


def get_results(tpu, y_ptr, rows, cols):
    return tpu.read_hbm(y_ptr, (rows, cols))


def softmax_kernel(builder, x_ptr, y_ptr, rows, cols):
    assert cols <= 128

    row_in = 0
    row_max = 128
    row_shifted = 256
    row_exp = 384
    row_denom = 512
    row_out = 640
    max_scalar = 768
    sum_scalar = 769

    b = builder
    b.s_load_imm(10, cols)
    b.v_set_length(10)

    for row in range(rows):
        row_ptr = x_ptr + row * cols
        out_ptr = y_ptr + row * cols

        b.s_load_imm(1, row_ptr)
        b.s_load_imm(2, row_in)
        b.s_load_imm(3, cols)
        b.dma_load(1, 2, 3)

        b.s_load_imm(1, max_scalar)
        b.s_load_imm(2, row_in)
        b.v_reduce_max(1, 2)

        b.s_load_imm(1, row_max)
        b.s_load_imm(2, max_scalar)
        b.v_vbroadcast(1, 2)

        b.s_load_imm(1, row_shifted)
        b.s_load_imm(2, row_in)
        b.s_load_imm(3, row_max)
        b.v_sub(1, 2, 3)

        b.s_load_imm(1, row_exp)
        b.s_load_imm(2, row_shifted)
        b.v_exp(1, 2)

        b.s_load_imm(1, sum_scalar)
        b.s_load_imm(2, row_exp)
        b.v_reduce_sum(1, 2)

        b.s_load_imm(1, row_denom)
        b.s_load_imm(2, sum_scalar)
        b.v_vbroadcast(1, 2)

        b.s_load_imm(1, row_denom)
        b.s_load_imm(2, row_denom)
        b.v_reciprocal(1, 2)

        b.s_load_imm(1, row_out)
        b.s_load_imm(2, row_exp)
        b.s_load_imm(3, row_denom)
        b.v_mult(1, 2, 3)

        b.s_load_imm(1, row_out)
        b.s_load_imm(2, out_ptr)
        b.s_load_imm(3, cols)
        b.dma_store(1, 2, 3)


if __name__ == "__main__":
    builder = ProgramBuilder()
    rows, cols = 8, 32
    tpu = SimTPU()
    x_ptr, y_ptr, torch_result = setup(tpu, rows, cols)
    softmax_kernel(builder, x_ptr, y_ptr, rows, cols)
    bundles = builder.build(OneBundleOneInstructionBundler())
    tpu.run(bundles)
    tpu_result = get_results(tpu, y_ptr, rows, cols)
    assert torch.allclose(torch_result, tpu_result, atol=1e-2, rtol=1e-2)
