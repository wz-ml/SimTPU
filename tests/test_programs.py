import torch

from source.builder import ProgramBuilder
from source.bundlers import OneBundleOneInstructionBundler
from source.programs.matmul import get_results as get_gemm_results
from source.programs.matmul import matmul_kernel, setup as setup_gemm
from source.programs.softmax import get_results as get_softmax_results
from source.programs.softmax import setup as setup_softmax
from source.programs.softmax import softmax_kernel
from source.sim import SimTPU


def test_matmul_kernel():
    torch.manual_seed(0)
    tpu = SimTPU(use_device=False)
    builder = ProgramBuilder()

    M, N, K = 256, 256, 256
    a_ptr, b_ptr, c_ptr, expected = setup_gemm(tpu, M, N, K)
    matmul_kernel(builder, a_ptr, b_ptr, c_ptr, M, N, K)

    bundles = builder.build(OneBundleOneInstructionBundler())
    tpu.run(bundles)

    actual = get_gemm_results(tpu, c_ptr, M, N)
    assert torch.allclose(actual.float(), expected.float(), atol=3e-1, rtol=1e-1)


def test_softmax_kernel():
    torch.manual_seed(0)
    tpu = SimTPU(use_device=False)
    builder = ProgramBuilder()

    rows, cols = 5, 37
    x_ptr, y_ptr, expected = setup_softmax(tpu, rows, cols)
    softmax_kernel(builder, x_ptr, y_ptr, rows, cols)

    bundles = builder.build(OneBundleOneInstructionBundler())
    tpu.run(bundles)

    actual = get_softmax_results(tpu, y_ptr, rows, cols)
    assert torch.allclose(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)
