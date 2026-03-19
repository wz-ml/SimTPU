from source.builder import ProgramBuilder
from source.bundlers import OneBundleOneInstructionBundler
from source.sim import SimTPU
import torch

def setup(tpu, M, N, K):
    A_ptr = 0; A_elems = M * K
    B_ptr = A_elems; B_elems = K * N
    C_ptr = B_ptr + B_elems; C_elems = M * N
    A = torch.randn(A_elems, dtype=torch.bfloat16)
    B = torch.randn(B_elems, dtype=torch.bfloat16)
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
    """
    C = A @ B
    A: M, K
    B: K, N
    C: M, N

    All matrices are in row-major order.
    Ptrs are HBM addresses.
    """
    b = builder
    pass # TODO
    
if __name__ == "__main__":
    builder = ProgramBuilder()
    M, N, K = 512, 512, 512
    tpu = SimTPU()
    A_ptr, B_ptr, C_ptr, torch_result = setup(tpu, M, N, K)
    matmul_kernel(builder, A_ptr, B_ptr, C_ptr, M, N, K)
    bundles = builder.build(OneBundleOneInstructionBundler())
    tpu.run(bundles)
    tpu_result = get_results(tpu, C_ptr, M, N)
    assert torch.allclose(torch_result, tpu_result)