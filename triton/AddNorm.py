import torch
import triton
import triton.language as tl

# Your Triton kernel (as defined earlier)
@triton.jit
def matmul_kernel(
    A, B, C, 
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(
            A + rm[:, None] * stride_am + (k + tl.arange(0, BLOCK_K))[None, :] * stride_ak,
            mask=(rm[:, None] < M) & ((k + tl.arange(0, BLOCK_K))[None, :] < K),
            other=0.0
        )
        b = tl.load(
            B + (k + tl.arange(0, BLOCK_K))[:, None] * stride_bk + rn[None, :] * stride_bn,
            mask=((k + tl.arange(0, BLOCK_K))[:, None] < K) & (rn[None, :] < N),
            other=0.0
        )
        acc += tl.dot(a, b)
    tl.store(
        C + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
        acc,
        mask=(rm[:, None] < M) & (rn[None, :] < N)
    )

def matmul_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Inner dimensions must match for matmul"
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (
        (M + BLOCK_M - 1) // BLOCK_M,
        (N + BLOCK_N - 1) // BLOCK_N,
    )
    matmul_kernel[grid](
        A, B, C, M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return C

# Define a custom autograd Function
class MatMulTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        C = matmul_triton(A, B)
        ctx.save_for_backward(A, B)
        return C

    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        # Compute gradients using the standard PyTorch matmul.
        # For a matmul: dL/dA = dL/dC @ B^T and dL/dB = A^T @ dL/dC.
        grad_A = grad_output.matmul(B.t())
        grad_B = A.t().matmul(grad_output)
        return grad_A, grad_B

# A wrapper function to use in your model.
def custom_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return MatMulTritonFunction.apply(A, B)

# Example usage in a model:
if __name__ == '__main__':
    A = torch.randn(256, 128, device='cuda', dtype=torch.float32, requires_grad=True)
    B = torch.randn(128, 256, device='cuda', dtype=torch.float32, requires_grad=True)
    
    C = custom_matmul(A, B)
    loss = C.sum()
    loss.backward()
    
    print("Gradients computed for A:", A.grad)
    print("Gradients computed for B:", B.grad)
