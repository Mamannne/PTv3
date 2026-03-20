import torch
import triton
import triton.language as tl

# --- 1. THE TRITON KERNEL (GPU Code) ---
@triton.jit
def sparse_conv_kernel(
    x_ptr,              # Pointer to Input Features
    weight_ptr,         # Pointer to Weights (27, C, C)
    out_ptr,            # Pointer to Output Features
    neighbor_table_ptr, # Pointer to Neighbor Table (N, 27)
    N,                  # Total number of points
    C: tl.constexpr,    # Number of Channels (Must be compile-time constant)
    BLOCK_SIZE_M: tl.constexpr # Block size (e.g., 32)
):
    # A. Identify which block of points we are processing
    pid = tl.program_id(axis=0)
    offs_m = tl.arange(0, BLOCK_SIZE_M)
    pt_idx = pid * BLOCK_SIZE_M + offs_m

    # Mask: Don't compute for indices beyond N (e.g. padding)
    check_N_mask = pt_idx < N

    # B. Setup Pointers for Channels
    offs_c = tl.arange(0, C) # [0, 1, ... C-1]
    
    # Initialize Accumulator (FP32)
    accumulator = tl.zeros((BLOCK_SIZE_M, C), dtype=tl.float32)

    # C. Loop over 27 Offsets (Fusion)
    for k in range(27):
        # 1. Load Neighbor IDs for this offset k
        n_ptrs = neighbor_table_ptr + (pt_idx * 27) + k
        neighbor_ids = tl.load(n_ptrs, mask=check_N_mask, other=-1)

        # 2. Check Validity (Is neighbor_id != -1?)
        valid_mask = (neighbor_ids != -1)

        # 3. Load Features (Gather)
        # Broadcasting: (BLOCK, 1) + (1, C) = (BLOCK, C)
        x_ptrs = x_ptr + (neighbor_ids[:, None] * C) + offs_c[None, :]

        features = tl.load(x_ptrs, mask=valid_mask[:, None], other=0.0)

        # 4. Load Weights (Broadcast)
        # Weight shape is (27, C, C). Flat index.
        # Start of matrix k: k * C * C
        k_start = weight_ptr + (k * C * C)
        
        # Grid of pointers for (C, C) matrix
        w_ptrs = k_start + (offs_c[:, None] * C) + offs_c[None, :]
        weights = tl.load(w_ptrs)

        # 5. Math (Matrix Multiply)
        accumulator += tl.dot(features, weights, allow_tf32=False)

    # D. Store Result
    out_ptrs = out_ptr + (pt_idx[:, None] * C) + offs_c[None, :]
    tl.store(out_ptrs, accumulator, mask=check_N_mask[:, None])


# --- 2. THE LAUNCHER (CPU Wrapper) ---
def apply_sparse_conv(x, weights, neighbor_table):
    """
    Args:
        x: (N, C) Float Tensor - Input Features
        weights: (27, C, C) Float Tensor - Kernel Weights
        neighbor_table: (N, 27) Int32 Tensor - Pre-computed neighbors
    Returns:
        out: (N, C) Float Tensor
    """
    # 1. Checks
    assert x.is_cuda and weights.is_cuda and neighbor_table.is_cuda
    N, C = x.shape
    
    # Ensure memory is contiguous (Triton requires this!)
    x = x.contiguous()
    weights = weights.contiguous()
    neighbor_table = neighbor_table.contiguous()
    
    # 2. Allocate Output
    out = torch.empty((N, C), device=x.device, dtype=x.dtype)
    
    # 3. Define Grid
    # How many blocks do we need?
    BLOCK_SIZE_M = 32 # You can tune this (32, 64, 128)
    grid = lambda META: ((N + META['BLOCK_SIZE_M'] - 1) // META['BLOCK_SIZE_M'], )
    
    # 4. Launch Kernel
    # num_stages=1: Disables aggressive prefetching. drastically reduces Shared Mem usage.
    # num_warps=4:  Standard thread count (128 threads per block).
    sparse_conv_kernel[grid](
        x_ptr=x,
        weight_ptr=weights,
        out_ptr=out,
        neighbor_table_ptr=neighbor_table,
        N=N,
        C=C, 
        BLOCK_SIZE_M=BLOCK_SIZE_M
    )
    
    return out