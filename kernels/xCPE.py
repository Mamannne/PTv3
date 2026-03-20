import torch
import torch.nn as nn
import math

from .sparseconv import NaiveSubMConv3d

# Try importing your Triton function
try:
    from .triton_ops import apply_sparse_conv
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("⚠️  Triton not found. 'triton' backend will fail.")

def build_neighbor_table(coords, batch_idx):
    """
    Builds (N, 27) neighbor table. 
    Handles batch separation by offsetting coordinates in hash space.
    """
    device = coords.device
    N = coords.shape[0]
    
    # 1. Coordinate Separation (The "Batch Trick")
    # We shift X coordinate by (batch_idx * 10000) so clouds don't touch in hash space
    aug_coords = coords.clone().long()
    aug_coords[:, 0] += batch_idx.long() * 100000 
    
    # 2. Bit Packing (Hash 3D -> 1D)
    x, y, z = aug_coords[:, 0], aug_coords[:, 1], aug_coords[:, 2]
    keys = (x << 40) | (y << 20) | z
    
    # 3. Sort
    sorted_keys, sort_idx = torch.sort(keys)
    
    # 4. Search 27 Offsets
    r = torch.arange(-1, 2, device=device)
    offsets = torch.stack(torch.meshgrid(r, r, r, indexing='ij'), dim=-1).reshape(-1, 3)
    table = torch.full((N, 27), -1, dtype=torch.int32, device=device)
    
    for k in range(27):
        target_coords = aug_coords + offsets[k]
        tx, ty, tz = target_coords[:, 0], target_coords[:, 1], target_coords[:, 2]
        target_keys = (tx << 40) | (ty << 20) | tz
        
        ptr = torch.searchsorted(sorted_keys, target_keys).clamp(0, N - 1)
        
        # Verify match
        found_key = sorted_keys[ptr]
        is_neighbor = (found_key == target_keys)
        
        if is_neighbor.any():
            real_idx = sort_idx[ptr]
            table[:, k] = torch.where(is_neighbor, real_idx.int(), table[:, k])
            
    return table

class TritonSubMConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
        super().__init__()
        assert kernel_size == 3, "Triton kernel only supports 3x3x3 currently"
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Weights: (27, C, C) - Layout optimized for Triton
        self.weight = nn.Parameter(torch.Tensor(27, in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # Standard Kaiming Init adapted for 27 offsets
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, features, coords, batch_idx):
        """
        Args:
            features: (N, C)
            coords: (N, 3) Integer Quantized Coords
            batch_idx: (N,)
        """
        # 1. Build Neighbor Table (This is the overhead, but Triton is so fast it pays off)
        neighbor_table = build_neighbor_table(coords, batch_idx)
        
        # 2. Run Triton Kernel
        out = apply_sparse_conv(features, self.weight, neighbor_table)
        if self.bias is not None:
            out = out + self.bias
        return out
    



class PointxCPE(nn.Module):
    def __init__(self, channels, grid_size=0.05, kernel_size=3, backend='triton'):
        super().__init__()
        self.grid_size = grid_size
        self.backend = backend

        # --- AUTO-SWITCH LOGIC ---
        # If channels are large (> 128), the weight matrix won't fit in Shared Memory.
        # We switch to Naive mode. 
        if self.backend == 'triton':
            if not TRITON_AVAILABLE:
                self.backend = 'naive'
            elif channels >= 128:  # Threshold for RTX 5070 Shared Memory
                self.backend = 'naive'
        # -------------------------

        if self.backend == 'naive':
            self.sparse_conv = NaiveSubMConv3d(channels, channels, kernel_size=kernel_size)
        else:
            self.sparse_conv = TritonSubMConv3d(channels, channels, kernel_size=kernel_size, bias=False)

    def forward(self, features, coords, batch_idx):
        """
        Args:
            features (Tensor): (N, C) The 1D feature list.
            coords (Tensor): (N, 3) The original 3D coordinates.
            batch_idx (Tensor): (N,) The batch index for each point.
        """
        # Quantize coordinates to integer grid
        quant_coords = torch.floor(coords / self.grid_size).int()
        
        # Shift to positive quadrant to ensure hashing safety
        min_xyz = quant_coords.min(dim=0)[0]
        quant_coords = quant_coords - min_xyz

        # Run the chosen backend
        out_features = self.sparse_conv(features, quant_coords, batch_idx)
        
        return features + out_features