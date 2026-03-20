import torch
import torch.nn as nn


class NaiveSubMConv3d(nn.Module):
    """
    A Pure PyTorch implementation of Submanifold Sparse Convolution.
    Mathematically equivalent to spconv.SubMConv3d but slower.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = 1 # SubM always has stride 1
        # 27 weights for 3x3x3 kernel
        # We store them as (Kernel_Vol, In, Out) to apply efficiently
        kernel_vol = kernel_size ** 3
        self.weight = nn.Parameter(torch.Tensor(kernel_vol, in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Init weights (Xavier)
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
            
        # Pre-compute offsets for 3x3x3 grid [-1, 0, 1]
        # Generates: [[-1,-1,-1], ... , [0,0,0], ... , [1,1,1]]
        k = kernel_size // 2
        range_k = torch.arange(-k, k+1)
        offsets = torch.meshgrid(range_k, range_k, range_k, indexing='ij')
        self.offsets = torch.stack([x.flatten() for x in offsets], dim=1).cuda() # (27, 3)

    def forward(self, x, coords, batch_idx):
        """
        x: (N, C_in) features
        coords: (N, 3) int32 grid coordinates
        batch_idx: (N,) int indices
        """
        N, C = x.shape
        device = x.device
        if self.offsets.device != device:
            self.offsets = self.offsets.to(device)

        # 1. Build Hash Map for Neighbor Search
        # We assume coordinates are int32. We can bit-pack them into int64 for hashing.
        # Key = batch << 55 | z << 38 | y << 19 | x 
        keys = self.pack_coords(batch_idx, coords)
        
        # Map: Key -> Index in x
        sorted_keys, sort_idx = torch.sort(keys)


        output = torch.zeros(N, self.weight.shape[2], device=device)

        # 2. Iterate over Kernel Offsets (27 iterations)
        for k_idx in range(self.offsets.shape[0]):
            offset = self.offsets[k_idx]
            
            # If offset is (0,0,0), it's just a linear projection of self
            if torch.all(offset == 0):
                w = self.weight[k_idx] # (Cin, Cout)
                output += x @ w
                continue
            
            # Calculate "Target" coordinates for this kernel weight
            target_coords = coords + offset
            target_keys = self.pack_coords(batch_idx, target_coords)
            
            # Find which target_keys exist in our active points (sorted_keys)
            indices = torch.searchsorted(sorted_keys, target_keys)
            
            # Check validity (clamp indices to bounds first)
            indices = indices.clamp(0, N - 1)
            found_keys = sorted_keys[indices]
            
            # Mask: True where the neighbor actually exists
            mask = (found_keys == target_keys)
            
            if mask.any():
                # Get the features of the neighbors
                # 'indices' points to sorted_keys. We need the original index in 'x'.
                neighbor_real_indices = sort_idx[indices]
                
                neighbor_feats = x[neighbor_real_indices]
                
                # Apply Weight
                w = self.weight[k_idx]
   
                valid_feats = neighbor_feats[mask]
                weighted_feats = valid_feats @ w
                
                output[mask] += weighted_feats

        # 3. Add Bias
        if self.bias is not None:
            output += self.bias
            
        return output

    def pack_coords(self, batch, xyz):
        # Packs (B, X, Y, Z) into single int64
        # Assumes coordinates fit in 18 bits (262k)
        b = batch.long()
        x, y, z = xyz[:, 0].long(), xyz[:, 1].long(), xyz[:, 2].long()
        return (b << 54) | (x << 36) | (y << 18) | z
    


