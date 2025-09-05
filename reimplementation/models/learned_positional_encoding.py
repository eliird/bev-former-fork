"""
LearnedPositionalEncoding implementation for BEVFormer.
Based on the config:
positional_encoding=dict(
    type='LearnedPositionalEncoding',
    num_feats=_pos_dim_,
    row_num_embed=bev_h_,
    col_num_embed=bev_w_,
)
"""

import torch
import torch.nn as nn
import math


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding for BEV features.
    
    This module generates learned positional embeddings for BEV (Bird's Eye View) features.
    It creates separate embeddings for row and column positions and combines them.
    
    Args:
        num_feats (int): Number of positional encoding features (usually embed_dims // 2)
        row_num_embed (int): Number of row embeddings (BEV height)
        col_num_embed (int): Number of column embeddings (BEV width)
        **kwargs: Additional arguments
    """
    
    def __init__(self,
                 num_feats=128,
                 row_num_embed=50,
                 col_num_embed=50,
                 **kwargs):
        super(LearnedPositionalEncoding, self).__init__()
        
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed
        
        # Learned embeddings for row and column positions
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize the weights of the positional embeddings."""
        # Initialize with small random values
        nn.init.uniform_(self.row_embed.weight, -0.1, 0.1)
        nn.init.uniform_(self.col_embed.weight, -0.1, 0.1)
    
    def forward(self, mask=None):
        """Forward function to generate positional encodings.
        
        Args:
            mask (Tensor, optional): Mask tensor. If provided, should have shape (batch_size, H, W).
                                   If None, generates full positional encoding.
                                   
        Returns:
            Tensor: Positional encoding with shape (batch_size, num_feats*2, H, W)
                   where num_feats*2 = embed_dims
        """
        h, w = self.row_num_embed, self.col_num_embed
        
        if mask is not None:
            # If mask is provided, use its spatial dimensions
            batch_size = mask.shape[0]
            h, w = mask.shape[-2:]
            # Ensure dimensions match the embedding tables
            assert h <= self.row_num_embed, f"Height {h} exceeds row_num_embed {self.row_num_embed}"
            assert w <= self.col_num_embed, f"Width {w} exceeds col_num_embed {self.col_num_embed}"
        else:
            # Default batch size is 1 if no mask provided
            batch_size = 1
        
        # Generate position indices
        # Shape: (h,) and (w,)
        y_pos = torch.arange(h, dtype=torch.long, device=self.row_embed.weight.device)
        x_pos = torch.arange(w, dtype=torch.long, device=self.col_embed.weight.device)
        
        # Get embeddings for each position
        # Shape: (h, num_feats) and (w, num_feats)
        y_embed = self.row_embed(y_pos)
        x_embed = self.col_embed(x_pos)
        
        # Create meshgrid and combine embeddings
        # Shape after unsqueeze: (h, 1, num_feats) and (1, w, num_feats)
        # Shape after expand: (h, w, num_feats) for both
        pos_embed = torch.cat([
            x_embed.unsqueeze(0).expand(h, w, self.num_feats),  # x positions
            y_embed.unsqueeze(1).expand(h, w, self.num_feats)   # y positions  
        ], dim=-1)
        
        # Reshape to (h, w, num_feats*2) -> (num_feats*2, h, w) -> (batch_size, num_feats*2, h, w)
        pos_embed = pos_embed.permute(2, 0, 1)  # (num_feats*2, h, w)
        pos_embed = pos_embed.unsqueeze(0).expand(batch_size, -1, -1, -1)  # (batch_size, num_feats*2, h, w)
        
        return pos_embed


def test_learned_positional_encoding():
    """Test LearnedPositionalEncoding module"""
    print("=" * 60)
    print("Testing LearnedPositionalEncoding")
    print("=" * 60)
    
    # Config parameters from BEVFormer
    embed_dims = 256
    num_feats = embed_dims // 2  # 128 - _pos_dim_
    bev_h = 50  # bev_h_  
    bev_w = 50  # bev_w_
    
    try:
        # Create LearnedPositionalEncoding
        pos_encoding = LearnedPositionalEncoding(
            num_feats=num_feats,
            row_num_embed=bev_h,
            col_num_embed=bev_w
        )
        
        print("✓ LearnedPositionalEncoding created successfully")
        print(f"  - num_feats: {num_feats}")
        print(f"  - row_num_embed: {bev_h}")
        print(f"  - col_num_embed: {bev_w}")
        print(f"  - expected output dims: {num_feats * 2}")
        
        # Test basic forward pass without mask
        batch_size = 2
        
        with torch.no_grad():
            pos_embed = pos_encoding()
        
        print("✓ Forward pass without mask successful")
        print(f"  - output shape: {pos_embed.shape}")
        print(f"  - expected shape: (1, {num_feats * 2}, {bev_h}, {bev_w})")
        
        # Verify output shape
        expected_shape = torch.Size([1, num_feats * 2, bev_h, bev_w])
        assert pos_embed.shape == expected_shape, f"Output shape {pos_embed.shape} != expected {expected_shape}"
        
        # Verify output is not NaN or Inf
        assert torch.isfinite(pos_embed).all(), "Output contains NaN or Inf values"
        
        # Test forward pass with mask
        mask = torch.ones(batch_size, bev_h, bev_w, dtype=torch.bool)
        
        with torch.no_grad():
            pos_embed_with_mask = pos_encoding(mask)
        
        print("✓ Forward pass with mask successful")
        print(f"  - output shape: {pos_embed_with_mask.shape}")
        print(f"  - expected shape: ({batch_size}, {num_feats * 2}, {bev_h}, {bev_w})")
        
        # Verify output shape with mask
        expected_mask_shape = torch.Size([batch_size, num_feats * 2, bev_h, bev_w])
        assert pos_embed_with_mask.shape == expected_mask_shape, f"Masked output shape {pos_embed_with_mask.shape} != expected {expected_mask_shape}"
        
        # Verify output is not NaN or Inf
        assert torch.isfinite(pos_embed_with_mask).all(), "Masked output contains NaN or Inf values"
        
        # Test with different spatial dimensions
        smaller_h, smaller_w = 25, 30
        smaller_mask = torch.ones(batch_size, smaller_h, smaller_w, dtype=torch.bool)
        
        with torch.no_grad():
            smaller_pos_embed = pos_encoding(smaller_mask)
        
        print("✓ Forward pass with smaller dimensions successful")
        print(f"  - output shape: {smaller_pos_embed.shape}")
        print(f"  - expected shape: ({batch_size}, {num_feats * 2}, {smaller_h}, {smaller_w})")
        
        expected_smaller_shape = torch.Size([batch_size, num_feats * 2, smaller_h, smaller_w])
        assert smaller_pos_embed.shape == expected_smaller_shape, f"Smaller output shape {smaller_pos_embed.shape} != expected {expected_smaller_shape}"
        
        # Test that different positions have different embeddings
        pos_embed_flat = pos_embed_with_mask[0].flatten(1)  # (embed_dims, h*w)
        
        # Check that position (0,0) is different from position (1,1)
        pos_00 = pos_embed_flat[:, 0]  # position (0,0)
        pos_11 = pos_embed_flat[:, bev_w + 1]  # position (1,1)
        
        assert not torch.allclose(pos_00, pos_11), "Different positions should have different embeddings"
        print("✓ Different positions have different embeddings")
        
        # Test that x and y components are properly separated
        # First half should be x embeddings, second half should be y embeddings
        x_component = pos_embed_with_mask[0, :num_feats]  # (num_feats, h, w)
        y_component = pos_embed_with_mask[0, num_feats:]  # (num_feats, h, w)
        
        # Same row should have same y component
        assert torch.allclose(y_component[:, 0, :], y_component[:, 0, :]), "Same row should have same y component"
        
        # Same column should have same x component
        assert torch.allclose(x_component[:, :, 0], x_component[:, :, 0]), "Same column should have same x component"
        
        # Different rows should have different y components
        if bev_h > 1:
            assert not torch.allclose(y_component[:, 0, 0], y_component[:, 1, 0]), "Different rows should have different y components"
        
        # Different columns should have different x components
        if bev_w > 1:
            assert not torch.allclose(x_component[:, 0, 0], x_component[:, 0, 1]), "Different columns should have different x components"
        
        print("✓ X and Y components are properly separated")
        
        # Test gradient flow
        pos_encoding.train()
        dummy_input = torch.randn(1, requires_grad=True)
        pos_embed_grad = pos_encoding()
        
        # Create a simple loss
        loss = pos_embed_grad.sum()
        loss.backward()
        
        # Check that gradients exist for embedding layers
        assert pos_encoding.row_embed.weight.grad is not None, "Row embeddings should have gradients"
        assert pos_encoding.col_embed.weight.grad is not None, "Column embeddings should have gradients"
        
        print("✓ Gradient flow test successful")
        
        # Test device consistency
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            pos_encoding_cuda = pos_encoding.to(device)
            
            with torch.no_grad():
                pos_embed_cuda = pos_encoding_cuda()
            
            assert pos_embed_cuda.device.type == 'cuda', "Output should be on CUDA device"
            print("✓ CUDA device test successful")
        
        print("✓ All assertions passed")
        print("🎉 LearnedPositionalEncoding test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Test FAILED with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_learned_positional_encoding()