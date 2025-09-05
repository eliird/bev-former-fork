import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiheadAttention(nn.Module):
    """Multi-Head Attention module compatible with DETR decoder.
    
    This is a wrapper around PyTorch's native MultiheadAttention with
    additional compatibility for transformer decoder use cases.
    
    Args:
        embed_dims (int): The embedding dimension. Default: 256
        num_heads (int): Number of attention heads. Default: 8
        dropout (float): Dropout rate. Default: 0.1
        batch_first (bool): If True, batch dimension is first. Default: False
    """
    
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.1,
                 batch_first=False,
                 **kwargs):
        super(MultiheadAttention, self).__init__()
        
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        
        # Use PyTorch's native MultiheadAttention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dims,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first
        )
        
        self.fp16_enabled = False
    
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for MultiheadAttention.
        
        Args:
            query (Tensor): Query tensor with shape (num_query, bs, embed_dims)
                           or (bs, num_query, embed_dims) if batch_first=True
            key (Tensor): Key tensor. If None, uses query (self-attention)
            value (Tensor): Value tensor. If None, uses key
            identity (Tensor): Identity tensor for residual connection
            query_pos (Tensor): Query positional encoding
            key_pos (Tensor): Key positional encoding  
            attn_mask (Tensor): Attention mask
            key_padding_mask (Tensor): Key padding mask
            
        Returns:
            Tensor: Output tensor with same shape as query
        """
        
        # Handle None cases
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
            
        # Add positional encodings
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos
            
        # Apply attention
        attn_output, _ = self.attention(
            query=query,
            key=key, 
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )
        
        # Residual connection
        return attn_output + identity


def test_multihead_attention():
    """Test MultiheadAttention module"""
    print("=" * 60)
    print("Testing MultiheadAttention")
    print("=" * 60)
    
    # Config parameters for decoder self-attention
    embed_dims = 256
    num_heads = 8
    dropout = 0.1
    
    try:
        # Create model
        model = MultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False  # Decoder convention
        )
        
        print(f" Model created successfully")
        print(f"  - embed_dims: {embed_dims}")
        print(f"  - num_heads: {num_heads}")
        print(f"  - dropout: {dropout}")
        print(f"  - batch_first: False")
        
        # Test inputs for decoder self-attention
        batch_size = 2
        num_queries = 900  # Object queries
        
        # Query tensor (object queries) - shape: (num_query, bs, embed_dims)
        query = torch.randn(num_queries, batch_size, embed_dims)
        
        # Query positional encoding
        query_pos = torch.randn(num_queries, batch_size, embed_dims)
        
        print(f" Test inputs created")
        print(f"  - query shape: {query.shape}")
        print(f"  - query_pos shape: {query_pos.shape}")
        
        # Test self-attention (key=value=None, should use query)
        with torch.no_grad():
            output = model(
                query=query,
                key=None,  # Self-attention
                value=None,  # Self-attention
                query_pos=query_pos
            )
        
        print(f" Self-attention forward pass successful")
        print(f"  - output shape: {output.shape}")
        print(f"  - expected shape: {query.shape}")
        
        # Verify output shape
        assert output.shape == query.shape, f"Output shape {output.shape} != expected {query.shape}"
        
        # Verify output is not NaN or Inf
        assert torch.isfinite(output).all(), "Output contains NaN or Inf values"
        
        # Test cross-attention with different key/value
        key_value = torch.randn(500, batch_size, embed_dims)  # Different sequence length
        key_pos = torch.randn(500, batch_size, embed_dims)
        
        with torch.no_grad():
            output_cross = model(
                query=query,
                key=key_value,
                value=key_value,
                query_pos=query_pos,
                key_pos=key_pos
            )
        
        print(f" Cross-attention forward pass successful")
        print(f"  - output shape: {output_cross.shape}")
        
        assert output_cross.shape == query.shape, f"Output shape {output_cross.shape} != expected {query.shape}"
        assert torch.isfinite(output_cross).all(), "Output contains NaN or Inf values"
        
        # Test with attention mask
        attn_mask = torch.zeros(num_queries, num_queries)
        # Create causal mask (lower triangular)
        attn_mask = torch.triu(torch.ones(num_queries, num_queries) * float('-inf'), diagonal=1)
        
        with torch.no_grad():
            output_masked = model(
                query=query,
                key=None,
                value=None,
                query_pos=query_pos,
                attn_mask=attn_mask
            )
        
        print(f" Masked self-attention forward pass successful")
        print(f"  - output shape: {output_masked.shape}")
        
        assert output_masked.shape == query.shape, f"Output shape {output_masked.shape} != expected {query.shape}"
        assert torch.isfinite(output_masked).all(), "Output contains NaN or Inf values"
        
        # Test with key padding mask
        key_padding_mask = torch.zeros(batch_size, num_queries, dtype=torch.bool)
        key_padding_mask[:, -100:] = True  # Mask last 100 positions
        
        with torch.no_grad():
            output_padded = model(
                query=query,
                key=None,
                value=None,
                query_pos=query_pos,
                key_padding_mask=key_padding_mask
            )
        
        print(f" Padded self-attention forward pass successful")
        print(f"  - output shape: {output_padded.shape}")
        
        assert output_padded.shape == query.shape, f"Output shape {output_padded.shape} != expected {query.shape}"
        assert torch.isfinite(output_padded).all(), "Output contains NaN or Inf values"
        
        print(" All assertions passed")
        print("<� MultiheadAttention test PASSED!")
        return True
        
    except Exception as e:
        print(f"L Test FAILED with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_multihead_attention()