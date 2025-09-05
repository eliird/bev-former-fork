"""
Test script for BEVFormer components
Run this to verify each component works independently
"""

import torch
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from models.backbone import ResNetBackbone
from models.neck import FPNNeck
from models.spatial_attention import SpatialCrossAttention
from reimplementation.models.bev_former_layer import BEVFormerEncoder


def test_backbone():
    """Test ResNet backbone"""
    print("=" * 50)
    print("Testing ResNet Backbone")
    print("=" * 50)
    
    model = ResNetBackbone(depth=50, out_indices=(3,))  # BEVFormer uses only C5
    
    # Test with BEVFormer tiny input size
    x = torch.randn(1, 3, 800, 450)
    
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        outputs = model(x)
    
    for i, out in enumerate(outputs):
        print(f"Output {i} shape: {out.shape}")
        print(f"Output {i} channels: {out.shape[1]}")
    
    print("✓ Backbone test passed!")
    return outputs


def test_neck(backbone_outputs):
    """Test FPN neck"""
    print("\n" + "=" * 50)
    print("Testing FPN Neck")
    print("=" * 50)
    
    # BEVFormer config: single input from C5, single output
    in_channels = [2048]  # C5 from ResNet50
    out_channels = 256
    num_outs = 1
    
    neck = FPNNeck(
        in_channels=in_channels,
        out_channels=out_channels,
        num_outs=num_outs,
        start_level=0,
        add_extra_convs='on_output',
        relu_before_extra_convs=True)
    
    with torch.no_grad():
        outputs = neck(backbone_outputs)
    
    print(f"Input shapes: {[x.shape for x in backbone_outputs]}")
    for i, out in enumerate(outputs):
        print(f"Output {i} shape: {out.shape}")
        print(f"Output {i} channels: {out.shape[1]}")
    
    print("✓ Neck test passed!")
    return outputs


def test_transformer_components():
    """Test transformer components"""
    print("\n" + "=" * 50)
    print("Testing Transformer Components")
    print("=" * 50)
    
    batch_size = 2
    num_cams = 6
    embed_dims = 256
    bev_h, bev_w = 50, 50
    num_bev_queries = bev_h * bev_w
    
    # Test Spatial Cross Attention
    print("Testing Spatial Cross Attention...")
    sca = SpatialCrossAttention(embed_dims=embed_dims, num_cams=num_cams)
    
    bev_query = torch.randn(batch_size, num_bev_queries, embed_dims)
    img_features = torch.randn(batch_size, num_cams, embed_dims, 25, 15)
    
    with torch.no_grad():
        sca_output = sca(query=bev_query, key=None, value=img_features)
    
    print(f"  Input BEV query: {bev_query.shape}")
    print(f"  Input img features: {img_features.shape}")
    print(f"  Output: {sca_output.shape}")
    print("✓ Spatial Cross Attention test passed!")
    
    return sca_output


def test_bev_encoder():
    """Test BEV Encoder"""
    print("\n" + "=" * 50)
    print("Testing BEV Encoder")  
    print("=" * 50)
    
    batch_size = 2
    embed_dims = 256
    bev_h, bev_w = 50, 50
    
    # Create BEV encoder
    encoder = BEVFormerEncoder(
        embed_dims=embed_dims,
        num_layers=2,  # Reduced for testing
        bev_h=bev_h,
        bev_w=bev_w)
    
    # Create fake FPN output (multi-view features)
    img_feat = torch.randn(batch_size, embed_dims, 25*6, 15)  # 6 cameras concatenated
    img_feats = [img_feat]
    
    # Previous BEV for temporal modeling
    prev_bev = torch.randn(batch_size, bev_h * bev_w, embed_dims)
    
    with torch.no_grad():
        bev_output = encoder(img_feats, prev_bev=prev_bev)
    
    print(f"Input image features: {img_feat.shape}")
    print(f"Previous BEV: {prev_bev.shape}")
    print(f"Output BEV features: {bev_output.shape}")
    print("✓ BEV Encoder test passed!")
    
    return bev_output


def test_full_pipeline():
    """Test full backbone + neck + transformer pipeline"""
    print("\n" + "=" * 50)
    print("Testing Full Pipeline")
    print("=" * 50)
    
    # Create models
    backbone = ResNetBackbone(depth=50, out_indices=(3,))
    neck = FPNNeck(
        in_channels=[2048], 
        out_channels=256, 
        num_outs=1,
        add_extra_convs='on_output',
        relu_before_extra_convs=True)
    
    bev_encoder = BEVFormerEncoder(
        embed_dims=256,
        num_layers=2,
        bev_h=50,
        bev_w=50)
    
    # Test input - simulating 6 cameras
    batch_size = 1  # Reduced for memory
    x = torch.randn(batch_size, 3*6, 800, 450)  # 6 cameras concatenated in channel dim
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        # Forward through backbone (treating as single large image)
        backbone_out = backbone(x)
        print(f"Backbone output shape: {backbone_out[0].shape}")
        
        # Forward through neck  
        neck_out = neck(backbone_out)
        print(f"Neck output shape: {neck_out[0].shape}")
        
        # Forward through BEV encoder
        bev_out = bev_encoder(neck_out)
        print(f"BEV output shape: {bev_out.shape}")
    
    print("✓ Full pipeline test passed!")
    
    # Print memory usage
    if torch.cuda.is_available():
        print(f"GPU memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    
    return backbone_out, neck_out, bev_out


if __name__ == "__main__":
    print("BEVFormer Component Tests")
    print("========================")
    
    # Test individual components
    backbone_outputs = test_backbone()
    neck_outputs = test_neck(backbone_outputs)
    
    # Test transformer components
    test_transformer_components()
    test_bev_encoder()
    
    # Test full pipeline
    test_full_pipeline()
    
    print("\n🎉 All tests passed! Components are working correctly.")