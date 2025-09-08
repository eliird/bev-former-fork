"""
Simplified GridMask implementation for BEVFormer
Based on the original GridMask paper and implementation
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image


class GridMask(nn.Module):
    """GridMask augmentation for images.
    
    Args:
        use_h (bool): Whether to use horizontal stripes
        use_w (bool): Whether to use vertical stripes  
        rotate (int): Maximum rotation angle
        offset (bool): Whether to add offset
        ratio (float): Ratio of mask to grid
        mode (int): 0 for mask, 1 for inverse mask
        prob (float): Probability of applying GridMask
    """
    
    def __init__(self, 
                 use_h=True, 
                 use_w=True, 
                 rotate=1, 
                 offset=False, 
                 ratio=0.5, 
                 mode=1, 
                 prob=0.7):
        super(GridMask, self).__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob
        
    def set_prob(self, epoch, max_epoch):
        """Adjust probability during training."""
        self.prob = self.st_prob * epoch / max_epoch
        
    def forward(self, x):
        """Apply GridMask to input images.
        
        Args:
            x (Tensor): Input images with shape (N, C, H, W) or (N*num_cams, C, H, W)
            
        Returns:
            Tensor: Augmented images
        """
        # Only apply during training and with probability
        if not self.training or np.random.rand() > self.prob:
            return x
            
        n, c, h, w = x.size()
        x_shape = x.shape
        
        # Flatten batch and channels for processing
        x = x.view(-1, h, w)
        
        # Create larger canvas for rotation
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        
        # Random grid size
        d = np.random.randint(2, min(h, w))
        
        # Calculate stripe width
        self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        
        # Create mask
        mask = np.ones((hh, ww), np.float32)
        
        # Random starting points
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        
        # Apply horizontal stripes
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
                
        # Apply vertical stripes
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0
                
        # Random rotation
        r = np.random.randint(self.rotate) if self.rotate > 1 else 0
        if r > 0:
            mask = Image.fromarray(np.uint8(mask * 255))
            mask = mask.rotate(r)
            mask = np.asarray(mask) / 255.0
            
        # Crop back to original size
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, 
                    (ww - w) // 2:(ww - w) // 2 + w]
        
        # Convert to tensor
        mask = torch.from_numpy(mask).float().to(x.device)
        
        # Inverse mask if mode == 1
        if self.mode == 1:
            mask = 1 - mask
            
        # Expand mask to match input shape
        mask = mask.unsqueeze(0).expand_as(x)
        
        # Apply mask with optional offset
        if self.offset:
            offset = torch.randn_like(x) * 0.1 * (1 - mask)
            x = x * mask + offset
        else:
            x = x * mask
            
        # Reshape back to original
        x = x.view(x_shape)
        
        return x


def test_grid_mask():
    """Test GridMask implementation"""
    print("Testing GridMask...")
    
    # Create GridMask instance
    grid_mask = GridMask(prob=1.0)  # Always apply for testing
    grid_mask.train()  # Set to training mode
    
    # Test single image
    img = torch.randn(1, 3, 224, 224)
    masked_img = grid_mask(img)
    assert masked_img.shape == img.shape
    print(f"✓ Single image test passed: {img.shape} -> {masked_img.shape}")
    
    # Test batch of images
    batch = torch.randn(4, 3, 224, 224)
    masked_batch = grid_mask(batch)
    assert masked_batch.shape == batch.shape
    print(f"✓ Batch test passed: {batch.shape} -> {masked_batch.shape}")
    
    # Test multi-camera batch (typical BEVFormer input)
    multi_cam = torch.randn(2 * 6, 3, 224, 224)  # 2 samples, 6 cameras
    masked_multi = grid_mask(multi_cam)
    assert masked_multi.shape == multi_cam.shape
    print(f"✓ Multi-camera test passed: {multi_cam.shape} -> {masked_multi.shape}")
    
    # Test eval mode (should not apply mask)
    grid_mask.eval()
    eval_out = grid_mask(img)
    assert torch.allclose(eval_out, img)
    print("✓ Eval mode test passed (no masking)")
    
    # Test probability
    grid_mask.train()
    grid_mask.prob = 0.0  # Never apply
    no_mask_out = grid_mask(img)
    assert torch.allclose(no_mask_out, img)
    print("✓ Probability test passed")
    
    print("All GridMask tests passed!")


if __name__ == "__main__":
    test_grid_mask()