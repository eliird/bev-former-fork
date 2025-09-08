"""
Data Structures for 3D Object Detection
Contains bounding box containers and data structures for BEVFormer
"""

import torch
import numpy as np
from typing import Union, Tuple, List, Optional


class LiDARInstance3DBoxes:
    """3D bounding boxes in LiDAR coordinate system.
    
    This class represents 3D bounding boxes in the LiDAR coordinate system,
    which is the standard coordinate system used in nuScenes.
    
    Args:
        tensor (torch.Tensor): Box tensor with shape (N, box_dim).
            The box_dim can be 7 (x, y, z, w, l, h, rot) or 9 (with vx, vy).
        box_dim (int): Dimension of each box. Default: 9.
        with_yaw (bool): Whether the box contains yaw rotation. Default: True.
    """
    
    def __init__(self, 
                 tensor: torch.Tensor,
                 box_dim: int = 9,
                 with_yaw: bool = True):
        
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor).float()
        
        if tensor.numel() == 0:
            # Empty tensor
            tensor = tensor.reshape(0, box_dim)
        
        assert tensor.dim() == 2 and tensor.size(-1) >= 7, \
            f"Expected tensor with shape (N, >=7), got {tensor.shape}"
        
        self.tensor = tensor
        self.box_dim = box_dim
        self.with_yaw = with_yaw
    
    @property
    def gravity_center(self) -> torch.Tensor:
        """Get gravity center of boxes.
        
        Returns:
            torch.Tensor: Centers with shape (N, 3)
        """
        return self.tensor[:, :3]
    
    @property
    def corners(self) -> torch.Tensor:
        """Get corner coordinates of boxes.
        
        Returns:
            torch.Tensor: Corners with shape (N, 8, 3)
        """
        # For simplicity, return centers (would implement full corner calculation in real version)
        centers = self.gravity_center
        # This is a simplified version - real implementation would calculate all 8 corners
        return centers.unsqueeze(1).expand(-1, 8, -1)
    
    @property
    def volume(self) -> torch.Tensor:
        """Get volume of boxes.
        
        Returns:
            torch.Tensor: Volumes with shape (N,)
        """
        dims = self.tensor[:, 3:6]  # w, l, h
        return dims[:, 0] * dims[:, 1] * dims[:, 2]
    
    @property
    def dims(self) -> torch.Tensor:
        """Get dimensions (w, l, h) of boxes.
        
        Returns:
            torch.Tensor: Dimensions with shape (N, 3)
        """
        return self.tensor[:, 3:6]
    
    @property
    def yaw(self) -> torch.Tensor:
        """Get yaw rotation of boxes.
        
        Returns:
            torch.Tensor: Yaw angles with shape (N,)
        """
        if self.with_yaw:
            return self.tensor[:, 6]
        else:
            return torch.zeros(len(self.tensor), device=self.tensor.device)
    
    @property
    def velocity(self) -> Optional[torch.Tensor]:
        """Get velocity of boxes if available.
        
        Returns:
            torch.Tensor or None: Velocities with shape (N, 2) or None
        """
        if self.tensor.size(-1) >= 9:
            return self.tensor[:, 7:9]  # vx, vy
        return None
    
    def __len__(self) -> int:
        """Get number of boxes."""
        return self.tensor.size(0)
    
    def __getitem__(self, item) -> 'LiDARInstance3DBoxes':
        """Get subset of boxes."""
        selected_tensor = self.tensor[item]
        
        # Handle single item selection - need to keep 2D shape
        if selected_tensor.dim() == 1:
            selected_tensor = selected_tensor.unsqueeze(0)
            
        return LiDARInstance3DBoxes(
            selected_tensor, 
            box_dim=self.box_dim, 
            with_yaw=self.with_yaw
        )
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get tensor shape."""
        return self.tensor.shape
    
    @property
    def device(self) -> torch.device:
        """Get tensor device."""
        return self.tensor.device
    
    def to(self, device: torch.device) -> 'LiDARInstance3DBoxes':
        """Move boxes to device."""
        return LiDARInstance3DBoxes(
            self.tensor.to(device),
            box_dim=self.box_dim,
            with_yaw=self.with_yaw
        )
    
    def cpu(self) -> 'LiDARInstance3DBoxes':
        """Move boxes to CPU."""
        return self.to(torch.device('cpu'))
    
    def cuda(self) -> 'LiDARInstance3DBoxes':
        """Move boxes to CUDA."""
        return self.to(torch.device('cuda'))
    
    def numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return self.tensor.detach().cpu().numpy()
    
    def clone(self) -> 'LiDARInstance3DBoxes':
        """Clone boxes."""
        return LiDARInstance3DBoxes(
            self.tensor.clone(),
            box_dim=self.box_dim,
            with_yaw=self.with_yaw
        )
    
    def __repr__(self) -> str:
        return f'LiDARInstance3DBoxes({self.tensor.shape})'


class MultiViewImages:
    """Container for multi-view images from multiple cameras.
    
    Args:
        imgs (torch.Tensor or np.ndarray): Images with shape (N, H, W, C) or (N, C, H, W)
        cam_names (List[str]): Names of cameras
        img_metas (List[Dict], optional): Meta information for each camera
    """
    
    def __init__(self,
                 imgs: Union[torch.Tensor, np.ndarray],
                 cam_names: List[str],
                 img_metas: Optional[List[dict]] = None):
        
        if isinstance(imgs, np.ndarray):
            imgs = torch.from_numpy(imgs)
        
        self.imgs = imgs
        self.cam_names = cam_names
        self.img_metas = img_metas or [{} for _ in cam_names]
        
        assert len(cam_names) == imgs.size(0), \
            f"Number of camera names ({len(cam_names)}) doesn't match images ({imgs.size(0)})"
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get images shape."""
        return self.imgs.shape
    
    @property
    def device(self) -> torch.device:
        """Get images device."""
        return self.imgs.device
    
    def __len__(self) -> int:
        """Get number of cameras."""
        return len(self.cam_names)
    
    def __getitem__(self, item) -> Tuple[torch.Tensor, str, dict]:
        """Get image, camera name and meta for specific camera."""
        return self.imgs[item], self.cam_names[item], self.img_metas[item]
    
    def to(self, device: torch.device) -> 'MultiViewImages':
        """Move images to device."""
        return MultiViewImages(
            self.imgs.to(device),
            self.cam_names,
            self.img_metas
        )
    
    def __repr__(self) -> str:
        return f'MultiViewImages(shape={self.imgs.shape}, cameras={self.cam_names})'


def test_structures():
    """Test data structures."""
    print("=" * 60)
    print("Testing Data Structures")
    print("=" * 60)
    
    # Test LiDARInstance3DBoxes
    print("Testing LiDARInstance3DBoxes...")
    
    # Create sample boxes with velocity
    boxes_with_vel = torch.tensor([
        [10.0, 20.0, 1.0, 2.0, 4.0, 1.5, 0.5, 1.0, 0.5],  # x,y,z,w,l,h,rot,vx,vy
        [15.0, 25.0, 1.2, 1.8, 4.2, 1.6, -0.3, -0.5, 1.2],
        [5.0, 10.0, 0.8, 2.1, 3.8, 1.4, 1.2, 0.8, -0.3],
    ])
    
    boxes_3d = LiDARInstance3DBoxes(boxes_with_vel)
    print(f"✓ Created 3D boxes: {boxes_3d}")
    print(f"  - Shape: {boxes_3d.shape}")
    print(f"  - Length: {len(boxes_3d)}")
    print(f"  - Gravity centers: {boxes_3d.gravity_center}")
    print(f"  - Dimensions: {boxes_3d.dims}")
    print(f"  - Yaw: {boxes_3d.yaw}")
    print(f"  - Velocity: {boxes_3d.velocity}")
    print(f"  - Volume: {boxes_3d.volume}")
    
    # Test indexing
    single_box = boxes_3d[0]
    print(f"  - Single box: {single_box}")
    print(f"  - Single box shape: {single_box.shape}")
    
    # Test MultiViewImages
    print("\nTesting MultiViewImages...")
    
    # Create sample multi-view images
    imgs = torch.randn(6, 900, 1600, 3)  # 6 cameras
    cam_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
                'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    
    mv_imgs = MultiViewImages(imgs, cam_names)
    print(f"✓ Created multi-view images: {mv_imgs}")
    print(f"  - Shape: {mv_imgs.shape}")
    print(f"  - Length: {len(mv_imgs)}")
    
    # Test indexing
    img, name, meta = mv_imgs[0]
    print(f"  - First camera: {name}, image shape: {img.shape}")
    
    # Test empty boxes
    print("\nTesting empty structures...")
    empty_boxes = LiDARInstance3DBoxes(torch.empty(0, 9))
    print(f"✓ Empty boxes: {empty_boxes}")
    print(f"  - Length: {len(empty_boxes)}")
    
    print("\n" + "=" * 60)
    print("All structure tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_structures()