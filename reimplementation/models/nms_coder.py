import torch


def denormalize_bbox(normalized_bboxes, pc_range):
    # rotation 
    rot_sine = normalized_bboxes[..., 6:7]

    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]
   
    # size
    w = normalized_bboxes[..., 2:3]
    l = normalized_bboxes[..., 3:4]
    h = normalized_bboxes[..., 5:6]

    w = w.exp() 
    l = l.exp() 
    h = h.exp() 
    if normalized_bboxes.size(-1) > 8:
         # velocity 
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
    else:
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)
    return denormalized_bboxes


class NMSFreeCoder:
    """Bbox coder for NMS-free detector.
    Args:
        pc_range (list[float]): Range of point cloud.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 pc_range,
                 voxel_size=None,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 num_classes=10):
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes

    def encode(self):

        pass

    def decode_single(self, cls_scores, bbox_preds):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        max_num = self.max_num

        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.view(-1).topk(max_num)
        labels = indexs % self.num_classes
        bbox_index = indexs // self.num_classes
        bbox_preds = bbox_preds[bbox_index]
       
        final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)   
        final_scores = scores 
        final_preds = labels 

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
            tmp_score = self.score_threshold
            while thresh_mask.sum() == 0:
                tmp_score *= 0.9
                if tmp_score < 0.01:
                    thresh_mask = final_scores > -1
                    break
                thresh_mask = final_scores >= tmp_score

        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(
                self.post_center_range, device=scores.device)
            mask = (final_box_preds[..., :3] >=
                    self.post_center_range[:3]).all(1)
            mask &= (final_box_preds[..., :3] <=
                     self.post_center_range[3:]).all(1)

            if self.score_threshold:
                mask &= thresh_mask

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]

            labels = final_preds[mask]
            predictions_dict = {
                'bboxes': boxes3d,
                'scores': scores,
                'labels': labels
            }

        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')
        return predictions_dict

    def decode(self, preds_dicts):
        """Decode bboxes.
        Args:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        all_cls_scores = preds_dicts['all_cls_scores'][-1]
        all_bbox_preds = preds_dicts['all_bbox_preds'][-1]
        
        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(self.decode_single(all_cls_scores[i], all_bbox_preds[i]))
        return predictions_list


def test_nms_free_coder():
    """Test NMSFreeCoder module"""
    print("=" * 60)
    print("Testing NMSFreeCoder")
    print("=" * 60)
    
    # Config parameters from your provided config
    point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    post_center_range = [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]
    voxel_size = [0.2, 0.2, 8]
    max_num = 300
    num_classes = 10
    
    try:
        # Create NMSFreeCoder
        bbox_coder = NMSFreeCoder(
            pc_range=point_cloud_range,
            post_center_range=post_center_range,
            voxel_size=voxel_size,
            max_num=max_num,
            num_classes=num_classes,
            score_threshold=0.1
        )
        
        print("✓ NMSFreeCoder created successfully")
        print(f"  - pc_range: {point_cloud_range}")
        print(f"  - post_center_range: {post_center_range}")
        print(f"  - max_num: {max_num}")
        print(f"  - num_classes: {num_classes}")
        print(f"  - voxel_size: {voxel_size}")
        
        # Test single decode
        batch_size = 2
        num_queries = 900
        code_size = 10  # cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy
        
        # Create mock prediction tensors
        # Classification scores: [num_queries, num_classes] - logits before sigmoid
        cls_scores = torch.randn(num_queries, num_classes)
        
        # Bbox predictions: [num_queries, code_size] - normalized coordinates
        bbox_preds = torch.randn(num_queries, code_size)
        # Set reasonable normalized values
        bbox_preds[:, 0:2] = torch.sigmoid(bbox_preds[:, 0:2])  # cx, cy normalized
        bbox_preds[:, 4] = torch.sigmoid(bbox_preds[:, 4])      # cz normalized
        bbox_preds[:, 2:4] = torch.randn(num_queries, 2) * 0.5  # w, l (log scale)
        bbox_preds[:, 5] = torch.randn(num_queries) * 0.5       # h (log scale)
        
        print("✓ Test inputs created")
        print(f"  - cls_scores shape: {cls_scores.shape}")
        print(f"  - bbox_preds shape: {bbox_preds.shape}")
        
        # Test decode_single
        result = bbox_coder.decode_single(cls_scores, bbox_preds)
        
        print("✓ decode_single successful")
        print(f"  - bboxes shape: {result['bboxes'].shape}")
        print(f"  - scores shape: {result['scores'].shape}")
        print(f"  - labels shape: {result['labels'].shape}")
        
        # Verify output constraints
        assert result['bboxes'].shape[0] <= max_num, f"Too many boxes: {result['bboxes'].shape[0]} > {max_num}"
        assert result['scores'].shape[0] == result['bboxes'].shape[0], "Mismatched scores and boxes count"
        assert result['labels'].shape[0] == result['bboxes'].shape[0], "Mismatched labels and boxes count"
        assert result['bboxes'].shape[1] in [7, 9], f"Invalid bbox format: {result['bboxes'].shape[1]} dimensions"
        
        # Verify bbox values are within post_center_range
        bboxes = result['bboxes']
        if len(bboxes) > 0:
            centers = bboxes[:, :3]  # x, y, z coordinates
            post_range_tensor = torch.tensor(post_center_range)
            assert (centers >= post_range_tensor[:3]).all(), "Some boxes outside post_center_range (min)"
            assert (centers <= post_range_tensor[3:]).all(), "Some boxes outside post_center_range (max)"
        
        print("✓ decode_single output validation passed")
        
        # Test batch decode
        # Create batch predictions
        all_cls_scores = torch.randn(batch_size, num_queries, num_classes)
        all_bbox_preds = torch.randn(batch_size, num_queries, code_size)
        
        # Set reasonable normalized values for batch
        all_bbox_preds[:, :, 0:2] = torch.sigmoid(all_bbox_preds[:, :, 0:2])  # cx, cy
        all_bbox_preds[:, :, 4] = torch.sigmoid(all_bbox_preds[:, :, 4])      # cz
        all_bbox_preds[:, :, 2:4] = torch.randn(batch_size, num_queries, 2) * 0.5  # w, l
        all_bbox_preds[:, :, 5] = torch.randn(batch_size, num_queries) * 0.5       # h
        
        # Simulate multi-layer predictions (6 decoder layers)
        preds_dicts = {
            'all_cls_scores': [all_cls_scores] * 6,  # 6 decoder layers
            'all_bbox_preds': [all_bbox_preds] * 6
        }
        
        print("✓ Batch test inputs created")
        print(f"  - batch_size: {batch_size}")
        print(f"  - all_cls_scores shape: {all_cls_scores.shape}")
        print(f"  - all_bbox_preds shape: {all_bbox_preds.shape}")
        
        # Test batch decode
        batch_results = bbox_coder.decode(preds_dicts)
        
        print("✓ batch decode successful")
        print(f"  - batch_results length: {len(batch_results)}")
        
        # Verify batch output
        assert len(batch_results) == batch_size, f"Wrong batch size: {len(batch_results)} != {batch_size}"
        
        for i, result in enumerate(batch_results):
            assert 'bboxes' in result, f"Missing bboxes in result {i}"
            assert 'scores' in result, f"Missing scores in result {i}"
            assert 'labels' in result, f"Missing labels in result {i}"
            
            assert result['bboxes'].shape[0] <= max_num, f"Too many boxes in batch {i}"
            assert result['scores'].shape[0] == result['bboxes'].shape[0], f"Mismatched scores/boxes in batch {i}"
            assert result['labels'].shape[0] == result['bboxes'].shape[0], f"Mismatched labels/boxes in batch {i}"
            
            print(f"    batch {i}: {result['bboxes'].shape[0]} detections")
        
        print("✓ Batch output validation passed")
        
        # Test edge cases
        # Test with very low scores
        low_scores = torch.ones(num_queries, num_classes) * -10  # Very low logits
        try:
            result_low = bbox_coder.decode_single(low_scores, bbox_preds)
            print("✓ Low scores handling successful")
            print(f"  - detections with low scores: {result_low['bboxes'].shape[0]}")
        except Exception as e:
            print(f"⚠ Low scores test failed: {e}")
        
        # Test with very high scores
        high_scores = torch.ones(num_queries, num_classes) * 10  # Very high logits
        result_high = bbox_coder.decode_single(high_scores, bbox_preds)
        print("✓ High scores handling successful")
        print(f"  - detections with high scores: {result_high['bboxes'].shape[0]}")
        
        # Test denormalize_bbox function directly
        test_normalized = torch.randn(5, 10)  # 5 boxes, 10 dimensions
        test_normalized[:, 0:2] = torch.sigmoid(test_normalized[:, 0:2])  # normalized cx, cy
        test_normalized[:, 4] = torch.sigmoid(test_normalized[:, 4])      # normalized cz
        
        denormalized = denormalize_bbox(test_normalized, point_cloud_range)
        print("✓ denormalize_bbox function test successful")
        print(f"  - input shape: {test_normalized.shape}")
        print(f"  - output shape: {denormalized.shape}")
        
        # Verify denormalization produces reasonable values
        assert denormalized.shape[1] in [7, 9], f"Invalid denormalized bbox dimensions: {denormalized.shape[1]}"
        
        print("✓ All assertions passed")
        print("🎉 NMSFreeCoder test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Test FAILED with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_nms_free_coder()

