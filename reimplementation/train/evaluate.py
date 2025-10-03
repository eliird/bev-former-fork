#!/usr/bin/env python3
"""
BEVFormer Evaluation Script
Compute mAP and NDS metrics on the complete validation dataset
"""

import argparse
import torch
from pathlib import Path
import time
import sys
import yaml
import signal

# Add parent directory to path
sys.path.append('..')

from models import BEVFormer
from dataset import NuScenesDataset, custom_collate_fn
from torch.utils.data import DataLoader
from evaluation import calculate_nds_map, extract_detections_from_model_output
from tqdm import tqdm


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config):
    """Create BEVFormer model from config."""
    model_config = config.get('model', {})
    data_config = config.get('data', {})

    # Model dimensions
    embed_dims = model_config.get('embed_dims', 256)
    encoder_layers = model_config.get('encoder_layers', 3)
    decoder_layers = model_config.get('decoder_layers', 3)
    num_query = model_config.get('num_query', 900)
    bev_h = model_config.get('bev_h', 200)
    bev_w = model_config.get('bev_w', 200)
    use_grid_mask = model_config.get('use_grid_mask', False)

    # Backbone configuration
    backbone_config = model_config.get('backbone', {})
    backbone_depth = backbone_config.get('depth', 50)
    num_stages = backbone_config.get('num_stages', 4)
    out_indices = backbone_config.get('out_indices', [1, 2, 3])
    frozen_stages = backbone_config.get('frozen_stages', 1)
    use_checkpoint = backbone_config.get('with_cp', False)

    # Neck configuration
    neck_config = model_config.get('neck', {})
    in_channels = neck_config.get('in_channels', [512, 1024, 2048])
    out_channels = neck_config.get('out_channels', 256)
    num_outs = neck_config.get('num_outs', 4)
    start_level = neck_config.get('start_level', 0)
    add_extra_convs = neck_config.get('add_extra_convs', 'on_output')
    relu_before_extra_convs = neck_config.get('relu_before_extra_convs', True)

    # Transformer configuration
    transformer_config = model_config.get('transformer', {})
    num_points_in_pillar = transformer_config.get('num_points_in_pillar', 4)
    return_intermediate_encoder = transformer_config.get('return_intermediate_encoder', False)
    return_intermediate_decoder = transformer_config.get('return_intermediate_decoder', True)
    rotate_prev_bev = transformer_config.get('rotate_prev_bev', True)
    use_shift = transformer_config.get('use_shift', True)
    use_can_bus = transformer_config.get('use_can_bus', True)

    # Bbox coder configuration
    bbox_coder_config = model_config.get('bbox_coder', {})
    post_center_range = bbox_coder_config.get('post_center_range', [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0])
    max_num = bbox_coder_config.get('max_num', 300)

    # Loss configuration
    loss_config = model_config.get('loss', {})
    cls_loss_config = loss_config.get('cls', {})
    bbox_loss_config = loss_config.get('bbox', {})
    iou_loss_config = loss_config.get('iou', {})

    # Point cloud range and class names from data config
    point_cloud_range = data_config.get('pc_range', [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
    class_names = data_config.get('class_names', [
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
        'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ])

    model_cfg = dict(
        img_backbone=dict(
            depth=backbone_depth,
            num_stages=num_stages,
            out_indices=out_indices,
            frozen_stages=frozen_stages,
            with_cp=use_checkpoint
        ),
        img_neck=dict(
            in_channels=in_channels,
            out_channels=out_channels,
            num_outs=num_outs,
            start_level=start_level,
            add_extra_convs=add_extra_convs,
            relu_before_extra_convs=relu_before_extra_convs
        ),
        pts_bbox_head=dict(
            num_classes=len(class_names),
            in_channels=embed_dims,
            num_query=num_query,
            bev_h=bev_h,
            bev_w=bev_w,
            transformer=dict(
                embed_dims=embed_dims,
                encoder=dict(
                    num_layers=encoder_layers,
                    pc_range=point_cloud_range,
                    num_points_in_pillar=num_points_in_pillar,
                    return_intermediate=return_intermediate_encoder
                ),
                decoder=dict(
                    num_layers=decoder_layers,
                    return_intermediate=return_intermediate_decoder
                ),
                rotate_prev_bev=rotate_prev_bev,
                use_shift=use_shift,
                use_can_bus=use_can_bus
            ),
            bbox_coder=dict(
                pc_range=point_cloud_range,
                post_center_range=post_center_range,
                max_num=max_num,
                num_classes=len(class_names)
            ),
            loss_cls=dict(
                use_sigmoid=cls_loss_config.get('use_sigmoid', True),
                gamma=cls_loss_config.get('gamma', 2.0),
                alpha=cls_loss_config.get('alpha', 0.25),
                loss_weight=cls_loss_config.get('loss_weight', 2.0)
            ),
            loss_bbox=dict(
                loss_weight=bbox_loss_config.get('loss_weight', 0.25)
            ),
            loss_iou=dict(
                loss_weight=iou_loss_config.get('loss_weight', 0.0)
            ),
            train_cfg=dict(
                pts=dict(
                    assigner=dict(
                        pc_range=point_cloud_range
                    )
                )
            )
        ),
        use_grid_mask=use_grid_mask,
        video_test_mode=False
    )

    model = BEVFormer(**model_cfg)
    return model


# Global variables for signal handling
evaluation_interrupted = False
predictions_partial = []
ground_truths_partial = []

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global evaluation_interrupted
    print(f"\n\n🛑 Evaluation interrupted by user (signal {signum})")
    print("Will compute metrics using partial results collected so far...")
    evaluation_interrupted = True

def evaluate_model(checkpoint_path, config_path, val_pkl, batch_size=1):
    """
    Evaluate BEVFormer model on validation dataset.

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to YAML config file
        val_pkl: Path to validation pickle file
        batch_size: Batch size for evaluation
    """
    # Load config
    config = load_config(config_path)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model using the same approach as training
    print("Creating model...")
    model = create_model(config)

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present (from DDP training)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    model.eval()

    # Create dataset
    print("Creating validation dataset...")
    data_config = config.get('data', {})
    queue_length = data_config.get('queue_length', 4)
    point_cloud_range = data_config.get('pc_range', [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
    class_names = data_config.get('class_names', [
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
        'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ])

    val_dataset = NuScenesDataset(
        data_file=val_pkl,
        queue_length=queue_length,
        training=False,
        point_cloud_range=point_cloud_range,
        class_names=class_names
    )

    # Create dataloader with custom collate function
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    print(f"Validation samples: {len(val_dataset)}")
    print(f"Validation batches: {len(val_loader)}")

    # Setup signal handler for graceful interruption
    global evaluation_interrupted, predictions_partial, ground_truths_partial
    evaluation_interrupted = False
    predictions_partial = []
    ground_truths_partial = []

    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    # Evaluation using the same temporal approach as training
    predictions = []
    ground_truths = []
    total_time = 0

    print("\nRunning evaluation with temporal sequences...")
    print("Using same temporal processing approach as training...")
    print("Press Ctrl+C to stop evaluation and compute metrics with partial results...")

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(val_loader, desc="Evaluating")):
            # Check for interruption
            if evaluation_interrupted:
                print(f"\nBreaking evaluation loop at batch {batch_idx}")
                break

            # Check if we have the required data
            if 'img' not in data or data['img'] is None:
                continue

            # Handle temporal sequences - same as training
            # img shape: [B, T, N, C, H, W] where T=queue_length (usually 4)
            batch_img = data['img']
            B = batch_img.shape[0]
            T = batch_img.shape[1] if len(batch_img.shape) > 5 else 1

            # Move to device
            if torch.cuda.is_available():
                batch_img = batch_img.cuda()

            for sample_idx in range(B):
                # Get ground truth for this sample (always for the last frame)
                gt_dict = {
                    'gt_bboxes_3d': data.get('gt_bboxes_3d', [None])[sample_idx] if 'gt_bboxes_3d' in data else torch.zeros((0, 9)),
                    'gt_labels_3d': data.get('gt_labels_3d', [None])[sample_idx] if 'gt_labels_3d' in data else torch.zeros(0, dtype=torch.long)
                }

                # Handle case where GT might be None
                if gt_dict['gt_bboxes_3d'] is None:
                    gt_dict['gt_bboxes_3d'] = torch.zeros((0, 9))
                if gt_dict['gt_labels_3d'] is None:
                    gt_dict['gt_labels_3d'] = torch.zeros(0, dtype=torch.long)

                ground_truths.append(gt_dict)

                # Run model inference - same approach as training
                start_time = time.time()

                try:
                    if T > 1:
                        # Use the training-style temporal processing
                        # Extract single sample as batch: [T, N, C, H, W] -> [1, T, N, C, H, W]
                        sample_img = batch_img[sample_idx:sample_idx+1]
                        sample_metas = [data['img_metas'][sample_idx]] if 'img_metas' in data else [[{}]*T]

                        # Split into previous and current frames (same as training)
                        prev_img = sample_img[:, :-1, ...]  # [1, T-1, N, C, H, W]
                        curr_img = sample_img[:, -1, ...]   # [1, N, C, H, W]

                        # Get previous BEV features using obtain_history_bev (same as training)
                        if prev_img.shape[1] > 0:  # If we have previous frames
                            # Clear cache before expensive BEV computation
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            prev_bev = model.obtain_history_bev(prev_img, sample_metas)
                            # Clear cache after BEV computation
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        else:
                            prev_bev = None

                        # Get current frame metadata
                        curr_meta = sample_metas[0][T-1] if len(sample_metas[0]) > T-1 else sample_metas[0][-1]

                        # Check if previous BEV should exist
                        if not curr_meta.get('prev_bev_exists', True):
                            prev_bev = None

                        # Run inference on current frame with accumulated BEV context
                        model_output = model.forward_test(
                            img=[curr_img[0]],  # Remove batch dim: [N, C, H, W]
                            img_metas=[[curr_meta]],
                            rescale=True
                        )

                        # Clear intermediate variables to save memory
                        del prev_img, curr_img, sample_img
                        if prev_bev is not None:
                            del prev_bev
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        # Single frame - use direct forward_test
                        curr_img = batch_img[sample_idx]  # [N, C, H, W]
                        curr_meta = data['img_metas'][sample_idx] if 'img_metas' in data else {}

                        model_output = model.forward_test(
                            img=[curr_img],
                            img_metas=[[curr_meta]],
                            rescale=True
                        )

                    # Extract detections
                    detections = extract_detections_from_model_output(model_output)
                    predictions.append(detections)

                    # Store in global variables for signal handler access
                    predictions_partial.append(detections)
                    ground_truths_partial.append(gt_dict)

                except Exception as e:
                    print(f"\nError in batch {batch_idx}, sample {sample_idx}: {e}")
                    # Add empty prediction on error
                    empty_pred = {
                        'boxes_3d': torch.zeros((0, 9)),
                        'scores_3d': torch.zeros(0),
                        'labels_3d': torch.zeros(0, dtype=torch.long)
                    }
                    predictions.append(empty_pred)
                    predictions_partial.append(empty_pred)
                    ground_truths_partial.append(gt_dict)

                total_time += time.time() - start_time

            # Clear GPU cache after each batch to prevent memory buildup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Progress update every 50 batches
            if (batch_idx + 1) % 50 == 0:
                avg_time = total_time / ((batch_idx + 1) * batch_size)
                print(f"Batch {batch_idx + 1}/{len(val_loader)} | Avg time: {avg_time:.3f}s/sample")

    # Use partial results if evaluation was interrupted
    final_predictions = predictions_partial if evaluation_interrupted else predictions
    final_ground_truths = ground_truths_partial if evaluation_interrupted else ground_truths

    # Calculate metrics
    print("\nCalculating metrics...")
    if evaluation_interrupted:
        print(f"🔄 Using partial results from interrupted evaluation")
    print(f"Total predictions collected: {len(final_predictions)}")
    print(f"Total ground truths collected: {len(final_ground_truths)}")

    if len(final_predictions) == 0:
        print("❌ No predictions collected. Cannot compute metrics.")
        return None

    class_names = config['data']['class_names']
    print(f"Using {len(class_names)} classes: {class_names}")

    try:
        print("🔄 Computing mAP and NDS metrics (this may take a few minutes)...")
        metrics = calculate_nds_map(final_predictions, final_ground_truths, class_names)
        print("✅ Metrics calculation completed successfully!")
    except KeyboardInterrupt:
        print(f"\n❌ Metrics calculation interrupted by user during computation.")
        print("Unfortunately, partial metrics cannot be computed. Try with fewer samples or simpler evaluation.")
        return None
    except Exception as e:
        print(f"❌ Error during metrics calculation: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Print results
    print("\n" + "="*60)
    if evaluation_interrupted:
        print("PARTIAL EVALUATION RESULTS (INTERRUPTED)")
    else:
        print("EVALUATION RESULTS")
    print("="*60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Total samples evaluated: {len(final_predictions)}")
    if evaluation_interrupted:
        print(f"⚠️  Evaluation was interrupted - results are based on partial data")
        print(f"   Coverage: {len(final_predictions)} / {len(val_dataset)} samples ({len(final_predictions)/len(val_dataset)*100:.1f}%)")
    print(f"Average inference time: {total_time/len(final_predictions):.3f}s/sample")
    print("-"*60)
    print(f"NDS: {metrics['NDS']:.4f}")
    print(f"mAP: {metrics['mAP']:.4f}")
    print("-"*60)
    print("Per-class AP:")
    for class_name, ap in metrics['per_class_AP'].items():
        print(f"  {class_name:30s}: {ap:.4f}")
    print("="*60)

    if evaluation_interrupted:
        print("\n💡 Note: These are partial results. For complete evaluation, run again without interruption.")

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate BEVFormer model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML file')
    parser.add_argument('--val-pkl', type=str,
                       default='../../data/nuscenes/nuscenes_infos_temporal_val.pkl',
                       help='Path to validation pickle file')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for evaluation')

    args = parser.parse_args()

    # Check if files exist
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    if not Path(args.config).exists():
        print(f"Error: Config not found: {args.config}")
        sys.exit(1)

    if not Path(args.val_pkl).exists():
        print(f"Error: Validation pickle not found: {args.val_pkl}")
        sys.exit(1)

    # Run evaluation
    evaluate_model(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        val_pkl=args.val_pkl,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()