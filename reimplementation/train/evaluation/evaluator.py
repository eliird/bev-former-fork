"""
BEVFormer Evaluator
Main evaluation orchestrator that handles model evaluation, metrics computation, and result aggregation
"""

import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .nuscenes_metrics import NuScenesMetrics
from ..utils.distributed import is_main_process, synchronize, DistributedMetrics


class BEVFormerEvaluator:
    """
    Comprehensive evaluator for BEVFormer model
    Handles evaluation loop, metrics computation, and result aggregation
    """

    def __init__(self,
                 model: nn.Module,
                 val_loader: DataLoader,
                 config: Optional[Dict[str, Any]] = None,
                 device: Optional[torch.device] = None,
                 logger: Optional[Any] = None):
        """
        Initialize BEVFormer evaluator

        Args:
            model: BEVFormer model
            val_loader: Validation data loader
            config: Evaluation configuration
            device: Device to run evaluation on
            logger: Logger instance
        """
        self.model = model
        self.val_loader = val_loader
        self.config = config or {}
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logger

        # Evaluation configuration
        eval_config = self.config.get('evaluation', {})
        self.max_eval_samples = eval_config.get('max_eval_samples', -1)
        self.score_threshold = eval_config.get('score_threshold', 0.1)
        self.nms_threshold = eval_config.get('nms_threshold', 0.2)
        self.max_detections = eval_config.get('max_num', 300)

        # Class names
        self.class_names = eval_config.get('class_names', [
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ])

        # Initialize metrics calculator
        self.metrics_calculator = NuScenesMetrics(
            class_names=self.class_names,
            score_threshold=self.score_threshold
        )

        # Results storage
        self.evaluation_results = []

    def evaluate(self,
                epoch: Optional[int] = None,
                save_results: bool = False,
                result_path: Optional[str] = None) -> Dict[str, float]:
        """
        Run full evaluation on validation set

        Args:
            epoch: Current epoch (for logging)
            save_results: Whether to save detailed results
            result_path: Path to save results

        Returns:
            Dictionary of computed metrics
        """
        if self.logger and is_main_process():
            self.logger.info(f"Starting evaluation{f' for epoch {epoch}' if epoch is not None else ''}...")

        start_time = time.time()

        # Set model to evaluation mode
        self.model.eval()

        # Reset metrics calculator
        self.metrics_calculator.reset()
        self.evaluation_results.clear()

        # Evaluation loop
        all_predictions = []
        all_ground_truths = []
        sample_tokens = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                # Limit evaluation samples for speed if specified
                if self.max_eval_samples > 0 and batch_idx >= self.max_eval_samples:
                    if self.logger and is_main_process():
                        self.logger.info(f"Limiting evaluation to {self.max_eval_samples} samples")
                    break

                # Move batch to device
                batch = self._move_batch_to_device(batch)

                # Forward pass
                predictions, ground_truths, batch_sample_tokens = self._evaluate_batch(batch)

                # Store results
                all_predictions.extend(predictions)
                all_ground_truths.extend(ground_truths)
                sample_tokens.extend(batch_sample_tokens)

                # Add to metrics calculator
                for pred, gt, token in zip(predictions, ground_truths, batch_sample_tokens):
                    self.metrics_calculator.add_sample(pred, gt, token)

                # Periodic logging
                if batch_idx % 100 == 0 and self.logger and is_main_process():
                    self.logger.info(f"  Evaluated {batch_idx + 1}/{len(self.val_loader)} batches")

        # Synchronize all processes before computing metrics
        synchronize()

        # Compute metrics
        if self.logger and is_main_process():
            self.logger.info("Computing evaluation metrics...")

        metrics = self.metrics_calculator.compute_metrics()

        # Distributed metric aggregation if needed
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            metrics = DistributedMetrics.gather_metrics(metrics)

        eval_time = time.time() - start_time

        # Log results
        if self.logger and is_main_process():
            self.logger.info(f"Evaluation completed in {eval_time:.2f}s")
            self._log_metrics(metrics, epoch)

        # Save detailed results if requested
        if save_results and is_main_process():
            self._save_evaluation_results(
                all_predictions, all_ground_truths, sample_tokens,
                metrics, result_path, epoch
            )

        return metrics

    def _evaluate_batch(self, batch: Dict[str, Any]) -> Tuple[List[Dict], List[Dict], List[str]]:
        """
        Evaluate a single batch

        Args:
            batch: Input batch

        Returns:
            Tuple of (predictions, ground_truths, sample_tokens)
        """
        # Extract inputs
        imgs = batch['img']
        img_metas = batch['img_metas']
        prev_bev = batch.get('prev_bev', None)

        # Forward pass
        if hasattr(self.model, 'module'):
            # Handle DDP wrapper
            outputs = self.model.module(imgs, img_metas, prev_bev=prev_bev)
        else:
            outputs = self.model(imgs, img_metas, prev_bev=prev_bev)

        # Get predictions
        if hasattr(self.model, 'module'):
            predictions = self.model.module.get_bboxes(outputs, img_metas)
        else:
            predictions = self.model.get_bboxes(outputs, img_metas)

        # Process predictions and ground truths
        batch_predictions = []
        batch_ground_truths = []
        batch_sample_tokens = []

        batch_size = len(predictions)

        for i in range(batch_size):
            # Process prediction
            pred_boxes, pred_scores, pred_labels = predictions[i]

            # Apply NMS if needed
            if self.nms_threshold > 0 and len(pred_boxes) > 0:
                keep_indices = self._apply_nms(pred_boxes, pred_scores, pred_labels)
                pred_boxes = pred_boxes[keep_indices]
                pred_scores = pred_scores[keep_indices]
                pred_labels = pred_labels[keep_indices]

            # Limit number of detections
            if len(pred_boxes) > self.max_detections:
                top_indices = torch.argsort(pred_scores, descending=True)[:self.max_detections]
                pred_boxes = pred_boxes[top_indices]
                pred_scores = pred_scores[top_indices]
                pred_labels = pred_labels[top_indices]

            # Convert to numpy for metrics calculation
            pred_dict = {
                'boxes': pred_boxes.cpu().numpy(),
                'scores': pred_scores.cpu().numpy(),
                'labels': pred_labels.cpu().numpy()
            }

            # Process ground truth
            if 'gt_bboxes_3d' in batch and 'gt_labels_3d' in batch:
                gt_boxes = batch['gt_bboxes_3d'][i]
                gt_labels = batch['gt_labels_3d'][i]

                gt_dict = {
                    'boxes': gt_boxes.cpu().numpy(),
                    'labels': gt_labels.cpu().numpy()
                }
            else:
                # No ground truth available
                gt_dict = {
                    'boxes': np.empty((0, 9), dtype=np.float32),
                    'labels': np.empty((0,), dtype=np.int64)
                }

            # Get sample token
            sample_token = f"sample_{len(batch_sample_tokens)}"
            if isinstance(img_metas, list) and i < len(img_metas):
                sample_token = img_metas[i].get('sample_token', sample_token)

            batch_predictions.append(pred_dict)
            batch_ground_truths.append(gt_dict)
            batch_sample_tokens.append(sample_token)

        return batch_predictions, batch_ground_truths, batch_sample_tokens

    def _apply_nms(self,
                   boxes: torch.Tensor,
                   scores: torch.Tensor,
                   labels: torch.Tensor) -> torch.Tensor:
        """
        Apply Non-Maximum Suppression

        Args:
            boxes: (N, 9) bounding boxes
            scores: (N,) confidence scores
            labels: (N,) class labels

        Returns:
            Indices of boxes to keep
        """
        if len(boxes) == 0:
            return torch.empty((0,), dtype=torch.long, device=boxes.device)

        # Simple implementation - can be improved with proper 3D NMS
        keep_indices = []

        # Apply NMS per class
        unique_labels = torch.unique(labels)
        for label in unique_labels:
            label_mask = labels == label
            if not torch.any(label_mask):
                continue

            label_indices = torch.where(label_mask)[0]
            label_boxes = boxes[label_indices]
            label_scores = scores[label_indices]

            # Sort by score
            sorted_indices = torch.argsort(label_scores, descending=True)

            for i, idx in enumerate(sorted_indices):
                if idx in keep_indices:
                    continue

                keep_indices.append(label_indices[idx].item())

                # Remove overlapping boxes (simplified)
                if i < len(sorted_indices) - 1:
                    current_box = label_boxes[idx]
                    remaining_indices = sorted_indices[i + 1:]

                    for j in remaining_indices:
                        other_box = label_boxes[j]
                        # Compute simple overlap (center distance)
                        distance = torch.norm(current_box[:3] - other_box[:3])
                        if distance < self.nms_threshold:
                            continue

        return torch.tensor(keep_indices, dtype=torch.long, device=boxes.device)

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch data to target device"""
        device_batch = {}

        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device, non_blocking=True)
            elif isinstance(value, list):
                # Handle lists of tensors
                device_batch[key] = [
                    item.to(self.device, non_blocking=True) if isinstance(item, torch.Tensor) else item
                    for item in value
                ]
            else:
                device_batch[key] = value

        return device_batch

    def _log_metrics(self, metrics: Dict[str, float], epoch: Optional[int] = None):
        """Log evaluation metrics"""
        if not self.logger:
            return

        epoch_str = f" - Epoch {epoch}" if epoch is not None else ""
        self.logger.info(f"Evaluation Results{epoch_str}")
        self.logger.info("=" * 80)

        # Main metrics
        main_metrics = ['NDS', 'mAP', 'mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE']
        for metric in main_metrics:
            if metric in metrics:
                self.logger.info(f"{metric}: {metrics[metric]:.4f}")

        # Per-class mAP
        self.logger.info("\nPer-class mAP:")
        for i, class_name in enumerate(self.class_names):
            key = f'mAP_{class_name}'
            if key in metrics:
                self.logger.info(f"  {class_name}: {metrics[key]:.4f}")

        self.logger.info("=" * 80)

    def _save_evaluation_results(self,
                                predictions: List[Dict],
                                ground_truths: List[Dict],
                                sample_tokens: List[str],
                                metrics: Dict[str, float],
                                result_path: Optional[str] = None,
                                epoch: Optional[int] = None):
        """Save detailed evaluation results"""
        if result_path is None:
            result_path = f"evaluation_results_epoch_{epoch}.json" if epoch is not None else "evaluation_results.json"

        result_path = Path(result_path)
        result_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare results for saving
        results = {
            'metrics': metrics,
            'evaluation_info': {
                'num_samples': len(predictions),
                'class_names': self.class_names,
                'score_threshold': self.score_threshold,
                'nms_threshold': self.nms_threshold,
                'max_detections': self.max_detections,
                'epoch': epoch
            },
            'detailed_results': []
        }

        # Add detailed per-sample results (limit to avoid huge files)
        max_detailed_samples = 100
        for i, (pred, gt, token) in enumerate(zip(predictions[:max_detailed_samples],
                                                ground_truths[:max_detailed_samples],
                                                sample_tokens[:max_detailed_samples])):
            sample_result = {
                'sample_token': token,
                'num_predictions': len(pred['boxes']),
                'num_ground_truths': len(gt['boxes']),
                'prediction_scores': pred['scores'].tolist() if len(pred['scores']) > 0 else [],
                'prediction_labels': pred['labels'].tolist() if len(pred['labels']) > 0 else [],
                'ground_truth_labels': gt['labels'].tolist() if len(gt['labels']) > 0 else []
            }
            results['detailed_results'].append(sample_result)

        # Save results
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        if self.logger:
            self.logger.info(f"Detailed evaluation results saved to: {result_path}")

    def evaluate_single_sample(self,
                              sample: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate a single sample (for debugging/analysis)

        Args:
            sample: Single data sample

        Returns:
            Dictionary of metrics for this sample
        """
        self.model.eval()

        with torch.no_grad():
            # Move to device
            sample = self._move_batch_to_device(sample)

            # Add batch dimension if needed
            if 'img' in sample and sample['img'].dim() == 4:
                for key, value in sample.items():
                    if isinstance(value, torch.Tensor):
                        sample[key] = value.unsqueeze(0)

            # Evaluate
            predictions, ground_truths, sample_tokens = self._evaluate_batch(sample)

            if len(predictions) > 0:
                # Compute metrics for this sample
                self.metrics_calculator.reset()
                self.metrics_calculator.add_sample(
                    predictions[0], ground_truths[0], sample_tokens[0]
                )
                metrics = self.metrics_calculator.compute_metrics()
                return metrics

        return {}

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """
        Get summary of evaluation configuration and capabilities

        Returns:
            Dictionary with evaluation summary
        """
        return {
            'evaluator_type': 'BEVFormerEvaluator',
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'score_threshold': self.score_threshold,
            'nms_threshold': self.nms_threshold,
            'max_detections': self.max_detections,
            'max_eval_samples': self.max_eval_samples,
            'supported_metrics': ['NDS', 'mAP', 'mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE'],
            'device': str(self.device)
        }


# Example usage and testing
if __name__ == "__main__":
    print("BEVFormer Evaluator - use with trained models for evaluation")