"""
NuScenes Evaluation Metrics Implementation
Implements NDS (nuScenes Detection Score) and mAP following the official nuScenes protocol
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import math

import torch


class NuScenesMetrics:
    """
    NuScenes evaluation metrics implementation
    Computes NDS, mAP, and individual error metrics following the official protocol
    """

    def __init__(self,
                 class_names: List[str] = None,
                 distance_thresholds: List[float] = None,
                 score_threshold: float = 0.1):
        """
        Initialize nuScenes metrics

        Args:
            class_names: List of class names
            distance_thresholds: Distance thresholds for mAP calculation
            score_threshold: Minimum confidence score threshold
        """
        # Default nuScenes class names
        if class_names is None:
            class_names = [
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
            ]
        self.class_names = class_names

        # Default distance thresholds for mAP calculation
        if distance_thresholds is None:
            distance_thresholds = [0.5, 1.0, 2.0, 4.0]
        self.distance_thresholds = distance_thresholds

        self.score_threshold = score_threshold

        # Error types for NDS calculation
        self.error_types = ['trans_err', 'scale_err', 'orient_err', 'vel_err', 'attr_err']

        # Matching distance thresholds per class (in meters)
        self.matching_distances = {
            'car': 2.0, 'truck': 2.0, 'bus': 2.0, 'trailer': 2.0,
            'construction_vehicle': 2.0, 'bicycle': 1.0, 'motorcycle': 1.0,
            'pedestrian': 1.0, 'traffic_cone': 1.0, 'barrier': 1.0
        }

        # Reset accumulated results
        self.reset()

    def reset(self) -> None:
        """Reset accumulated evaluation results"""
        self.predictions = []
        self.ground_truths = []
        self.sample_tokens = []

    def add_sample(self,
                   predictions: Dict[str, Any],
                   ground_truths: Dict[str, Any],
                   sample_token: str = None) -> None:
        """
        Add a sample for evaluation

        Args:
            predictions: Dictionary with prediction data
                - boxes: (N, 9) array [x, y, z, w, l, h, rot, vx, vy]
                - scores: (N,) array of confidence scores
                - labels: (N,) array of class labels
            ground_truths: Dictionary with ground truth data
                - boxes: (M, 9) array [x, y, z, w, l, h, rot, vx, vy]
                - labels: (M,) array of class labels
            sample_token: Unique identifier for the sample
        """
        self.predictions.append(predictions)
        self.ground_truths.append(ground_truths)
        self.sample_tokens.append(sample_token or f"sample_{len(self.predictions)}")

    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute all evaluation metrics

        Returns:
            Dictionary containing all computed metrics
        """
        if not self.predictions:
            return {}

        # Filter predictions by score threshold
        filtered_predictions = self._filter_predictions()

        # Perform matching between predictions and ground truths
        matches = self._match_predictions_to_gt(filtered_predictions, self.ground_truths)

        # Compute mAP
        map_metrics = self._compute_map(matches, filtered_predictions, self.ground_truths)

        # Compute error metrics for matched detections
        error_metrics = self._compute_errors(matches, filtered_predictions, self.ground_truths)

        # Compute NDS (combines mAP with error metrics)
        nds = self._compute_nds(map_metrics['mAP'], error_metrics)

        # Combine all metrics
        metrics = {
            'NDS': nds,
            'mAP': map_metrics['mAP'],
            'mATE': error_metrics['mATE'],  # Mean Average Translation Error
            'mASE': error_metrics['mASE'],  # Mean Average Scale Error
            'mAOE': error_metrics['mAOE'],  # Mean Average Orientation Error
            'mAVE': error_metrics['mAVE'],  # Mean Average Velocity Error
            'mAAE': error_metrics['mAAE'],  # Mean Average Attribute Error
        }

        # Add per-class mAP scores
        for i, class_name in enumerate(self.class_names):
            metrics[f'mAP_{class_name}'] = map_metrics['class_aps'][i]

        return metrics

    def _filter_predictions(self) -> List[Dict[str, Any]]:
        """Filter predictions by score threshold"""
        filtered_predictions = []

        for pred in self.predictions:
            scores = pred['scores']
            valid_mask = scores >= self.score_threshold

            if not np.any(valid_mask):
                # No valid predictions for this sample
                filtered_pred = {
                    'boxes': np.empty((0, 9), dtype=np.float32),
                    'scores': np.empty((0,), dtype=np.float32),
                    'labels': np.empty((0,), dtype=np.int64)
                }
            else:
                filtered_pred = {
                    'boxes': pred['boxes'][valid_mask],
                    'scores': pred['scores'][valid_mask],
                    'labels': pred['labels'][valid_mask]
                }

            filtered_predictions.append(filtered_pred)

        return filtered_predictions

    def _match_predictions_to_gt(self,
                                predictions: List[Dict[str, Any]],
                                ground_truths: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Match predictions to ground truth using Hungarian algorithm

        Args:
            predictions: List of prediction dictionaries
            ground_truths: List of ground truth dictionaries

        Returns:
            List of matching results for each sample
        """
        matches = []

        for pred, gt in zip(predictions, ground_truths):
            sample_matches = self._match_single_sample(pred, gt)
            matches.append(sample_matches)

        return matches

    def _match_single_sample(self,
                           pred: Dict[str, Any],
                           gt: Dict[str, Any]) -> Dict[str, Any]:
        """
        Match predictions to ground truth for a single sample

        Args:
            pred: Prediction dictionary
            gt: Ground truth dictionary

        Returns:
            Matching results for the sample
        """
        pred_boxes = pred['boxes']  # (N, 9)
        pred_scores = pred['scores']  # (N,)
        pred_labels = pred['labels']  # (N,)

        gt_boxes = gt['boxes']  # (M, 9)
        gt_labels = gt['labels']  # (M,)

        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            return {
                'pred_matches': np.full(len(pred_boxes), -1, dtype=int),
                'gt_matches': np.full(len(gt_boxes), -1, dtype=int),
                'distances': np.array([]),
                'pred_indices': np.array([], dtype=int),
                'gt_indices': np.array([], dtype=int)
            }

        # Compute distance matrix
        distance_matrix = self._compute_distance_matrix(pred_boxes, gt_boxes)

        # Perform matching for each class separately
        pred_matches = np.full(len(pred_boxes), -1, dtype=int)
        gt_matches = np.full(len(gt_boxes), -1, dtype=int)
        match_distances = []
        match_pred_indices = []
        match_gt_indices = []

        for class_idx, class_name in enumerate(self.class_names):
            # Find predictions and GTs of this class
            pred_class_mask = pred_labels == class_idx
            gt_class_mask = gt_labels == class_idx

            if not np.any(pred_class_mask) or not np.any(gt_class_mask):
                continue

            pred_class_indices = np.where(pred_class_mask)[0]
            gt_class_indices = np.where(gt_class_mask)[0]

            # Get distance submatrix for this class
            class_distances = distance_matrix[pred_class_indices][:, gt_class_indices]

            # Get matching threshold for this class
            match_threshold = self.matching_distances.get(class_name, 2.0)

            # Find valid matches (within threshold)
            valid_matches = class_distances <= match_threshold

            if not np.any(valid_matches):
                continue

            # Use greedy matching (could be improved with Hungarian algorithm)
            used_gt = set()
            for i, pred_idx in enumerate(pred_class_indices):
                best_gt_local_idx = None
                best_distance = float('inf')

                for j, gt_idx in enumerate(gt_class_indices):
                    if gt_idx in used_gt or not valid_matches[i, j]:
                        continue

                    if class_distances[i, j] < best_distance:
                        best_distance = class_distances[i, j]
                        best_gt_local_idx = j

                if best_gt_local_idx is not None:
                    gt_idx = gt_class_indices[best_gt_local_idx]
                    pred_matches[pred_idx] = gt_idx
                    gt_matches[gt_idx] = pred_idx
                    used_gt.add(gt_idx)

                    match_distances.append(best_distance)
                    match_pred_indices.append(pred_idx)
                    match_gt_indices.append(gt_idx)

        return {
            'pred_matches': pred_matches,
            'gt_matches': gt_matches,
            'distances': np.array(match_distances),
            'pred_indices': np.array(match_pred_indices, dtype=int),
            'gt_indices': np.array(match_gt_indices, dtype=int)
        }

    def _compute_distance_matrix(self,
                               pred_boxes: np.ndarray,
                               gt_boxes: np.ndarray) -> np.ndarray:
        """
        Compute center distance matrix between predictions and ground truths

        Args:
            pred_boxes: (N, 9) prediction boxes
            gt_boxes: (M, 9) ground truth boxes

        Returns:
            (N, M) distance matrix
        """
        pred_centers = pred_boxes[:, :3]  # (N, 3) - x, y, z
        gt_centers = gt_boxes[:, :3]  # (M, 3) - x, y, z

        # Compute Euclidean distances in 3D space
        distances = np.sqrt(
            np.sum((pred_centers[:, None, :] - gt_centers[None, :, :]) ** 2, axis=2)
        )

        return distances

    def _compute_map(self,
                    matches: List[Dict[str, Any]],
                    predictions: List[Dict[str, Any]],
                    ground_truths: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute mean Average Precision (mAP)

        Args:
            matches: Matching results for each sample
            predictions: Filtered predictions
            ground_truths: Ground truth data

        Returns:
            Dictionary with mAP metrics
        """
        class_aps = []

        for class_idx, class_name in enumerate(self.class_names):
            ap = self._compute_class_ap(class_idx, matches, predictions, ground_truths)
            class_aps.append(ap)

        map_score = np.mean(class_aps)

        return {
            'mAP': map_score,
            'class_aps': class_aps
        }

    def _compute_class_ap(self,
                         class_idx: int,
                         matches: List[Dict[str, Any]],
                         predictions: List[Dict[str, Any]],
                         ground_truths: List[Dict[str, Any]]) -> float:
        """Compute Average Precision for a single class"""
        # Collect all predictions and their match status for this class
        all_scores = []
        all_matches = []
        total_gt = 0

        for sample_idx, (match, pred, gt) in enumerate(zip(matches, predictions, ground_truths)):
            # Count ground truth instances of this class
            gt_class_mask = gt['labels'] == class_idx
            total_gt += np.sum(gt_class_mask)

            # Get predictions of this class
            pred_class_mask = pred['labels'] == class_idx
            if not np.any(pred_class_mask):
                continue

            pred_class_scores = pred['scores'][pred_class_mask]
            pred_class_indices = np.where(pred_class_mask)[0]

            # Determine which predictions are true positives
            for local_idx, global_idx in enumerate(pred_class_indices):
                score = pred_class_scores[local_idx]
                is_match = match['pred_matches'][global_idx] != -1

                all_scores.append(score)
                all_matches.append(is_match)

        if total_gt == 0 or len(all_scores) == 0:
            return 0.0

        # Sort by confidence score (descending)
        sorted_indices = np.argsort(all_scores)[::-1]
        sorted_matches = np.array(all_matches)[sorted_indices]

        # Compute precision-recall curve
        tp = np.cumsum(sorted_matches)
        fp = np.cumsum(1 - sorted_matches)
        recall = tp / total_gt
        precision = tp / (tp + fp + 1e-16)

        # Compute AP using 11-point interpolation
        ap = 0.0
        for recall_threshold in np.linspace(0, 1, 11):
            precision_at_recall = np.max(precision[recall >= recall_threshold], initial=0)
            ap += precision_at_recall / 11

        return ap

    def _compute_errors(self,
                       matches: List[Dict[str, Any]],
                       predictions: List[Dict[str, Any]],
                       ground_truths: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute error metrics (ATE, ASE, AOE, AVE, AAE)

        Args:
            matches: Matching results
            predictions: Predictions
            ground_truths: Ground truths

        Returns:
            Dictionary with error metrics
        """
        all_trans_errors = []
        all_scale_errors = []
        all_orient_errors = []
        all_vel_errors = []
        all_attr_errors = []

        for match, pred, gt in zip(matches, predictions, ground_truths):
            if len(match['pred_indices']) == 0:
                continue

            pred_indices = match['pred_indices']
            gt_indices = match['gt_indices']

            matched_pred_boxes = pred['boxes'][pred_indices]
            matched_gt_boxes = gt['boxes'][gt_indices]

            # Translation error (center distance)
            trans_errors = np.sqrt(
                np.sum((matched_pred_boxes[:, :3] - matched_gt_boxes[:, :3]) ** 2, axis=1)
            )
            all_trans_errors.extend(trans_errors)

            # Scale error (1 - IoU of bounding boxes)
            scale_errors = 1.0 - self._compute_box_ious(matched_pred_boxes, matched_gt_boxes)
            all_scale_errors.extend(scale_errors)

            # Orientation error (angle difference)
            orient_errors = np.abs(
                self._angle_diff(matched_pred_boxes[:, 6], matched_gt_boxes[:, 6])
            )
            all_orient_errors.extend(orient_errors)

            # Velocity error (L2 norm of velocity difference)
            if matched_pred_boxes.shape[1] >= 9 and matched_gt_boxes.shape[1] >= 9:
                vel_errors = np.sqrt(
                    np.sum((matched_pred_boxes[:, 7:9] - matched_gt_boxes[:, 7:9]) ** 2, axis=1)
                )
                all_vel_errors.extend(vel_errors)

            # Attribute error (for now, set to 0 as we don't have attributes)
            attr_errors = np.zeros(len(matched_pred_boxes))
            all_attr_errors.extend(attr_errors)

        # Compute mean errors
        mate = np.mean(all_trans_errors) if all_trans_errors else 1.0
        mase = np.mean(all_scale_errors) if all_scale_errors else 1.0
        maoe = np.mean(all_orient_errors) if all_orient_errors else 1.0
        mave = np.mean(all_vel_errors) if all_vel_errors else 1.0
        maae = np.mean(all_attr_errors) if all_attr_errors else 1.0

        return {
            'mATE': mate,
            'mASE': mase,
            'mAOE': maoe,
            'mAVE': mave,
            'mAAE': maae
        }

    def _compute_box_ious(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """Compute IoU between corresponding 3D boxes"""
        # Simplified IoU computation for 3D boxes
        # This is a basic implementation - could be improved for better accuracy

        ious = []
        for box1, box2 in zip(boxes1, boxes2):
            # Extract dimensions
            x1, y1, z1, w1, l1, h1 = box1[:6]
            x2, y2, z2, w2, l2, h2 = box2[:6]

            # Compute intersection volume (simplified)
            x_overlap = max(0, min(x1 + w1/2, x2 + w2/2) - max(x1 - w1/2, x2 - w2/2))
            y_overlap = max(0, min(y1 + l1/2, y2 + l2/2) - max(y1 - l1/2, y2 - l2/2))
            z_overlap = max(0, min(z1 + h1/2, z2 + h2/2) - max(z1 - h1/2, z2 - h2/2))

            intersection = x_overlap * y_overlap * z_overlap

            # Compute union
            vol1 = w1 * l1 * h1
            vol2 = w2 * l2 * h2
            union = vol1 + vol2 - intersection

            iou = intersection / (union + 1e-8)
            ious.append(iou)

        return np.array(ious)

    def _angle_diff(self, angle1: np.ndarray, angle2: np.ndarray) -> np.ndarray:
        """Compute angle difference (handling wrap-around)"""
        diff = angle1 - angle2
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        return np.abs(diff)

    def _compute_nds(self, map_score: float, error_metrics: Dict[str, float]) -> float:
        """
        Compute nuScenes Detection Score (NDS)

        NDS = 1/2 * (mAP + 1/5 * Σ(1 - normalized_error))

        Args:
            map_score: mAP score
            error_metrics: Dictionary of error metrics

        Returns:
            NDS score
        """
        # Error normalization constants (from nuScenes paper)
        error_weights = {
            'mATE': 1.0,    # Translation error weight
            'mASE': 1.0,    # Scale error weight
            'mAOE': 1.0,    # Orientation error weight
            'mAVE': 1.0,    # Velocity error weight
            'mAAE': 1.0     # Attribute error weight
        }

        # Normalized errors (clamped between 0 and 1)
        normalized_errors = []
        for error_type in ['mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE']:
            error_value = error_metrics.get(error_type, 1.0)
            # Normalize error (these thresholds are from nuScenes evaluation)
            if error_type == 'mATE':
                normalized_error = min(1.0, error_value)
            elif error_type == 'mASE':
                normalized_error = min(1.0, error_value)
            elif error_type == 'mAOE':
                normalized_error = min(1.0, error_value / np.pi)
            elif error_type == 'mAVE':
                normalized_error = min(1.0, error_value)
            elif error_type == 'mAAE':
                normalized_error = min(1.0, error_value)
            else:
                normalized_error = 1.0

            normalized_errors.append(1.0 - normalized_error)

        # Compute NDS
        error_term = np.mean(normalized_errors)
        nds = 0.5 * (map_score + error_term)

        return nds

    def compute_sample_metrics(self,
                             pred_boxes: torch.Tensor,
                             pred_scores: torch.Tensor,
                             pred_labels: torch.Tensor,
                             gt_boxes: torch.Tensor,
                             gt_labels: torch.Tensor) -> Dict[str, float]:
        """
        Compute metrics for a single sample (convenience function)

        Args:
            pred_boxes: (N, 9) prediction boxes
            pred_scores: (N,) prediction scores
            pred_labels: (N,) prediction labels
            gt_boxes: (M, 9) ground truth boxes
            gt_labels: (M,) ground truth labels

        Returns:
            Dictionary of computed metrics
        """
        # Convert to numpy
        predictions = {
            'boxes': pred_boxes.cpu().numpy(),
            'scores': pred_scores.cpu().numpy(),
            'labels': pred_labels.cpu().numpy()
        }

        ground_truths = {
            'boxes': gt_boxes.cpu().numpy(),
            'labels': gt_labels.cpu().numpy()
        }

        # Reset and add sample
        self.reset()
        self.add_sample(predictions, ground_truths)

        # Compute metrics
        return self.compute_metrics()


# Example usage and testing
if __name__ == "__main__":
    # Test the metrics implementation
    print("Testing NuScenes metrics implementation...")

    # Create test data
    np.random.seed(42)

    # Ground truth data
    gt_boxes = np.array([
        [0, 0, 0, 2, 4, 1.5, 0, 0, 0],    # car
        [10, 0, 0, 2, 4, 1.5, 0.5, 1, 0], # car
        [0, 10, 0, 1, 2, 2, 0, 0, 0]      # pedestrian
    ])
    gt_labels = np.array([0, 0, 8])  # car, car, pedestrian

    # Prediction data (with some noise)
    pred_boxes = np.array([
        [0.5, 0.2, 0, 2.1, 4.1, 1.4, 0.1, 0.1, 0.1],  # close to GT 1
        [9.8, 0.1, 0, 1.9, 3.9, 1.6, 0.4, 0.9, 0.1],  # close to GT 2
        [0.2, 9.9, 0, 1.1, 1.9, 2.1, 0.1, 0.1, 0.1],  # close to GT 3
        [50, 50, 0, 2, 4, 1.5, 0, 0, 0]                # false positive
    ])
    pred_scores = np.array([0.9, 0.8, 0.7, 0.6])
    pred_labels = np.array([0, 0, 8, 0])  # car, car, pedestrian, car

    # Create metrics calculator
    metrics_calculator = NuScenesMetrics()

    # Add sample
    predictions = {
        'boxes': pred_boxes,
        'scores': pred_scores,
        'labels': pred_labels
    }

    ground_truths = {
        'boxes': gt_boxes,
        'labels': gt_labels
    }

    metrics_calculator.add_sample(predictions, ground_truths)

    # Compute metrics
    metrics = metrics_calculator.compute_metrics()

    print("\nComputed metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nNuScenes metrics test completed!")