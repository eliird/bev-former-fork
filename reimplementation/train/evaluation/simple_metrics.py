"""
Simple NDS and mAP calculation for nuScenes evaluation
Simplified implementation without complex dependencies
"""

import torch
import numpy as np
from typing import List, Dict, Tuple


def calculate_nds_map(predictions: List[Dict], ground_truths: List[Dict],
                     class_names: List[str], distance_thresholds: List[float] = None) -> Dict[str, float]:
    """
    Calculate NDS and mAP metrics for nuScenes evaluation.

    Args:
        predictions: List of prediction dicts with keys: 'boxes_3d', 'scores_3d', 'labels_3d'
        ground_truths: List of GT dicts with keys: 'gt_bboxes_3d', 'gt_labels_3d'
        class_names: List of class names
        distance_thresholds: Distance thresholds for matching (default: [0.5, 1.0, 2.0, 4.0])

    Returns:
        Dict with 'NDS' and 'mAP' values
    """
    if distance_thresholds is None:
        distance_thresholds = [0.5, 1.0, 2.0, 4.0]

    if len(predictions) == 0 or len(ground_truths) == 0:
        return {'NDS': 0.0, 'mAP': 0.0}

    # Calculate mAP for each class and threshold
    class_aps = []

    for class_idx, class_name in enumerate(class_names):
        # Collect all predictions and GTs for this class
        class_predictions = []
        class_gts = []

        for pred, gt in zip(predictions, ground_truths):
            # Get predictions for this class
            if 'labels_3d' in pred and len(pred['labels_3d']) > 0:
                class_mask = pred['labels_3d'] == class_idx
                if class_mask.sum() > 0:
                    class_pred_boxes = pred['boxes_3d'][class_mask]
                    class_pred_scores = pred['scores_3d'][class_mask]
                    for box, score in zip(class_pred_boxes, class_pred_scores):
                        class_predictions.append({
                            'box': box.cpu().numpy() if torch.is_tensor(box) else box,
                            'score': score.item() if torch.is_tensor(score) else score
                        })

            # Get ground truths for this class
            if 'gt_labels_3d' in gt and len(gt['gt_labels_3d']) > 0:
                gt_class_mask = gt['gt_labels_3d'] == class_idx
                if gt_class_mask.sum() > 0:
                    gt_class_boxes = gt['gt_bboxes_3d'][gt_class_mask]
                    for box in gt_class_boxes:
                        class_gts.append({
                            'box': box.cpu().numpy() if torch.is_tensor(box) else box,
                            'matched': False
                        })

        # Calculate AP for this class across all thresholds
        if len(class_predictions) == 0 and len(class_gts) == 0:
            class_ap = 1.0  # Perfect score if no predictions and no GTs
        elif len(class_gts) == 0:
            class_ap = 0.0  # No GTs but have predictions = bad
        else:
            class_ap = calculate_class_ap(class_predictions, class_gts, distance_thresholds)

        class_aps.append(class_ap)

    # Calculate overall mAP
    mAP = np.mean(class_aps)

    # For simplified NDS, we'll approximate it as mAP
    # (proper NDS needs translation/scale/orientation errors which are complex)
    NDS = mAP  # Simplified - in reality NDS = (mAP + other_error_metrics) / 2

    return {
        'NDS': float(NDS),
        'mAP': float(mAP),
        'per_class_AP': {class_names[i]: ap for i, ap in enumerate(class_aps)}
    }


def calculate_class_ap(predictions: List[Dict], ground_truths: List[Dict],
                      distance_thresholds: List[float]) -> float:
    """Calculate AP for a single class."""
    if len(predictions) == 0:
        return 0.0 if len(ground_truths) > 0 else 1.0

    # Sort predictions by confidence score (descending)
    predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)

    # Calculate AP for each distance threshold, then average
    threshold_aps = []

    for dist_thresh in distance_thresholds:
        # Reset GT matching status
        for gt in ground_truths:
            gt['matched'] = False

        tp = []  # True positives
        fp = []  # False positives

        for pred in predictions:
            # Find closest GT
            best_distance = float('inf')
            best_gt_idx = -1

            for gt_idx, gt in enumerate(ground_truths):
                if gt['matched']:
                    continue

                # Calculate center distance (simplified - uses only x,y)
                pred_center = pred['box'][:2]  # x, y
                gt_center = gt['box'][:2]      # x, y
                distance = np.linalg.norm(pred_center - gt_center)

                if distance < best_distance:
                    best_distance = distance
                    best_gt_idx = gt_idx

            # Check if match is within threshold
            if best_gt_idx >= 0 and best_distance <= dist_thresh:
                ground_truths[best_gt_idx]['matched'] = True
                tp.append(1)
                fp.append(0)
            else:
                tp.append(0)
                fp.append(1)

        # Calculate precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        num_gts = len(ground_truths)
        recalls = tp_cumsum / max(num_gts, 1)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)

        # Calculate AP using 11-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            # Find precisions for recalls >= t
            valid_precisions = precisions[recalls >= t]
            if len(valid_precisions) > 0:
                ap += np.max(valid_precisions)
        ap /= 11.0

        threshold_aps.append(ap)

    return np.mean(threshold_aps)


def extract_detections_from_model_output(model_output) -> Dict:
    """
    Extract detection results from model forward_test output.

    Args:
        model_output: Output from model.forward_test()

    Returns:
        Dict with 'boxes_3d', 'scores_3d', 'labels_3d'
    """
    # Handle different possible output formats
    if isinstance(model_output, list) and len(model_output) > 0:
        # Take first item if it's a list
        detection = model_output[0]
    elif isinstance(model_output, dict):
        detection = model_output
    else:
        # Fallback - empty detection
        return {
            'boxes_3d': torch.zeros((0, 9)),
            'scores_3d': torch.zeros(0),
            'labels_3d': torch.zeros(0, dtype=torch.long)
        }

    # Extract components with fallbacks
    if isinstance(detection, dict):
        # Check if detection has pts_bbox key (BEVFormer format)
        if 'pts_bbox' in detection:
            pts_bbox = detection['pts_bbox']
            boxes_3d = pts_bbox.get('boxes_3d', torch.zeros((0, 9)))
            scores_3d = pts_bbox.get('scores_3d', torch.zeros(0))
            labels_3d = pts_bbox.get('labels_3d', torch.zeros(0, dtype=torch.long))
        else:
            # Direct format
            boxes_3d = detection.get('boxes_3d', torch.zeros((0, 9)))
            scores_3d = detection.get('scores_3d', torch.zeros(0))
            labels_3d = detection.get('labels_3d', torch.zeros(0, dtype=torch.long))
    else:
        # Unknown format - return empty
        boxes_3d = torch.zeros((0, 9))
        scores_3d = torch.zeros(0)
        labels_3d = torch.zeros(0, dtype=torch.long)

    return {
        'boxes_3d': boxes_3d,
        'scores_3d': scores_3d,
        'labels_3d': labels_3d
    }