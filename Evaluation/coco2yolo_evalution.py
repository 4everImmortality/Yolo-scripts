# -*- coding: utf-8 -*-
"""
A standalone script for evaluating object detection model performance using COCO-style metrics.

This script calculates precision, recall, and mean Average Precision (mAP)
by comparing model predictions against ground-truth labels. It is designed to be
a self-contained, dependency-light alternative to official evaluation tools,
making it easy to integrate and understand.

It can process predictions from different frameworks, such as YOLO (JSON output)
and MMDetection (pickle output), ensuring a consistent evaluation standard.

Usage:
    python this_script_name.py --label_file /path/to/coco_labels.json \
                               --pred_file /path/to/predictions.json
"""

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from prettytable import PrettyTable
from tqdm import tqdm

# --- Utility Functions ---

def clip_boxes(boxes: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """Clips bounding boxes (xyxy) to image shape (height, width)."""
    boxes[..., 0].clamp_(0, shape[1])  # x1
    boxes[..., 1].clamp_(0, shape[0])  # y1
    boxes[..., 2].clamp_(0, shape[1])  # x2
    boxes[..., 3].clamp_(0, shape[0])  # y2
    return boxes

def scale_boxes(
    img1_shape: Tuple[int, int],
    boxes: torch.Tensor,
    img0_shape: Tuple[int, int],
    ratio_pad: Tuple[Tuple[float, float], Tuple[float, float]] = None
) -> torch.Tensor:
    """Rescales bounding boxes (xyxy) from img1_shape to img0_shape."""
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (
            (img1_shape[1] - img0_shape[1] * gain) / 2,
            (img1_shape[0] - img0_shape[0] * gain) / 2,
        )
    else:
        gain, pad = ratio_pad[0][0], ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes

def box_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Calculates Intersection-over-Union (IoU) of two sets of boxes.
    
    Args:
        box1 (torch.Tensor): A tensor of shape (N, 4) of boxes in (x1, y1, x2, y2) format.
        box2 (torch.Tensor): A tensor of shape (M, 4) of boxes in (x1, y1, x2, y2) format.
        eps (float): A small value to prevent division by zero.

    Returns:
        torch.Tensor: An (N, M) tensor with pairwise IoU values.
    """
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    union = (a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps
    return inter / union

# --- Core Evaluator Class ---

class CocoEvaluator:
    """
    A class to evaluate object detection models using COCO metrics.
    """
    IOU_THRESHOLDS = torch.linspace(0.5, 0.95, 10)
    CONF_THRESHOLD = 0.001
    NUM_INTERP_POINTS = 101

    def __init__(self, label_file: str):
        """
        Initializes the CocoEvaluator.

        Args:
            label_file (str): Path to the ground-truth COCO JSON file.
        """
        self.ground_truths, self.class_names, self.image_metadata = self._load_ground_truth(label_file)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.IOU_THRESHOLDS = self.IOU_THRESHOLDS.to(self.device)
        print(f"Evaluator initialized on device: {self.device}")

    def _load_ground_truth(self, label_file: str) -> Tuple[Dict[int, np.ndarray], List[str], Dict[int, Dict]]:
        """Loads and parses the ground-truth COCO annotation file."""
        if not Path(label_file).exists():
            raise FileNotFoundError(f"Ground truth file not found: {label_file}")
            
        with open(label_file) as f:
            data = json.load(f)

        class_names = [category['name'] for category in data['categories']]
        
        image_metadata = {img['id']: {'height': img['height'], 'width': img['width']} for img in data['images']}

        ground_truths = {}
        print("Loading ground truth annotations...")
        for ann in tqdm(data['annotations']):
            img_id = ann['image_id']
            if img_id not in ground_truths:
                ground_truths[img_id] = []
            
            cat_id = ann['category_id']
            x_min, y_min, w, h = ann['bbox']
            x_max, y_max = x_min + w, y_min + h
            ground_truths[img_id].append([cat_id, x_min, y_min, x_max, y_max])
        
        # Convert lists to numpy arrays for efficiency
        for img_id in ground_truths:
            ground_truths[img_id] = np.array(ground_truths[img_id])
            
        return ground_truths, class_names, image_metadata

    def _load_predictions(self, pred_file: str) -> Dict[int, np.ndarray]:
        """Loads predictions from either a COCO-style JSON or MMDetection pickle file."""
        if not Path(pred_file).exists():
            raise FileNotFoundError(f"Prediction file not found: {pred_file}")

        predictions = {}
        print("Loading model predictions...")
        if pred_file.endswith('.json'):
            with open(pred_file) as f:
                data = json.load(f)
            for pred in tqdm(data):
                if pred['score'] < self.CONF_THRESHOLD:
                    continue
                img_id = pred['image_id']
                if img_id not in predictions:
                    predictions[img_id] = []
                cat_id = pred['category_id']
                x_min, y_min, w, h = pred['bbox']
                x_max, y_max = x_min + w, y_min + h
                predictions[img_id].append([x_min, y_min, x_max, y_max, pred['score'], cat_id])

        elif pred_file.endswith('.pickle') or pred_file.endswith('.pkl'):
            with open(pred_file, 'rb') as f:
                data = pickle.load(f)
            for result in tqdm(data):
                img_id_str = Path(result['img_path']).stem
                img_id = int(img_id_str) if img_id_str.isdigit() else img_id_str

                if img_id not in predictions:
                    predictions[img_id] = []
                
                pred_instances = result['pred_instances']
                for i in range(len(pred_instances['scores'])):
                    score = pred_instances['scores'][i].item()
                    if score < self.CONF_THRESHOLD:
                        continue
                    label = pred_instances['labels'][i].item()
                    bbox = pred_instances['bboxes'][i].cpu().numpy()
                    x_min, y_min, x_max, y_max = bbox
                    predictions[img_id].append([x_min, y_min, x_max, y_max, score, label])
        else:
            raise ValueError("Unsupported prediction file format. Use .json or .pickle")
        
        # Convert lists to numpy arrays for efficiency
        for img_id in predictions:
            predictions[img_id] = np.array(predictions[img_id])

        return predictions
    
    def _process_single_image(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculates the correctness matrix for detections on a single image.

        Args:
            predictions (torch.Tensor): Detections for an image, shape (N, 6) -> [x1, y1, x2, y2, conf, class].
            labels (torch.Tensor): Ground truth for an image, shape (M, 5) -> [class, x1, y1, x2, y2].

        Returns:
            torch.Tensor: A boolean tensor of shape (N, num_iou_thresholds) indicating true positives.
        """
        correct = torch.zeros(predictions.shape[0], self.IOU_THRESHOLDS.shape[0], dtype=torch.bool, device=self.device)
        
        if labels.shape[0] == 0:
            return correct # All predictions are false positives if no labels exist

        iou = box_iou(labels[:, 1:], predictions[:, :4])
        correct_class = labels[:, 0:1] == predictions[:, 5]

        for i, iou_thresh in enumerate(self.IOU_THRESHOLDS):
            # Find potential matches for this IoU threshold
            matches_found = torch.where((iou >= iou_thresh) & correct_class)
            
            if matches_found[0].shape[0]:
                # Create a tensor of [label_idx, detection_idx, iou_value]
                matches = torch.cat(
                    (torch.stack(matches_found, 1), iou[matches_found[0], matches_found[1]][:, None]), 1
                )
                # Sort matches by IoU in descending order to prioritize best overlaps
                matches = matches[matches[:, 2].argsort(descending=True)]
                
                # Enforce one-to-one matching
                # Keep only the first (best) match for each detection
                matches = matches[torch.unique(matches[:, 1], return_inverse=False, return_counts=False)[1]]
                # Keep only the first (best) match for each ground truth label
                matches = matches[torch.unique(matches[:, 0], return_inverse=False, return_counts=False)[1]]
                
                # Mark the successful matches as correct
                correct[matches[:, 1].long(), i] = True
        return correct

    def _calculate_ap_per_class(self, stats: List[np.ndarray]) -> Tuple:
        """
        Computes the Average Precision (AP) for each class.
        
        Args:
            stats (List[np.ndarray]): A list containing [true_positives, confidences, pred_classes, target_classes].

        Returns:
            A tuple containing (true_positives_per_class, false_positives_per_class, precision, 
                               recall, f1, ap_per_class, unique_classes).
        """
        true_positives, confidences, pred_classes, target_classes = stats
        
        # Sort by confidence
        sort_indices = np.argsort(-confidences)
        true_positives = true_positives[sort_indices]
        confidences = confidences[sort_indices]
        pred_classes = pred_classes[sort_indices]

        unique_classes, target_counts = np.unique(target_classes, return_counts=True)
        num_classes = unique_classes.shape[0]

        # Initialize storage
        ap = np.zeros((num_classes, true_positives.shape[1])) # AP for each class and IoU threshold
        px, py = np.linspace(0, 1, self.NUM_INTERP_POINTS), []  # For PR curve plotting
        precision_curve = np.zeros((num_classes, self.NUM_INTERP_POINTS))
        recall_curve = np.zeros((num_classes, self.NUM_INTERP_POINTS))

        for class_idx, class_val in enumerate(unique_classes):
            is_class = pred_classes == class_val
            num_targets = target_counts[class_idx]
            num_predictions = np.sum(is_class)

            if num_predictions == 0 or num_targets == 0:
                continue

            # Cumulative sums of true positives and false positives
            fp_cumsum = (1 - true_positives[is_class]).cumsum(0)
            tp_cumsum = true_positives[is_class].cumsum(0)
            
            # Recall and Precision
            recall = tp_cumsum / (num_targets + 1e-16)
            precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)
            
            # Store PR curve for mAP@0.5
            precision_curve[class_idx] = np.interp(-px, -confidences[is_class], precision[:, 0], left=1)
            recall_curve[class_idx] = np.interp(-px, -confidences[is_class], recall[:, 0], left=0)

            # Calculate AP for each IoU threshold
            for j in range(true_positives.shape[1]):
                ap[class_idx, j], _, _ = self._compute_ap(recall[:, j], precision[:, j])

        # Compute F1 score
        f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-16)
        
        # Find confidence threshold that gives best F1
        best_f1_idx = np.argmax(f1_scores.mean(0))
        final_precision = precision_curve[:, best_f1_idx]
        final_recall = recall_curve[:, best_f1_idx]
        final_f1 = f1_scores[:, best_f1_idx]

        tp_at_best_f1 = (final_recall * target_counts).round()
        fp_at_best_f1 = (tp_at_best_f1 / (final_precision + 1e-16) - tp_at_best_f1).round()

        return tp_at_best_f1, fp_at_best_f1, final_precision, final_recall, final_f1, ap, unique_classes.astype(int)

    def _compute_ap(self, recall: np.ndarray, precision: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute the average precision from recall and precision curves.

        Args:
            recall (np.ndarray): The recall curve.
            precision (np.ndarray): The precision curve.

        Returns:
            A tuple containing (average_precision, precision_envelope, recall_with_sentinels).
        """
        # Append sentinel values
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))

        # Create a precision envelope (monotonic decreasing)
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

        # Integrate area under curve using 101-point COCO-style interpolation
        x = np.linspace(0, 1, self.NUM_INTERP_POINTS)
        ap = np.trapz(np.interp(x, mrec, mpre), x)
        return ap, mpre, mrec

    def run(self, prediction_file: str) -> None:
        """
        Executes the full evaluation pipeline.

        Args:
            prediction_file (str): Path to the prediction file (.json or .pickle).
        """
        predictions = self._load_predictions(prediction_file)
        
        stats = []
        print("Matching predictions to ground truth...")
        for img_id in tqdm(self.image_metadata.keys()):
            # Get labels and predictions for the current image
            labels_np = self.ground_truths.get(img_id, np.empty((0, 5)))
            preds_np = predictions.get(img_id, np.empty((0, 6)))

            labels = torch.from_numpy(labels_np).to(self.device)
            preds = torch.from_numpy(preds_np).to(self.device)

            if preds.shape[0] == 0:
                if labels.shape[0] != 0:
                    # No predictions but labels exist: all labels are false negatives
                    stats.append(
                        (torch.empty((0, self.IOU_THRESHOLDS.shape[0]), dtype=torch.bool, device=self.device),
                         torch.empty((0), device=self.device),
                         torch.empty((0), device=self.device),
                         labels[:, 0])
                    )
                continue
            
            correctness_matrix = self._process_single_image(preds, labels)
            stats.append(
                (correctness_matrix, preds[:, 4], preds[:, 5], labels[:, 0])
            )

        # Concatenate stats from all images
        if not stats:
            print("No predictions or labels found to evaluate.")
            return

        stats_cat = [torch.cat([s[i] for s in stats]).cpu().numpy() for i in range(4)]
        
        metrics = self._calculate_ap_per_class(stats_cat)
        self.display_results(metrics)

    def display_results(self, metrics: Tuple) -> None:
        """
        Formats and prints the final evaluation metrics in a table.

        Args:
            metrics (Tuple): The output from _calculate_ap_per_class.
        """
        tp, fp, p, r, f1, ap, classes_present = metrics

        table = PrettyTable()
        table.title = "Object Detection Metrics"
        table.field_names = ["Class", "Images", "Instances", "P", "R", "F1", "mAP@.5", "mAP@.5:.95"]

        # Map class indices to class names
        class_map = {i: name for i, name in enumerate(self.class_names)}
        
        # Count instances per class
        _, instance_counts = np.unique([c for gt_list in self.ground_truths.values() for c in gt_list[:, 0]], return_counts=True)
        
        # Count images per class
        images_per_class = {i: 0 for i in range(len(self.class_names))}
        for gt_list in self.ground_truths.values():
            for class_id in np.unique(gt_list[:,0]):
                 images_per_class[class_id] +=1
        
        # Overall metrics
        mean_p = p.mean()
        mean_r = r.mean()
        mean_f1 = f1.mean()
        mean_ap50 = ap[:, 0].mean()
        mean_ap50_95 = ap.mean()
        
        table.add_row([
            "all",
            len(self.image_metadata),
            int(np.sum(instance_counts)),
            f"{mean_p:.3f}",
            f"{mean_r:.3f}",
            f"{mean_f1:.3f}",
            f"{mean_ap50:.3f}",
            f"{mean_ap50_95:.3f}",
        ])
        
        # Per-class metrics
        for i, class_id in enumerate(classes_present):
            class_name = class_map.get(class_id, f"class_{class_id}")
            table.add_row([
                class_name,
                images_per_class.get(class_id, 0),
                int(tp[i]+fp[i]), # Total instances for this class from predictions
                f"{p[i]:.3f}",
                f"{r[i]:.3f}",
                f"{f1[i]:.3f}",
                f"{ap[i, 0]:.3f}",
                f"{ap[i, :].mean():.3f}",
            ])
            
        print(table)


def main():
    """Main function to run the evaluation."""
    parser = argparse.ArgumentParser(description="COCO Metrics Evaluator for Object Detection")
    parser.add_argument('--label_file', type=str, required=True, help='Path to the ground truth COCO annotation file.')
    parser.add_argument('--pred_file', type=str, required=True, help='Path to the prediction file (JSON or Pickle).')
    args = parser.parse_args()

    try:
        evaluator = CocoEvaluator(args.label_file)
        evaluator.run(args.pred_file)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
