import numpy as np
from pathlib import Path
import torch # Need torch for box_iou and tensor operations
from torchvision.ops.boxes import box_iou # Using the pre-built function
import statistics
import random # just for testing. delete later
import os
# --- Constants ---
IOU_THRESHOLD = 0.8 # threshold for TP/FP: means if IoU >= 0.8, it's a TP otherwise FP
NUM_CLASSES = 2 # 0: plane, 1: ship (Adjust if different)

# Fault Classification Thresholds
MASKED_TOLERANCE = 0.001          # Tolerance for floating-point differences (MASKED detection)
MATCHING_IOU_THRESHOLD = 0.7      # Minimum IoU to match boxes between golden and faulty runs (lower than IOU_THRESHOLD to detect location errors)

# --- Helper Function: Convert YOLO format to xyxy ---
def yolo_to_xyxy(yolo_box):
    """Converts YOLO box [xc, yc, w, h] (normalized) to [x1, y1, x2, y2] (normalized)."""
    xc, yc, w, h = yolo_box
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    # Clamp values to be within [0, 1] in case of slight errors
    return [max(0.0, x1), max(0.0, y1), min(1.0, x2), min(1.0, y2)]

# --- Ground Truth Loading ---
def load_ground_truth(gt_path_str):
    """Loads ground truth boxes from a YOLO format .txt file.
    Returns a list of dicts: [{'label_id': int, 'box_xyxy': [x1, y1, x2, y2]}, ...]
    Returns empty list on error.
    """
    gt_path = Path(gt_path_str)
    ground_truths = []
    if not gt_path.is_file():
        print(f"    GT Load Error: File not found: {gt_path}")
        return ground_truths # Return empty list

    try:
        with open(gt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    # Assuming parts[1] to parts[4] are xc, yc, w, h as strings
                    yolo_coords = [float(p) for p in parts[1:]]
                    # Convert to normalized xyxy
                    xyxy_coords = yolo_to_xyxy(yolo_coords)
                    ground_truths.append({'label_id': class_id, 'box_xyxy': xyxy_coords})
                else:
                    print(f"    GT Load Warning: Skipping invalid line in {gt_path}: {line.strip()}")
    except Exception as e:
         print(f"    GT Load Error: Failed loading {gt_path}: {e}")
         return [] # Return empty list on error

    return ground_truths

# --- Main AP Calculation ---
def calculate_ap_for_image(predictions_list, ground_truth_path_str):
    """
    Calculates Average Precision (AP) for a single image at a specific IoU threshold.

    Args:
        predictions_list (list): List of prediction dicts from the worker.
                                 Assumes dicts contain 'box_xyxyn', 'label_id', 'score'.
        ground_truth_path_str (str): Absolute path to the ground truth file.

    Returns:
        float: Average Precision for this image (average across classes), or -1.0 on error.
    """
    try:
        ground_truths = load_ground_truth(ground_truth_path_str)
        # Handle case where GT file is empty or unreadable
        if ground_truths is None: return -1.0 
        
        # Handle case with no GT objects correctly
        if not ground_truths:
             return 1.0 if not predictions_list else 0.0

        # --- Standard AP Calculation Steps (Using torchvision.ops.box_iou) ---
        
        ap_per_class = []

        for class_id in range(NUM_CLASSES):
            
            # 1. Filter predictions and GT for the current class
            # Ensure predictions are sorted by score descending
            class_preds = sorted([p for p in predictions_list if p['label_id'] == class_id], 
                                 key=lambda x: x['score'], reverse=True)
            class_gts = [gt for gt in ground_truths if gt['label_id'] == class_id]
            num_gt = len(class_gts)

            # Handle cases where no GT or no predictions for this class
            if num_gt == 0:
                 ap_per_class.append(1.0 if not class_preds else 0.0)
                 continue 
            if not class_preds:
                 ap_per_class.append(0.0)
                 continue

            # Convert to tensors for box_iou
            # Prediction boxes (use xyxyn directly if GTs are also normalized xyxy)
            pred_boxes_tensor = torch.tensor([p['box_xyxyn'] for p in class_preds], dtype=torch.float32)
            # Ground truth boxes (already converted to xyxy in load_ground_truth)
            gt_boxes_tensor = torch.tensor([gt['box_xyxy'] for gt in class_gts], dtype=torch.float32)

            # 2. Match predictions to GTs using box_iou
            tp = np.zeros(len(class_preds))
            fp = np.zeros(len(class_preds))
            gt_matched = np.zeros(num_gt) # Keep track of which GTs have been matched

            if gt_boxes_tensor.numel() == 0 or pred_boxes_tensor.numel() == 0:
                # If either set is empty after filtering, handle appropriately
                 if pred_boxes_tensor.numel() > 0: # Predictions exist but no GTs for this class
                     fp = np.ones(len(class_preds)) # All predictions are false positives
                 # If no predictions, tp/fp remain zeros, AP calc below handles it
            
            else:
                 # Calculate IoU matrix: rows=predictions, cols=ground_truths
                 iou_matrix = box_iou(pred_boxes_tensor, gt_boxes_tensor) # Shape: [num_preds, num_gts]

                 # Iterate through predictions (sorted by score)
                 for i in range(len(class_preds)):
                     # Get IoUs for this prediction against all GTs
                     ious = iou_matrix[i, :]
                     best_iou, best_gt_idx = torch.max(ious, dim=0)

                     # Check if the best match meets threshold and GT wasn't matched
                     if best_iou >= IOU_THRESHOLD:
                         if gt_matched[best_gt_idx] == 0:
                             tp[i] = 1.0
                             gt_matched[best_gt_idx] = 1 # Mark GT as matched
                         else: # Matched an already claimed GT
                             fp[i] = 1.0
                     else: # No match met threshold
                         fp[i] = 1.0

            # 3. Calculate Precision-Recall curve points
            tp_cumulative = np.cumsum(tp)
            fp_cumulative = np.cumsum(fp)
            
            recall = tp_cumulative / num_gt if num_gt > 0 else np.zeros_like(tp_cumulative)
            # Add small epsilon to avoid division by zero if no positives found yet
            precision = tp_cumulative / (tp_cumulative + fp_cumulative + 1e-16) 

            # 4. Calculate AP using numerical integration (Area Under Curve)
            # Use all-point interpolation (PASCAL VOC 2010+ method)
            precision = np.concatenate(([0.], precision, [0.]))
            recall = np.concatenate(([0.], recall, [1.]))
            
            # Ensure precision is monotonically decreasing
            for k in range(len(precision) - 2, -1, -1):
                precision[k] = np.maximum(precision[k], precision[k+1])
                
            # Find indices where recall changes
            indices = np.where(recall[1:] != recall[:-1])[0] + 1
            
            # Sum areas of rectangles
            class_ap = np.sum((recall[indices] - recall[indices-1]) * precision[indices])
            ap_per_class.append(class_ap)

        # 5. Average AP across classes for this image
        if not ap_per_class: 
            return 0.0 
        
        image_mean_ap = statistics.mean(ap_per_class)
        return image_mean_ap

    except Exception as e:
        # Print detailed error, including traceback maybe?
        import traceback
        print(f"    --- Unhandled Error during AP calculation for {ground_truth_path_str} ---")
        traceback.print_exc() 
        print(f"    Error details: {e}")
        print(f"    --- End Error ---")
        return -1.0 # Indicate calculation error

# --- Fault Classification Functions ---

def boxes_are_similar(pred1, pred2, tolerance=MASKED_TOLERANCE):
    """
    Check if two prediction dictionaries are similar within tolerance.

    Args:
        pred1 (dict): First prediction with 'box_xyxyn', 'label_id', 'score'
        pred2 (dict): Second prediction with 'box_xyxyn', 'label_id', 'score'
        tolerance (float): Maximum allowed difference for floating-point values

    Returns:
        bool: True if predictions are similar within tolerance
    """
    # Check if label_id matches (must be exact)
    if pred1['label_id'] != pred2['label_id']:
        return False

    # Check if score is within tolerance
    if abs(pred1['score'] - pred2['score']) > tolerance:
        return False

    # Check if all box coordinates are within tolerance
    box1 = pred1['box_xyxyn']
    box2 = pred2['box_xyxyn']
    for coord1, coord2 in zip(box1, box2):
        if abs(coord1 - coord2) > tolerance:
            return False

    return True

def predictions_are_masked(golden_preds, faulty_preds, tolerance=MASKED_TOLERANCE):
    """
    Check if two prediction lists are effectively identical (MASKED fault).

    Args:
        golden_preds (list): Golden run predictions
        faulty_preds (list): Faulty run predictions
        tolerance (float): Tolerance for floating-point comparison

    Returns:
        bool: True if predictions are masked (identical within tolerance)
    """
    # Must have same number of predictions
    if len(golden_preds) != len(faulty_preds):
        return False

    # Empty predictions are considered masked
    if len(golden_preds) == 0:
        return True

    # Sort both lists by score (descending) to ensure consistent comparison
    golden_sorted = sorted(golden_preds, key=lambda x: x['score'], reverse=True)
    faulty_sorted = sorted(faulty_preds, key=lambda x: x['score'], reverse=True)

    # Check if all predictions match
    for g_pred, f_pred in zip(golden_sorted, faulty_sorted):
        if not boxes_are_similar(g_pred, f_pred, tolerance):
            return False

    return True

def match_boxes(golden_preds, faulty_preds, iou_threshold=MATCHING_IOU_THRESHOLD):
    """
    Match boxes between golden and faulty predictions based on IoU and class.

    Args:
        golden_preds (list): Golden run predictions
        faulty_preds (list): Faulty run predictions
        iou_threshold (float): Minimum IoU to consider a match

    Returns:
        dict: {
            'matched_pairs': [(g_idx, f_idx, iou, same_class), ...],
            'unmatched_golden': [g_idx, ...],
            'unmatched_faulty': [f_idx, ...]
        }
    """
    if not golden_preds or not faulty_preds:
        return {
            'matched_pairs': [],
            'unmatched_golden': list(range(len(golden_preds))),
            'unmatched_faulty': list(range(len(faulty_preds)))
        }

    # Convert to tensors
    golden_boxes = torch.tensor([p['box_xyxyn'] for p in golden_preds], dtype=torch.float32)
    faulty_boxes = torch.tensor([p['box_xyxyn'] for p in faulty_preds], dtype=torch.float32)

    # Calculate IoU matrix: [num_golden, num_faulty]
    iou_matrix = box_iou(golden_boxes, faulty_boxes)

    # Track matched indices
    matched_golden = set()
    matched_faulty = set()
    matched_pairs = []

    # Greedy matching: process from highest to lowest IoU
    # Flatten the IoU matrix with indices
    matches = []
    for g_idx in range(len(golden_preds)):
        for f_idx in range(len(faulty_preds)):
            iou_val = iou_matrix[g_idx, f_idx].item()
            if iou_val >= iou_threshold:
                same_class = (golden_preds[g_idx]['label_id'] == faulty_preds[f_idx]['label_id'])
                matches.append((g_idx, f_idx, iou_val, same_class))

    # Sort by IoU (descending) for greedy matching
    matches.sort(key=lambda x: x[2], reverse=True)

    # Assign matches greedily (highest IoU first, no double-matching)
    for g_idx, f_idx, iou_val, same_class in matches:
        if g_idx not in matched_golden and f_idx not in matched_faulty:
            matched_pairs.append((g_idx, f_idx, iou_val, same_class))
            matched_golden.add(g_idx)
            matched_faulty.add(f_idx)

    # Find unmatched boxes
    unmatched_golden = [i for i in range(len(golden_preds)) if i not in matched_golden]
    unmatched_faulty = [i for i in range(len(faulty_preds)) if i not in matched_faulty]

    return {
        'matched_pairs': matched_pairs,
        'unmatched_golden': unmatched_golden,
        'unmatched_faulty': unmatched_faulty
    }

def classify_fault_outcome(golden_predictions, faulty_predictions, status="SUCCESS"):
    """
    Classify the outcome of a fault injection run by comparing golden vs faulty predictions.

    Uses a composite approach to detect multiple types of Silent Data Corruption (SDC):
    - SDC_M: Missed detections (golden boxes not found in faulty)
    - SDC_P: Phantom detections (faulty boxes not found in golden)
    - SDC_L: Location errors (matched boxes with poor IoU)
    - SDC_C: Classification errors (matched boxes with different class)

    Args:
        golden_predictions (list): Predictions from golden (fault-free) run.
                                    List of dicts with 'box_xyxyn', 'label_id', 'score'.
        faulty_predictions (list or None): Predictions from faulty run.
                                            None or empty if crashed/hung.
        status (str): Run status - "SUCCESS", "CRASH", or "HANG"

    Returns:
        str: Classification result. Examples:
             - "MASKED" if no detectable differences
             - "SDC_M,SDC_L" if missed detections and location errors
             - "CRASH" if application crashed
             - "HANG" if application hung/timed out
    """
    # 1. Handle CRASH/HANG cases
    if status == "CRASH":
        return "CRASH"
    if status == "HANG":
        return "HANG"

    # Handle None or empty faulty predictions (also treated as crash-like)
    if faulty_predictions is None:
        return "CRASH"

    # 2. Check if fault was MASKED (no visible effect)
    if predictions_are_masked(golden_predictions, faulty_predictions):
        return "MASKED"

    # 3. Match boxes between golden and faulty predictions
    match_result = match_boxes(golden_predictions, faulty_predictions)

    # 4. Detect different types of SDC
    sdc_types = []

    # SDC_M: Missed detections (golden boxes with no match in faulty)
    if match_result['unmatched_golden']:
        sdc_types.append("SDC_M")

    # SDC_P: Phantom detections (faulty boxes with no match in golden)
    if match_result['unmatched_faulty']:
        sdc_types.append("SDC_P")

    # SDC_L and SDC_C: Check matched pairs for location and classification errors
    has_location_errors = False
    has_classification_errors = False

    for g_idx, f_idx, iou_val, same_class in match_result['matched_pairs']:
        # SDC_L: Location error (matched but IoU below accuracy threshold)
        if iou_val < IOU_THRESHOLD:
            has_location_errors = True

        # SDC_C: Classification error (matched by location but different class)
        if not same_class:
            has_classification_errors = True

    if has_location_errors:
        sdc_types.append("SDC_L")

    if has_classification_errors:
        sdc_types.append("SDC_C")

    # 5. Return result
    if sdc_types:
        return ",".join(sdc_types)
    else:
        # Edge case: predictions differ but no specific SDC detected
        # (e.g., only score differences within matched boxes)
        return "MASKED"