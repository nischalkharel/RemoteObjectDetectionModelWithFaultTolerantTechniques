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

# --- Example Usage (for testing this script directly) ---
if __name__ == "__main__":
     # Create dummy data matching worker output and GT format
     dummy_predictions = [
         {'box_xyxyn': [0.1, 0.1, 0.3, 0.3], 'label_id': 1, 'score': 0.95},
         {'box_xyxyn': [0.5, 0.5, 0.7, 0.7], 'label_id': 0, 'score': 0.88},
         {'box_xyxyn': [0.15, 0.15, 0.35, 0.35], 'label_id': 1, 'score': 0.70}, # FP likely
     ]
     # Create a dummy GT file
     dummy_gt_path = "dummy_gt.txt"
     with open(dummy_gt_path, "w") as f:
         f.write("1 0.2 0.2 0.2 0.2\n") # GT Ship (overlaps well with pred 0, poorly with 2)
         f.write("0 0.6 0.6 0.2 0.2\n") # GT Plane (overlaps well with pred 1)
         
     print(f"Calculating AP for dummy data with GT file '{dummy_gt_path}'...")
     ap_score = calculate_ap_for_image(dummy_predictions, dummy_gt_path)
     
     if ap_score >= 0:
         print(f"Calculated AP score: {ap_score:.4f}")
     else:
         print("AP calculation failed.")
         
     # Clean up dummy file
     os.remove(dummy_gt_path)