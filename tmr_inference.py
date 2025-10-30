"""
TMR (Triple Modular Redundancy) Inference Worker Script
Implements "Run 3, Vote 1" fault tolerance technique for YOLOv8.

Runs inference on the same image with 3 independent model instances,
then uses a voting algorithm to merge results. Only detections where
at least 2 out of 3 models agree are included in the final output.

Expected to be called by run_evaluations.py with --image and --model arguments.
Outputs JSON to stdout with predictions and inference time.
"""

import argparse
import json
import sys
import time
from pathlib import Path

try:
    from ultralytics import YOLO
    import torch
    from torchvision.ops.boxes import box_iou
except ImportError as e:
    # If imports fail, output error JSON and exit
    error_output = {
        "error": f"Missing required libraries: {e}. Install with: pip install ultralytics torch torchvision",
        "predictions": [],
        "inference_time_ms": 0
    }
    print(json.dumps(error_output))
    sys.exit(1)

# --- TMR Configuration ---
VOTING_IOU_THRESHOLD = 0.7  # IoU threshold for clustering predictions during voting


def cluster_predictions_for_voting(all_predictions):
    """
    Cluster predictions from 3 models based on class and IoU overlap.

    Args:
        all_predictions (list): List of 3 prediction lists from models A, B, C
                               Each prediction: {'box_xyxyn': [...], 'label_id': int, 'score': float}

    Returns:
        list: Clusters where each cluster is a list of (model_idx, pred_idx, prediction) tuples
    """
    # Flatten all predictions with their source info
    flat_preds = []
    for model_idx, preds in enumerate(all_predictions):
        for pred_idx, pred in enumerate(preds):
            flat_preds.append((model_idx, pred_idx, pred))

    if not flat_preds:
        return []

    clusters = []
    used = set()

    # For each prediction, try to find matching predictions
    for i, (m1, p1, pred1) in enumerate(flat_preds):
        if i in used:
            continue

        # Start new cluster with this prediction
        cluster = [(m1, p1, pred1)]
        used.add(i)

        # Find all predictions that match this one
        for j, (m2, p2, pred2) in enumerate(flat_preds):
            if j in used or j <= i:
                continue

            # Check if same class
            if pred1['label_id'] != pred2['label_id']:
                continue

            # Check IoU
            box1 = torch.tensor([pred1['box_xyxyn']], dtype=torch.float32)
            box2 = torch.tensor([pred2['box_xyxyn']], dtype=torch.float32)
            iou = box_iou(box1, box2).item()

            if iou >= VOTING_IOU_THRESHOLD:
                cluster.append((m2, p2, pred2))
                used.add(j)

        clusters.append(cluster)

    return clusters


def vote_and_merge(clusters):
    """
    Apply voting logic: only keep clusters with 2 or 3 votes (2/3 agreement).
    For valid clusters, average the box coordinates and scores.

    Args:
        clusters (list): List of clusters from cluster_predictions_for_voting()

    Returns:
        list: Final voted predictions in same format as baseline_inference.py
    """
    voted_predictions = []

    for cluster in clusters:
        # Only keep if at least 2 models agree (2 or 3 predictions in cluster)
        if len(cluster) < 2:
            continue

        # Extract predictions from cluster
        preds = [pred for (_, _, pred) in cluster]

        # Average box coordinates
        boxes = [pred['box_xyxyn'] for pred in preds]
        avg_box = [
            sum(box[i] for box in boxes) / len(boxes)
            for i in range(4)
        ]

        # Average scores
        avg_score = sum(pred['score'] for pred in preds) / len(preds)

        # Use class from first prediction (all should be same due to clustering)
        label_id = preds[0]['label_id']

        voted_predictions.append({
            'box_xyxyn': avg_box,
            'label_id': label_id,
            'score': avg_score
        })

    return voted_predictions


def run_tmr_inference(image_path, model_path):
    """
    Run TMR (Triple Modular Redundancy) inference on a single image.

    Loads model 3 times, runs inference 3 times, and applies voting.

    Args:
        image_path (str): Absolute path to the image file
        model_path (str): Absolute path to the YOLOv8 model file (.pt)

    Returns:
        dict: JSON-compatible dict with predictions and inference_time_ms
    """
    try:
        # Validate paths
        image_path = Path(image_path)
        model_path = Path(model_path)

        if not image_path.is_file():
            return {
                "error": f"Image file not found: {image_path}",
                "predictions": [],
                "inference_time_ms": 0
            }

        if not model_path.is_file():
            return {
                "error": f"Model file not found: {model_path}",
                "predictions": [],
                "inference_time_ms": 0
            }

        # --- TMR: Load model 3 times as independent instances ---
        model_a = YOLO(str(model_path))
        model_b = YOLO(str(model_path))
        model_c = YOLO(str(model_path))

        # --- TMR: Run inference 3 times and measure total time ---
        start_time = time.perf_counter()

        results_a = model_a(str(image_path), verbose=False)
        results_b = model_b(str(image_path), verbose=False)
        results_c = model_c(str(image_path), verbose=False)

        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000

        # --- Extract predictions from all 3 models ---
        all_predictions = []

        for results in [results_a, results_b, results_c]:
            result = results[0]  # First (and only) image result
            predictions = []

            if result.boxes is not None and len(result.boxes) > 0:
                boxes_xyxyn = result.boxes.xyxyn.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()

                for box, cls, score in zip(boxes_xyxyn, classes, scores):
                    predictions.append({
                        "box_xyxyn": box.tolist(),
                        "label_id": int(cls),
                        "score": float(score)
                    })

            all_predictions.append(predictions)

        # --- TMR: Apply voting algorithm ---
        clusters = cluster_predictions_for_voting(all_predictions)
        final_predictions = vote_and_merge(clusters)

        # Return successful result
        return {
            "predictions": final_predictions,
            "inference_time_ms": round(inference_time_ms, 2)
        }

    except Exception as e:
        # Handle any unexpected errors
        return {
            "error": f"TMR inference error: {str(e)}",
            "predictions": [],
            "inference_time_ms": 0
        }


def main():
    """
    Main entry point for the TMR inference worker.
    Parses command-line arguments and runs TMR inference.
    """
    parser = argparse.ArgumentParser(description="YOLOv8 TMR (Triple Modular Redundancy) Inference Worker")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLOv8 model (.pt file)")

    args = parser.parse_args()

    # Run TMR inference
    result = run_tmr_inference(args.image, args.model)

    # Output JSON to stdout (captured by run_evaluations.py)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
