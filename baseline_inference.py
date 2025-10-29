"""
Baseline Inference Worker Script
Runs YOLOv8 inference on a single image and outputs predictions in JSON format.

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
except ImportError as e:
    # If imports fail, output error JSON and exit
    error_output = {
        "error": f"Missing required libraries: {e}. Install with: pip install ultralytics torch torchvision",
        "predictions": [],
        "inference_time_ms": 0
    }
    print(json.dumps(error_output))
    sys.exit(1)


def run_inference(image_path, model_path):
    """
    Run YOLOv8 inference on a single image.

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

        # Load model
        model = YOLO(str(model_path))

        # Run inference and measure time
        start_time = time.perf_counter()
        results = model(str(image_path), verbose=False)  # verbose=False to suppress YOLO logs
        end_time = time.perf_counter()

        inference_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds

        # Extract predictions from results
        predictions = []

        # YOLOv8 results object
        result = results[0]  # First (and only) image result

        if result.boxes is not None and len(result.boxes) > 0:
            # Extract boxes, classes, and scores
            boxes_xyxyn = result.boxes.xyxyn.cpu().numpy()  # Normalized xyxy format
            classes = result.boxes.cls.cpu().numpy()  # Class IDs
            scores = result.boxes.conf.cpu().numpy()  # Confidence scores

            # Format predictions for compare_results.py
            for box, cls, score in zip(boxes_xyxyn, classes, scores):
                predictions.append({
                    "box_xyxyn": box.tolist(),  # [x1, y1, x2, y2] normalized
                    "label_id": int(cls),  # 0: plane, 1: ship
                    "score": float(score)
                })

        # Return successful result
        return {
            "predictions": predictions,
            "inference_time_ms": round(inference_time_ms, 2)
        }

    except Exception as e:
        # Handle any unexpected errors
        return {
            "error": f"Inference error: {str(e)}",
            "predictions": [],
            "inference_time_ms": 0
        }


def main():
    """
    Main entry point for the baseline inference worker.
    Parses command-line arguments and runs inference.
    """
    parser = argparse.ArgumentParser(description="YOLOv8 Baseline Inference Worker")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLOv8 model (.pt file)")

    args = parser.parse_args()

    # Run inference
    result = run_inference(args.image, args.model)

    # Output JSON to stdout (captured by run_evaluations.py)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
