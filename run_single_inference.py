"""
Single-Image Inference Worker for NVBitFI Fault Injection

This script runs inference on a single image using a specified fault tolerance technique.
It's designed to be called repeatedly by NVBitFI for fault injection campaigns.

Each invocation:
- Runs inference on ONE image (randomly selected or specified)
- Saves results to a JSON file
- Exits cleanly for NVBitFI to process

Usage:
  python run_single_inference.py \
    --technique notechnique \
    --model Models/yolov8n.pt \
    --dataset-list validation_dataset_list.txt \
    --random-image \
    --output results/result_001.json
"""

import argparse
import json
import sys
import random
import importlib
from pathlib import Path
from datetime import datetime


def load_dataset_list(dataset_list_path):
    """
    Load image/label pairs from dataset list file.

    Args:
        dataset_list_path (str): Path to validation_dataset_list.txt

    Returns:
        list: List of (image_path, label_path) tuples
    """
    pairs = []
    try:
        with open(dataset_list_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and ',' in line:
                    parts = line.split(',')
                    if len(parts) == 2:
                        pairs.append((parts[0], parts[1]))
    except Exception as e:
        print(f"Error reading dataset list: {e}", file=sys.stderr)
        return []

    return pairs


def select_image(dataset_list_path, specific_image=None, use_random=False, seed=None):
    """
    Select an image from the dataset.

    Args:
        dataset_list_path (str): Path to dataset list file
        specific_image (str): Specific image path to use (overrides random)
        use_random (bool): Randomly select from dataset
        seed (int): Random seed for reproducibility

    Returns:
        tuple: (image_path, label_path) or (None, None) on error
    """
    # If specific image provided, use it directly
    if specific_image:
        # Need to find corresponding label
        pairs = load_dataset_list(dataset_list_path)
        for img_path, lbl_path in pairs:
            if img_path == specific_image or Path(img_path).name == Path(specific_image).name:
                return (img_path, lbl_path)
        # If not found in list, try to infer label path
        # Assume label has same name but in Validation_Labels folder
        label_path = specific_image.replace('Validation_Images', 'Validation_Labels').replace('.png', '.txt')
        return (specific_image, label_path)

    # Random selection
    if use_random:
        pairs = load_dataset_list(dataset_list_path)
        if not pairs:
            return (None, None)

        if seed is not None:
            random.seed(seed)

        return random.choice(pairs)

    # Default: use first image
    pairs = load_dataset_list(dataset_list_path)
    if pairs:
        return pairs[0]

    return (None, None)


def run_single_inference(technique, model_path, image_path, label_path, output_path):
    """
    Run inference using specified technique and save results.

    Args:
        technique (str): Technique name (notechnique, tmr, etc.)
        model_path (str): Path to model file
        image_path (str): Path to image file
        label_path (str): Path to label file
        output_path (str): Path to save output JSON

    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    try:
        # Dynamically import the technique module
        module_name = f"{technique}_inference"
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            error_result = {
                "technique": technique,
                "image_path": str(image_path),
                "label_path": str(label_path),
                "predictions": [],
                "inference_time_ms": 0,
                "timestamp": datetime.now().isoformat(),
                "error": f"Failed to import technique module '{module_name}': {e}"
            }
            with open(output_path, 'w') as f:
                json.dump(error_result, f, indent=2)
            return 1

        # Get the inference function
        # All techniques now use consistent naming: run_{technique}_inference
        function_name = f"run_{technique}_inference"
        try:
            inference_function = getattr(module, function_name)
        except AttributeError:
            error_result = {
                "technique": technique,
                "image_path": str(image_path),
                "label_path": str(label_path),
                "predictions": [],
                "inference_time_ms": 0,
                "timestamp": datetime.now().isoformat(),
                "error": f"Function '{function_name}' not found in module '{module_name}'"
            }
            with open(output_path, 'w') as f:
                json.dump(error_result, f, indent=2)
            return 1

        # Validate paths
        image_path = Path(image_path)
        if not image_path.is_file():
            error_result = {
                "technique": technique,
                "image_path": str(image_path),
                "label_path": str(label_path),
                "predictions": [],
                "inference_time_ms": 0,
                "timestamp": datetime.now().isoformat(),
                "error": f"Image file not found: {image_path}"
            }
            with open(output_path, 'w') as f:
                json.dump(error_result, f, indent=2)
            return 1

        # Run inference
        result = inference_function(str(image_path), str(model_path))

        # Add metadata to result
        result["technique"] = technique
        result["image_path"] = str(image_path)
        result["label_path"] = str(label_path)
        result["timestamp"] = datetime.now().isoformat()

        # Ensure error field exists
        if "error" not in result:
            result["error"] = None

        # Save result to output file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        # Return exit code based on whether inference succeeded
        if result.get("error"):
            return 1

        return 0

    except Exception as e:
        # Catch-all error handler
        error_result = {
            "technique": technique,
            "image_path": str(image_path),
            "label_path": str(label_path),
            "predictions": [],
            "inference_time_ms": 0,
            "timestamp": datetime.now().isoformat(),
            "error": f"Unexpected error: {str(e)}"
        }
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(error_result, f, indent=2)
        except:
            print(json.dumps(error_result), file=sys.stderr)

        return 1


def main():
    """Main entry point for single-image inference worker."""
    parser = argparse.ArgumentParser(
        description="Single-image inference worker for NVBitFI fault injection"
    )

    # Required arguments
    parser.add_argument("--technique", type=str, required=True,
                       help="Technique module name (notechnique, tmr, etc.)")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to YOLOv8 model (.pt file)")
    parser.add_argument("--dataset-list", type=str, required=True,
                       help="Path to validation dataset list file")
    parser.add_argument("--output", type=str, required=True,
                       help="Output JSON file path")

    # Image selection (mutually exclusive)
    image_group = parser.add_mutually_exclusive_group()
    image_group.add_argument("--image", type=str,
                            help="Specific image path to process")
    image_group.add_argument("--random-image", action="store_true",
                            help="Randomly select image from dataset")

    # Optional arguments
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")

    args = parser.parse_args()

    # Select image
    image_path, label_path = select_image(
        args.dataset_list,
        specific_image=args.image,
        use_random=args.random_image,
        seed=args.seed
    )

    if image_path is None or label_path is None:
        error_result = {
            "technique": args.technique,
            "image_path": None,
            "label_path": None,
            "predictions": [],
            "inference_time_ms": 0,
            "timestamp": datetime.now().isoformat(),
            "error": "Failed to select image from dataset"
        }
        with open(args.output, 'w') as f:
            json.dump(error_result, f, indent=2)
        sys.exit(1)

    # Run inference
    exit_code = run_single_inference(
        args.technique,
        args.model,
        image_path,
        label_path,
        args.output
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
