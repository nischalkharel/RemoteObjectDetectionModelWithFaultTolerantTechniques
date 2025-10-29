import json
import os
from pathlib import Path
import statistics
import random # Needed for random selection

# Import AP calculation function from compare_results.py
from compare_results import calculate_ap_for_image

# Import inference function from baseline_inference.py
from baseline_inference import run_inference

# --- Helper Functions ---
def xyxyn_to_yolo_format(box_xyxyn):
    """
    Convert normalized xyxy format to YOLO format (xc, yc, w, h) normalized.

    Args:
        box_xyxyn (list): [x1, y1, x2, y2] normalized coordinates

    Returns:
        list: [xc, yc, w, h] normalized coordinates
    """
    x1, y1, x2, y2 = box_xyxyn
    w = x2 - x1
    h = y2 - y1
    xc = x1 + w / 2
    yc = y1 + h / 2
    return [xc, yc, w, h]

def save_predictions_as_yolo(predictions_list, output_path):
    """
    Save predictions in YOLO .txt format (same as ground truth format).

    Args:
        predictions_list (list): List of predictions with 'box_xyxyn', 'label_id', 'score'
        output_path (Path): Path to save the .txt file
    """
    try:
        with open(output_path, 'w') as f:
            for pred in predictions_list:
                # Convert xyxyn to YOLO format
                xc, yc, w, h = xyxyn_to_yolo_format(pred['box_xyxyn'])
                class_id = pred['label_id']
                # Write in YOLO format: class_id xc yc w h
                f.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
    except Exception as e:
        print(f"  Warning: Could not save golden predictions to {output_path}: {e}")

# --- Configuration ---
# This script should be inside RemoteObjectDetectionModelWithFaultTolerantTechniques
BASE_DIR = Path.cwd()
WORKER_SCRIPT_NAME = "baseline_inference.py"
MODEL_RELATIVE_PATH = Path("Plane_Ship_Detection/Plane_Ship_Model.pt")
IMAGE_LIST_RELATIVE_PATH = Path("validation_dataset_list.txt")

# Output configuration
EXPERIMENT_NAME = "baseline_nofault"  # Name of experiment folder
OUTPUT_FILE_NAME = "baseline_nofault.json"  # Name of results JSON file
GOLDEN_PREDICTIONS_FOLDER = "golden_predictions"  # Folder to store golden run predictions

# <<< --- Image Selection Mode --- >>>
# Set to "sequential" for standard mAP evaluation (process each image once)
# Set to "random" to randomly sample images (like the fault injection campaign)
SELECTION_MODE = "sequential"
# If random, how many total runs (can be more than the number of images)
NUM_RANDOM_RUNS = 10000

# --- Construct absolute paths ---
worker_script_abs_path = BASE_DIR / WORKER_SCRIPT_NAME
model_abs_path = BASE_DIR / MODEL_RELATIVE_PATH
image_list_abs_path = BASE_DIR / IMAGE_LIST_RELATIVE_PATH

# Create nested output directory structure
output_base_dir = BASE_DIR / "output"
experiment_dir = output_base_dir / EXPERIMENT_NAME
golden_predictions_dir = experiment_dir / GOLDEN_PREDICTIONS_FOLDER
output_file_path = experiment_dir / OUTPUT_FILE_NAME

# Create all directories
output_base_dir.mkdir(exist_ok=True)
experiment_dir.mkdir(exist_ok=True)
golden_predictions_dir.mkdir(exist_ok=True)

# --- Check if files/scripts exist ---
# (Error checking code remains the same as before...)
if not worker_script_abs_path.is_file(): print(f"Error: Worker script not found: {worker_script_abs_path}"); exit()
if not model_abs_path.is_file(): print(f"Error: Model not found: {model_abs_path}"); exit()
if not image_list_abs_path.is_file(): print(f"Error: Image list not found: {image_list_abs_path}"); exit()


# --- Read Image List ---
try:
    with open(image_list_abs_path, "r") as f:
        # Read all pairs into memory
        image_label_pairs = [line.strip().split(',') for line in f if line.strip() and len(line.strip().split(',')) == 2] 
except Exception as e:
    print(f"Error reading image list file: {e}")
    exit()

if not image_label_pairs:
    print(f"Error: No valid image/label pairs found in {image_list_abs_path}")
    exit()

total_unique_images = len(image_label_pairs)
print(f"Found {total_unique_images} unique image/label pairs.")

# --- Determine Loop Iterations based on Mode ---
if SELECTION_MODE.lower() == "random":
    num_iterations = NUM_RANDOM_RUNS
    print(f"Running in RANDOM mode for {num_iterations} iterations.")
elif SELECTION_MODE.lower() == "sequential":
    num_iterations = total_unique_images
    print(f"Running in SEQUENTIAL mode for {num_iterations} iterations (once per image).")
else:
    print(f"Error: Invalid SELECTION_MODE '{SELECTION_MODE}'. Use 'sequential' or 'random'.")
    exit()

# --- Main Evaluation Loop ---
all_predictions_data = []
all_ground_truth_paths = []
all_inference_times = []
all_ap_scores = []  # Store AP score for each image 

print(f"Starting evaluation using model: {MODEL_RELATIVE_PATH.as_posix()}")

for i in range(num_iterations):
    
    # <<< --- Select Image Pair based on Mode --- >>>
    if SELECTION_MODE.lower() == "random":
        # Pick a random pair from the list for this iteration
        image_rel_path_str, label_rel_path_str = random.choice(image_label_pairs)
        iteration_label = f"Iteration {i+1}/{num_iterations} (Random)"
    else: # Sequential mode
        # Get the pair for this specific index
        image_rel_path_str, label_rel_path_str = image_label_pairs[i]
        iteration_label = f"Image {i+1}/{num_iterations} (Sequential)"
        
    image_abs_path = BASE_DIR / Path(image_rel_path_str)
    label_abs_path = BASE_DIR / Path(label_rel_path_str) 

    # --- File existence checks ---
    if not image_abs_path.is_file() or not label_abs_path.is_file():
        print(f"Warning: Skipping {iteration_label}. Image or Label not found: {image_abs_path} and {label_abs_path}")
        # In random mode, we just continue; in sequential, this image is skipped
        if SELECTION_MODE.lower() == "random": continue
        else:
             all_predictions_data.append({"image_path": image_rel_path_str, "predictions": []}) # Add placeholder
             all_ground_truth_paths.append(str(label_abs_path))
             all_inference_times.append(0)
             all_ap_scores.append(-1.0)  # Mark as error
             continue 

    # --- Call Inference Function Directly ---
    print(f"Processing {iteration_label}: {image_rel_path_str}...")

    try:
        # Call run_inference function directly instead of subprocess
        worker_data = run_inference(str(image_abs_path), str(model_abs_path))

        if worker_data.get("error"):
            print(f"  Worker Error: {worker_data['error']}")
            # Store placeholder results even on worker error
            all_predictions_data.append({"image_path": image_rel_path_str, "predictions": []})
            all_ground_truth_paths.append(str(label_abs_path))
            all_inference_times.append(0)
            all_ap_scores.append(-1.0)  # Mark as error
            continue

        # --- Store Results ---
        predictions_list = worker_data.get("predictions", [])
        all_predictions_data.append({
            "image_path": image_rel_path_str,
            "predictions": predictions_list
        })
        all_ground_truth_paths.append(str(label_abs_path))
        all_inference_times.append(worker_data.get("inference_time_ms", 0))

        # --- Calculate AP for this image ---
        ap_score = calculate_ap_for_image(predictions_list, str(label_abs_path))
        all_ap_scores.append(ap_score)
        if ap_score >= 0:
            print(f"  AP: {ap_score:.4f}")

        # --- Save Golden Predictions ---
        # Extract image filename without extension (e.g., "P0019" from "Images/Validation_Images/P0019.png")
        image_filename_stem = Path(image_rel_path_str).stem  # Gets filename without extension
        golden_pred_path = golden_predictions_dir / f"{image_filename_stem}.txt"
        save_predictions_as_yolo(predictions_list, golden_pred_path)

    except Exception as e:
        print(f"  Error during inference for {image_rel_path_str}: {e}")
        all_predictions_data.append({"image_path": image_rel_path_str, "predictions": []})
        all_ground_truth_paths.append(str(label_abs_path))
        all_inference_times.append(0)
        all_ap_scores.append(-1.0)  # Mark as error
        continue 

# --- Final Calculations ---

# Calculate average inference time
average_time_ms = 0
successful_runs = [t for t in all_inference_times if t > 0] # Exclude errors/skips
if successful_runs:
    average_time_ms = statistics.mean(successful_runs)

# Calculate mAP (mean Average Precision)
# Filter out error cases (AP = -1.0)
valid_ap_scores = [ap for ap in all_ap_scores if ap >= 0]

if valid_ap_scores:
    # For YOLOv8, we're calculating AP at IoU=0.8 (which is close to mAP@0.50 metric)
    # Since we only use one IoU threshold (0.8), we report it as mAP
    mAP = statistics.mean(valid_ap_scores)
else:
    mAP = -1.0 

print("\n--- Evaluation Complete ---")
print(f"Mode: {SELECTION_MODE.upper()}")
print(f"Total iterations attempted: {num_iterations}")
print(f"Total images processed successfully: {len(successful_runs)}")
print(f"Total valid AP calculations: {len(valid_ap_scores)}")
print(f"Average Inference Time (successful runs): {average_time_ms:.2f} ms")
print(f"mAP@0.8 (IoU threshold): {mAP:.4f}")

# --- Save Results to Output File ---
output_data = {
    "configuration": {
        "worker_script": WORKER_SCRIPT_NAME,
        "model_path": MODEL_RELATIVE_PATH.as_posix(),
        "image_list": IMAGE_LIST_RELATIVE_PATH.as_posix(),
        "selection_mode": SELECTION_MODE,
        "iou_threshold": 0.8
    },
    "results": {
        "total_iterations_attempted": num_iterations,
        "total_images_processed": len(successful_runs),
        "total_valid_ap_calculations": len(valid_ap_scores),
        "average_inference_time_ms": round(average_time_ms, 2),
        "mAP": round(mAP, 4)
    },
    "per_image_results": [
        {
            "image_path": data["image_path"],
            "ap_score": round(all_ap_scores[idx], 4) if all_ap_scores[idx] >= 0 else -1.0,
            "inference_time_ms": all_inference_times[idx],
            "num_detections": len(data["predictions"])
        }
        for idx, data in enumerate(all_predictions_data)
    ]
}

try:
    with open(output_file_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_file_path}")
except Exception as e:
    print(f"\nError saving results to file: {e}")