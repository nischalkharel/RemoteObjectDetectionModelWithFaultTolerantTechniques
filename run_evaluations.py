import json
import os
from pathlib import Path
import statistics
import random # Needed for random selection

# Import AP calculation and fault classification functions from compare_results.py
from compare_results import calculate_ap_for_image, classify_fault_outcome

# Import inference function from inference
from baseline_inference import run_inference
from tmr_inference import run_tmr_inference as run_inference

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

def yolo_format_to_xyxyn(yolo_box):
    """
    Convert YOLO format (xc, yc, w, h) to normalized xyxy format.

    Args:
        yolo_box (list): [xc, yc, w, h] normalized coordinates

    Returns:
        list: [x1, y1, x2, y2] normalized coordinates
    """
    xc, yc, w, h = yolo_box
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return [x1, y1, x2, y2]

def load_golden_predictions(golden_pred_path):
    """
    Load golden predictions from YOLO .txt format file.

    Args:
        golden_pred_path (Path): Path to the golden predictions .txt file

    Returns:
        list or None: List of predictions with 'box_xyxyn', 'label_id', 'score' (score=1.0 for golden)
                      Returns empty list [] if file exists but has no detections (valid case)
                      Returns None if file doesn't exist or on error.
    """
    if not golden_pred_path.is_file():
        return None  # File not found - this is an error

    predictions = []
    try:
        with open(golden_pred_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:  # class_id xc yc w h
                    class_id = int(parts[0])
                    yolo_coords = [float(p) for p in parts[1:5]]
                    xyxyn_coords = yolo_format_to_xyxyn(yolo_coords)
                    predictions.append({
                        'box_xyxyn': xyxyn_coords,
                        'label_id': class_id,
                        'score': 1.0  # Golden predictions don't have scores, use 1.0
                    })
    except Exception as e:
        print(f"  Warning: Could not load golden predictions from {golden_pred_path}: {e}")
        return None  # Read error - this is an error

    return predictions  # Valid result: could be [] (no detections) or list of predictions

# --- Configuration ---
# This script should be inside RemoteObjectDetectionModelWithFaultTolerantTechniques
BASE_DIR = Path.cwd()
#WORKER_SCRIPT_NAME = "baseline_inference.py"
WORKER_SCRIPT_NAME = "tmr_inference.py"
MODEL_RELATIVE_PATH = Path("Plane_Ship_Detection/Plane_Ship_Model.pt")
IMAGE_LIST_RELATIVE_PATH = Path("validation_dataset_list.txt")

# Run type configuration
RUN_TYPE = "faultinjection"  # Options: "nofault", "faultinjection"

# Output configuration
#EXPERIMENT_NAME = "baseline_nofault"  # Name of experiment folder
EXPERIMENT_NAME = "tmr_fault"
#OUTPUT_FILE_NAME = "baseline_nofault.json"  # Name of results JSON file
OUTPUT_FILE_NAME = "tmr_fault.json"
GOLDEN_PREDICTIONS_FOLDER = "golden_predictions"  # Folder to store golden run predictions

# Fault injection configuration (only used when RUN_TYPE = "faultinjection")
GOLDEN_SOURCE_EXPERIMENT = "tmr_nofault"  # Which experiment's golden predictions to compare against

# <<< --- Image Selection Mode --- >>>
# Set to "sequential" for standard mAP evaluation (process each image once)
# Set to "random" to randomly sample images (like the fault injection campaign)
SELECTION_MODE = "random"  # Options: "sequential", "random"
# If random, how many total runs (can be more than the number of images)
NUM_RANDOM_RUNS = 500

# --- Construct absolute paths ---
worker_script_abs_path = BASE_DIR / WORKER_SCRIPT_NAME
model_abs_path = BASE_DIR / MODEL_RELATIVE_PATH
image_list_abs_path = BASE_DIR / IMAGE_LIST_RELATIVE_PATH

# Create nested output directory structure
output_base_dir = BASE_DIR / "output"
experiment_dir = output_base_dir / EXPERIMENT_NAME
golden_predictions_dir = experiment_dir / GOLDEN_PREDICTIONS_FOLDER  # For saving golden predictions
output_file_path = experiment_dir / OUTPUT_FILE_NAME

# Path to golden predictions source (for fault injection mode)
golden_source_dir = output_base_dir / GOLDEN_SOURCE_EXPERIMENT / GOLDEN_PREDICTIONS_FOLDER

# Create directories based on run type
output_base_dir.mkdir(exist_ok=True)
experiment_dir.mkdir(exist_ok=True)

if RUN_TYPE == "nofault":
    # Only create golden predictions folder in nofault mode
    golden_predictions_dir.mkdir(exist_ok=True)
elif RUN_TYPE == "faultinjection":
    # Verify golden source directory exists
    if not golden_source_dir.is_dir():
        print(f"Error: Golden predictions directory not found: {golden_source_dir}")
        print(f"Make sure to run a 'nofault' experiment first to generate golden predictions.")
        exit()

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
all_results = []  # Store AP scores (nofault) or fault outcomes (faultinjection) for each image

print(f"Starting evaluation using model: {MODEL_RELATIVE_PATH.as_posix()}")
print(f"Run Type: {RUN_TYPE.upper()}")

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
             all_results.append(-1.0 if RUN_TYPE == "nofault" else "ERROR")  # Mark as error
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
            all_results.append(-1.0 if RUN_TYPE == "nofault" else "CRASH")  # Mark as crash
            continue

        # --- Store Results ---
        predictions_list = worker_data.get("predictions", [])
        all_predictions_data.append({
            "image_path": image_rel_path_str,
            "predictions": predictions_list
        })
        all_ground_truth_paths.append(str(label_abs_path))
        all_inference_times.append(worker_data.get("inference_time_ms", 0))

        # Extract image filename without extension (e.g., "P0019" from "Images/Validation_Images/P0019.png")
        image_filename_stem = Path(image_rel_path_str).stem

        # --- Mode-specific processing ---
        if RUN_TYPE == "nofault":
            # Calculate AP for this image
            ap_score = calculate_ap_for_image(predictions_list, str(label_abs_path))
            all_results.append(ap_score)
            if ap_score >= 0:
                print(f"  AP: {ap_score:.4f}")

            # Save Golden Predictions
            golden_pred_path = golden_predictions_dir / f"{image_filename_stem}.txt"
            save_predictions_as_yolo(predictions_list, golden_pred_path)

        elif RUN_TYPE == "faultinjection":
            # Load golden predictions for this image
            golden_pred_path = golden_source_dir / f"{image_filename_stem}.txt"
            golden_predictions = load_golden_predictions(golden_pred_path)

            # Check if golden predictions file was missing or had read error (None)
            if golden_predictions is None:
                print(f"  Warning: No golden predictions file found for {image_filename_stem}")
                all_results.append("ERROR")
                continue

            # Classify fault outcome by comparing against golden predictions
            # Note: golden_predictions can be [] (empty list) which is valid - means no objects detected
            fault_outcome = classify_fault_outcome(golden_predictions, predictions_list, status="SUCCESS")
            all_results.append(fault_outcome)
            print(f"  Fault Outcome: {fault_outcome}")

    except Exception as e:
        print(f"  Error during inference for {image_rel_path_str}: {e}")
        all_predictions_data.append({"image_path": image_rel_path_str, "predictions": []})
        all_ground_truth_paths.append(str(label_abs_path))
        all_inference_times.append(0)
        all_results.append(-1.0 if RUN_TYPE == "nofault" else "CRASH")  # Mark as crash
        continue 

# --- Final Calculations ---

# Calculate average inference time
average_time_ms = 0
successful_runs = [t for t in all_inference_times if t > 0] # Exclude errors/skips
if successful_runs:
    average_time_ms = statistics.mean(successful_runs)

print("\n--- Evaluation Complete ---")
print(f"Run Type: {RUN_TYPE.upper()}")
print(f"Selection Mode: {SELECTION_MODE.upper()}")
print(f"Total iterations attempted: {num_iterations}")
print(f"Total images processed successfully: {len(successful_runs)}")
print(f"Average Inference Time (successful runs): {average_time_ms:.2f} ms")

# Mode-specific calculations and output
if RUN_TYPE == "nofault":
    # Calculate mAP (mean Average Precision)
    # Filter out error cases (AP = -1.0)
    valid_ap_scores = [ap for ap in all_results if isinstance(ap, (int, float)) and ap >= 0]

    if valid_ap_scores:
        # For YOLOv8, we're calculating AP at IoU=0.8
        # Since we only use one IoU threshold (0.8), we report it as mAP
        mAP = statistics.mean(valid_ap_scores)
    else:
        mAP = -1.0

    print(f"Total valid AP calculations: {len(valid_ap_scores)}")
    print(f"mAP@0.8 (IoU threshold): {mAP:.4f}")

elif RUN_TYPE == "faultinjection":
    # Count fault outcomes
    from collections import Counter
    fault_outcomes = [outcome for outcome in all_results if isinstance(outcome, str)]
    outcome_counts = Counter(fault_outcomes)

    # Calculate rates
    total_runs = len(all_results)
    crash_count = outcome_counts.get("CRASH", 0) + outcome_counts.get("ERROR", 0)
    masked_count = outcome_counts.get("MASKED", 0)

    # SDC count: any outcome containing "SDC"
    sdc_count = sum(1 for outcome in fault_outcomes if "SDC" in outcome and outcome != "MASKED")

    crash_rate = (crash_count / total_runs * 100) if total_runs > 0 else 0
    masked_rate = (masked_count / total_runs * 100) if total_runs > 0 else 0
    sdc_rate = (sdc_count / total_runs * 100) if total_runs > 0 else 0

    print(f"\n--- Fault Injection Statistics ---")
    print(f"Total runs: {total_runs}")
    print(f"MASKED: {masked_count} ({masked_rate:.2f}%)")
    print(f"SDC (any type): {sdc_count} ({sdc_rate:.2f}%)")
    print(f"CRASH/ERROR: {crash_count} ({crash_rate:.2f}%)")
    print(f"\nDetailed outcome breakdown:")
    for outcome, count in sorted(outcome_counts.items()):
        percentage = (count / total_runs * 100) if total_runs > 0 else 0
        print(f"  {outcome}: {count} ({percentage:.2f}%)")

# --- Save Results to Output File ---
output_data = {
    "configuration": {
        "run_type": RUN_TYPE,
        "worker_script": WORKER_SCRIPT_NAME,
        "model_path": MODEL_RELATIVE_PATH.as_posix(),
        "image_list": IMAGE_LIST_RELATIVE_PATH.as_posix(),
        "selection_mode": SELECTION_MODE,
        "iou_threshold": 0.8
    },
    "results": {
        "total_iterations_attempted": num_iterations,
        "total_images_processed": len(successful_runs),
        "average_inference_time_ms": round(average_time_ms, 2)
    }
}

# Add mode-specific results
if RUN_TYPE == "nofault":
    output_data["results"]["total_valid_ap_calculations"] = len(valid_ap_scores)
    output_data["results"]["mAP"] = round(mAP, 4)

    output_data["per_image_results"] = [
        {
            "image_path": data["image_path"],
            "ap_score": round(all_results[idx], 4) if isinstance(all_results[idx], (int, float)) and all_results[idx] >= 0 else -1.0,
            "inference_time_ms": all_inference_times[idx],
            "num_detections": len(data["predictions"])
        }
        for idx, data in enumerate(all_predictions_data)
    ]

elif RUN_TYPE == "faultinjection":
    output_data["configuration"]["golden_source_experiment"] = GOLDEN_SOURCE_EXPERIMENT

    output_data["results"]["fault_injection_statistics"] = {
        "total_runs": total_runs,
        "masked_count": masked_count,
        "masked_rate_percent": round(masked_rate, 2),
        "sdc_count": sdc_count,
        "sdc_rate_percent": round(sdc_rate, 2),
        "crash_count": crash_count,
        "crash_rate_percent": round(crash_rate, 2),
        "detailed_outcome_counts": dict(outcome_counts)
    }

    output_data["per_image_results"] = [
        {
            "image_path": data["image_path"],
            "fault_outcome": all_results[idx] if isinstance(all_results[idx], str) else "ERROR",
            "inference_time_ms": all_inference_times[idx],
            "num_detections": len(data["predictions"])
        }
        for idx, data in enumerate(all_predictions_data)
    ]

try:
    with open(output_file_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_file_path}")
except Exception as e:
    print(f"\nError saving results to file: {e}")