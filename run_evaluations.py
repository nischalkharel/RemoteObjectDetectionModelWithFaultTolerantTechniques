import subprocess
import json
import os
from pathlib import Path
import statistics
import random # Needed for random selection

# Import AP calculation function from compare_results.py
from compare_results import calculate_ap_for_image

# --- Configuration ---
# This script should be inside RemoteObjectDetectionModelWithFaultTolerantTechniques
BASE_DIR = Path.cwd() 
WORKER_SCRIPT_NAME = "baseline_inference.py" 
MODEL_RELATIVE_PATH = Path("Plane_Ship_Detection/Plane_Ship_Model.pt")
IMAGE_LIST_RELATIVE_PATH = Path("validation_dataset_list.txt")
OUTPUT_FILE_NAME = "baseline_nofault.json"
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
output_dir = BASE_DIR / "output"
output_file_path = output_dir / OUTPUT_FILE_NAME

# Create output directory if it doesn't exist
output_dir.mkdir(exist_ok=True)

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

    # --- Call the Worker Script ---
    command = [
        "python", 
        str(worker_script_abs_path),
        "--image", str(image_abs_path), 
        "--model", str(model_abs_path)
    ]
    
    print(f"Processing {iteration_label}: {image_rel_path_str}...")
    
    try:
        process_result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8') # Added encoding
        worker_output_json = process_result.stdout #getting the outputs
        
        # --- Parse Worker Output ---
        try:
            worker_data = json.loads(worker_output_json)
            
            if worker_data.get("error"):
                print(f"  Worker Error: {worker_data['error']}")
                # Store placeholder results even on worker error
                # Storing placeholders helps keep lists aligned for mAP calc if needed.
                all_predictions_data.append({"image_path": image_rel_path_str, "predictions": []})
                all_ground_truth_paths.append(str(label_abs_path))
                all_inference_times.append(0) # Or mark as error?
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

        except json.JSONDecodeError:
            print(f"  Error: Could not decode JSON from worker: {worker_output_json[:200]}...")
            all_predictions_data.append({"image_path": image_rel_path_str, "predictions": []})
            all_ground_truth_paths.append(str(label_abs_path))
            all_inference_times.append(0)
            all_ap_scores.append(-1.0)  # Mark as error
            continue 

    except subprocess.CalledProcessError as e:
        print(f"  Error running worker script for {image_rel_path_str}:")
        # (Error printing code remains the same...)
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