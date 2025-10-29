import os
from pathlib import Path

# --- Configuration ---
image_dir_relative = Path("Images/Validation_Images") 
label_dir_relative = Path("Images/Validation_Labels")
output_file = "validation_dataset_list.txt"
image_extension = ".png" # Assuming images are PNGs
label_extension = ".txt" # Assuming labels are TXT files

# --- Get the current working directory ---
# This assumes the script is run from RemoteObjectDetectionModelWithFaultTolerantTechniques
base_dir = Path.cwd() 

# --- Construct absolute paths for searching ---
image_dir_abs = base_dir / image_dir_relative
label_dir_abs = base_dir / label_dir_relative

# --- Find image files and create path pairs ---
path_pairs = []

# Check if the image directory exists
if not image_dir_abs.is_dir():
    print(f"Error: Image directory not found at {image_dir_abs}")
else:
    # Iterate through files in the image directory
    for image_file_abs in image_dir_abs.glob(f"*{image_extension}"):
        # Get the base filename without extension (e.g., "P0019")
        base_filename = image_file_abs.stem 
        
        # Construct the expected label file path (absolute)
        label_file_abs = label_dir_abs / f"{base_filename}{label_extension}"
        
        # Check if the corresponding label file exists
        if label_file_abs.is_file():
            # Create the relative paths for the output file
            image_path_relative = image_dir_relative / image_file_abs.name
            label_path_relative = label_dir_relative / label_file_abs.name
            
            # Use forward slashes for better cross-platform compatibility in paths
            path_pairs.append(f"{image_path_relative.as_posix()},{label_path_relative.as_posix()}")
        else:
            print(f"Warning: Label file not found for image {image_file_abs.name}")

# --- Write the paths to the output file ---
if path_pairs:
    try:
        with open(output_file, "w") as f:
            for pair in sorted(path_pairs): # Sort for consistent order
                f.write(pair + "\n")
        print(f"Successfully created '{output_file}' with {len(path_pairs)} image/label pairs.")
    except Exception as e:
        print(f"Error writing to output file '{output_file}': {e}")
elif image_dir_abs.is_dir():
     print("No image/label pairs found to write.")