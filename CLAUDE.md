# Project Guidelines

## User Preferences
- **Keep it simple**: No unnecessary files, scripts, or automation
- **Focus on the experiment**: Don't create extra documentation, READMEs, or setup files unless absolutely necessary
- **Clean workspace**: Delete temporary files and avoid clutter
- **Minimal approach**: One script to do the job, not multiple helper scripts

## Project Goal
**Evaluate fault tolerance techniques for YOLOv8 object detection under GPU fault injection using NVBitFI**

### Techniques Being Tested:
1. `notechnique` - Baseline (no fault tolerance)
2. `tmr` - Triple Modular Redundancy (Run 3, Vote 1)

### Experimental Setup:
- Platform: Jetson Orin Nano (sm_87, CUDA 12.6)
- Model: YOLOv8n
- Dataset: 112 validation images
- Fault Injection: NVBitFI (10,000 injections per technique)
- Fault Model: Single-bit flips in general-purpose (GP) instructions

### Key Files:
- `notechnique_inference.py` - Baseline inference
- `tmr_inference.py` - TMR inference
- `run_single_inference.py` - Worker called by NVBitFI 10K times
- `run_evaluations.py` - Run without faults for testing
- `setup_and_run_nvbitfi.sh` - Complete NVBitFI setup and execution
- `compare_results.py` - Analyze predictions
- `aggregate_fault_results.py` - Aggregate 10K results

## Development Guidelines
- Keep adding any new things being installed into requirements.txt and also in the venv
- Always update existing files instead of creating new ones
- One setup script maximum - don't create multiple automation scripts
- Don't create documentation files (*.md, README) unless explicitly requested