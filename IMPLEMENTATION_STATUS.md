# NVBitFI Integration - Implementation Status

## Overview
This document tracks the implementation status of the NVBitFI fault injection integration for YOLOv8 object detection with fault tolerance techniques.

**Date:** October 31, 2025
**Status:** ✅ Core scripts completed - Ready for testing

## Important Notes from context.txt

### 1. Image Count Correction
- **Note:** Dataset has **112 images**, not 111 (as mentioned in line 1 of context.txt)
- Validation_dataset_list.txt should contain 112 entries

### 2. run_evaluations.py
- **Important:** Leave run_evaluations.py as-is (line 2 of context.txt)
- The single-image logic has been extracted to run_single_inference.py
- The loop iteration is now handled at a higher level (by NVBitFI framework)

## Completed Components ✅

### 1. run_single_inference.py
**Purpose:** Single-image inference worker for NVBitFI fault injection campaigns

**Features:**
- Dynamically loads technique modules (baseline, tmr, etc.)
- Supports random or specific image selection
- Outputs JSON results compatible with NVBitFI
- Handles errors gracefully with proper exit codes

**Usage:**
```bash
python run_single_inference.py \
  --technique baseline \
  --model Models/yolov8n.pt \
  --dataset-list validation_dataset_list.txt \
  --random-image \
  --output results/result_001.json
```

### 2. nvbitfi_config.ini
**Purpose:** Central configuration for all fault injection campaigns

**Sections:**
- `[paths]` - NVBitFI, CUDA, and project paths
- `[env]` - Environment variables
- `[build]` - Build configuration (ARCH=87 for sm_87)
- `[runner]` - NVBitFI pipeline commands
- `[injection]` - Fault injection parameters (10,000 injections)
- `[techniques]` - List of techniques to evaluate (baseline, tmr)
- `[logging]` - Output directories
- `[golden]` - Golden prediction configuration

**Note:** Paths are configured for Jetson Orin Nano deployment. Adjust before running.

### 3. aggregate_fault_results.py
**Purpose:** Aggregate 10,000+ fault injection results and generate statistics

**Features:**
- Loads all result JSONs from campaign
- Compares against golden predictions
- Classifies outcomes (MASKED, SDC_M, SDC_P, SDC_L, SDC_C, CRASH)
- Calculates aggregate statistics and AP metrics
- Generates comprehensive summary report

**Usage:**
```bash
python aggregate_fault_results.py \
  --technique baseline \
  --results-dir output/baseline_fault/results \
  --golden-dir output/baseline_nofault/golden_predictions \
  --output output/baseline_fault/summary_report.json \
  --iou-threshold 0.8 \
  --ap-tolerance 0.05
```

### 4. nvbitfi_wrapper.sh
**Purpose:** Main orchestrator for entire fault injection pipeline

**Features:**
- Parses nvbitfi_config.ini configuration
- Checks dependencies (nvcc, python3, PyTorch CUDA)
- Builds NVBitFI tools (injector, profiler, pf_injector)
- Runs campaigns for multiple techniques
- Calls aggregate_fault_results.py automatically
- Supports dry-run and small-test modes

**Usage:**
```bash
# Full campaign with all techniques
./nvbitfi_wrapper.sh --config nvbitfi_config.ini

# Single technique
./nvbitfi_wrapper.sh --technique baseline

# Small test (10 injections)
./nvbitfi_wrapper.sh --small-test --technique baseline

# Dry run (show commands without executing)
./nvbitfi_wrapper.sh --dry-run
```

### 5. requirements.txt
**Purpose:** Python dependencies for the entire project

**Key Dependencies:**
- torch==2.9.0, torchvision==0.24.0
- ultralytics==8.3.222 (YOLOv8)
- opencv-python==4.12.0.88
- numpy, scipy, matplotlib, PyYAML

**Installation:**
```bash
pip install -r requirements.txt
```

## Existing Components (Unchanged)

- ✅ `baseline_inference.py` - Baseline inference worker
- ✅ `tmr_inference.py` - TMR (Triple Modular Redundancy) inference worker
- ✅ `compare_results.py` - AP calculation & fault classification
- ✅ `run_evaluations.py` - Full evaluation orchestrator (no-fault mode)
- ✅ Golden predictions - `output/baseline_nofault/golden_predictions/` (112 files)

## File Structure

```
RemoteObjectDetectionModelWithFaultTolerantTechniques/
│
├── Models/
│   └── yolov8n.pt                     # YOLOv8 model weights
│
├── Images/
│   ├── Validation_Images/             # 112 PNG images
│   └── Validation_Labels/             # 112 TXT label files
│
├── validation_dataset_list.txt        # Image,Label paths (112 lines)
│
├── EXISTING SCRIPTS:
├── baseline_inference.py
├── tmr_inference.py
├── compare_results.py
├── run_evaluations.py
│
├── NEW SCRIPTS (Created Today):
├── run_single_inference.py            ✅ NEW
├── aggregate_fault_results.py         ✅ NEW
├── nvbitfi_wrapper.sh                 ✅ NEW
├── nvbitfi_config.ini                 ✅ NEW
├── requirements.txt                   ✅ UPDATED
│
└── output/
    ├── baseline_nofault/              # Part 1: No-fault baseline (COMPLETE)
    │   ├── baseline_nofault.json
    │   └── golden_predictions/        # 112 golden .txt files
    │
    ├── baseline_fault/                # Part 2: To be generated
    │   ├── results/                   # 10,000 result JSONs
    │   ├── nvbitfi_logs/
    │   └── summary_report.json
    │
    └── tmr_fault/                     # Part 3: To be generated
        ├── results/
        ├── nvbitfi_logs/
        └── summary_report.json
```

## Next Steps

### Testing Phase 1: Unit Testing (No NVBitFI)

1. **Test run_single_inference.py:**
   ```bash
   python run_single_inference.py \
     --technique baseline \
     --model Models/yolov8n.pt \
     --dataset-list validation_dataset_list.txt \
     --random-image \
     --output test_result.json

   # Verify test_result.json exists and contains valid predictions
   ```

2. **Test aggregate_fault_results.py:**
   ```bash
   # Create a few test results, then:
   python aggregate_fault_results.py \
     --technique baseline \
     --results-dir test_results/ \
     --golden-dir output/baseline_nofault/golden_predictions \
     --output test_summary.json
   ```

### Testing Phase 2: Small-Scale Integration

1. **Run small test campaign:**
   ```bash
   ./nvbitfi_wrapper.sh --small-test --technique baseline
   ```

2. **Verify outputs:**
   - Check that result JSONs are created
   - Verify summary_report.json is generated
   - Review statistics for correctness

### Testing Phase 3: Deployment to Jetson

1. **Transfer files to Jetson Orin Nano**

2. **Update nvbitfi_config.ini paths** to match Jetson environment

3. **Make nvbitfi_wrapper.sh executable:**
   ```bash
   chmod +x nvbitfi_wrapper.sh
   ```

4. **Activate correct conda environment:**
   ```bash
   conda activate jetson_env
   ```

5. **Run small test first:**
   ```bash
   ./nvbitfi_wrapper.sh --small-test --technique baseline
   ```

6. **Run full campaign (10,000 injections):**
   ```bash
   ./nvbitfi_wrapper.sh --technique baseline
   ```

## Integration with NVBitFI

The wrapper script currently includes placeholders for NVBitFI integration steps:

- **Step 1:** Profiling (run_profiler.py)
- **Step 2:** Injection list generation (generate_injection_list.py)
- **Step 3:** Injection campaign (run_injections.py)
- **Step 4:** Result parsing (parse_results.py)
- **Step 5:** Application-level aggregation (aggregate_fault_results.py) ✅

**Note:** Steps 1-4 require NVBitFI-specific configuration and will be completed during Jetson deployment.

## Troubleshooting

### Common Issues

1. **Module import errors:**
   - Ensure PYTHONPATH includes project root
   - Check that technique modules exist (baseline_inference.py, tmr_inference.py)

2. **CUDA not available:**
   - Verify PyTorch CUDA installation
   - Check CUDA_HOME and LD_LIBRARY_PATH

3. **Golden predictions not found:**
   - Run baseline no-fault evaluation first
   - Verify output/baseline_nofault/golden_predictions/ contains 112 .txt files

4. **Permission denied on .sh script:**
   - Run: `chmod +x nvbitfi_wrapper.sh`

## References

- **Design Document:** context.txt (comprehensive specifications)
- **NVBitFI Documentation:** https://github.com/NVlabs/NVBitFI
- **System Configuration:** Jetson Orin Nano, JetPack 6.0, CUDA 12.6, sm_87

---

**Status:** All core scripts implemented and ready for testing ✅
