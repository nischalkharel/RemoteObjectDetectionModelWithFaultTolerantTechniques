#!/bin/bash
################################################################################
# NVBitFI Fault Injection Wrapper Script
#
# Main orchestrator for running fault injection campaigns on YOLOv8 object
# detection with multiple fault tolerance techniques.
#
# Usage:
#   ./nvbitfi_wrapper.sh [--config CONFIG_FILE] [--technique TECHNIQUE] [--dry-run]
#
# This script:
#   1. Loads configuration from nvbitfi_config.ini
#   2. Sets up environment variables
#   3. Builds NVBitFI tools if needed
#   4. Runs fault injection campaigns for each technique
#   5. Aggregates and reports results
################################################################################

set -e  # Exit on any error
set -u  # Exit on undefined variables

# --- Default Configuration ---
CONFIG_FILE="./nvbitfi_config.ini"
TECHNIQUE=""
DRY_RUN=false
SKIP_BUILD=false
SMALL_TEST=false
SMALL_TEST_SIZE=10

# --- Color Output (optional) ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# --- Helper Functions ---

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --config FILE       Path to nvbitfi_config.ini (default: ./nvbitfi_config.ini)"
    echo "  --technique NAME    Run only specific technique (default: all from config)"
    echo "  --dry-run           Print commands without executing"
    echo "  --skip-build        Skip NVBitFI tool build step"
    echo "  --small-test        Run small test with 10 injections"
    echo "  --help              Show this help message"
    echo ""
    exit 1
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse INI file using Python
parse_ini() {
    local file=$1
    local section=$2
    local key=$3
    python3 -c "
import configparser
import sys
config = configparser.ConfigParser()
config.read('$file')
try:
    print(config['$section']['$key'])
except:
    sys.exit(1)
"
}

# --- Argument Parsing ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --technique)
            TECHNIQUE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --small-test)
            SMALL_TEST=true
            shift
            ;;
        --help)
            usage
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            ;;
    esac
done

# --- Check Prerequisites ---
check_dependencies() {
    log_info "Checking dependencies..."

    # Check nvcc
    if ! command -v nvcc &> /dev/null; then
        log_error "nvcc not found. Install CUDA toolkit."
        exit 1
    fi
    log_success "nvcc: $(nvcc --version | grep 'release' | head -1)"

    # Check python3
    if ! command -v python3 &> /dev/null; then
        log_error "python3 not found."
        exit 1
    fi
    log_success "python3: $(python3 --version)"

    # Check conda environment (if applicable)
    if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
        log_success "Conda environment: $CONDA_DEFAULT_ENV"
    else
        log_warning "No conda environment active. Make sure correct Python environment is active."
    fi

    # Check PyTorch CUDA
    if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        log_error "PyTorch CUDA not available."
        exit 1
    fi
    log_success "PyTorch CUDA: Available"

    # Check config file exists
    if [[ ! -f "$CONFIG_FILE" ]]; then
        log_error "Config file not found: $CONFIG_FILE"
        exit 1
    fi
    log_success "Config file: $CONFIG_FILE"

    echo ""
}

# --- Build NVBitFI Tools ---
build_nvbitfi() {
    local nvbitfi_root=$1

    log_info "Building NVBitFI tools..."

    if [[ ! -d "$nvbitfi_root" ]]; then
        log_error "NVBitFI root directory not found: $nvbitfi_root"
        exit 1
    fi

    cd "$nvbitfi_root"

    # Build injector
    log_info "Building injector..."
    cd injector
    if [[ "$DRY_RUN" = false ]]; then
        make clean && make || { log_error "Failed to build injector"; exit 1; }
    else
        echo "  [DRY RUN] make clean && make"
    fi
    cd ..

    # Build profiler
    log_info "Building profiler..."
    cd profiler
    if [[ "$DRY_RUN" = false ]]; then
        make clean && make || { log_error "Failed to build profiler"; exit 1; }
    else
        echo "  [DRY RUN] make clean && make"
    fi
    cd ..

    # Build pf_injector (if directory exists)
    if [[ -d "pf_injector" ]]; then
        log_info "Building pf_injector..."
        cd pf_injector
        if [[ "$DRY_RUN" = false ]]; then
            make clean && make || { log_error "Failed to build pf_injector"; exit 1; }
        else
            echo "  [DRY RUN] make clean && make"
        fi
        cd ..
    fi

    log_success "NVBitFI tools built successfully"
    echo ""
}

# --- Run Fault Injection Campaign ---
run_campaign() {
    local technique=$1
    local project_root=$2
    local nvbitfi_root=$3
    local model_path=$4
    local dataset_list=$5
    local num_injections=$6

    echo ""
    echo "=========================================="
    log_info "RUNNING CAMPAIGN: $technique"
    echo "=========================================="

    # Create output directory
    local output_dir="$project_root/output/${technique}_fault"
    local results_dir="$output_dir/results"
    local nvbitfi_logs_dir="$output_dir/nvbitfi_logs"

    log_info "Creating output directories..."
    if [[ "$DRY_RUN" = false ]]; then
        mkdir -p "$results_dir"
        mkdir -p "$nvbitfi_logs_dir"
    else
        echo "  [DRY RUN] mkdir -p $results_dir"
        echo "  [DRY RUN] mkdir -p $nvbitfi_logs_dir"
    fi

    # Create NVBitFI workload directory
    local workload_dir="$nvbitfi_root/logs/$technique"
    log_info "Creating NVBitFI workload directory..."
    if [[ "$DRY_RUN" = false ]]; then
        mkdir -p "$workload_dir"
    else
        echo "  [DRY RUN] mkdir -p $workload_dir"
    fi

    # --- Step 1: Profiling ---
    log_info "Step 1: Profiling application..."
    echo "  NOTE: This step would run the profiling phase of NVBitFI"
    echo "  Command would be similar to:"
    echo "  python3 $nvbitfi_root/scripts/run_profiler.py -a $technique"
    echo "  This requires NVBitFI-specific setup and is marked as TODO"

    # --- Step 2: Generate Injection List ---
    log_info "Step 2: Generating injection list..."
    echo "  NOTE: This step would generate the fault injection sites"
    echo "  Command would be similar to:"
    echo "  python3 $nvbitfi_root/scripts/generate_injection_list.py -a $technique"
    echo "  This requires NVBitFI-specific setup and is marked as TODO"

    # --- Step 3: Run Injections ---
    log_info "Step 3: Running $num_injections fault injections..."
    echo "  NOTE: This step would run the actual fault injection campaign"
    echo "  Each injection would call:"
    echo "  python3 run_single_inference.py \\"
    echo "    --technique $technique \\"
    echo "    --model $model_path \\"
    echo "    --dataset-list $dataset_list \\"
    echo "    --random-image \\"
    echo "    --output $results_dir/result_XXXX.json"
    echo "  This would be wrapped by NVBitFI's injection framework"

    # For testing purposes, we can run a few trials manually
    if [[ "$SMALL_TEST" = true && "$DRY_RUN" = false ]]; then
        log_info "Running small test with $SMALL_TEST_SIZE trials..."
        for i in $(seq 1 $SMALL_TEST_SIZE); do
            result_file=$(printf "$results_dir/result_%04d.json" $i)
            echo "  Trial $i/$SMALL_TEST_SIZE -> $result_file"
            python3 "$project_root/run_single_inference.py" \
                --technique "$technique" \
                --model "$model_path" \
                --dataset-list "$dataset_list" \
                --random-image \
                --output "$result_file" \
                --seed $i || log_warning "Trial $i failed"
        done
        log_success "Small test complete"
    fi

    # --- Step 4: Parse NVBitFI Results ---
    log_info "Step 4: Parsing NVBitFI results..."
    echo "  NOTE: This step would parse NVBitFI's internal logs"
    echo "  Command would be similar to:"
    echo "  python3 $nvbitfi_root/scripts/parse_results.py -a $technique"

    # --- Step 5: Aggregate Application Results ---
    log_info "Step 5: Aggregating application-level results..."

    local golden_dir="$project_root/output/notechnique_nofault/golden_predictions"
    local summary_output="$output_dir/summary_report.json"

    if [[ "$DRY_RUN" = false ]]; then
        if [[ -d "$results_dir" ]] && [[ $(ls -A "$results_dir" 2>/dev/null | wc -l) -gt 0 ]]; then
            python3 "$project_root/aggregate_fault_results.py" \
                --technique "$technique" \
                --results-dir "$results_dir" \
                --golden-dir "$golden_dir" \
                --output "$summary_output" \
                --iou-threshold 0.8 \
                --ap-tolerance 0.05
            log_success "Results aggregated: $summary_output"
        else
            log_warning "No result files found in $results_dir - skipping aggregation"
        fi
    else
        echo "  [DRY RUN] python3 aggregate_fault_results.py \\"
        echo "    --technique $technique \\"
        echo "    --results-dir $results_dir \\"
        echo "    --golden-dir $golden_dir \\"
        echo "    --output $summary_output"
    fi

    log_success "Campaign complete: $technique"
    echo ""
}

# --- Main Execution ---
main() {
    echo "=========================================="
    echo "NVBITFI FAULT INJECTION WRAPPER"
    echo "=========================================="
    echo "Config: $CONFIG_FILE"
    if [[ "$DRY_RUN" = true ]]; then
        echo "Mode: DRY RUN (commands will not be executed)"
    fi
    if [[ "$SMALL_TEST" = true ]]; then
        echo "Mode: SMALL TEST ($SMALL_TEST_SIZE injections)"
    fi
    echo ""

    # Check dependencies
    check_dependencies

    # Parse configuration
    log_info "Loading configuration..."
    NVBITFI_ROOT=$(parse_ini "$CONFIG_FILE" "paths" "nvbitfi_root") || {
        log_error "Failed to parse nvbitfi_root from config"
        exit 1
    }
    PROJECT_ROOT=$(parse_ini "$CONFIG_FILE" "paths" "project_root") || {
        log_error "Failed to parse project_root from config"
        exit 1
    }
    MODEL_PATH=$(parse_ini "$CONFIG_FILE" "paths" "model_path") || {
        log_error "Failed to parse model_path from config"
        exit 1
    }
    DATASET_LIST=$(parse_ini "$CONFIG_FILE" "paths" "dataset_list") || {
        log_error "Failed to parse dataset_list from config"
        exit 1
    }

    if [[ "$SMALL_TEST" = true ]]; then
        NUM_INJECTIONS=$SMALL_TEST_SIZE
    else
        NUM_INJECTIONS=$(parse_ini "$CONFIG_FILE" "injection" "num_injections") || {
            log_error "Failed to parse num_injections from config"
            exit 1
        }
    fi

    if [[ -z "$TECHNIQUE" ]]; then
        TECHNIQUES=$(parse_ini "$CONFIG_FILE" "techniques" "techniques") || {
            log_error "Failed to parse techniques from config"
            exit 1
        }
    else
        TECHNIQUES="$TECHNIQUE"
    fi

    log_success "Configuration loaded"
    echo "  NVBitFI Root: $NVBITFI_ROOT"
    echo "  Project Root: $PROJECT_ROOT"
    echo "  Model Path: $MODEL_PATH"
    echo "  Dataset List: $DATASET_LIST"
    echo "  Techniques: $TECHNIQUES"
    echo "  Injections per technique: $NUM_INJECTIONS"
    echo ""

    # Export environment variables
    log_info "Setting environment variables..."
    export NVBITFI_HOME="$NVBITFI_ROOT"
    export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
    log_success "Environment variables set"
    echo ""

    # Build NVBitFI tools
    if [[ "$SKIP_BUILD" = false ]]; then
        read -p "Build NVBitFI tools? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            build_nvbitfi "$NVBITFI_ROOT"
        else
            log_info "Skipping build"
        fi
    else
        log_info "Skipping build (--skip-build flag set)"
    fi

    # Run campaigns for each technique
    IFS=',' read -ra TECH_ARRAY <<< "$TECHNIQUES"
    for tech in "${TECH_ARRAY[@]}"; do
        tech=$(echo "$tech" | xargs)  # Trim whitespace
        run_campaign "$tech" "$PROJECT_ROOT" "$NVBITFI_ROOT" "$MODEL_PATH" "$DATASET_LIST" "$NUM_INJECTIONS"
    done

    echo "=========================================="
    log_success "ALL CAMPAIGNS COMPLETE"
    echo "=========================================="
}

# Run main function
main
