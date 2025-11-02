#!/bin/bash
################################################################################
# NVBitFI YOLOv8 Fault Injection - Complete Setup and Run Script
#
# USAGE:
#   ./setup_and_run_nvbitfi.sh setup              # First time setup only
#   ./setup_and_run_nvbitfi.sh run <technique>    # Run 10K fault injection
#
# EXAMPLE:
#   ./setup_and_run_nvbitfi.sh setup
#   ./setup_and_run_nvbitfi.sh run notechnique
#   ./setup_and_run_nvbitfi.sh run tmr
################################################################################

set -e

# ============================================================================
# CONFIGURATION - CHANGE THESE IF NEEDED
# ============================================================================
PROJECT_ROOT="/home/shrec/NischalNVBitFi/RemoteObjectDetectionModelWithFaultTolerantTechniques"
NVBITFI_ROOT="/home/shrec/NischalNVBitFi/nvbit_release_aarch64/nvbitfi"
MODEL_PATH="${PROJECT_ROOT}/Models/yolov8n.pt"
DATASET_LIST="${PROJECT_ROOT}/validation_dataset_list.txt"

# ============================================================================
# SETUP FUNCTION - Run this once
# ============================================================================
setup() {
    echo "=========================================="
    echo "NVBitFI Setup"
    echo "=========================================="

    cd ${PROJECT_ROOT}

    # Create Models directory
    echo "Creating Models directory..."
    mkdir -p Models
    [ -f yolov8n.pt ] && mv yolov8n.pt Models/ || echo "Model already in place"

    # Rename baseline to notechnique if needed
    if [ -f baseline_inference.py ] && [ ! -f notechnique_inference.py ]; then
        echo "Renaming baseline_inference.py to notechnique_inference.py..."
        mv baseline_inference.py notechnique_inference.py
    fi

    # Create workload directories for each technique
    for TECH in notechnique tmr; do
        echo "Setting up yolov8_${TECH}..."
        WORKLOAD_DIR="${NVBITFI_ROOT}/test-apps/yolov8_${TECH}"
        mkdir -p ${WORKLOAD_DIR}

        # Create run.sh
        cat > ${WORKLOAD_DIR}/run.sh <<EOF
#!/bin/bash
eval \${PRELOAD_FLAG} python3 ${PROJECT_ROOT}/run_single_inference.py \\
    --technique ${TECH} \\
    --model ${MODEL_PATH} \\
    --dataset-list ${DATASET_LIST} \\
    --random-image \\
    --output result.json > stdout.txt 2> stderr.txt
EOF
        chmod +x ${WORKLOAD_DIR}/run.sh

        # Create sdc_check.sh
        cat > ${WORKLOAD_DIR}/sdc_check.sh <<'EOF'
#!/bin/bash
diff stdout.txt ${APP_DIR}/golden_stdout.txt > stdout_diff.log 2>&1 || touch stdout_diff.log
diff stderr.txt ${APP_DIR}/golden_stderr.txt > stderr_diff.log 2>&1 || touch stderr_diff.log
touch diff.log
touch special_check.log
EOF
        chmod +x ${WORKLOAD_DIR}/sdc_check.sh

        # Generate golden output
        echo "Generating golden output for ${TECH}..."
        cd ${WORKLOAD_DIR}
        export PRELOAD_FLAG=""
        export BIN_DIR=$(pwd)
        export APP_DIR=$(pwd)
        bash run.sh
        mv stdout.txt golden_stdout.txt 2>/dev/null || true
        mv stderr.txt golden_stderr.txt 2>/dev/null || true
        echo "✓ Golden output created for ${TECH}"
    done

    # Update NVBitFI params.py
    echo "Updating NVBitFI params.py..."
    PARAMS_FILE="${NVBITFI_ROOT}/scripts/params.py"
    cp ${PARAMS_FILE} ${PARAMS_FILE}.backup

    # Check if already updated
    if grep -q "yolov8_notechnique" ${PARAMS_FILE}; then
        echo "✓ params.py already updated"
    else
        # Add workloads to apps dictionary
        python3 << 'PYEOF'
import sys
params_file = "/home/shrec/NischalNVBitFi/nvbit_release_aarch64/nvbitfi/scripts/params.py"

with open(params_file, 'r') as f:
    content = f.read()

# Add yolov8 workloads before the closing brace of apps dict
new_apps = """        'yolov8_notechnique': [
                        NVBITFI_HOME + '/test-apps/yolov8_notechnique',
                        'run.sh',
                        NVBITFI_HOME + '/test-apps/yolov8_notechnique/',
                        5,
                        ""
                ],
        'yolov8_tmr': [
                        NVBITFI_HOME + '/test-apps/yolov8_tmr',
                        'run.sh',
                        NVBITFI_HOME + '/test-apps/yolov8_tmr/',
                        15,
                        ""
                ],"""

content = content.replace(
    "                ],\n}",
    "                ],\n" + new_apps + "\n}"
)

# Update injection counts
content = content.replace("NUM_INJECTIONS = 1000", "NUM_INJECTIONS = 10000")
content = content.replace("THRESHOLD_JOBS = 25", "THRESHOLD_JOBS = 10000")

with open(params_file, 'w') as f:
    f.write(content)

print("✓ params.py updated")
PYEOF
    fi

    echo ""
    echo "=========================================="
    echo "Setup Complete!"
    echo "=========================================="
    echo ""
    echo "Run fault injection with:"
    echo "  ./setup_and_run_nvbitfi.sh run notechnique"
    echo "  ./setup_and_run_nvbitfi.sh run tmr"
}

# ============================================================================
# RUN FUNCTION - Run 10K fault injection campaign
# ============================================================================
run_campaign() {
    TECHNIQUE=$1

    if [ -z "$TECHNIQUE" ]; then
        echo "Error: Please specify technique (notechnique or tmr)"
        echo "Usage: ./setup_and_run_nvbitfi.sh run <technique>"
        exit 1
    fi

    WORKLOAD_NAME="yolov8_${TECHNIQUE}"

    echo "=========================================="
    echo "Running NVBitFI Campaign: ${WORKLOAD_NAME}"
    echo "=========================================="

    cd ${NVBITFI_ROOT}/scripts

    # Step 1: Profile
    echo ""
    echo "Step 1/4: Profiling..."
    python3 run_profiler.py -a ${WORKLOAD_NAME}

    # Step 2: Generate injection list
    echo ""
    echo "Step 2/4: Generating injection list..."
    python3 generate_injection_list.py -a ${WORKLOAD_NAME}

    # Step 3: Run injections
    echo ""
    echo "Step 3/4: Running 10,000 fault injections (THIS WILL TAKE A LONG TIME)..."
    sudo -E python3 run_injections.py standalone -a ${WORKLOAD_NAME}

    # Step 4: Parse results
    echo ""
    echo "Step 4/4: Parsing results..."
    python3 parse_results.py -a ${WORKLOAD_NAME}

    echo ""
    echo "=========================================="
    echo "Campaign Complete!"
    echo "=========================================="
    echo "Results: ${NVBITFI_ROOT}/logs/${WORKLOAD_NAME}/"
}

# ============================================================================
# MAIN
# ============================================================================
case "$1" in
    setup)
        setup
        ;;
    run)
        run_campaign $2
        ;;
    *)
        echo "Usage:"
        echo "  $0 setup              # First time setup"
        echo "  $0 run <technique>    # Run fault injection"
        echo ""
        echo "Example:"
        echo "  $0 setup"
        echo "  $0 run notechnique"
        echo "  $0 run tmr"
        exit 1
        ;;
esac
