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
# CONFIGURATION - ADJUST FOR YOUR XAVIER NX SETUP
# ============================================================================
# Xavier NX paths (using ~ for home directory on Xavier: shrenx@ubuntu)
PROJECT_ROOT="$HOME/Nischal_NVBitFi/RemoteObjectDetectionWithFaultTolerantTechniques"
NVBITFI_ROOT="$HOME/Nischal_NVBitFi/nvbit_release/tools/nvbitfi"
MODEL_PATH="${PROJECT_ROOT}/Plane_Ship_Detection/Plane_Ship_Model.pt"
DATASET_LIST="${PROJECT_ROOT}/validation_dataset_list.txt"

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
# Set NVBITFI_HOME if not already set (required by NVBitFI)
if [ -z "$NVBITFI_HOME" ]; then
    export NVBITFI_HOME="${NVBITFI_ROOT}"
    echo "Setting NVBITFI_HOME=${NVBITFI_HOME}"
fi

# Verify NVBITFI_HOME is set correctly
if [ ! -d "$NVBITFI_HOME" ]; then
    echo "Error: NVBITFI_HOME directory does not exist: $NVBITFI_HOME"
    echo "Please check your paths and try again."
    exit 1
fi

# ============================================================================
# SETUP FUNCTION - Run this once
# ============================================================================
setup() {
    echo "=========================================="
    echo "NVBitFI Setup"
    echo "=========================================="

    cd ${PROJECT_ROOT} #go to project root

    # Create workload directories for each technique
    for TECH in notechnique tmr; do #TODO: DO NOT REMOVE: WIll need to add more techniques later
        echo "Setting up ${TECH}..."
        WORKLOAD_DIR="${NVBITFI_ROOT}/test-apps/${TECH}"
        mkdir -p ${WORKLOAD_DIR}

        # Create output directory for this technique's fault results in project folder
        TECHNIQUE_OUTPUT_DIR="${PROJECT_ROOT}/output/${TECH}_fault/results"
        mkdir -p ${TECHNIQUE_OUTPUT_DIR}

        # Create run.sh with injection counter
        cat > ${WORKLOAD_DIR}/run.sh <<EOF
#!/bin/bash
# NVBitFI doesn't set RUN_ID automatically, so we use a counter file
COUNTER_FILE="${TECHNIQUE_OUTPUT_DIR}/.run_counter"
if [ -f "\${COUNTER_FILE}" ]; then
    RUN_ID=\$(cat "\${COUNTER_FILE}")
else
    RUN_ID=0
fi
# Increment for next run
echo \$((RUN_ID + 1)) > "\${COUNTER_FILE}"

OUTPUT_FILE="${TECHNIQUE_OUTPUT_DIR}/result_\${RUN_ID}.json"

eval \${PRELOAD_FLAG} python3 ${PROJECT_ROOT}/run_single_inference.py \\
    --technique ${TECH} \\
    --model ${MODEL_PATH} \\
    --dataset-list ${DATASET_LIST} \\
    --random-image \\
    --output "\${OUTPUT_FILE}" > stdout.txt 2> stderr.txt
EOF
        chmod +x ${WORKLOAD_DIR}/run.sh

        # Create minimal sdc_check.sh
        # Only checks if program crashed - real SDC analysis done post-processing
        cat > ${WORKLOAD_DIR}/sdc_check.sh <<'EOF'
#!/bin/bash
# Minimal check - NVBitFI requires these files
touch diff.log
touch special_check.log
# NVBitFI will detect crashes via exit codes
# Real SDC classification happens later with aggregate_fault_results.py
EOF
        chmod +x ${WORKLOAD_DIR}/sdc_check.sh

        # Generate golden output (for NVBitFI's internal checks)
        echo "Generating golden output for ${TECH}..."
        cd ${WORKLOAD_DIR}
        export PRELOAD_FLAG=""
        export BIN_DIR=$(pwd)
        export APP_DIR=$(pwd)
        bash run.sh
        mv stdout.txt golden_stdout.txt 2>/dev/null || true
        mv stderr.txt golden_stderr.txt 2>/dev/null || true
        echo "Golden output created for ${TECH}"

        # Reset counter for actual fault injection runs
        rm -f "${TECHNIQUE_OUTPUT_DIR}/.run_counter"
        echo "0" > "${TECHNIQUE_OUTPUT_DIR}/.run_counter"
    done

    echo ""
    echo "=========================================="
    echo "IMPORTANT: Generate Golden Predictions"
    echo "=========================================="
    echo "Before running fault injection, generate golden predictions by running:"
    echo "  python3 run_evaluations.py"
    echo ""
    echo "Make sure to configure run_evaluations.py with:"
    echo "  - RUN_TYPE = 'nofault'"
    echo "  - WORKER_SCRIPT_NAME for each technique"
    echo ""
    echo "Golden predictions will be saved to:"
    echo "  output/notechnique_nofault/golden_predictions/"
    echo "  output/tmr_nofault/golden_predictions/"

    # Update NVBitFI params.py
    echo "Updating NVBitFI params.py..."
    PARAMS_FILE="${NVBITFI_ROOT}/scripts/params.py"
    cp ${PARAMS_FILE} ${PARAMS_FILE}.backup

#TODO: DO NOT REMOVE: will need to add other techniques info later down in new_apps

    # Check if already updated
    if grep -q "notechnique" ${PARAMS_FILE}; then
        echo "params.py already updated"
    else
        # Add workloads to apps dictionary
        python3 << PYEOF
import sys
import os
params_file = os.path.expanduser("~/Nischal_NVBitFi/nvbit_release/tools/nvbitfi/scripts/params.py")

with open(params_file, 'r') as f:
    content = f.read()

# Add model workloads before the closing brace of apps dict
new_apps = """        'notechnique': [
                        NVBITFI_HOME + '/test-apps/notechnique',
                        'run.sh',
                        NVBITFI_HOME + '/test-apps/notechnique/',
                        5,
                        ""
                ],
        'tmr': [
                        NVBITFI_HOME + '/test-apps/tmr',
                        'run.sh',
                        NVBITFI_HOME + '/test-apps/tmr/',
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

print("params.py updated")
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
        echo "Error: Please specify technique (notechnique or tmr)" #TODO
        echo "Usage: ./setup_and_run_nvbitfi.sh run <technique>"
        exit 1
    fi

    WORKLOAD_NAME="${TECHNIQUE}"

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
       echo "NVBitFI logs: ${NVBITFI_ROOT}/logs/${WORKLOAD_NAME}/"
    echo "Result JSONs: ${PROJECT_ROOT}/output/${TECHNIQUE}_fault/results/"
    echo ""
    echo "To analyze fault outcomes, run:"
    echo "  python3 aggregate_fault_results.py \\"
    echo "    --technique ${TECHNIQUE} \\"
    echo "    --results-dir output/${TECHNIQUE}_fault/results \\"
    echo "    --golden-dir output/${TECHNIQUE}_nofault/golden_predictions \\"
    echo "    --output output/${TECHNIQUE}_fault/summary_report.json"
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
