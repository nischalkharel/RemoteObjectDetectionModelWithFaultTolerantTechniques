#!/bin/bash
################################################################################
# Linux Setup Script for Object Detection Project
# For standard x86_64 Linux systems
# For NVIDIA Orin Nano, see orin_nano_setup.txt
################################################################################

echo ""
echo "========================================"
echo "Object Detection Project Setup (Linux)"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Virtual environment already exists."
    echo "To recreate it, delete the 'venv' folder first."
    echo ""
else
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create virtual environment."
        echo "Make sure Python 3.8+ is installed."
        echo "Try: sudo apt-get install python3 python3-venv python3-pip"
        exit 1
    fi
    echo "Virtual environment created successfully."
    echo ""
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo ""
echo "Installing dependencies..."
echo "This may take several minutes..."
echo ""

# Upgrade pip first
python3 -m pip install --upgrade pip

# Install PyTorch with CUDA support (for NVIDIA GPUs)
echo "Installing PyTorch with CUDA support..."
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
echo "Installing other dependencies..."
pip3 install numpy

echo ""
echo "Verifying PyTorch installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "To activate the virtual environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run the project:"
echo "  python3 compare_results.py"
echo "  or"
echo "  python3 main.py"
echo ""
echo "To save current package versions:"
echo "  pip3 freeze > requirements_frozen.txt"
echo ""
