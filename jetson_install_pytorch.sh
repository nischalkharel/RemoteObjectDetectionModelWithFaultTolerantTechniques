#!/bin/bash
###############################################################################
# PyTorch Installation Script for NVIDIA Jetson Nano
###############################################################################
#
# This script installs PyTorch and TorchVision using NVIDIA's pre-built wheels
# optimized for Jetson platforms.
#
# IMPORTANT: Do NOT use 'pip install torch' - it will install the wrong version!
#
# PREREQUISITES:
# - NVIDIA Jetson Nano with JetPack 4.6.x
# - Python 3.6+
# - System dependencies installed (run jetson_setup.sh first)
#
# USAGE:
#   chmod +x jetson_install_pytorch.sh
#   ./jetson_install_pytorch.sh
#
###############################################################################

set -e  # Exit on error

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}PyTorch Installation for Jetson Nano${NC}"
echo -e "${GREEN}========================================${NC}"

# Check Python version
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
print_status "Python version: $PYTHON_VERSION"

# Determine PyTorch wheel based on Python version
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" != "3" ]; then
    print_error "Python 3.x is required. Current version: $PYTHON_VERSION"
    exit 1
fi

# Create temp directory for downloads
TEMP_DIR="/tmp/jetson_pytorch_install"
mkdir -p $TEMP_DIR
cd $TEMP_DIR

print_status "Downloading PyTorch wheel for Jetson Nano (JetPack 4.6)..."

# PyTorch v1.10.0 for JetPack 4.6 (most stable for Jetson Nano)
if [ "$PYTHON_MINOR" == "6" ]; then
    # Python 3.6
    TORCH_WHEEL="torch-1.10.0-cp36-cp36m-linux_aarch64.whl"
    TORCH_URL="https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl"
elif [ "$PYTHON_MINOR" == "8" ]; then
    # Python 3.8
    print_warning "Python 3.8 detected. Using Python 3.6 wheel (should work)."
    TORCH_WHEEL="torch-1.10.0-cp36-cp36m-linux_aarch64.whl"
    TORCH_URL="https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl"
else
    print_warning "Python version $PYTHON_VERSION detected. Using Python 3.6 wheel."
    TORCH_WHEEL="torch-1.10.0-cp36-cp36m-linux_aarch64.whl"
    TORCH_URL="https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl"
fi

# Download PyTorch wheel
if [ -f "$TORCH_WHEEL" ]; then
    print_status "PyTorch wheel already downloaded."
else
    print_status "Downloading PyTorch wheel from NVIDIA..."
    wget $TORCH_URL -O $TORCH_WHEEL
fi

# Install PyTorch dependencies
print_status "Installing PyTorch dependencies..."
sudo apt-get install -y \
    libopenblas-base \
    libopenmpi-dev

# Install PyTorch
print_status "Installing PyTorch..."
pip3 install $TORCH_WHEEL

# Verify PyTorch installation
print_status "Verifying PyTorch installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')" || {
    print_error "PyTorch import failed!"
    exit 1
}

# Check CUDA availability
print_status "Checking CUDA availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" || {
    print_warning "Could not check CUDA availability"
}

# Install TorchVision dependencies
print_status "Installing TorchVision dependencies..."
sudo apt-get install -y \
    libjpeg-dev \
    zlib1g-dev \
    libpython3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev

# Build TorchVision from source (v0.11.0 compatible with PyTorch 1.10.0)
print_status "Building TorchVision from source (this may take 15-30 minutes)..."
cd $TEMP_DIR

if [ -d "torchvision" ]; then
    rm -rf torchvision
fi

git clone --branch v0.11.0 https://github.com/pytorch/vision torchvision
cd torchvision

# Set build version
export BUILD_VERSION=0.11.0

print_status "Compiling TorchVision... Please be patient."
print_warning "This process may use significant RAM. If it fails, increase swap space."

# Build and install
python3 setup.py install --user

# Verify TorchVision installation
print_status "Verifying TorchVision installation..."
python3 -c "import torchvision; print(f'TorchVision version: {torchvision.__version__}')" || {
    print_error "TorchVision import failed!"
    exit 1
}

# Clean up
print_status "Cleaning up temporary files..."
cd ~
rm -rf $TEMP_DIR

print_status "PyTorch and TorchVision installation complete!"
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Installation Summary${NC}"
echo -e "${GREEN}========================================${NC}"
python3 << END
import torch
import torchvision
print(f"PyTorch version: {torch.__version__}")
print(f"TorchVision version: {torchvision.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
END
echo ""
echo -e "${GREEN}Next step:${NC} Run jetson_install_packages.sh to install remaining Python packages"
echo ""
