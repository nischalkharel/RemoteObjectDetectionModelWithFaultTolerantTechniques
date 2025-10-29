# Object Detection Project - Setup Guide

This project is designed to run on both development machines and NVIDIA Orin Nano devices. All dependencies are tracked and documented for easy deployment.

## Quick Start (Current System)

### Windows
```bash
# Run the automated setup script
setup_windows.bat

# Or manually:
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Linux (Standard x86_64)
```bash
# Make the script executable (first time only)
chmod +x setup_linux.sh

# Run the automated setup script
./setup_linux.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

## Running the Project

Once setup is complete:

```bash
# Activate virtual environment (if not already active)
# Windows:
venv\Scripts\activate
# Linux:
source venv/bin/activate

# Run the main scripts
python compare_results.py
# or
python main.py
```

## Project Files

### Core Python Scripts
- `compare_results.py` - Average Precision (AP) calculation for object detection
- `main.py` - Main evaluation loop for processing images
- `image_list_maker.py` - Utility for managing image lists

### Configuration Files
- `requirements.txt` - Python package dependencies (high-level)
- `requirements_frozen.txt` - Exact versions of installed packages (auto-generated)
- `validation_dataset_list.txt` - List of validation images and labels

### Setup Scripts
- `setup_windows.bat` - Automated setup for Windows
- `setup_linux.sh` - Automated setup for Linux x86_64
- `orin_nano_setup.txt` - Comprehensive guide for NVIDIA Orin Nano deployment

### Model and Data
- `Plane_Ship_Detection/Plane_Ship_Model.pt` - YOLOv8 model for plane/ship detection

## Dependencies

### Core Requirements
- Python 3.8 or later
- PyTorch 2.0+
- TorchVision 0.15+
- NumPy 1.24+

### Current Installation
All installed packages and their exact versions are tracked in `requirements_frozen.txt`.

To update this file after installing new packages:
```bash
pip freeze > requirements_frozen.txt
```

## Deploying to NVIDIA Orin Nano

The NVIDIA Orin Nano requires special setup due to its ARM architecture and CUDA support. See `orin_nano_setup.txt` for detailed instructions.

### Key Points for Orin Nano:
1. Use NVIDIA's official PyTorch wheels for Jetson devices
2. Match PyTorch version to your JetPack version
3. Cannot directly copy venv from x86_64 to ARM - must reinstall
4. CUDA support is critical for performance

Quick deployment steps:
```bash
# 1. Copy project to Orin Nano
scp -r RemoteObjectDetectionModelWithFaultTolerantTechniques/ user@orin-nano-ip:~/

# 2. SSH into Orin Nano
ssh user@orin-nano-ip

# 3. Follow instructions in orin_nano_setup.txt
cd ~/RemoteObjectDetectionModelWithFaultTolerantTechniques
cat orin_nano_setup.txt
```

## Keeping Dependencies Updated

This project automatically tracks all dependencies:

1. **requirements.txt** - Lists main dependencies with minimum versions
2. **requirements_frozen.txt** - Lists exact versions currently installed

### Adding New Dependencies

When you install a new package:
```bash
# 1. Install the package
pip install package-name

# 2. Add it to requirements.txt
echo "package-name>=version" >> requirements.txt

# 3. Update frozen requirements
pip freeze > requirements_frozen.txt

# 4. Commit both files to git
git add requirements.txt requirements_frozen.txt
git commit -m "Add package-name dependency"
```

## Portability

### Same Architecture
If moving between machines with the same architecture (e.g., Windows to Windows, or Linux x86_64 to Linux x86_64):

```bash
# On target machine:
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements_frozen.txt
```

### Different Architecture
When moving to different architecture (e.g., Windows/Linux x86_64 to ARM-based Orin Nano):

- Follow platform-specific instructions in `orin_nano_setup.txt`
- Cannot reuse the same virtual environment
- Must install architecture-specific packages

## Troubleshooting

### Virtual Environment Issues
```bash
# Delete and recreate if corrupted
rm -rf venv  # or rmdir /s venv on Windows
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### CUDA Not Available
The CPU-only version of PyTorch was installed by default. For GPU support:

**Windows/Linux x86_64:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**NVIDIA Orin Nano:**
See `orin_nano_setup.txt` for Jetson-specific PyTorch wheels.

### Import Errors
Make sure virtual environment is activated:
```bash
# Check which Python is being used
which python  # Linux
where python  # Windows

# Should show path inside venv folder
# If not, activate the virtual environment
```

## Performance Optimization

### On NVIDIA Orin Nano
- Run `sudo jetson_clocks` for maximum performance
- Monitor with `sudo tegrastats`
- See `orin_nano_setup.txt` for detailed optimization tips

### Memory Management
- Reduce batch sizes if encountering OOM errors
- Monitor GPU memory usage
- Consider using mixed precision training/inference

## Project Structure

```
RemoteObjectDetectionModelWithFaultTolerantTechniques/
├── venv/                          # Virtual environment (not in git)
├── Plane_Ship_Detection/
│   └── Plane_Ship_Model.pt       # Trained model
├── compare_results.py             # AP calculation script
├── main.py                        # Main evaluation script
├── image_list_maker.py            # Utility script
├── requirements.txt               # Dependency list
├── requirements_frozen.txt        # Exact versions (auto-generated)
├── setup_windows.bat              # Windows setup script
├── setup_linux.sh                 # Linux setup script
├── orin_nano_setup.txt           # Orin Nano deployment guide
├── README_SETUP.md               # This file
└── validation_dataset_list.txt   # Image list for validation
```

## Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [NVIDIA Jetson PyTorch](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)
- [Python Virtual Environments](https://docs.python.org/3/tutorial/venv.html)

## Support

For issues specific to:
- **NVIDIA Orin Nano**: See `orin_nano_setup.txt`
- **PyTorch**: Check [PyTorch Forums](https://discuss.pytorch.org/)
- **Project-specific**: Check git history and commit messages
