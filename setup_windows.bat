@echo off
REM ============================================================================
REM Windows Setup Script for Object Detection Project
REM ============================================================================

echo.
echo ========================================
echo Object Detection Project Setup (Windows)
echo ========================================
echo.

REM Check if virtual environment exists
if exist "venv\" (
    echo Virtual environment already exists.
    echo To recreate it, delete the 'venv' folder first.
    echo.
) else (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        echo Make sure Python 3.8+ is installed and in your PATH.
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.
    echo.
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Installing dependencies...
echo This may take several minutes...
echo.

REM Upgrade pip first
python -m pip install --upgrade pip

REM Install PyTorch with CUDA support (for NVIDIA GPUs)
echo Installing PyTorch with CUDA support...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

REM Install other requirements
echo Installing other dependencies...
pip install numpy

echo.
echo ========================================
echo Setup complete!
echo ========================================
echo.
echo To activate the virtual environment:
echo   venv\Scripts\activate
echo.
echo To run the project:
echo   python compare_results.py
echo   or
echo   python main.py
echo.
echo To save current package versions:
echo   pip freeze ^> requirements_frozen.txt
echo.
pause
