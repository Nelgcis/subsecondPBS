@echo off
echo === RTD Integration Quick Test ===

REM Check if CUDA is available
nvcc --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: CUDA not found. Please install CUDA Toolkit.
    echo Download from: https://developer.nvidia.com/cuda-downloads
    exit /b 1
)

REM Check if GPU is available
nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: NVIDIA GPU not found or driver not installed.
    exit /b 1
)

REM Display GPU information
echo GPU Information:
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits

REM Check if we're in the right directory
if not exist "compile_all.bat" (
    echo Error: compile_all.bat not found. Please run this script from the raytracedicom_integration directory.
    exit /b 1
)

REM Compile
echo Compiling RTD Integration...
call compile_all.bat

if %errorlevel% equ 0 (
    echo Compilation successful!
    
    REM Run test
    echo Running RTD test...
    bin\test_raytracedicom.exe
    
    if %errorlevel% equ 0 (
        echo.
        echo ✅ Test completed successfully!
        echo RTD Integration is working correctly.
    ) else (
        echo.
        echo ❌ Test failed!
        echo Please check the error messages above.
        exit /b 1
    )
) else (
    echo.
    echo ❌ Compilation failed!
    echo Please check the error messages above.
    echo.
    echo Possible solutions:
    echo 1. Install Visual Studio Build Tools
    echo 2. Use WSL: wsl --install
    echo 3. Use Docker with NVIDIA support
    exit /b 1
)

echo.
echo === Test Complete ===
