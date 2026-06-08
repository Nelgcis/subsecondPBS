#!/bin/bash
# RTD Integration Quick Test for Linux/Mac

echo "=== RTD Integration Quick Test ==="

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo "Error: CUDA not found. Please install CUDA Toolkit."
    echo "Download from: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

# Check if GPU is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: NVIDIA GPU not found or driver not installed."
    exit 1
fi

# Display GPU information
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ]; then
    echo "Error: CMakeLists.txt not found. Please run this script from the raytracedicom_integration directory."
    exit 1
fi

# Create build directory if it doesn't exist
if [ ! -d "build" ]; then
    echo "Creating build directory..."
    mkdir build
fi

cd build

# Configure CMake
echo "Configuring CMake..."
cmake ..
if [ $? -ne 0 ]; then
    echo "Error: CMake configuration failed!"
    exit 1
fi

# Compile
echo "Compiling RTD Integration..."
make -j$(nproc)
if [ $? -ne 0 ]; then
    echo "Error: Compilation failed!"
    exit 1
fi

echo "Compilation successful!"

# Run test
echo "Running RTD test..."
./bin/test_raytracedicom

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Test completed successfully!"
    echo "RTD Integration is working correctly."
else
    echo ""
    echo "❌ Test failed!"
    echo "Please check the error messages above."
    exit 1
fi

echo ""
echo "=== Test Complete ==="