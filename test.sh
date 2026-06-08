#!/bin/bash
# Simple test script for RTD Integration

echo "=== RTD Integration Simple Test ==="

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ]; then
    echo "Error: CMakeLists.txt not found. Please run this script from the raytracedicom_integration directory."
    exit 1
fi

# Create build directory
echo "Creating build directory..."
mkdir -p build
cd build

# Configure CMake
echo "Configuring CMake..."
cmake -DCMAKE_BUILD_TYPE=Release ..
if [ $? -ne 0 ]; then
    echo "Error: CMake configuration failed!"
    exit 1
fi

# Compile
echo "Compiling..."
make -j$(nproc)
if [ $? -ne 0 ]; then
    echo "Error: Compilation failed!"
    exit 1
fi

echo "✅ Compilation successful!"
echo "You can now run: ./bin/test_raytracedicom"

