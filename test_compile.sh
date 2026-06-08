#!/bin/bash
# Simple compilation test script

echo "=== Testing RayTraceDicom Integration Compilation ==="

# Clean build directory
if [ -d "build" ]; then
    echo "Cleaning build directory..."
    rm -rf build
fi

# Create build directory
echo "Creating build directory..."
mkdir build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

if [ $? -ne 0 ]; then
    echo "❌ CMake configuration failed!"
    exit 1
fi

# Build the project
echo "Building project..."
make -j4

if [ $? -eq 0 ]; then
    echo "✅ Compilation successful!"
    echo "Build artifacts:"
    ls -la bin/ 2>/dev/null || echo "No bin directory found"
    exit 0
else
    echo "❌ Compilation failed!"
    exit 1
fi
