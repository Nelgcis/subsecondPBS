#!/bin/bash

# RayTraceDicom Debug Runner Script
# This script sets up the environment and runs the debug program

# Set library paths
export LD_LIBRARY_PATH="/root/raytracedicom_updated/bin:/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"

# Set CUDA paths
export CUDA_HOME="/usr/local/cuda"
export PATH="/usr/local/cuda/bin:$PATH"

# Run the debug program
echo "=== RayTraceDicom Debug Runner ==="
echo "Environment setup:"
echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "  CUDA_HOME: $CUDA_HOME"
echo "  PATH: $PATH"
echo ""

# Check if debug program exists
if [ ! -f "bin/debug_raytracedicom" ]; then
    echo "Error: debug_raytracedicom not found. Please run 'make debug' first."
    exit 1
fi

# Check if shared library exists
if [ ! -f "bin/libraytracedicom.so" ]; then
    echo "Error: libraytracedicom.so not found. Please run 'make debug' first."
    exit 1
fi

echo "Running debug program..."
echo ""

# Run the debug program
./bin/debug_raytracedicom

echo ""
echo "=== Debug Runner Complete ==="
