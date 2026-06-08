#!/bin/bash
# RayTraceDicom集成项目 - 快速调试脚本

echo "RayTraceDicom Integration - Quick Debug Script"
echo "============================================="

# 检查CUDA环境
echo "Checking CUDA environment..."
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: CUDA not found! Please install CUDA toolkit."
    exit 1
fi
nvcc --version

# 创建调试构建目录
echo "Creating debug build directory..."
mkdir -p build_debug
cd build_debug

# 配置CMake
echo "Configuring CMake for debug build..."
cmake -DCMAKE_BUILD_TYPE=Debug ..
if [ $? -ne 0 ]; then
    echo "ERROR: CMake configuration failed!"
    exit 1
fi

# 编译项目
echo "Building project..."
make -j$(nproc)
if [ $? -ne 0 ]; then
    echo "ERROR: Build failed!"
    exit 1
fi

echo "Build completed successfully!"
echo

# 运行调试测试
echo "Running debug tests..."
echo "====================="
./bin/debug_test
if [ $? -ne 0 ]; then
    echo "ERROR: Debug test failed!"
    exit 1
fi

echo
echo "Debug tests completed successfully!"
echo

# 运行主测试
echo "Running main test..."
echo "==================="
./bin/test_raytracedicom
if [ $? -ne 0 ]; then
    echo "ERROR: Main test failed!"
    exit 1
fi

echo
echo "All tests completed successfully!"
echo

# 提供调试选项
echo "Debug Options:"
echo "1. Run with cuda-memcheck"
echo "2. Run with GDB"
echo "3. Run with CUDA-GDB"
echo "4. Exit"
echo
read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo "Running with cuda-memcheck..."
        cuda-memcheck ./bin/test_raytracedicom
        ;;
    2)
        echo "Running with GDB..."
        gdb ./bin/test_raytracedicom
        ;;
    3)
        echo "Running with CUDA-GDB..."
        cuda-gdb ./bin/test_raytracedicom
        ;;
    4)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice!"
        ;;
esac


