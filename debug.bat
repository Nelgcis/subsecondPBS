@echo off
REM RayTraceDicom集成项目 - 快速调试脚本

echo RayTraceDicom Integration - Quick Debug Script
echo =============================================

REM 检查CUDA环境
echo Checking CUDA environment...
nvcc --version
if %errorlevel% neq 0 (
    echo ERROR: CUDA not found! Please install CUDA toolkit.
    pause
    exit /b 1
)

REM 创建调试构建目录
echo Creating debug build directory...
if not exist build_debug mkdir build_debug
cd build_debug

REM 配置CMake
echo Configuring CMake for debug build...
cmake -DCMAKE_BUILD_TYPE=Debug ..
if %errorlevel% neq 0 (
    echo ERROR: CMake configuration failed!
    pause
    exit /b 1
)

REM 编译项目
echo Building project...
cmake --build . --config Debug
if %errorlevel% neq 0 (
    echo ERROR: Build failed!
    pause
    exit /b 1
)

echo Build completed successfully!
echo.

REM 运行调试测试
echo Running debug tests...
echo =====================
bin\debug_test.exe
if %errorlevel% neq 0 (
    echo ERROR: Debug test failed!
    pause
    exit /b 1
)

echo.
echo Debug tests completed successfully!
echo.

REM 运行主测试
echo Running main test...
echo ===================
bin\test_raytracedicom.exe
if %errorlevel% neq 0 (
    echo ERROR: Main test failed!
    pause
    exit /b 1
)

echo.
echo All tests completed successfully!
echo.

REM 提供调试选项
echo Debug Options:
echo 1. Run with cuda-memcheck
echo 2. Run with GDB
echo 3. Run with CUDA-GDB
echo 4. Exit
echo.
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo Running with cuda-memcheck...
    cuda-memcheck bin\test_raytracedicom.exe
) else if "%choice%"=="2" (
    echo Running with GDB...
    gdb bin\test_raytracedicom.exe
) else if "%choice%"=="3" (
    echo Running with CUDA-GDB...
    cuda-gdb bin\test_raytracedicom.exe
) else if "%choice%"=="4" (
    echo Exiting...
    exit /b 0
) else (
    echo Invalid choice!
)

pause


