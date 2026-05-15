@echo off
echo === RTD Integration Compilation ===

REM Create build directories
if not exist build mkdir build
if not exist bin mkdir bin

echo Compiling CUDA source files...

REM Compile CUDA source files
nvcc -std=c++14 -O2 -I./include -c src/ray_tracing.cu -o build/ray_tracing.o
if %errorlevel% neq 0 (
    echo Error compiling ray_tracing.cu
    exit /b 1
)

nvcc -std=c++14 -O2 -I./include -c src/idd_sigma_calculation.cu -o build/idd_sigma_calculation.o
if %errorlevel% neq 0 (
    echo Error compiling idd_sigma_calculation.cu
    exit /b 1
)

nvcc -std=c++14 -O2 -I./include -c src/superposition_kernels.cu -o build/superposition_kernels.o
if %errorlevel% neq 0 (
    echo Error compiling superposition_kernels.cu
    exit /b 1
)

nvcc -std=c++14 -O2 -I./include -c src/utils.cu -o build/utils.o
if %errorlevel% neq 0 (
    echo Error compiling utils.cu
    exit /b 1
)

nvcc -std=c++14 -O2 -I./include -c src/raytracedicom_wrapper.cu -o build/raytracedicom_wrapper.o
if %errorlevel% neq 0 (
    echo Error compiling raytracedicom_wrapper.cu
    exit /b 1
)

echo Compiling C++ source files...

REM Compile C++ source files
g++ -std=c++14 -O2 -I./include -c src/test_raytracedicom.cpp -o build/test_raytracedicom.o
if %errorlevel% neq 0 (
    echo Error compiling test_raytracedicom.cpp
    exit /b 1
)

echo Linking executable...

REM Link executable
nvcc build/*.o -o bin/test_raytracedicom -lcudart
if %errorlevel% neq 0 (
    echo Error linking executable
    exit /b 1
)

echo Compilation successful!
echo Executable created: bin/test_raytracedicom.exe

echo.
echo To run the program:
echo   bin\test_raytracedicom.exe
echo.
echo To clean build files:
echo   del /s build\*.o
echo   del bin\test_raytracedicom.exe
