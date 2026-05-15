#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

int main() {
    // 记录程序开始时间
    auto programStart = std::chrono::high_resolution_clock::now();
    
    std::cout << "=== CUDA Diagnostic Program ===" << std::endl;
    
    // 1. Test CUDA Runtime Version
    std::cout << "\n1. Testing CUDA Runtime Version..." << std::endl;
    int runtimeVersion = 0;
    cudaError_t error = cudaRuntimeGetVersion(&runtimeVersion);
    if (error == cudaSuccess) {
        int major = runtimeVersion / 1000;
        int minor = (runtimeVersion % 1000) / 10;
        std::cout << "CUDA Runtime Version: " << major << "." << minor << std::endl;
    } else {
        std::cout << "cudaRuntimeGetVersion failed: " << cudaGetErrorString(error) << std::endl;
    }
    
    // 2. Test CUDA Driver Version
    std::cout << "\n2. Testing CUDA Driver Version..." << std::endl;
    int driverVersion = 0;
    error = cudaDriverGetVersion(&driverVersion);
    if (error == cudaSuccess) {
        int major = driverVersion / 1000;
        int minor = (driverVersion % 1000) / 10;
        std::cout << "CUDA Driver Version: " << major << "." << minor << std::endl;
    } else {
        std::cout << "cudaDriverGetVersion failed: " << cudaGetErrorString(error) << std::endl;
    }
    
    // 3. Test Device Count
    std::cout << "\n3. Testing Device Count..." << std::endl;
    int deviceCount = 0;
    error = cudaGetDeviceCount(&deviceCount);
    if (error == cudaSuccess) {
        std::cout << "CUDA Device Count: " << deviceCount << std::endl;
    } else {
        std::cout << "cudaGetDeviceCount failed: " << cudaGetErrorString(error) << " (code: " << error << ")" << std::endl;
        
        // Try to get more information about the error
        const char* errorName = cudaGetErrorName(error);
        const char* errorString = cudaGetErrorString(error);
        std::cout << "Error name: " << errorName << std::endl;
        std::cout << "Error string: " << errorString << std::endl;
        
        return 1;
    }
    
    if (deviceCount == 0) {
        std::cout << "No CUDA devices found!" << std::endl;
        return 1;
    }
    
    // 4. Test Device Properties
    std::cout << "\n4. Testing Device Properties..." << std::endl;
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        error = cudaGetDeviceProperties(&deviceProp, i);
        if (error == cudaSuccess) {
            std::cout << "Device " << i << ": " << deviceProp.name << std::endl;
            std::cout << "  Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
            std::cout << "  Total Memory: " << deviceProp.totalGlobalMem / (1024*1024) << " MB" << std::endl;
            std::cout << "  Multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
        } else {
            std::cout << "cudaGetDeviceProperties failed for device " << i << ": " << cudaGetErrorString(error) << std::endl;
        }
    }
    
    // 5. Test Device Selection
    std::cout << "\n5. Testing Device Selection..." << std::endl;
    error = cudaSetDevice(0);
    if (error == cudaSuccess) {
        std::cout << "cudaSetDevice(0) successful" << std::endl;
        
        int currentDevice = -1;
        error = cudaGetDevice(&currentDevice);
        if (error == cudaSuccess) {
            std::cout << "Current device: " << currentDevice << std::endl;
        } else {
            std::cout << "cudaGetDevice failed: " << cudaGetErrorString(error) << std::endl;
        }
    } else {
        std::cout << "cudaSetDevice(0) failed: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }
    
    // 6. Test Memory Allocation
    std::cout << "\n6. Testing Memory Allocation..." << std::endl;
    float* d_test = nullptr;
    error = cudaMalloc(&d_test, 1024 * sizeof(float));
    if (error == cudaSuccess) {
        std::cout << "cudaMalloc successful" << std::endl;
        
        // Test memory copy
        std::vector<float> h_data(1024, 1.0f);
        error = cudaMemcpy(d_test, h_data.data(), 1024 * sizeof(float), cudaMemcpyHostToDevice);
        if (error == cudaSuccess) {
            std::cout << "Host to device copy successful" << std::endl;
            
            // Test device to host copy
            std::vector<float> h_result(1024);
            error = cudaMemcpy(h_result.data(), d_test, 1024 * sizeof(float), cudaMemcpyDeviceToHost);
            if (error == cudaSuccess) {
                std::cout << "Device to host copy successful" << std::endl;
                
                // Verify data
                bool correct = true;
                for (int i = 0; i < 1024; i++) {
                    if (h_result[i] != 1.0f) {
                        correct = false;
                        break;
                    }
                }
                if (correct) {
                    std::cout << "Memory copy verification: SUCCESS" << std::endl;
                } else {
                    std::cout << "Memory copy verification: FAILED" << std::endl;
                }
            } else {
                std::cout << "Device to host copy failed: " << cudaGetErrorString(error) << std::endl;
            }
        } else {
            std::cout << "Host to device copy failed: " << cudaGetErrorString(error) << std::endl;
        }
        
        cudaFree(d_test);
    } else {
        std::cout << "cudaMalloc failed: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }
    
    // 7. Test Synchronization
    std::cout << "\n7. Testing Synchronization..." << std::endl;
    error = cudaDeviceSynchronize();
    if (error == cudaSuccess) {
        std::cout << "cudaDeviceSynchronize successful" << std::endl;
    } else {
        std::cout << "cudaDeviceSynchronize failed: " << cudaGetErrorString(error) << std::endl;
    }
    
    // 计算总耗时
    auto programEnd = std::chrono::high_resolution_clock::now();
    auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(programEnd - programStart);
    
    std::cout << "\n=== CUDA Diagnostic Complete ===" << std::endl;
    std::cout << "[TIMING] Total diagnostic execution time: " << totalDuration.count() << " ms" << std::endl;
    return 0;
}

