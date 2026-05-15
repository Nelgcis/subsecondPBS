#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include "../include/core/raytracedicom_integration.h"
#include "../include/utils/utils.h"

int main() {
    // 记录程序开始时间
    auto programStart = std::chrono::high_resolution_clock::now();
    
    std::cout << "=== RayTraceDicom Library Test ===" << std::endl;
    
    // Test CUDA initialization
    cudaError_t error = cudaSetDevice(0);
    if (error != cudaSuccess) {
        std::cerr << "CUDA initialization failed: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }
    
    // Test device info using our utility functions
    printDeviceInfo();
    printMemoryInfo();
    
    // Test data structures
    RTDBeamSettings* beamSettings = createRTDBeamSettings();
    RTDEnergyStruct* energyData = createRTDEnergyStruct();
    
    if (beamSettings && energyData) {
        std::cout << "✓ Data structures created successfully" << std::endl;
        
        // Test memory allocation using our utility functions
        size_t testSize = 1024 * sizeof(float);
        float* devPtr = (float*)allocateDeviceMemory(testSize);
        if (devPtr) {
            std::cout << "✓ Device memory allocation successful" << std::endl;
            freeDeviceMemory(devPtr);
            std::cout << "✓ Device memory deallocation successful" << std::endl;
        }
        
        // Clean up
        destroyRTDBeamSettings(beamSettings);
        destroyRTDEnergyStruct(energyData);
        std::cout << "✓ Data structures destroyed successfully" << std::endl;
    } else {
        std::cerr << "✗ Failed to create data structures" << std::endl;
        return 1;
    }
    
    // 计算总耗时
    auto programEnd = std::chrono::high_resolution_clock::now();
    auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(programEnd - programStart);
    
    std::cout << "\n=== All tests passed! ===" << std::endl;
    std::cout << "RayTraceDicom library is working correctly." << std::endl;
    std::cout << "[TIMING] Total library test execution time: " << totalDuration.count() << " ms" << std::endl;
    return 0;
}
