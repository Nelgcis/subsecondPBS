#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "../include/core/raytracedicom_integration.h"

int main() {
    std::cout << "=== RayTraceDicom Integration Test ===" << std::endl;
    
    // Test CUDA initialization
    cudaError_t error = cudaSetDevice(0);
    if (error != cudaSuccess) {
        std::cerr << "CUDA initialization failed: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }
    
    // Test device info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "CUDA Device: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Global Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    
    // Test data structures
    RTDBeamSettings* beamSettings = createRTDBeamSettings();
    RTDEnergyStruct* energyData = createRTDEnergyStruct();
    
    if (beamSettings && energyData) {
        std::cout << "Data structures created successfully" << std::endl;
        
        // Clean up
        destroyRTDBeamSettings(beamSettings);
        destroyRTDEnergyStruct(energyData);
        std::cout << "Data structures destroyed successfully" << std::endl;
    } else {
        std::cerr << "Failed to create data structures" << std::endl;
        return 1;
    }
    
    std::cout << "=== Test completed successfully ===" << std::endl;
    return 0;
}
