#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "../include/core/common.cuh"
#include "../include/utils/utils.h"

int main() {
    std::cout << "=== Subspot Data Debug ===" << std::endl;
    
    // text
    int numLayers = 3;
    int maxSubspotsPerLayer = 2;
    std::vector<float> subspotData(numLayers * maxSubspotsPerLayer * 5);
    
    for (int layer = 0; layer < numLayers; layer++) {
        for (int subspot = 0; subspot < maxSubspotsPerLayer; subspot++) {
            int baseIdx = (layer * maxSubspotsPerLayer + subspot) * 5;
            subspotData[baseIdx + 0] = 0.0f;  // deltaX
            subspotData[baseIdx + 1] = 0.0f;  // deltaY
            subspotData[baseIdx + 2] = 10.0f; // weight
            subspotData[baseIdx + 3] = 1.0f;  // sigmaX
            subspotData[baseIdx + 4] = 1.0f;  // sigmaY
        }
    }
    
    std::cout << "Subspot data (first 3 subspots, layer 0):" << std::endl;
    for (int subspot = 0; subspot < 3; subspot++) {
        int baseIdx = subspot * 5;
        std::cout << "Subspot " << subspot << ": ";
        std::cout << "deltaX=" << subspotData[baseIdx + 0] << ", ";
        std::cout << "deltaY=" << subspotData[baseIdx + 1] << ", ";
        std::cout << "weight=" << subspotData[baseIdx + 2] << ", ";
        std::cout << "sigmaX=" << subspotData[baseIdx + 3] << ", ";
        std::cout << "sigmaY=" << subspotData[baseIdx + 4] << std::endl;
    }
    
    // create texture
    cudaTextureObject_t subspotTex = create3DTexture(
        subspotData.data(), 
        make_int3(5, maxSubspotsPerLayer, numLayers),
        cudaFilterModeLinear, 
        cudaAddressModeClamp
    );
    
    std::cout << "Texture created successfully" << std::endl;
    
    // 清理
    destroyTextureObjectAndArray(subspotTex);
    
    return 0;
}
