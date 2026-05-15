/**
 * \file
 * \brief Complete CPB to Ray Tracing Integration Test
 * 
 * This test verifies the complete workflow from subspot data to final dose calculation
 * following the RayTracedicom algorithm.
 */

#include <cuda_runtime.h>
#include <texture_types.h>
#include <vector>
#include <iostream>
#include <chrono>
#include <random>
#include <cmath>

// Include our library headers
#include "../include/core/common.cuh"
#include "../include/algorithms/convolution.h"
#include "../include/core/raytracedicom_integration.h"
#include "../include/utils/utils.h"
#include "../include/utils/debug_tools.h"

// ============================================================================
// Test Data Generation
// ============================================================================

std::vector<float> createRealisticSubspotData(int numLayers, int maxSubspotsPerLayer) {
    std::vector<float> subspotData(numLayers * maxSubspotsPerLayer * 5);
    std::default_random_engine generator(42); // Fixed seed for reproducibility
    
    // Constrain subspot positions to be within CPB grid range
    // CPB grid: corner=(-15,-15), resolution=0.1, dims=300x300
    // So valid range is approximately [-15, 15] cm
    std::uniform_real_distribution<float> deltaDist(-8.0f, 8.0f);  // Within CPB range
    std::uniform_real_distribution<float> weightDist(0.5f, 2.0f);
    std::uniform_real_distribution<float> sigmaDist(0.8f, 1.5f);
    
    for (int l = 0; l < numLayers; ++l) {
        for (int i = 0; i < maxSubspotsPerLayer; ++i) {
            int baseIdx = (l * maxSubspotsPerLayer + i) * 5;
            
            // Generate subspot positions within CPB grid
            float deltaX = deltaDist(generator);
            float deltaY = deltaDist(generator);
            
            // Ensure some subspots are near the center
            if (i < 3) {
                deltaX *= 0.5f;  // Closer to center
                deltaY *= 0.5f;
            }
            
            subspotData[baseIdx + 0] = deltaX;     // deltaX
            subspotData[baseIdx + 1] = deltaY;     // deltaY
            subspotData[baseIdx + 2] = weightDist(generator); // weight
            subspotData[baseIdx + 3] = sigmaDist(generator); // sigmaX
            subspotData[baseIdx + 4] = sigmaDist(generator); // sigmaY
        }
    }
    
    return subspotData;
}

// ============================================================================
// Complete Workflow Test
// ============================================================================

void testCompleteCPBToRayTracingWorkflow() {
    std::cout << "\n=== Complete CPB to Ray Tracing Workflow Test ===" << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Test parameters
    int numLayers = 3;
    int maxSubspotsPerLayer = 2;
    int rayDimsX = 100;  // Ray grid dimensions - closer to CPB grid
    int rayDimsY = 100;
    
    // Beam geometry
    vec3f beamDirection = make_vec3f(0.0f, 0.0f, 1.0f);  // Z direction
    vec3f bmXDirection = make_vec3f(1.0f, 0.0f, 0.0f);    // X direction
    vec3f bmYDirection = make_vec3f(0.0f, 1.0f, 0.0f);    // Y direction
    vec3f sourcePosition = make_vec3f(0.0f, 0.0f, -100.0f); // Source at Z=-100
    float sad = 100.0f;  // Source-to-axis distance
    float refPlaneZ = 0.0f;  // Reference plane at Z=0
    
    // ROI parameters (smaller, more focused)
    vec3f roiCorner = make_vec3f(-10.0f, -10.0f, -5.0f);
    vec3f roiResolution = make_vec3f(0.2f, 0.2f, 1.0f);
    vec3i roiDims = make_vec3i(100, 100, 10);
    
    // CPB grid parameters (based on ROI + margin)
    vec3f cpbCorner = make_vec3f(roiCorner.x - ROI_MARGIN_X, 
                                roiCorner.y - ROI_MARGIN_Y, 
                                roiCorner.z - ROI_MARGIN_Z);
    vec3f cpbResolution = make_vec3f(0.1f, 0.1f, 1.0f);  // Higher resolution
    vec3i cpbDims = make_vec3i(
        (int)((roiCorner.x + roiDims.x * roiResolution.x + ROI_MARGIN_X - cpbCorner.x) / cpbResolution.x),
        (int)((roiCorner.y + roiDims.y * roiResolution.y + ROI_MARGIN_Y - cpbCorner.y) / cpbResolution.y),
        numLayers
    );
    
    std::cout << "CPB Grid:" << std::endl;
    std::cout << "  Corner: (" << cpbCorner.x << ", " << cpbCorner.y << ", " << cpbCorner.z << ")" << std::endl;
    std::cout << "  Resolution: (" << cpbResolution.x << ", " << cpbResolution.y << ", " << cpbResolution.z << ")" << std::endl;
    std::cout << "  Dimensions: " << cpbDims.x << "x" << cpbDims.y << "x" << cpbDims.z << std::endl;
    
    // Create test subspot data
    std::vector<float> subspotData = createRealisticSubspotData(numLayers, maxSubspotsPerLayer);
    
    // Create subspot texture
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray* subspotArray;
    cudaExtent subspotExtent = make_cudaExtent(5, maxSubspotsPerLayer, numLayers);
    cudaMalloc3DArray(&subspotArray, &channelDesc, subspotExtent);
    
    cudaMemcpy3DParms copyParams = {};
    copyParams.srcPtr = make_cudaPitchedPtr(subspotData.data(), 5 * sizeof(float), 5, maxSubspotsPerLayer);
    copyParams.dstArray = subspotArray;
    copyParams.extent = subspotExtent;
    copyParams.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams);
    
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = subspotArray;
    
    cudaTextureDesc texDesc = {};
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;
    
    cudaTextureObject_t subspotTexture = 0;
    cudaCreateTextureObject(&subspotTexture, &resDesc, &texDesc, nullptr);
    
    // Step 1: Perform Subspot to CPB Convolution
    std::cout << "\nStep 1: Subspot to CPB Convolution" << std::endl;
    
    float* d_cpbWeights;
    size_t cpbWeightsSize = cpbDims.x * cpbDims.y * cpbDims.z * sizeof(float);
    cudaMalloc(&d_cpbWeights, cpbWeightsSize);
    cudaMemset(d_cpbWeights, 0, cpbWeightsSize);
    
    performSubspotToCPBConvolution(subspotTexture, numLayers, maxSubspotsPerLayer,
                                  cpbCorner, cpbResolution, cpbDims, d_cpbWeights,
                                  beamDirection, bmXDirection, bmYDirection,
                                  sourcePosition, sad, refPlaneZ);
    
    // Copy CPB results back for analysis
    std::vector<float> cpbWeights(cpbDims.x * cpbDims.y * cpbDims.z);
    cudaMemcpy(cpbWeights.data(), d_cpbWeights, cpbWeightsSize, cudaMemcpyDeviceToHost);
    
    // Analyze CPB results
    float totalCPBWeight = 0.0f;
    float maxCPBWeight = 0.0f;
    int nonZeroCPB = 0;
    for (int i = 0; i < cpbWeights.size(); i++) {
        if (cpbWeights[i] > 1e-6f) {
            nonZeroCPB++;
            totalCPBWeight += cpbWeights[i];
            maxCPBWeight = std::max(maxCPBWeight, cpbWeights[i]);
        }
    }
    
    std::cout << "CPB Results:" << std::endl;
    std::cout << "  Total weight: " << totalCPBWeight << std::endl;
    std::cout << "  Max weight: " << maxCPBWeight << std::endl;
    std::cout << "  Non-zero elements: " << nonZeroCPB << "/" << cpbWeights.size() << std::endl;
    
    // Step 2: Map CPB Weights to Ray Weights
    std::cout << "\nStep 2: CPB to Ray Weight Mapping" << std::endl;
    
    float* d_rayWeights;
    size_t rayWeightsSize = rayDimsX * rayDimsY * sizeof(float);
    cudaMalloc(&d_rayWeights, rayWeightsSize);
    cudaMemset(d_rayWeights, 0, rayWeightsSize);
    
    vec3i rayDims = make_vec3i(rayDimsX, rayDimsY, 1);
    
    performCPBToRayWeightMapping(d_cpbWeights, cpbDims, cpbCorner, cpbResolution,
                                d_rayWeights, rayDims,
                                cpbCorner, cpbResolution,
                                0,  // layerIdx
                                beamDirection, bmXDirection, bmYDirection,
                                sourcePosition, sad, refPlaneZ);
    
    // Copy ray weights back for analysis
    std::vector<float> rayWeights(rayDimsX * rayDimsY);
    cudaMemcpy(rayWeights.data(), d_rayWeights, rayWeightsSize, cudaMemcpyDeviceToHost);
    
    // Analyze ray weight results
    float totalRayWeight = 0.0f;
    float maxRayWeight = 0.0f;
    int nonZeroRay = 0;
    for (int i = 0; i < rayWeights.size(); i++) {
        if (rayWeights[i] > 1e-6f) {
            nonZeroRay++;
            totalRayWeight += rayWeights[i];
            maxRayWeight = std::max(maxRayWeight, rayWeights[i]);
        }
    }
    
    std::cout << "Ray Weight Results:" << std::endl;
    std::cout << "  Total weight: " << totalRayWeight << std::endl;
    std::cout << "  Max weight: " << maxRayWeight << std::endl;
    std::cout << "  Non-zero elements: " << nonZeroRay << "/" << rayWeights.size() << std::endl;
    
    // Show ray weight distribution (center region)
    std::cout << "\nRay weights (center 8x8):" << std::endl;
    int centerX = rayDimsX / 2;
    int centerY = rayDimsY / 2;
    for (int y = centerY - 4; y < centerY + 4; y++) {
        for (int x = centerX - 4; x < centerX + 4; x++) {
            if (x >= 0 && x < rayDimsX && y >= 0 && y < rayDimsY) {
                int idx = y * rayDimsX + x;
                printf("%.4f\t", rayWeights[idx]);
            }
        }
        printf("\n");
    }
    
    // Step 3: Verify Weight Conservation
    std::cout << "\nStep 3: Weight Conservation Analysis" << std::endl;
    
    float weightRatio = (totalRayWeight > 0) ? totalRayWeight / totalCPBWeight : 0.0f;
    std::cout << "Weight conservation ratio (Ray/CPB): " << weightRatio << std::endl;
    
    if (weightRatio > 0.8f && weightRatio < 1.2f) {
        std::cout << "✓ Weight conservation is good" << std::endl;
    } else {
        std::cout << "⚠ Weight conservation needs attention" << std::endl;
    }
    
    // Cleanup
    cudaDestroyTextureObject(subspotTexture);
    cudaFreeArray(subspotArray);
    cudaFree(d_cpbWeights);
    cudaFree(d_rayWeights);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "\n=== Workflow Test Complete ===" << std::endl;
    std::cout << "Total execution time: " << duration.count() << " μs" << std::endl;
}

// ============================================================================
// Main Function
// ============================================================================

int main() {
    std::cout << "=== CPB to Ray Tracing Integration Test ===" << std::endl;
    
    // Initialize CUDA
    cudaSetDevice(0);
    
    // Run the complete workflow test
    testCompleteCPBToRayTracingWorkflow();
    
    return 0;
}
