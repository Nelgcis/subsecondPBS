/**
 * \file
 * \brief CPB Convolution Debug Program
 * 
 * This file implements a comprehensive debug program to test CPB convolution
 * with proper ROI range calculation and beam intersection checking
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cuda_runtime.h>
#include <texture_types.h>

#include "../include/core/common.cuh"
#include "../include/core/Macro.cuh"
#include "../include/algorithms/convolution.h"
#include "../include/algorithms/cuda_final_dose.h"

// Debug function to test SubspotInfo construction
void debugSubspotInfoConstruction() {
    std::cout << "\n=== Debug SubspotInfo Construction ===" << std::endl;
    
    // Create test subspot data
    int numLayers = 3;
    int maxSubspotsPerLayer = 5;
    std::vector<float> subspotData(numLayers * maxSubspotsPerLayer * 5);
    
    // Fill with test data (not all zeros or ones, with fractional deltas)
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> deltaRange(-1.5f, 1.5f);
    std::uniform_real_distribution<float> weightRange(0.5f, 3.0f);
    std::uniform_real_distribution<float> sigmaRange(0.8f, 1.5f);
    
    for (int layer = 0; layer < numLayers; layer++) {
        for (int subspot = 0; subspot < maxSubspotsPerLayer; subspot++) {
            int baseIdx = (layer * maxSubspotsPerLayer + subspot) * 5;
            subspotData[baseIdx + 0] = deltaRange(gen);  // deltaX
            subspotData[baseIdx + 1] = deltaRange(gen);  // deltaY
            subspotData[baseIdx + 2] = weightRange(gen); // weight
            subspotData[baseIdx + 3] = sigmaRange(gen);  // sigmaX
            subspotData[baseIdx + 4] = sigmaRange(gen);  // sigmaY
        }
    }
    
    // Print input data
    std::cout << "Input subspot data:" << std::endl;
    for (int layer = 0; layer < numLayers; layer++) {
        std::cout << "Layer " << layer << ":" << std::endl;
        for (int subspot = 0; subspot < maxSubspotsPerLayer; subspot++) {
            int baseIdx = (layer * maxSubspotsPerLayer + subspot) * 5;
            std::cout << "  Subspot " << subspot << ": "
                      << "deltaX=" << subspotData[baseIdx + 0]
                      << ", deltaY=" << subspotData[baseIdx + 1]
                      << ", weight=" << subspotData[baseIdx + 2]
                      << ", sigmaX=" << subspotData[baseIdx + 3]
                      << ", sigmaY=" << subspotData[baseIdx + 4] << std::endl;
        }
    }
    
    // Create texture
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray* devArray;
    cudaExtent extent = make_cudaExtent(5, maxSubspotsPerLayer, numLayers);
    cudaMalloc3DArray(&devArray, &channelDesc, extent);
    
    // Copy data to device
    float* devSubspotData;
    cudaMalloc(&devSubspotData, subspotData.size() * sizeof(float));
    cudaMemcpy(devSubspotData, subspotData.data(), subspotData.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    // Copy to 3D array - 修复纹理布局
    cudaMemcpy3DParms copyParams = {};
    copyParams.srcPtr = make_cudaPitchedPtr(devSubspotData, 5 * sizeof(float), 5, maxSubspotsPerLayer);
    copyParams.dstArray = devArray;
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyDeviceToDevice;
    cudaMemcpy3D(&copyParams);
    
    // Create texture object
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = devArray;
    
    cudaTextureDesc texDesc = {};
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    
    cudaTextureObject_t subspotTexture;
    cudaCreateTextureObject(&subspotTexture, &resDesc, &texDesc, NULL);
    
    // Test SubspotInfo construction on GPU
    SubspotInfo* d_subspotInfoArray;
    cudaMalloc(&d_subspotInfoArray, maxSubspotsPerLayer * sizeof(SubspotInfo));
    
    // Beam parameters
    vec3f beamDirection = vec3f(0.0f, 0.0f, -1.0f);
    vec3f bmXDirection = vec3f(1.0f, 0.0f, 0.0f);
    vec3f bmYDirection = vec3f(0.0f, 1.0f, 0.0f);
    vec3f sourcePosition = vec3f(0.0f, 0.0f, 100.0f);
    float sad = 100.0f;
    float refPlaneZ = 0.0f;
    
    // Launch kernel to construct SubspotInfo using library function
    printf("Calling performSubspotToCPBConvolution...\n");
    
    // Allocate CPB weights for testing
    float* d_cpbWeights;
    vec3f cpbCorner = vec3f(-200.0f, -200.0f, 0.0f);
    vec3f cpbResolution = vec3f(2.0f, 2.0f, 1.0f);
    vec3i cpbDims = vec3i(200, 200, 1);
    size_t cpbWeightsSize = cpbDims.x * cpbDims.y * cpbDims.z * sizeof(float);
    cudaMalloc(&d_cpbWeights, cpbWeightsSize);
    cudaMemset(d_cpbWeights, 0, cpbWeightsSize);
    
    performSubspotToCPBConvolution(subspotTexture, 1, maxSubspotsPerLayer,
                                  cpbCorner, cpbResolution, cpbDims, d_cpbWeights,
                                  beamDirection, bmXDirection, bmYDirection,
                                  sourcePosition, sad, refPlaneZ);
    
    // Copy results back
    std::vector<float> cpbWeights(cpbDims.x * cpbDims.y * cpbDims.z);
    cudaMemcpy(cpbWeights.data(), d_cpbWeights, cpbWeightsSize, cudaMemcpyDeviceToHost);
    
    // Print CPB weights
    printf("CPB weights (first 4x4):\n");
    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            int idx = y * cpbDims.x + x;
            printf("%.6f\t", cpbWeights[idx]);
        }
        printf("\n");
    }
    
    // Analyze non-zero elements distribution
    printf("\n=== CPB Weight Analysis ===\n");
    float totalWeight = 0.0f;
    float maxWeight = 0.0f;
    int nonZeroCount = 0;
    int totalElements = cpbDims.x * cpbDims.y * cpbDims.z;
    
    // Find weight distribution
    for (int i = 0; i < totalElements; i++) {
        if (cpbWeights[i] > 1e-6f) {
            nonZeroCount++;
            totalWeight += cpbWeights[i];
            maxWeight = max(maxWeight, cpbWeights[i]);
        }
    }
    
    printf("Total weight: %.6f\n", totalWeight);
    printf("Max weight: %.6f\n", maxWeight);
    printf("Non-zero elements: %d/%d\n", nonZeroCount, totalElements);
    
    // Find the center of mass of non-zero weights
    float centerX = 0.0f, centerY = 0.0f;
    float weightSum = 0.0f;
    
    for (int y = 0; y < cpbDims.y; y++) {
        for (int x = 0; x < cpbDims.x; x++) {
            int idx = y * cpbDims.x + x;
            if (cpbWeights[idx] > 1e-6f) {
                float worldX = cpbCorner.x + (x + 0.5f) * cpbResolution.x;
                float worldY = cpbCorner.y + (y + 0.5f) * cpbResolution.y;
                centerX += worldX * cpbWeights[idx];
                centerY += worldY * cpbWeights[idx];
                weightSum += cpbWeights[idx];
            }
        }
    }
    
    if (weightSum > 0) {
        centerX /= weightSum;
        centerY /= weightSum;
        printf("Center of mass: (%.3f, %.3f)\n", centerX, centerY);
        
        // Check if center is near beam axis (should be around (0,0) for our test)
        float distanceFromBeamAxis = sqrtf(centerX * centerX + centerY * centerY);
        printf("Distance from beam axis: %.3f cm\n", distanceFromBeamAxis);
    }
    
    cudaFree(d_cpbWeights);
}

// ROI range calculation function
struct ROIRange {
    vec3f minCorner;
    vec3f maxCorner;
    vec3i dims;
    bool isValid;
};

ROIRange calculateROIRange(const std::vector<vec3i>& roiIndices, 
                          const vec3f& doseGridResolution,
                          const vec3f& doseGridCorner,
                          const vec3f& margin) {
    ROIRange roiRange;
    
    if (roiIndices.empty()) {
        roiRange.isValid = false;
        return roiRange;
    }
    
    // Find min/max indices
    vec3i minIdx = roiIndices[0];
    vec3i maxIdx = roiIndices[0];
    
    for (const auto& idx : roiIndices) {
        minIdx.x = min(minIdx.x, idx.x);
        minIdx.y = min(minIdx.y, idx.y);
        minIdx.z = min(minIdx.z, idx.z);
        maxIdx.x = max(maxIdx.x, idx.x);
        maxIdx.y = max(maxIdx.y, idx.y);
        maxIdx.z = max(maxIdx.z, idx.z);
    }
    
    // Convert to world coordinates
    roiRange.minCorner = vec3f(
        doseGridCorner.x + minIdx.x * doseGridResolution.x,
        doseGridCorner.y + minIdx.y * doseGridResolution.y,
        doseGridCorner.z + minIdx.z * doseGridResolution.z
    );
    
    roiRange.maxCorner = vec3f(
        doseGridCorner.x + (maxIdx.x + 1) * doseGridResolution.x,
        doseGridCorner.y + (maxIdx.y + 1) * doseGridResolution.y,
        doseGridCorner.z + (maxIdx.z + 1) * doseGridResolution.z
    );
    
    // Add margin
    roiRange.minCorner.x -= margin.x;
    roiRange.minCorner.y -= margin.y;
    roiRange.minCorner.z -= margin.z;
    roiRange.maxCorner.x += margin.x;
    roiRange.maxCorner.y += margin.y;
    roiRange.maxCorner.z += margin.z;
    
    // Calculate dimensions
    roiRange.dims = vec3i(
        maxIdx.x - minIdx.x + 1,
        maxIdx.y - minIdx.y + 1,
        maxIdx.z - minIdx.z + 1
    );
    
    roiRange.isValid = true;
    return roiRange;
}

// Beam-ROI intersection check
bool checkBeamROIIntersection(const vec3f& beamDirection,
                             const vec3f& sourcePosition,
                             const ROIRange& roiRange) {
    // Simple intersection check: if beam direction points towards ROI
    vec3f roiCenter = vec3f(
        (roiRange.minCorner.x + roiRange.maxCorner.x) * 0.5f,
        (roiRange.minCorner.y + roiRange.maxCorner.y) * 0.5f,
        (roiRange.minCorner.z + roiRange.maxCorner.z) * 0.5f
    );
    
    vec3f toROI = roiCenter - sourcePosition;
    float dotProduct = dot(beamDirection, toROI);
    
    return dotProduct > 0.0f; // Beam points towards ROI
}

// Test CPB convolution with proper ROI calculation
void testCPBConvolutionWithROI() {
    std::cout << "\n=== Test CPB Convolution with ROI ===" << std::endl;
    
    // Create test ROI indices (large grid to cover subspot positions)
    std::vector<vec3i> roiIndices;
    for (int z = 0; z < 3; z++) {
        for (int y = 0; y < 200; y++) {
            for (int x = 0; x < 200; x++) {
                roiIndices.push_back(vec3i(x, y, z));
            }
        }
    }
    
    // Dose grid parameters - make it large enough to cover subspot positions
    vec3f doseGridResolution = vec3f(2.0f, 2.0f, 1.0f);
    vec3f doseGridCorner = vec3f(-200.0f, -200.0f, 0.0f);
    vec3f margin = vec3f(ROI_MARGIN_X, ROI_MARGIN_Y, ROI_MARGIN_Z);
    
    // Calculate ROI range
    ROIRange roiRange = calculateROIRange(roiIndices, doseGridResolution, doseGridCorner, margin);
    
    std::cout << "ROI Range:" << std::endl;
    std::cout << "  Min Corner: (" << roiRange.minCorner.x << ", " << roiRange.minCorner.y << ", " << roiRange.minCorner.z << ")" << std::endl;
    std::cout << "  Max Corner: (" << roiRange.maxCorner.x << ", " << roiRange.maxCorner.y << ", " << roiRange.maxCorner.z << ")" << std::endl;
    std::cout << "  Dimensions: " << roiRange.dims.x << "x" << roiRange.dims.y << "x" << roiRange.dims.z << std::endl;
    
    // Beam parameters
    vec3f beamDirection = vec3f(0.0f, 0.0f, -1.0f);
    vec3f sourcePosition = vec3f(0.0f, 0.0f, 100.0f);
    
    // Check beam-ROI intersection
    bool intersects = checkBeamROIIntersection(beamDirection, sourcePosition, roiRange);
    std::cout << "Beam-ROI intersection: " << (intersects ? "YES" : "NO") << std::endl;
    
    if (!intersects) {
        std::cout << "ERROR: Beam does not intersect with ROI!" << std::endl;
        return;
    }
    
    // Create CPB grid based on ROI
    vec3f cpbCorner = vec3f(roiRange.minCorner.x, roiRange.minCorner.y, 0.0f);
    vec3f cpbResolution = vec3f(0.5f, 0.5f, 1.0f);
    vec3i cpbDims = vec3i(
        (int)((roiRange.maxCorner.x - roiRange.minCorner.x) / cpbResolution.x),
        (int)((roiRange.maxCorner.y - roiRange.minCorner.y) / cpbResolution.y),
        3 // 3 energy layers
    );
    
    std::cout << "CPB Grid:" << std::endl;
    std::cout << "  Corner: (" << cpbCorner.x << ", " << cpbCorner.y << ", " << cpbCorner.z << ")" << std::endl;
    std::cout << "  Resolution: (" << cpbResolution.x << ", " << cpbResolution.y << ", " << cpbResolution.z << ")" << std::endl;
    std::cout << "  Dimensions: " << cpbDims.x << "x" << cpbDims.y << "x" << cpbDims.z << std::endl;
    
    // Create test subspot data
    int numLayers = 3;
    int maxSubspotsPerLayer = 5;
    std::vector<float> subspotData(numLayers * maxSubspotsPerLayer * 5);
    
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> deltaRange(-1.0f, 1.0f);
    std::uniform_real_distribution<float> weightRange(1.0f, 5.0f);
    std::uniform_real_distribution<float> sigmaRange(0.5f, 1.5f);
    
    for (int layer = 0; layer < numLayers; layer++) {
        for (int subspot = 0; subspot < maxSubspotsPerLayer; subspot++) {
            int baseIdx = (layer * maxSubspotsPerLayer + subspot) * 5;
            subspotData[baseIdx + 0] = deltaRange(gen);
            subspotData[baseIdx + 1] = deltaRange(gen);
            subspotData[baseIdx + 2] = weightRange(gen);
            subspotData[baseIdx + 3] = sigmaRange(gen);
            subspotData[baseIdx + 4] = sigmaRange(gen);
        }
    }
    
    // Create texture and run convolution
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray* devArray;
    cudaExtent extent = make_cudaExtent(5, maxSubspotsPerLayer, numLayers);
    cudaMalloc3DArray(&devArray, &channelDesc, extent);
    
    float* devSubspotData;
    cudaMalloc(&devSubspotData, subspotData.size() * sizeof(float));
    cudaMemcpy(devSubspotData, subspotData.data(), subspotData.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaMemcpy3DParms copyParams = {};
    copyParams.srcPtr = make_cudaPitchedPtr(devSubspotData, 5 * sizeof(float), 5, maxSubspotsPerLayer);
    copyParams.dstArray = devArray;
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyDeviceToDevice;
    cudaMemcpy3D(&copyParams);
    
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = devArray;
    
    cudaTextureDesc texDesc = {};
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    
    cudaTextureObject_t subspotTexture;
    cudaCreateTextureObject(&subspotTexture, &resDesc, &texDesc, NULL);
    
    // Allocate CPB weights
    float* devCpbWeights;
    size_t cpbWeightsSize = cpbDims.x * cpbDims.y * cpbDims.z * sizeof(float);
    cudaMalloc(&devCpbWeights, cpbWeightsSize);
    cudaMemset(devCpbWeights, 0, cpbWeightsSize);
    
    // Run convolution
    vec3f bmXDirection = vec3f(1.0f, 0.0f, 0.0f);
    vec3f bmYDirection = vec3f(0.0f, 1.0f, 0.0f);
    float sad = 100.0f;
    float refPlaneZ = 0.0f;
    
    auto start = std::chrono::high_resolution_clock::now();
    performSubspotToCPBConvolution(subspotTexture, numLayers, maxSubspotsPerLayer,
                                  cpbCorner, cpbResolution, cpbDims, devCpbWeights,
                                  beamDirection, bmXDirection, bmYDirection,
                                  sourcePosition, sad, refPlaneZ);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Copy results back
    std::vector<float> cpbWeights(cpbDims.x * cpbDims.y * cpbDims.z);
    cudaMemcpy(cpbWeights.data(), devCpbWeights, cpbWeightsSize, cudaMemcpyDeviceToHost);
    
    // Analyze results
    float totalWeight = 0.0f;
    float maxWeight = 0.0f;
    int nonZeroElements = 0;
    
    for (size_t i = 0; i < cpbWeights.size(); i++) {
        totalWeight += cpbWeights[i];
        maxWeight = std::max(maxWeight, cpbWeights[i]);
        if (cpbWeights[i] > 1e-6f) {
            nonZeroElements++;
        }
    }
    
    std::cout << "\nConvolution Results:" << std::endl;
    std::cout << "  Execution time: " << duration.count() << " μs" << std::endl;
    std::cout << "  Total weight: " << totalWeight << std::endl;
    std::cout << "  Max weight: " << maxWeight << std::endl;
    std::cout << "  Non-zero elements: " << nonZeroElements << "/" << cpbWeights.size() << std::endl;
    
    // Find the center of mass of non-zero weights
    float centerX = 0.0f, centerY = 0.0f;
    float weightSum = 0.0f;
    
    for (int y = 0; y < cpbDims.y; y++) {
        for (int x = 0; x < cpbDims.x; x++) {
            int idx = y * cpbDims.x + x;
            if (cpbWeights[idx] > 1e-6f) {
                float worldX = cpbCorner.x + (x + 0.5f) * cpbResolution.x;
                float worldY = cpbCorner.y + (y + 0.5f) * cpbResolution.y;
                centerX += worldX * cpbWeights[idx];
                centerY += worldY * cpbWeights[idx];
                weightSum += cpbWeights[idx];
            }
        }
    }
    
    if (weightSum > 0) {
        centerX /= weightSum;
        centerY /= weightSum;
        std::cout << "  Center of mass: (" << centerX << ", " << centerY << ")" << std::endl;
        
        // Check if center is near beam axis (should be around (0,0) for our test)
        float distanceFromBeamAxis = sqrtf(centerX * centerX + centerY * centerY);
        std::cout << "  Distance from beam axis: " << distanceFromBeamAxis << " cm" << std::endl;
        
        // Check if center is within ROI
        bool withinROI = (centerX >= roiRange.minCorner.x && centerX <= roiRange.maxCorner.x &&
                         centerY >= roiRange.minCorner.y && centerY <= roiRange.maxCorner.y);
        std::cout << "  Center within ROI: " << (withinROI ? "YES" : "NO") << std::endl;
    }
    
    // Show first layer weights
    std::cout << "\nCPB weights (layer 0, first 5x5):" << std::endl;
    for (int y = 0; y < std::min(5, cpbDims.y); y++) {
        for (int x = 0; x < std::min(5, cpbDims.x); x++) {
            int idx = y * cpbDims.x + x;
            std::cout << cpbWeights[idx] << "\t";
        }
        std::cout << std::endl;
    }
    
    // Cleanup
    cudaDestroyTextureObject(subspotTexture);
    cudaFreeArray(devArray);
    cudaFree(devSubspotData);
    cudaFree(devCpbWeights);
}

int main() {
    std::cout << "=== CPB Convolution Debug Program ===" << std::endl;
    
    // Test 1: SubspotInfo construction
    debugSubspotInfoConstruction();
    
    // Test 2: CPB convolution with ROI
    testCPBConvolutionWithROI();
    
    std::cout << "\n=== Debug Program Complete ===" << std::endl;
    return 0;
}
