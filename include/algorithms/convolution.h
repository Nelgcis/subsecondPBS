/**
 * \file
 * \brief Unified Convolution Algorithm Headers
 * 
 * This file provides unified headers for all convolution-related algorithms
 */

#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include "../core/common.cuh"
#include "../core/Macro.cuh"
#include <cuda_runtime.h>

// ============================================================================
// Kernel Declarations
// ============================================================================

// Subspot data reading kernel
__global__ void readSubspotDataKernel(
    cudaTextureObject_t subspotData,
    SubspotInfo* subspotInfoArray,
    int nsubspot,
    int layerIdx,
    vec3f beamDirection,
    vec3f bmXDirection,
    vec3f bmYDirection,
    vec3f sourcePosition,
    float sad,
    float refPlaneZ
);

// Subspot range calculation kernel
__global__ void calculateSubspotRangeKernel(
    const SubspotInfo* subspotInfoArray,
    int nsubspot,
    vec3f cpbCorner,
    vec3f cpbResolution,
    vec3i cpbDims,
    int* subspotRanges
);

// Subspot to CPB convolution kernel
__global__ void subspotToCPBConvolutionOptimizedKernel(
    const SubspotInfo* subspotInfoArray,
    const int* subspotRanges,
    int nsubspot,
    int layerIdx,
    vec3f cpbCorner,
    vec3f cpbResolution,
    vec3i cpbDims,
    float* cpbWeights
);

// GPU 2D convolution kernel
__global__ void gpuConvolution2DKernel(
    float* input,
    float* output,
    int width,
    int height,
    float* kernel,
    int kernelSize,
    int padding
);

__global__ void xConvGathResampGpu(
    float* in,
    float* out,
    float2* sigma,
    int inWidth,
    int outWidth,
    int height,
    float pixelSp,
    float inOutOffset,
    float inOutDelta
);

__global__ void yConvGathResampGpu(
    float* in,
    float* out,
    float2* sigma,
    int width,
    int inHeight,
    int outHeight,
    float pixelSp,
    float inOutOffset,
    float inOutDelta
);

// CPB to ray weight mapping kernel (scalar args to avoid struct ABI issues)
__global__ void mapCPBWeightsToRayWeightsKernel(
    const float* cpbWeights,
    int cpbDimsX, int cpbDimsY, int cpbDimsZ,
    float cpbCornerX, float cpbCornerY, float cpbCornerZ,
    float cpbResolutionX, float cpbResolutionY, float cpbResolutionZ,
    float* rayWeights,
    int rayDimsX, int rayDimsY,
    float rayCornerX, float rayCornerY, float rayCornerZ,
    float rayResolutionX, float rayResolutionY, float rayResolutionZ,
    int layerIdx,
    float beamDirX, float beamDirY, float beamDirZ,
    float bmXDirX, float bmXDirY, float bmXDirZ,
    float bmYDirX, float bmYDirY, float bmYDirZ,
    float sourcePosX, float sourcePosY, float sourcePosZ,
    float sad,
    float refPlaneZ
);

// ============================================================================
// Function Declarations
// ============================================================================

// Subspot to CPB Convolution
void performSubspotToCPBConvolution(
    cudaTextureObject_t subspotData,
    int numLayers,
    int maxSubspotsPerLayer,
    vec3f cpbCorner,
    vec3f cpbResolution,
    vec3i cpbDims,
    float* cpbWeights,
    vec3f beamDirection,
    vec3f bmXDirection,
    vec3f bmYDirection,
    vec3f sourcePosition,
    float sad,
    float refPlaneZ
);

// GPU 2D Convolution
void performGPUConvolution2D(
    float* input,
    float* output,
    int width,
    int height,
    float* kernel,
    int kernelSize,
    int padding
);

void performExactRTDConvolution2D(
    float* spotWeights,
    float* convInterm,
    float* rayWeights,
    float2* entrySigmas,
    uint3 spotGridDims,
    uint3 rayGridDims,
    float3 spotDelta,
    float3 spotOffset,
    float3 rayDelta,
    float3 rayOffset,
    float2 pxSpMult
);

// CPB to Ray Weight Mapping
void performCPBToRayWeightMapping(
    float* cpbWeights,
    vec3i cpbDims,
    vec3f cpbCorner,
    vec3f cpbResolution,
    float* rayWeights,
    vec3i rayDims,
    vec3f rayCorner,
    vec3f rayResolution,
    int layerIdx,  // Current energy layer index
    vec3f beamDirection,
    vec3f bmXDirection,
    vec3f bmYDirection,
    vec3f sourcePosition,
    float sad,
    float refPlaneZ
);

#endif // CONVOLUTION_H
