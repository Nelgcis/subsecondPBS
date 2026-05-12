/**
 * \file
 * \brief Backward Compatibility Headers for Test Files
 * 
 * This file provides backward compatibility function declarations
 */

#ifndef BACKWARD_COMPATIBILITY_H
#define BACKWARD_COMPATIBILITY_H

#include "../core/common.cuh"

// ============================================================================
// Backward Compatibility Function Declarations
// ============================================================================

// Backward compatible function for subspotToCPBConvolutionGPU
int subspotToCPBConvolutionGPU(
    const float* subspotData,
    int numLayers,
    int maxSubspotsPerLayer,
    vec3f cpbCorner,
    vec3f cpbResolution,
    vec3i cpbDims,
    float* cpbWeights,
    int gpuId
);

// Backward compatible function for initializeRayWeightsFromSubspotDataGPU
int initializeRayWeightsFromSubspotDataGPU(
    float* rayWeights,
    vec3i rayDims,
    vec3f beamDirection,
    vec3f bmXDirection,
    vec3f bmYDirection,
    vec3f sourcePosition,
    float sad,
    float refPlaneZ,
    int gpuId
);

#endif // BACKWARD_COMPATIBILITY_H
