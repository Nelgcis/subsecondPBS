/**
 * \file
 * \brief Unified IDD and Sigma Calculation Headers
 * 
 * This file provides unified headers for IDD and sigma calculation algorithms
 */

#ifndef IDD_SIGMA_H
#define IDD_SIGMA_H

#include "../core/common.cuh"
#include "../core/Macro.cuh"
#include <cuda_runtime.h>

// Forward declarations
struct FillIddAndSigmaParams;

// Kernel function declarations
#ifdef NUCLEAR_CORR
__global__ void fillIddAndSigmaKernel(
    float* bevDensity,
    float* bevCumulSp,
    float* bevIdd,
    float* bevRSigmaEff,
    float* bevNucIdd,
    float* bevNucRSigmaEff,
    const float* nucRayWeights,
    const int* nucIdcs,
    float* rayWeights,
    int* firstInside,
    int* firstOutside,
    int* firstPassive,
    FillIddAndSigmaParams params,
    int rayDimsX,
    int rayDimsY,
    int steps,
    cudaTextureObject_t cumulIddTex,
    cudaTextureObject_t rRadiationLengthTex,
    cudaTextureObject_t nucWeightTex,
    cudaTextureObject_t nucSqSigmaTex
);
#else
__global__ void fillIddAndSigmaKernel(
    float* bevDensity,
    float* bevCumulSp,
    float* bevIdd,
    float* bevRSigmaEff,
    float* rayWeights,
    int* firstInside,
    int* firstOutside,
    int* firstPassive,
    FillIddAndSigmaParams params,
    int rayDimsX,
    int rayDimsY,
    int steps,
    cudaTextureObject_t cumulIddTex,
    cudaTextureObject_t rRadiationLengthTex
);
#endif

// Legacy kernel (deprecated)
__global__ void simpleIddCalculationKernel(
    float* bevDensity,
    float* bevCumulSp,
    float* bevIdd,
    float* bevRSigmaEff,
    float* rayWeights,
    int* firstInside,
    int* firstOutside,
    int rayDimsX,
    int rayDimsY,
    int steps,
    cudaTextureObject_t cumulIddTex,
    cudaTextureObject_t rRadiationLengthTex
);

// ============================================================================
// Function Declarations
// ============================================================================

// IDD and Sigma Calculation
void performIddAndSigmaCalculation(
    float* bevDensity,
    float* bevCumulSp,
    float* bevIdd,
    float* bevRSigmaEff,
    float* rayWeights,
    int* firstInside,
    int* firstOutside,
    int* firstPassive,
    const FillIddAndSigmaParams& params,
    cudaTextureObject_t cumulIddTex,
    cudaTextureObject_t rRadiationLengthTex
);

// Sigma Texture Calculation
void performSigmaTextureCalculation(
    cudaTextureObject_t subspotData,
    float* sigmaXTexture,
    float* sigmaYTexture,
    vec3f cpbCorner,
    vec3f cpbResolution,
    vec3i cpbDims,
    vec3f beamDirection,
    vec3f bmXDirection,
    vec3f bmYDirection,
    vec3f sourcePosition,
    float sad,
    float refPlaneZ,
    int numLayers,
    int maxSubspotsPerLayer
);

#endif // IDD_SIGMA_H
