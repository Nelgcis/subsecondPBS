/**
 * \file
 * \brief Simplified IDD and Sigma Calculation Implementation
 * 
 * This file provides simplified implementations for IDD and sigma calculation
 */

#include "../include/algorithms/idd_sigma.h"
#include "../include/algorithms/fill_idd_and_sigma_params.cuh"
#include "../include/core/common.cuh"
#include "../include/core/Macro.cuh"
#include "../include/utils/debug_tools.h"
#include <cuda_runtime.h>
#include <texture_indirect_functions.h>

// ============================================================================
// Simplified IDD and Sigma Calculation
// ============================================================================

// Correct fillIddAndSigma kernel following RayTracedicom algorithm
// Uses energyIdx to query IDD lookup table (not density!)
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
)
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
)
#endif
{
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= (unsigned)rayDimsX || y >= (unsigned)rayDimsY) return;

    const unsigned int memStep = (unsigned int)(rayDimsY * rayDimsX);
    const unsigned int idx2d = y * (unsigned int)rayDimsX + x;

    const unsigned int firstIn = (unsigned int)firstInside[idx2d];
    unsigned int afterLast = (unsigned int)min(firstOutside[idx2d], (int)params.getAfterLastStep());

    const float rayWeight = rayWeights[idx2d];
    bool beamLive = true;

    if (rayWeight < 1e-6f || afterLast < (unsigned int)params.getFirstStep()) {
        beamLive = false;
        afterLast = 0;
    }

    float cumulSp = 0.0f;
    float cumulSpOld = 0.0f;
    float cumulDose = 0.0f;
    float cumulDoseOld = 0.0f;

#ifdef NUCLEAR_CORR
    const bool nuclearEnabled =
        bevNucIdd != nullptr &&
        bevNucRSigmaEff != nullptr &&
        nucRayWeights != nullptr &&
        nucIdcs != nullptr &&
        nucWeightTex != 0 &&
        nucSqSigmaTex != 0 &&
        params.getNucMemStep() > 0 &&
        params.getSpotDist() > 0.0f;
#endif

    params.initStepAndAirDiv();

    const float pInv = 0.5649718f;
    const float eCoef = 8.639415f;
    const float sqrt2 = 1.41421356f;
    float eRefSq = 198.81f;
    float sigmaDelta = 0.21f;

#ifdef NUCLEAR_CORR
    if (nuclearEnabled) {
#if NUCLEAR_CORR == SOUKUP
        eRefSq = 190.44f;
        sigmaDelta = 0.0f;
#elif NUCLEAR_CORR == FLUKA
        eRefSq = 216.09f;
        sigmaDelta = 0.08f;
#elif NUCLEAR_CORR == GAUSS_FIT
        eRefSq = 169.00f;
        sigmaDelta = 0.06f;
#endif
    }
#endif

    float incScat = 0.0f;
    float incincScat = 0.0f;
    float incDiv = params.getSigmaSqAirLin() +
                   (2.0f * float(params.getFirstStep()) - 1.0f) * params.getSigmaSqAirQuad();
    // RTD-main semantics: primary entry sigma is already accounted for in the
    // ray-weight convolution stage. The transport sigma evolution starts from
    // -incDiv so the first loop iteration adds the correct air-divergence term
    // without double-counting entry broadening.
    float sigmaSq = -incDiv;
    bool suppressDistalSigmaDip = false;

#ifdef NUCLEAR_CORR
    float nucRes = 0.0f;
    float nucRSigmaEff = __int_as_float(0x7f800000);
    int nucIdx = -1;
    float nucRayWeight = 0.0f;
    if (nuclearEnabled) {
        nucIdx = nucIdcs[idx2d];
        if (nucIdx >= 0) {
            nucRayWeight = nucRayWeights[nucIdx];
        }
        nucIdx += static_cast<int>(params.getFirstStep() * params.getNucMemStep());
#if NUCLEAR_CORR == GAUSS_FIT
        suppressDistalSigmaDip = true;
#endif
    }
#endif

    unsigned int idx = idx2d + (unsigned int)params.getFirstStep() * memStep;

    for (unsigned int stepNo = (unsigned int)params.getFirstStep();
         stepNo < (unsigned int)params.getAfterLastStep();
         ++stepNo) {

        float res = 0.0f;
        float rSigmaEff = __int_as_float(0x7f800000);
#ifdef NUCLEAR_CORR
        nucRes = 0.0f;
        nucRSigmaEff = __int_as_float(0x7f800000);
#endif

        if (beamLive) {
            cumulSp = bevCumulSp[idx];
            const float depthIdx = cumulSp * params.getEnergyScaleFact() + HALF;
            const float energyIdx = params.getEnergyIdx() + HALF;
            cumulDose = tex2D<float>(cumulIddTex, depthIdx, energyIdx);

            const float density = bevDensity[idx];

            if (cumulSp < params.getPeakDepth()) {
                const float resE = eCoef * __powf(params.getPeakDepth() - HALF * (cumulSp + cumulSpOld), pInv);
                const float betaP = resE + 938.3f - 938.3f * 938.3f / (resE + 938.3f);
                const float rRl = density * tex1D<float>(rRadiationLengthTex, density * params.getRRlScale() + HALF);
                const float thetaSq = eRefSq / (betaP * betaP) * params.getStepLength() * rRl;

                sigmaSq += incScat + incDiv;
                incincScat += 2.0f * thetaSq * params.getStepLength() * params.getStepLength();
                incScat += incincScat;
                incDiv += 2.0f * params.getSigmaSqAirQuad();
            } else {
                if (!suppressDistalSigmaDip) {
                    sigmaSq -= 1.5f * (incScat + incDiv) * density;
                }
            }

            rSigmaEff = HALF * (params.voxelWidth(stepNo).x + params.voxelWidth(stepNo).y) /
                        (sqrt2 * (sqrtf(sigmaSq) + sigmaDelta));

#ifdef DOSE_TO_WATER
            const float mass = (cumulSp - cumulSpOld) * params.stepVol(stepNo);
#else
            const float mass = density * params.stepVol(stepNo);
#endif

            if (mass > 1e-2f) {
#ifdef NUCLEAR_CORR
                if (nuclearEnabled) {
                    const float depthMidIdx = HALF * (cumulSp + cumulSpOld) * params.getEnergyScaleFact() + HALF;
                    const float energyTexIdx = params.getEnergyIdx() + HALF;
                    const float nucWeight = tex2D<float>(nucWeightTex, depthMidIdx, energyTexIdx);
                    res = (1.0f - nucWeight) * rayWeight * (cumulDose - cumulDoseOld) / mass;
                    if (nucIdx >= 0) {
                        nucRes = nucWeight * nucRayWeight * (cumulDose - cumulDoseOld) /
                                 (mass * params.getSpotDist() * params.getSpotDist());
                    }
                } else {
                    res = rayWeight * (cumulDose - cumulDoseOld) / mass;
                }
#else
                res = rayWeight * (cumulDose - cumulDoseOld) / mass;
#endif
            }

#ifdef NUCLEAR_CORR
            if (nuclearEnabled && nucIdx >= 0) {
                const float depthMidIdx = HALF * (cumulSp + cumulSpOld) * params.getEnergyScaleFact() + HALF;
                const float energyTexIdx = params.getEnergyIdx() + HALF;
                const float nucSqSigma = tex2D<float>(nucSqSigmaTex, depthMidIdx, energyTexIdx);
                nucRSigmaEff = HALF * params.getSpotDist() *
                               (params.voxelWidth(stepNo).x + params.voxelWidth(stepNo).y) /
                               (sqrt2 * sqrtf(sigmaSq + nucSqSigma + params.getEntrySigmaSq()));
            }
#endif

            cumulSpOld = cumulSp;
            cumulDoseOld = cumulDose;

            if (cumulSp > params.getRangeStopDepth() || stepNo == afterLast) {
                beamLive = false;
                afterLast = stepNo;
            }
        }

        if (!beamLive || (int)stepNo < ((int)firstIn - 1)) {
            res = 0.0f;
            rSigmaEff = __int_as_float(0x7f800000);
#ifdef NUCLEAR_CORR
            nucRes = 0.0f;
            nucRSigmaEff = __int_as_float(0x7f800000);
#endif
        }

        bevIdd[idx] = res;
        bevRSigmaEff[idx] = rSigmaEff;
#ifdef NUCLEAR_CORR
        if (nuclearEnabled && nucIdx >= 0) {
            bevNucIdd[nucIdx] = nucRes;
            bevNucRSigmaEff[nucIdx] = nucRSigmaEff;
            nucIdx += static_cast<int>(params.getNucMemStep());
        }
#endif

        idx += memStep;
    }

    firstPassive[idx2d] = (int)afterLast;
}

// Legacy simplified kernel (kept for compatibility, but should not be used)
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
) {
    // This kernel is deprecated - use fillIddAndSigmaKernel instead
    // Kept only for backward compatibility
}

// ============================================================================
// Simplified Sigma Texture Calculation
// ============================================================================

// Simplified sigma texture calculation kernel
__global__ void simpleSigmaTextureKernel(
    cudaTextureObject_t subspotData,
    float* sigmaXTexture,
    float* sigmaYTexture,
    vec3f cpbCorner,
    vec3f cpbResolution,
    vec3i cpbDims,
    int layerIdx,
    int layerSize
) {
    int cpbIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalCPBPoints = cpbDims.x * cpbDims.y;
    
    if (cpbIdx >= totalCPBPoints) return;
    
    int cpbX = cpbIdx % cpbDims.x;
    int cpbY = cpbIdx / cpbDims.x;
    
    // Calculate CPB grid point position
    vec3f cpbPos = vec3f(
        cpbCorner.x + (cpbX + 0.5f) * cpbResolution.x,
        cpbCorner.y + (cpbY + 0.5f) * cpbResolution.y,
        cpbCorner.z
    );
    
    float totalWeight = 0.0f;
    float weightedSigmaX = 0.0f;
    float weightedSigmaY = 0.0f;
    
    // Calculate weighted sigma from subspots
    for (int subspotIdx = 0; subspotIdx < layerSize; subspotIdx++) {
        float deltaX = tex3D<float>(subspotData, 0.0f, float(subspotIdx), float(layerIdx));
        float deltaY = tex3D<float>(subspotData, 1.0f, float(subspotIdx), float(layerIdx));
        float weight = tex3D<float>(subspotData, 2.0f, float(subspotIdx), float(layerIdx));
        float sigmaX = tex3D<float>(subspotData, 3.0f, float(subspotIdx), float(layerIdx));
        float sigmaY = tex3D<float>(subspotData, 4.0f, float(subspotIdx), float(layerIdx));
        
        if (weight < 0.001f) continue; // Simple cutoff
        
        // Simple distance-based weighting
        float dx = cpbPos.x - deltaX;
        float dy = cpbPos.y - deltaY;
        float distance = sqrtf(dx * dx + dy * dy);
        float distanceWeight = expf(-distance * distance / 100.0f); // Simplified weighting
        
        float totalSubspotWeight = weight * distanceWeight;
        totalWeight += totalSubspotWeight;
        weightedSigmaX += totalSubspotWeight * sigmaX;
        weightedSigmaY += totalSubspotWeight * sigmaY;
    }
    
    // Calculate weighted average sigma
    if (totalWeight > 0.001f) {
        sigmaXTexture[cpbIdx] = weightedSigmaX / totalWeight;
        sigmaYTexture[cpbIdx] = weightedSigmaY / totalWeight;
    } else {
        sigmaXTexture[cpbIdx] = 0.0f;
        sigmaYTexture[cpbIdx] = 0.0f;
    }
}

// ============================================================================
// Host Functions
// ============================================================================

// Simplified IDD and sigma calculation
void performIddAndSigmaCalculation(
    float* bevDensity,
    float* bevCumulSp,
    float* bevIdd,
    float* bevRSigmaEff,
    float* rayWeights,
    int* firstInside,
    int* firstOutside,
    int* firstPassive,
    int rayDimsX,
    int rayDimsY,
    int steps,
    cudaTextureObject_t cumulIddTex,
    cudaTextureObject_t rRadiationLengthTex
) {
    GPU_TIMER_START();
    
    dim3 blockSize(16, 16);
    dim3 gridSize((rayDimsX + blockSize.x - 1) / blockSize.x,
                  (rayDimsY + blockSize.y - 1) / blockSize.y);
    
    simpleIddCalculationKernel<<<gridSize, blockSize>>>(
        bevDensity, bevCumulSp, bevIdd, bevRSigmaEff, rayWeights,
        firstInside, firstOutside, rayDimsX, rayDimsY, steps,
        cumulIddTex, rRadiationLengthTex
    );
    checkCudaErrors(cudaDeviceSynchronize());
    
    GPU_TIMER_END("Simplified IDD and Sigma Calculation");
}

// Simplified sigma texture calculation
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
) {
    GPU_TIMER_START();
    
    dim3 blockSize(256);
    int totalCPBPoints = cpbDims.x * cpbDims.y;
    dim3 gridSize((totalCPBPoints + blockSize.x - 1) / blockSize.x);
    
    // Calculate sigma texture for each energy layer
    for (int layerIdx = 0; layerIdx < numLayers; layerIdx++) {
        simpleSigmaTextureKernel<<<gridSize, blockSize>>>(
            subspotData, sigmaXTexture, sigmaYTexture,
            cpbCorner, cpbResolution, cpbDims,
            layerIdx, maxSubspotsPerLayer
        );
        checkCudaErrors(cudaDeviceSynchronize());
    }
    
    GPU_TIMER_END("Simplified Sigma Texture Calculation");
}
