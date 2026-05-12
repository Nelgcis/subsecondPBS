/**
 * \file
 * \brief Unified Superposition Algorithm Headers
 * 
 * This file provides unified headers for all superposition-related algorithms
 */

#ifndef SUPERPOSITION_H
#define SUPERPOSITION_H

#include "../core/common.cuh"
#include "../core/Macro.cuh"
#include <cuda_runtime.h>

// ============================================================================
// Function Declarations
// ============================================================================

// Enhanced Superposition Algorithm
void performEnhancedSuperposition(
    float* inDose,
    float* inRSigmaEff,
    float* outDose,
    int inDosePitch,
    int rayDimsX,
    int rayDimsY,
    int numLayers,
    int startZ
);

// Kernel Superposition Algorithm
void performKernelSuperposition(
    float* inDose,
    float* inRSigmaEff,
    float* outDose,
    int inDosePitch,
    int rayDimsX,
    int rayDimsY,
    int radius
);

// Tile-Based Superposition Algorithms (RayTraceDicom-main aligned)
//
// Following upstream kernel_wrapper.cu (lines 985-1108), the primary and nuclear
// branches are SEPARATE functions with branch-specific overflow policies and
// independent counters. This split is intentional and matches RTD-main; the
// duplication is the design, not an oversight.
//
//   performPrimaryTileBasedSuperposition: clamp+warn on overflow.
//     Primary CPB convolution σ may exceed kMaxSuperpR for Carbon-class plans
//     where pre-CPB-convolution effective σ is wider than the proton-era RTD
//     contract assumed. Clamping into the max-radius bucket truncates the
//     extreme tails but keeps the central dose computable. Overflow is logged.
//
//   performNuclearTileBasedSuperposition: throw on overflow (RTD-main contract).
//     Mirrors kernel_wrapper.cu:1004 exactly. Nuclear σ is typically WIDER than
//     primary, so overflow is more likely; silent clamping would systematically
//     truncate the halo lateral umbrella and produce undetected physics error.
//     Fail-fast is the upstream-correct behavior.
//
// Both accept an optional stream parameter so the caller may execute them
// concurrently (each branch has independent buffers; concurrent execution
// requires only that downstream nucTransfDiv/primTransfDiv waits for the
// respective stream to complete).

void performPrimaryTileBasedSuperposition(
    float* devPrimIdd,
    float* devPrimRSigmaEff,
    float* devBevPrimDose,
    int rayDimsX,
    int rayDimsY,
    int steps,
    int beamFirstInside,
    int beamFirstCalculatedPassive,
    cudaStream_t stream = 0
);

void performNuclearTileBasedSuperposition(
    float* devNucIdd,
    float* devNucRSigmaEff,
    float* devBevNucDose,
    int nucRayDimsX,
    int nucRayDimsY,
    int steps,
    int beamFirstInside,
    int beamFirstCalculatedPassive,
    cudaStream_t stream = 0
);

// Helper kernels for reduction
template<typename T, int blockSize>
__global__ void sliceMinVar(T* const devIn, T* const devOut, const int n);

template<typename T, int blockSize>
__global__ void sliceMaxVar(T* const devIn, T* const devOut, const int n);

#endif // SUPERPOSITION_H
