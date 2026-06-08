#include "../include/algorithms/superposition.h"
#include "../include/utils/debug_tools.h"

#include <vector>
#include <iostream>
#include <algorithm>
#include <climits>
#include <string>
#include <cfloat>
#include <cuda_runtime.h>

namespace {
constexpr int kMaxSuperpR = MAX_SUPERP_RADIUS;
constexpr int kSuperpTileX = SUPERP_TILE_X;
constexpr int kSuperpTileY = SUPERP_TILE_Y;
// Reference uses 4 and assumes SUPERP_TILE_Y % blockY == 0
constexpr int kTileRadBlockY = 4;
bool rtdSuperpOverflowDebugEnabled() {
    const char* v = std::getenv("RTD_SUPERP_OVERFLOW_DEBUG");
    if (!v) return false;
    const std::string s(v);
    return !(s == "0" || s == "false" || s == "FALSE" || s == "off" || s == "OFF");
}

bool rtdPerfProfileEnabled() {
    const char* v = std::getenv("RTD_PERF_PROFILE");
    if (!v) return false;
    const std::string s(v);
    return !(s == "0" || s == "false" || s == "FALSE" || s == "off" || s == "OFF");
}

bool rtdHaloAuditEnabled() {
    const char* v = std::getenv("RTD_HALO_AUDIT");
    if (!v) return false;
    const std::string s(v);
    return !(s == "0" || s == "false" || s == "FALSE" || s == "off" || s == "OFF");
}

#define LAUNCH_SUPERP_KERNEL(R) \
    if (batchedPrimTileRadCtrs[R] > 0) { \
        kernelSuperposition<R><<<batchedPrimTileRadCtrs[R], superpBlockDim>>>( \
            devRayIdd, devRayRSigmaEff, devBevPrimDose, rayDimsX, devPrimInOutIdcs, maxNoPrimTiles, devTilePrimRadCtrs); \
    }

struct TileOverflowDebug {
    int z = -1;
    int tileX = -1;
    int tileY = -1;
    float minRSigmaEff = INF;
    float minFiniteRSigmaEff = INF;
    float maxFiniteRSigmaEff = 0.0f;
    float approxMaxSigma = 0.0f;
    float approxMinSigma = 0.0f;
    int finiteCount = 0;
    int infCount = 0;
    int nanCount = 0;
    int belowThresholdCount = 0;
    int positiveIddCount = 0;
    int lowDensityPositiveIddCount = 0;
    int lowMassPositiveIddCount = 0;
    int belowRangeStopPositiveIddCount = 0;
    int aboveRangeStopPositiveIddCount = 0;
    float maxIdd = 0.0f;
    float minCumulSp = FLT_MAX;
    float maxCumulSp = -FLT_MAX;
    float minDensity = FLT_MAX;
    float maxDensity = -FLT_MAX;
    float minMass = FLT_MAX;
    float maxMass = -FLT_MAX;
    float minPositiveIdd = FLT_MAX;
    float minPositiveIddDensity = FLT_MAX;
    float maxPositiveIddDensity = -FLT_MAX;
    float minPositiveIddMass = FLT_MAX;
    float maxPositiveIddMass = -FLT_MAX;
    bool hasTransportContext = false;
    std::string causeHint = "no_transport_context";
};

inline TileOverflowDebug inspectOverflowTile(const float* devRayRSigmaEff,
                                             const float* devRayIdd,
                                             const float* devBevCumulSp,
                                             const float* devBevDensity,
                                             int rayDimsX,
                                             int rayDimsY,
                                             int inIdx,
                                             float threshold,
                                             const SuperpositionDebugContext* debugContext) {
    TileOverflowDebug dbg;
    const int slicePitch = rayDimsX * rayDimsY;
    dbg.z = inIdx / slicePitch;
    const int rem = inIdx - dbg.z * slicePitch;
    const int y0 = rem / rayDimsX;
    const int x0 = rem - y0 * rayDimsX;
    dbg.tileX = x0 / kSuperpTileX;
    dbg.tileY = y0 / kSuperpTileY;

    std::vector<float> sigmaRow(kSuperpTileX, INF);
    std::vector<float> iddRow(kSuperpTileX, 0.0f);
    dbg.hasTransportContext = (devBevCumulSp != nullptr &&
                               devBevDensity != nullptr &&
                               debugContext != nullptr &&
                               debugContext->originalRayDimsX > 0 &&
                               debugContext->originalRayDimsY > 0);
    for (int row = 0; row < kSuperpTileY; ++row) {
        const int rowIdx = dbg.z * slicePitch + (y0 + row) * rayDimsX + x0;
        checkCudaErrors(cudaMemcpy(sigmaRow.data(),
                                   devRayRSigmaEff + rowIdx,
                                   kSuperpTileX * sizeof(float),
                                   cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(iddRow.data(),
                                   devRayIdd + rowIdx,
                                   kSuperpTileX * sizeof(float),
                                   cudaMemcpyDeviceToHost));
        for (int col = 0; col < kSuperpTileX; ++col) {
            const float sig = sigmaRow[col];
            const float idd = iddRow[col];
            float cumulSp = 0.0f;
            float density = 0.0f;
            float mass = 0.0f;
            const int rayX = x0 + col;
            const int rayY = y0 + row;
            const bool hasOriginalRay =
                dbg.hasTransportContext &&
                rayX < debugContext->originalRayDimsX &&
                rayY < debugContext->originalRayDimsY;
            if (hasOriginalRay) {
                const int originalSlicePitch =
                    debugContext->originalRayDimsX * debugContext->originalRayDimsY;
                const int originalIdx =
                    dbg.z * originalSlicePitch +
                    rayY * debugContext->originalRayDimsX +
                    rayX;
                checkCudaErrors(cudaMemcpy(&cumulSp,
                                           devBevCumulSp + originalIdx,
                                           sizeof(float),
                                           cudaMemcpyDeviceToHost));
                checkCudaErrors(cudaMemcpy(&density,
                                           devBevDensity + originalIdx,
                                           sizeof(float),
                                           cudaMemcpyDeviceToHost));
                mass = density * (debugContext->stepVolConst +
                                  float(dbg.z) * debugContext->stepVolLin +
                                  float(dbg.z) * float(dbg.z) * debugContext->stepVolSq);
                if (std::isfinite(cumulSp)) {
                    if (cumulSp < dbg.minCumulSp) dbg.minCumulSp = cumulSp;
                    if (cumulSp > dbg.maxCumulSp) dbg.maxCumulSp = cumulSp;
                }
                if (std::isfinite(density)) {
                    if (density < dbg.minDensity) dbg.minDensity = density;
                    if (density > dbg.maxDensity) dbg.maxDensity = density;
                }
                if (std::isfinite(mass)) {
                    if (mass < dbg.minMass) dbg.minMass = mass;
                    if (mass > dbg.maxMass) dbg.maxMass = mass;
                }
            }
            if (sig < dbg.minRSigmaEff) dbg.minRSigmaEff = sig;
            if (std::isnan(sig)) {
                dbg.nanCount++;
            } else if (!std::isfinite(sig)) {
                dbg.infCount++;
            } else {
                dbg.finiteCount++;
                if (sig < dbg.minFiniteRSigmaEff) dbg.minFiniteRSigmaEff = sig;
                if (sig > dbg.maxFiniteRSigmaEff) dbg.maxFiniteRSigmaEff = sig;
                if (sig < threshold) dbg.belowThresholdCount++;
            }
            if (idd > 0.0f) {
                dbg.positiveIddCount++;
                if (idd < dbg.minPositiveIdd) dbg.minPositiveIdd = idd;
                if (idd > dbg.maxIdd) dbg.maxIdd = idd;
                if (hasOriginalRay) {
                    if (density < dbg.minPositiveIddDensity) dbg.minPositiveIddDensity = density;
                    if (density > dbg.maxPositiveIddDensity) dbg.maxPositiveIddDensity = density;
                    if (mass < dbg.minPositiveIddMass) dbg.minPositiveIddMass = mass;
                    if (mass > dbg.maxPositiveIddMass) dbg.maxPositiveIddMass = mass;
                    if (density <= 1.0e-4f) dbg.lowDensityPositiveIddCount++;
                    if (mass <= 1.0e-2f) dbg.lowMassPositiveIddCount++;
                    if (cumulSp <= debugContext->rangeStopDepthMm) {
                        dbg.belowRangeStopPositiveIddCount++;
                    } else {
                        dbg.aboveRangeStopPositiveIddCount++;
                    }
                }
            }
        }
    }
    if (dbg.minFiniteRSigmaEff == INF) dbg.minFiniteRSigmaEff = 0.0f;
    if (dbg.minFiniteRSigmaEff > 0.0f) {
        dbg.approxMaxSigma = rsqrtf(2.0f) / dbg.minFiniteRSigmaEff;
    }
    if (dbg.maxFiniteRSigmaEff > 0.0f) {
        dbg.approxMinSigma = rsqrtf(2.0f) / dbg.maxFiniteRSigmaEff;
    }
    if (dbg.minCumulSp == FLT_MAX) dbg.minCumulSp = 0.0f;
    if (dbg.maxCumulSp == -FLT_MAX) dbg.maxCumulSp = 0.0f;
    if (dbg.minDensity == FLT_MAX) dbg.minDensity = 0.0f;
    if (dbg.maxDensity == -FLT_MAX) dbg.maxDensity = 0.0f;
    if (dbg.minMass == FLT_MAX) dbg.minMass = 0.0f;
    if (dbg.maxMass == -FLT_MAX) dbg.maxMass = 0.0f;
    if (dbg.minPositiveIdd == FLT_MAX) dbg.minPositiveIdd = 0.0f;
    if (dbg.minPositiveIddDensity == FLT_MAX) dbg.minPositiveIddDensity = 0.0f;
    if (dbg.maxPositiveIddDensity == -FLT_MAX) dbg.maxPositiveIddDensity = 0.0f;
    if (dbg.minPositiveIddMass == FLT_MAX) dbg.minPositiveIddMass = 0.0f;
    if (dbg.maxPositiveIddMass == -FLT_MAX) dbg.maxPositiveIddMass = 0.0f;
    if (dbg.hasTransportContext) {
        if (dbg.positiveIddCount == 0) {
            dbg.causeHint = "zero_idd_tile";
        } else if (dbg.aboveRangeStopPositiveIddCount > 0) {
            dbg.causeHint = "range_stop_mismatch";
        } else if (dbg.lowMassPositiveIddCount > 0) {
            dbg.causeHint = "mass_amplified_tail";
        } else if (dbg.lowDensityPositiveIddCount > 0) {
            dbg.causeHint = "low_density_tail";
        } else {
            dbg.causeHint = "positive_idd_tail";
        }
    }
    return dbg;
}
} // namespace

// -----------------------------------------------------------------------------
// sliceMinVar / sliceMaxVar
// -----------------------------------------------------------------------------
// NOTE: The reduction is applied per Z-slice. Each blockIdx.z selects one slice.
template<typename T, int blockSize>
__global__ void sliceMinVar(T* const devIn, T* const devOut, const int n) {
    __shared__ T sdata[blockSize];

    const int tid = threadIdx.x;
    const int base = n * blockIdx.z;  // offset to this slice
    int idx = base + tid;

    // Initialize with a large value
    T myMin = (T)INT_MAX;

    // Stride across the slice
    while (idx < base + n) {
        T v = devIn[idx];
        if (v < myMin) myMin = v;
        idx += blockSize;
    }

    sdata[tid] = myMin;
    __syncthreads();

    // Reduce within the block
    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            T other = sdata[tid + s];
            if (other < sdata[tid]) sdata[tid] = other;
        }
        __syncthreads();
    }

    if (tid == 0) {
        devOut[blockIdx.z] = sdata[0];
    }
}

template<typename T, int blockSize>
__global__ void sliceMaxVar(T* const devIn, T* const devOut, const int n) {
    __shared__ T sdata[blockSize];

    const int tid = threadIdx.x;
    const int base = n * blockIdx.z;  // offset to this slice
    int idx = base + tid;

    // Initialize with a small value
    T myMax = (T)INT_MIN;

    // Stride across the slice
    while (idx < base + n) {
        T v = devIn[idx];
        if (v > myMax) myMax = v;
        idx += blockSize;
    }

    sdata[tid] = myMax;
    __syncthreads();

    // Reduce within the block
    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            T other = sdata[tid + s];
            if (other > sdata[tid]) sdata[tid] = other;
        }
        __syncthreads();
    }

    if (tid == 0) {
        devOut[blockIdx.z] = sdata[0];
    }
}

// Explicit template instantiations for int with blockSize=1024
template __global__ void sliceMinVar<int, 1024>(int* const, int* const, const int);
template __global__ void sliceMaxVar<int, 1024>(int* const, int* const, const int);
// Explicit template instantiation needed for WEPL min-reduction (per-step)
template __global__ void sliceMinVar<float, 128>(float* const, float* const, const int);


// -----------------------------------------------------------------------------
// Complete Tile-Based Superposition (RayTraceDicom reference implementation)
// -----------------------------------------------------------------------------
// IMPORTANT: This implementation assumes:
//   - rayDimsX % SUPERP_TILE_X == 0
//   - rayDimsY % SUPERP_TILE_Y == 0
// The wrapper pads (IDD, rSigmaEff) arrays accordingly.
//
// tileRadCalc computes, for each (tile, z), the required radius bucket and stores
// (inIdx, outIdx) pairs in inOutIdcs arranged as [rad][tileIdxWithinRad].
// -----------------------------------------------------------------------------
template<unsigned int blockY>
__global__ void tileRadCalc(float const* __restrict__ devIn,
                            const int startZ,
                            int* const tilePrimRadCtrs,
                            int2* const inOutIdcs,
                            const int noTiles) {
    __shared__ float tile[kSuperpTileX * blockY];

    const int tileIdx = kSuperpTileX * threadIdx.y + threadIdx.x;
    const int pitch = gridDim.x * kSuperpTileX;
    const int inIdx =
        (startZ + blockIdx.z) * (gridDim.y * kSuperpTileY * pitch) +
        (blockIdx.y * kSuperpTileY + threadIdx.y) * pitch +
        blockIdx.x * kSuperpTileX + threadIdx.x;

    // One block per tile. Each thread scans superpTileY/blockY rows and reduces.
    float minVal = devIn[inIdx];
#pragma unroll
    for (int i = 1; i < (kSuperpTileY / blockY); ++i) {
        const int idx = inIdx + i * blockY * pitch;
        const float testVal = devIn[idx];
        if (testVal < minVal) { minVal = testVal; }
    }
    tile[tileIdx] = minVal;
    __syncthreads();

    // Reduction over y
    for (int maxIdxY = blockY / 2; maxIdxY > 0; maxIdxY >>= 1) {
        if (threadIdx.y < maxIdxY && tile[tileIdx + maxIdxY * kSuperpTileX] < tile[tileIdx]) {
            tile[tileIdx] = tile[tileIdx + maxIdxY * kSuperpTileX];
        }
        __syncthreads();
    }

    // Reduction over x (only y==0 threads participate)
    if (threadIdx.y == 0) {
        for (int maxIdxX = kSuperpTileX / 2; maxIdxX > 0; maxIdxX >>= 1) {
            if (threadIdx.x < maxIdxX && tile[threadIdx.x + maxIdxX] < tile[threadIdx.x]) {
                tile[threadIdx.x] = tile[threadIdx.x + maxIdxX];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            // Calc rad, atomically increment corresponding counter, and write indices and pitches
            int rad = min(int(KS_SIGMA_CUTOFF / (sqrtf(2.0f) * tile[0]) + HALF), kMaxSuperpR + 1);
            int radIdx = atomicAdd(tilePrimRadCtrs + rad, 1);

            const int outIdx =
                (startZ + blockIdx.z) *
                    ((gridDim.y * kSuperpTileY + 2 * kMaxSuperpR) *
                     (gridDim.x * kSuperpTileX + 2 * kMaxSuperpR)) +
                (blockIdx.y * kSuperpTileY) * (gridDim.x * kSuperpTileX + 2 * kMaxSuperpR) +
                blockIdx.x * kSuperpTileX;

            inOutIdcs[rad * noTiles + radIdx] = make_int2(inIdx, outIdx);
        }
    }
}

template<unsigned int blockY>
__global__ void tileRadCalcDoseGated(float const* __restrict__ devDose,
                                     float const* __restrict__ devRSigmaEff,
                                     const int startZ,
                                     int* const tilePrimRadCtrs,
                                     int2* const inOutIdcs,
                                     const int noTiles) {
    __shared__ float tileMin[kSuperpTileX * blockY];
    __shared__ int tileHasDose[kSuperpTileX * blockY];

    const int tileIdx = kSuperpTileX * threadIdx.y + threadIdx.x;
    const int pitch = gridDim.x * kSuperpTileX;
    const int inIdx =
        (startZ + blockIdx.z) * (gridDim.y * kSuperpTileY * pitch) +
        (blockIdx.y * kSuperpTileY + threadIdx.y) * pitch +
        blockIdx.x * kSuperpTileX + threadIdx.x;

    float minVal = INF;
    int hasDose = 0;
#pragma unroll
    for (int i = 0; i < (kSuperpTileY / blockY); ++i) {
        const int idx = inIdx + i * blockY * pitch;
        const float dose = devDose[idx];
        const float testVal = devRSigmaEff[idx];
        if (dose > 0.0f && testVal > 0.0f && !isnan(testVal)) {
            hasDose = 1;
            if (testVal < minVal) { minVal = testVal; }
        }
    }
    tileMin[tileIdx] = minVal;
    tileHasDose[tileIdx] = hasDose;
    __syncthreads();

    for (int maxIdxY = blockY / 2; maxIdxY > 0; maxIdxY >>= 1) {
        if (threadIdx.y < maxIdxY) {
            const int otherIdx = tileIdx + maxIdxY * kSuperpTileX;
            tileHasDose[tileIdx] |= tileHasDose[otherIdx];
            if (tileMin[otherIdx] < tileMin[tileIdx]) {
                tileMin[tileIdx] = tileMin[otherIdx];
            }
        }
        __syncthreads();
    }

    if (threadIdx.y == 0) {
        for (int maxIdxX = kSuperpTileX / 2; maxIdxX > 0; maxIdxX >>= 1) {
            if (threadIdx.x < maxIdxX) {
                const int otherIdx = threadIdx.x + maxIdxX;
                tileHasDose[threadIdx.x] |= tileHasDose[otherIdx];
                if (tileMin[otherIdx] < tileMin[threadIdx.x]) {
                    tileMin[threadIdx.x] = tileMin[otherIdx];
                }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0 && tileHasDose[0]) {
            const int rad = min(int(KS_SIGMA_CUTOFF / (sqrtf(2.0f) * tileMin[0]) + HALF), kMaxSuperpR + 1);
            const int radIdx = atomicAdd(tilePrimRadCtrs + rad, 1);

            const int outIdx =
                (startZ + blockIdx.z) *
                    ((gridDim.y * kSuperpTileY + 2 * kMaxSuperpR) *
                     (gridDim.x * kSuperpTileX + 2 * kMaxSuperpR)) +
                (blockIdx.y * kSuperpTileY) * (gridDim.x * kSuperpTileX + 2 * kMaxSuperpR) +
                blockIdx.x * kSuperpTileX;

            inOutIdcs[rad * noTiles + radIdx] = make_int2(inIdx, outIdx);
        }
    }
}

/**
 * \brief Tile-based kernel superposition (RayTraceDicom reference implementation)
 *
 * NOTE:
 *   - blockDim.x must be SUPERP_TILE_X
 *   - blockDim.y must divide SUPERP_TILE_Y (we use SUPERP_TILE_Y)
 */
template <int rad>
__global__ void kernelSuperposition(float const* __restrict__ inDose,
                                    float const* __restrict__ inRSigmaEff,
                                    float* const outDose,
                                    const int inDosePitch,
                                    int2* const inOutIdcs,
                                    const int inOutIdxPitch,
                                    int* const tileCtrs) {
    volatile __shared__ float tile[(kSuperpTileX + 2 * rad) * (kSuperpTileY + 2 * rad)];

    // Clear shared tile buffer
    for (int i = threadIdx.y * blockDim.x + threadIdx.x;
         i < (kSuperpTileY + 2 * rad) * (kSuperpTileX + 2 * rad);
         i += blockDim.x * blockDim.y) {
        tile[i] = 0.0f;
    }
    __syncthreads();

    int tileIdx = blockIdx.x;
    int radIdx = rad;
    while (tileIdx >= tileCtrs[radIdx]) {
        tileIdx -= tileCtrs[radIdx];
        radIdx -= 1;
    }

    // __syncthreads inside this block requires that blockDim.y divides superpTileY
    for (int row = threadIdx.y; row < kSuperpTileY; row += blockDim.y) {
        int inIdx = inOutIdcs[radIdx * inOutIdxPitch + tileIdx].x +
                    row * inDosePitch + threadIdx.x;

        float dose = inDose[inIdx];

        if (__syncthreads_or(dose > 0.0f)) {
            float rSigmaEff = inRSigmaEff[inIdx];
            float erfNew = erff(rSigmaEff * HALF);
            float erfOld = -erfNew;
            float erfDiffs[rad + 1];

#pragma unroll
            for (int i = 0; i <= rad; ++i) {
                erfDiffs[i] = HALF * (erfNew - erfOld);
                erfOld = erfNew;
                erfNew = erff(rSigmaEff * (float(i) + 1.5f));
            }

#pragma unroll
            for (int i = 0; i < 2 * rad + 1; ++i) {
#pragma unroll
                for (int j = 0; j < 2 * rad + 1; ++j) {
                    tile[(row + i) * (kSuperpTileX + 2 * rad) + threadIdx.x + j] +=
                        dose * erfDiffs[abs(rad - i)] * erfDiffs[abs(rad - j)];
                }
                __syncthreads(); // Must not be inside a conditional branch
            }
        }
    }

    for (int row = threadIdx.y - rad + kMaxSuperpR;
         row < kSuperpTileY + rad + kMaxSuperpR;
         row += blockDim.y) {
        for (int col = threadIdx.x - rad + kMaxSuperpR;
             col < kSuperpTileX + rad + kMaxSuperpR;
             col += kSuperpTileX) {
            int outPitch = inDosePitch + 2 * kMaxSuperpR;
            int outIdx = inOutIdcs[radIdx * inOutIdxPitch + tileIdx].y +
                         row * outPitch + col;
            atomicAdd(outDose + outIdx,
                      tile[(row + rad - kMaxSuperpR) * (kSuperpTileX + 2 * rad) +
                           col + rad - kMaxSuperpR]);
        }
    }
}

// =============================================================================
// Tile-based superposition: PRIMARY branch
// =============================================================================
// Mirrors RayTraceDicom-main kernel_wrapper.cu lines ~985-1075 (primary path).
// Overflow policy: clamp + warn (Carbon-class plans may exceed kMaxSuperpR for
// pre-CPB-convolution effective sigma; truncating tails is acceptable for
// primary because primary sigma is much narrower than nuclear sigma in physical
// space).
//
// Local variables intentionally named generically (devIdd, devTileRadCtrs, etc.)
// so the LAUNCH_SUPERP_KERNEL macro works without naming churn. The outer name
// `Primary` lives at the API boundary; inside the function body, this is just a
// superposition over a per-branch buffer set.

#undef LAUNCH_SUPERP_KERNEL
#define LAUNCH_SUPERP_KERNEL(R) \
    if (batchedTileRadCtrs[R] > 0) { \
        kernelSuperposition<R><<<batchedTileRadCtrs[R], superpBlockDim, 0, stream>>>( \
            devIdd, devRSigmaEff, devBevDose, rayDimsX, devInOutIdcs, maxNoTiles, devTileRadCtrs); \
    }

void performPrimaryTileBasedSuperposition(float* devIdd,
                                          float* devRSigmaEff,
                                          float* devBevDose,
                                          int rayDimsX,
                                          int rayDimsY,
                                          int steps,
                                          int beamFirstInside,
                                          int beamFirstCalculatedPassive,
                                          cudaStream_t stream,
                                          const SuperpositionDebugContext* debugContext) {
    GPU_TIMER_START();
    (void)steps;

    if (beamFirstCalculatedPassive <= beamFirstInside) {
        std::cerr << "Warning: beamFirstCalculatedPassive <= beamFirstInside, skipping primary superposition\n";
        return;
    }

    if (rayDimsX % kSuperpTileX != 0 || rayDimsY % kSuperpTileY != 0) {
        std::cerr << "Error: Primary ray dimensions must be multiples of superposition tile size.\n"
                  << "  rayDims: (" << rayDimsX << ", " << rayDimsY << ")\n"
                  << "  tile:    (" << kSuperpTileX << ", " << kSuperpTileY << ")\n";
        return;
    }

    const int numTilesX = rayDimsX / kSuperpTileX;
    const int numTilesY = rayDimsY / kSuperpTileY;
    const int numStepsInside = beamFirstCalculatedPassive - beamFirstInside;
    const int maxNoTiles = numStepsInside * numTilesX * numTilesY;

    int* devTileRadCtrs = nullptr;
    int2* devInOutIdcs = nullptr;
    checkCudaErrors(cudaMalloc(&devTileRadCtrs, (kMaxSuperpR + 2) * sizeof(int)));
    checkCudaErrors(cudaMalloc(&devInOutIdcs, (kMaxSuperpR + 2) * maxNoTiles * sizeof(int2)));
    checkCudaErrors(cudaMemsetAsync(devTileRadCtrs, 0, (kMaxSuperpR + 2) * sizeof(int), stream));

    const dim3 tileRadGridDim(numTilesX, numTilesY, numStepsInside);
    const dim3 tileRadBlockDim(kSuperpTileX, kTileRadBlockY);

    tileRadCalcDoseGated<kTileRadBlockY><<<tileRadGridDim, tileRadBlockDim, 0, stream>>>(
        devIdd, devRSigmaEff, beamFirstInside, devTileRadCtrs, devInOutIdcs, maxNoTiles);
    checkCudaErrors(cudaStreamSynchronize(stream));

    std::vector<int> tileRadCtrs(kMaxSuperpR + 2, 0);
    checkCudaErrors(cudaMemcpyAsync(tileRadCtrs.data(),
                                    devTileRadCtrs,
                                    (kMaxSuperpR + 2) * sizeof(int),
                                    cudaMemcpyDeviceToHost,
                                    stream));
    checkCudaErrors(cudaStreamSynchronize(stream));

    // [PRIMARY OVERFLOW POLICY] Clamp + warn.
    // Carbon-class primary plans can have pre-CPB effective sigma wider than
    // the proton-era RTD radius assumption. Clamp into the max-radius bucket
    // and continue; document the truncation. See halo-energy-conservation.md
    // (RC5) for why nuclear must use a different policy.
    if (tileRadCtrs[kMaxSuperpR + 1] > 0) {
        const int overflowCount = tileRadCtrs[kMaxSuperpR + 1];
        const int maxBucketBase = tileRadCtrs[kMaxSuperpR];
        const float overflowThreshold = KS_SIGMA_CUTOFF / (sqrtf(2.0f) * (float(kMaxSuperpR) + 0.5f));

        std::cerr << "Warning: Primary superposition required radius larger than max radius; clamping to max radius\n"
                  << "  maxRadius=" << kMaxSuperpR
                  << " overflowTiles=" << overflowCount
                  << " threshold_rSigmaEff<" << overflowThreshold
                  << std::endl;
        if (debugContext != nullptr) {
            std::cerr << "  layer=" << debugContext->layerIdx << "/" << debugContext->numLayers
                      << " energy=" << debugContext->energy
                      << " energyIdx=" << debugContext->energyIdx
                      << " peakDepthMm=" << debugContext->peakDepthMm
                      << " layerCutoffMm=" << debugContext->layerCutoffMm
                      << std::endl;
            std::cerr << "  activeSteps=[" << beamFirstInside << "," << (beamFirstCalculatedPassive - 1) << "]"
                      << " originalRayDims=(" << debugContext->originalRayDimsX << "," << debugContext->originalRayDimsY << ")"
                      << " superpRayDims=(" << debugContext->superpRayDimsX << "," << debugContext->superpRayDimsY << ")"
                      << std::endl;
            std::cerr << "  meanVoxelWidthMm(first,mid,last)=("
                      << debugContext->meanVoxelWidthFirstMm << ","
                      << debugContext->meanVoxelWidthMidMm << ","
                      << debugContext->meanVoxelWidthLastMm << ")"
                      << " sigmaAtRad32Mm(first,mid,last)=("
                      << debugContext->sigmaAtRad32FirstMm << ","
                      << debugContext->sigmaAtRad32MidMm << ","
                      << debugContext->sigmaAtRad32LastMm << ")"
                      << std::endl;
            std::cerr << "  spotSigmaMm=(" << debugContext->spotSigmaXmm << ","
                      << debugContext->spotSigmaYmm << ")"
                      << " profileRowIdx=" << debugContext->profileRowIdx
                      << std::endl;
        }
        const int sampleCount = rtdSuperpOverflowDebugEnabled() ? std::min(overflowCount, 4) : 0;
        if (sampleCount > 0) {
            std::vector<int2> overflowSamples(sampleCount);
            checkCudaErrors(cudaMemcpy(overflowSamples.data(),
                                       devInOutIdcs + (kMaxSuperpR + 1) * maxNoTiles,
                                       sampleCount * sizeof(int2),
                                       cudaMemcpyDeviceToHost));
            for (int i = 0; i < sampleCount; ++i) {
                const TileOverflowDebug dbg = inspectOverflowTile(
                    devRSigmaEff,
                    devIdd,
                    debugContext != nullptr ? debugContext->devBevCumulSp : nullptr,
                    debugContext != nullptr ? debugContext->devBevDensity : nullptr,
                    rayDimsX,
                    rayDimsY,
                    overflowSamples[i].x,
                    overflowThreshold,
                    debugContext);
                std::cerr << "  overflowSample[" << i << "]"
                          << " z=" << dbg.z
                          << " tile=(" << dbg.tileX << "," << dbg.tileY << ")"
                          << " minRSigmaEff=" << dbg.minRSigmaEff
                          << " minFiniteRSigmaEff=" << dbg.minFiniteRSigmaEff
                          << " maxFiniteRSigmaEff=" << dbg.maxFiniteRSigmaEff
                          << " approxSigmaRange=[" << dbg.approxMinSigma << "," << dbg.approxMaxSigma << "]"
                          << " finite=" << dbg.finiteCount
                          << " inf=" << dbg.infCount
                          << " nan=" << dbg.nanCount
                          << " belowThreshold=" << dbg.belowThresholdCount
                          << " positiveIdd=" << dbg.positiveIddCount
                          << " maxIdd=" << dbg.maxIdd
                          << std::endl;
                if (dbg.hasTransportContext) {
                    std::cerr << "    transport"
                              << " cumulSpRange=[" << dbg.minCumulSp << "," << dbg.maxCumulSp << "]"
                              << " densityRange=[" << dbg.minDensity << "," << dbg.maxDensity << "]"
                              << " massRange=[" << dbg.minMass << "," << dbg.maxMass << "]"
                              << " rangeStopDepth=" << debugContext->rangeStopDepthMm
                              << std::endl;
                    std::cerr << "    positiveIddTransport"
                              << " iddRange=[" << dbg.minPositiveIdd << "," << dbg.maxIdd << "]"
                              << " densityRange=[" << dbg.minPositiveIddDensity << "," << dbg.maxPositiveIddDensity << "]"
                              << " massRange=[" << dbg.minPositiveIddMass << "," << dbg.maxPositiveIddMass << "]"
                              << " lowDensityPositiveIdd=" << dbg.lowDensityPositiveIddCount
                              << " lowMassPositiveIdd=" << dbg.lowMassPositiveIddCount
                              << " belowRangeStopPositiveIdd=" << dbg.belowRangeStopPositiveIddCount
                              << " aboveRangeStopPositiveIdd=" << dbg.aboveRangeStopPositiveIddCount
                              << " causeHint=" << dbg.causeHint
                              << std::endl;
                }
            }
        }

        checkCudaErrors(cudaMemcpyAsync(
            devInOutIdcs + kMaxSuperpR * maxNoTiles + maxBucketBase,
            devInOutIdcs + (kMaxSuperpR + 1) * maxNoTiles,
            overflowCount * sizeof(int2),
            cudaMemcpyDeviceToDevice,
            stream));
        tileRadCtrs[kMaxSuperpR] += overflowCount;
        tileRadCtrs[kMaxSuperpR + 1] = 0;
        checkCudaErrors(cudaMemcpyAsync(devTileRadCtrs,
                                        tileRadCtrs.data(),
                                        (kMaxSuperpR + 2) * sizeof(int),
                                        cudaMemcpyHostToDevice,
                                        stream));
    }

    int layerMaxSuperpR = 0;
    for (int i = 0; i < kMaxSuperpR + 2; ++i) {
        if (tileRadCtrs[i] > 0) { layerMaxSuperpR = i; }
    }

    int totalTiles = 0;
    for (int rad = 0; rad <= kMaxSuperpR; ++rad) totalTiles += tileRadCtrs[rad];
    const int skippedNoDoseTiles = maxNoTiles - totalTiles;

    if (rtdVerboseFineTiming()) {
        std::cout << "  Primary superposition tile summary: totalTiles=" << totalTiles
                  << ", skippedNoDoseTiles=" << skippedNoDoseTiles
                  << ", layerMaxSuperpR=" << layerMaxSuperpR
                  << ", rad0=" << tileRadCtrs[0]
                  << ", rad1=" << tileRadCtrs[1]
                  << ", rad2=" << tileRadCtrs[2]
                  << std::endl;
    }
    if (rtdPerfProfileEnabled()) {
        std::cout << "[PERF_SUPERP] layer="
                  << (debugContext != nullptr ? debugContext->layerIdx : -1)
                  << " totalTiles=" << totalTiles
                  << " skippedNoDoseTiles=" << skippedNoDoseTiles
                  << " layerMaxSuperpR=" << layerMaxSuperpR
                  << " rad0=" << tileRadCtrs[0]
                  << " rad1=" << tileRadCtrs[1]
                  << " rad2=" << tileRadCtrs[2]
                  << " rad4=" << tileRadCtrs[4]
                  << " rad8=" << tileRadCtrs[8]
                  << " rad16=" << tileRadCtrs[16]
                  << " rad24=" << tileRadCtrs[24]
                  << " rad32=" << tileRadCtrs[32]
                  << std::endl;
    }

    std::vector<int> batchedTileRadCtrs(kMaxSuperpR + 1, 0);
    batchedTileRadCtrs[0] = tileRadCtrs[0];
    int recRad = layerMaxSuperpR;
    for (int rad = layerMaxSuperpR; rad > 0; --rad) {
        batchedTileRadCtrs[recRad] += tileRadCtrs[rad];
        if (batchedTileRadCtrs[recRad] >= MIN_TILES_IN_BATCH) {
            recRad = rad - 1;
        }
    }

    if (rtdVerboseFineTiming()) {
        std::cout << "  Primary superposition batched launches:";
        for (int r = 0; r <= layerMaxSuperpR; ++r) {
            if (batchedTileRadCtrs[r] > 0) {
                std::cout << " [rad=" << r << ":" << batchedTileRadCtrs[r] << "]";
            }
        }
        std::cout << std::endl;
    }

    const dim3 superpBlockDim(kSuperpTileX, kSuperpTileY);

    LAUNCH_SUPERP_KERNEL(0);
    LAUNCH_SUPERP_KERNEL(1);
    LAUNCH_SUPERP_KERNEL(2);
    LAUNCH_SUPERP_KERNEL(3);
    LAUNCH_SUPERP_KERNEL(4);
    LAUNCH_SUPERP_KERNEL(5);
    LAUNCH_SUPERP_KERNEL(6);
    LAUNCH_SUPERP_KERNEL(7);
    LAUNCH_SUPERP_KERNEL(8);
    LAUNCH_SUPERP_KERNEL(9);
    LAUNCH_SUPERP_KERNEL(10);
    LAUNCH_SUPERP_KERNEL(11);
    LAUNCH_SUPERP_KERNEL(12);
    LAUNCH_SUPERP_KERNEL(13);
    LAUNCH_SUPERP_KERNEL(14);
    LAUNCH_SUPERP_KERNEL(15);
    LAUNCH_SUPERP_KERNEL(16);
    LAUNCH_SUPERP_KERNEL(17);
    LAUNCH_SUPERP_KERNEL(18);
    LAUNCH_SUPERP_KERNEL(19);
    LAUNCH_SUPERP_KERNEL(20);
    LAUNCH_SUPERP_KERNEL(21);
    LAUNCH_SUPERP_KERNEL(22);
    LAUNCH_SUPERP_KERNEL(23);
    LAUNCH_SUPERP_KERNEL(24);
    LAUNCH_SUPERP_KERNEL(25);
    LAUNCH_SUPERP_KERNEL(26);
    LAUNCH_SUPERP_KERNEL(27);
    LAUNCH_SUPERP_KERNEL(28);
    LAUNCH_SUPERP_KERNEL(29);
    LAUNCH_SUPERP_KERNEL(30);
    LAUNCH_SUPERP_KERNEL(31);
    LAUNCH_SUPERP_KERNEL(32);

    checkCudaErrors(cudaStreamSynchronize(stream));

    checkCudaErrors(cudaFree(devTileRadCtrs));
    checkCudaErrors(cudaFree(devInOutIdcs));

    GPU_TIMER_END("Primary Tile-Based Superposition");
}

// =============================================================================
// Tile-based superposition: NUCLEAR branch
// =============================================================================
// Mirrors RayTraceDicom-main kernel_wrapper.cu lines ~997-1108 (nuclear path).
// Overflow policy: THROW (matches kernel_wrapper.cu:1004 exactly).
//   if (tileNucRadCtrs[maxSuperpR+1] > 0) { throw("Found larger than allowed
//                                                  kernel superposition radius"); }
// Nuclear sigma is typically WIDER than primary; silent clamping would
// systematically truncate the halo lateral umbrella. Fail-fast forces the
// caller to address the lattice geometry (canonical PB grid via 9.23) or
// raise MAX_SUPERP_RADIUS rather than mask the physics error.

void performNuclearTileBasedSuperposition(float* devIdd,
                                          float* devRSigmaEff,
                                          float* devBevDose,
                                          int rayDimsX,
                                          int rayDimsY,
                                          int steps,
                                          int beamFirstInside,
                                          int beamFirstCalculatedPassive,
                                          cudaStream_t stream) {
    GPU_TIMER_START();
    (void)steps;

    if (beamFirstCalculatedPassive <= beamFirstInside) {
        std::cerr << "Warning: beamFirstCalculatedPassive <= beamFirstInside, skipping nuclear superposition\n";
        return;
    }

    if (rayDimsX % kSuperpTileX != 0 || rayDimsY % kSuperpTileY != 0) {
        throw std::runtime_error(
            "[9.44] Nuclear ray dimensions must be multiples of superposition tile size; got rayDims=(" +
            std::to_string(rayDimsX) + "," + std::to_string(rayDimsY) + ") tile=(" +
            std::to_string(kSuperpTileX) + "," + std::to_string(kSuperpTileY) + ")");
    }

    const int numTilesX = rayDimsX / kSuperpTileX;
    const int numTilesY = rayDimsY / kSuperpTileY;
    const int numStepsInside = beamFirstCalculatedPassive - beamFirstInside;
    const int maxNoTiles = numStepsInside * numTilesX * numTilesY;

    int* devTileRadCtrs = nullptr;
    int2* devInOutIdcs = nullptr;
    checkCudaErrors(cudaMalloc(&devTileRadCtrs, (kMaxSuperpR + 2) * sizeof(int)));
    checkCudaErrors(cudaMalloc(&devInOutIdcs, (kMaxSuperpR + 2) * maxNoTiles * sizeof(int2)));
    checkCudaErrors(cudaMemsetAsync(devTileRadCtrs, 0, (kMaxSuperpR + 2) * sizeof(int), stream));

    const dim3 tileRadGridDim(numTilesX, numTilesY, numStepsInside);
    const dim3 tileRadBlockDim(kSuperpTileX, kTileRadBlockY);

    tileRadCalc<kTileRadBlockY><<<tileRadGridDim, tileRadBlockDim, 0, stream>>>(
        devRSigmaEff, beamFirstInside, devTileRadCtrs, devInOutIdcs, maxNoTiles);
    checkCudaErrors(cudaStreamSynchronize(stream));

    std::vector<int> tileRadCtrs(kMaxSuperpR + 2, 0);
    checkCudaErrors(cudaMemcpyAsync(tileRadCtrs.data(),
                                    devTileRadCtrs,
                                    (kMaxSuperpR + 2) * sizeof(int),
                                    cudaMemcpyDeviceToHost,
                                    stream));
    checkCudaErrors(cudaStreamSynchronize(stream));

    // [9.44 NUCLEAR OVERFLOW POLICY] Strict throw, matches RTD-main exactly.
    if (tileRadCtrs[kMaxSuperpR + 1] > 0) {
        const int overflowCount = tileRadCtrs[kMaxSuperpR + 1];
        const float overflowThreshold = KS_SIGMA_CUTOFF / (sqrtf(2.0f) * (float(kMaxSuperpR) + 0.5f));

        // Free local allocations before throwing so they don't leak.
        cudaFree(devTileRadCtrs);
        cudaFree(devInOutIdcs);

        throw std::runtime_error(
            std::string("[9.44] Nuclear superposition kernel radius exceeded kMaxSuperpR=") +
            std::to_string(kMaxSuperpR) +
            "; overflowTiles=" + std::to_string(overflowCount) +
            " threshold_rSigmaEff<" + std::to_string(overflowThreshold) +
            ". Silent clamping would truncate the halo lateral umbrella and produce "
            "undetected physics error. Fix options: (1) land canonical physical PB "
            "lattice (task 9.23) so spotDist > 1 reduces in-cell radius; "
            "(2) raise MAX_SUPERP_RADIUS at build time; (3) verify nuclear LUT depth "
            "axis aligns to primary (task 9.14).");
    }

    int layerMaxSuperpR = 0;
    for (int i = 0; i < kMaxSuperpR + 2; ++i) {
        if (tileRadCtrs[i] > 0) { layerMaxSuperpR = i; }
    }

    int totalTiles = 0;
    for (int rad = 0; rad <= kMaxSuperpR; ++rad) totalTiles += tileRadCtrs[rad];

    if (rtdVerboseFineTiming() || rtdHaloAuditEnabled()) {
        std::cout << "  Nuclear superposition tile summary: totalTiles=" << totalTiles
                  << ", layerMaxSuperpR=" << layerMaxSuperpR
                  << ", rad0=" << tileRadCtrs[0]
                  << ", rad1=" << tileRadCtrs[1]
                  << ", rad2=" << tileRadCtrs[2]
                  << ", rad4=" << tileRadCtrs[4]
                  << ", rad8=" << tileRadCtrs[8]
                  << ", rad16=" << tileRadCtrs[16]
                  << ", rad24=" << tileRadCtrs[24]
                  << ", rad32=" << tileRadCtrs[32]
                  << std::endl;
    }

    std::vector<int> batchedTileRadCtrs(kMaxSuperpR + 1, 0);
    batchedTileRadCtrs[0] = tileRadCtrs[0];
    int recRad = layerMaxSuperpR;
    for (int rad = layerMaxSuperpR; rad > 0; --rad) {
        batchedTileRadCtrs[recRad] += tileRadCtrs[rad];
        if (batchedTileRadCtrs[recRad] >= MIN_TILES_IN_BATCH) {
            recRad = rad - 1;
        }
    }

    if (rtdVerboseFineTiming() || rtdHaloAuditEnabled()) {
        std::cout << "  Nuclear superposition batched launches:";
        for (int r = 0; r <= layerMaxSuperpR; ++r) {
            if (batchedTileRadCtrs[r] > 0) {
                std::cout << " [rad=" << r << ":" << batchedTileRadCtrs[r] << "]";
            }
        }
        std::cout << std::endl;
    }

    const dim3 superpBlockDim(kSuperpTileX, kSuperpTileY);

    LAUNCH_SUPERP_KERNEL(0);
    LAUNCH_SUPERP_KERNEL(1);
    LAUNCH_SUPERP_KERNEL(2);
    LAUNCH_SUPERP_KERNEL(3);
    LAUNCH_SUPERP_KERNEL(4);
    LAUNCH_SUPERP_KERNEL(5);
    LAUNCH_SUPERP_KERNEL(6);
    LAUNCH_SUPERP_KERNEL(7);
    LAUNCH_SUPERP_KERNEL(8);
    LAUNCH_SUPERP_KERNEL(9);
    LAUNCH_SUPERP_KERNEL(10);
    LAUNCH_SUPERP_KERNEL(11);
    LAUNCH_SUPERP_KERNEL(12);
    LAUNCH_SUPERP_KERNEL(13);
    LAUNCH_SUPERP_KERNEL(14);
    LAUNCH_SUPERP_KERNEL(15);
    LAUNCH_SUPERP_KERNEL(16);
    LAUNCH_SUPERP_KERNEL(17);
    LAUNCH_SUPERP_KERNEL(18);
    LAUNCH_SUPERP_KERNEL(19);
    LAUNCH_SUPERP_KERNEL(20);
    LAUNCH_SUPERP_KERNEL(21);
    LAUNCH_SUPERP_KERNEL(22);
    LAUNCH_SUPERP_KERNEL(23);
    LAUNCH_SUPERP_KERNEL(24);
    LAUNCH_SUPERP_KERNEL(25);
    LAUNCH_SUPERP_KERNEL(26);
    LAUNCH_SUPERP_KERNEL(27);
    LAUNCH_SUPERP_KERNEL(28);
    LAUNCH_SUPERP_KERNEL(29);
    LAUNCH_SUPERP_KERNEL(30);
    LAUNCH_SUPERP_KERNEL(31);
    LAUNCH_SUPERP_KERNEL(32);

    checkCudaErrors(cudaStreamSynchronize(stream));

    checkCudaErrors(cudaFree(devTileRadCtrs));
    checkCudaErrors(cudaFree(devInOutIdcs));

    GPU_TIMER_END("Nuclear Tile-Based Superposition");
}

#undef LAUNCH_SUPERP_KERNEL
