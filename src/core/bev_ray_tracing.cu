/**
 * \file
 * \brief BEV发散坐标系下的射线追踪实现
 */

#include "../include/core/common.cuh"
#include "../include/core/ray_tracing.h"
#include "../include/algorithms/fill_idd_and_sigma_params.cuh"
#include "../include/algorithms/superposition.h"
#include "../include/algorithms/convolution.h"
#include "../include/algorithms/idd_sigma.h"
#include "../include/algorithms/transfer_param_struct_div3.cuh"
#include "../include/utils/debug_tools.h"
#include <cuda_runtime.h>
#include <texture_indirect_functions.h>

__device__ inline float sample3DLinearBorder(
    const float* vol,
    int3 dims,
    float x,
    float y,
    float z
) {
    if (!vol) return 0.0f;

    const float fx = x - 0.5f;
    const float fy = y - 0.5f;
    const float fz = z - 0.5f;

    const int x0 = static_cast<int>(floorf(fx));
    const int y0 = static_cast<int>(floorf(fy));
    const int z0 = static_cast<int>(floorf(fz));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const int z1 = z0 + 1;

    const float tx = fx - float(x0);
    const float ty = fy - float(y0);
    const float tz = fz - float(z0);

    auto at = [&](int xi, int yi, int zi) -> float {
        if (xi < 0 || yi < 0 || zi < 0 || xi >= dims.x || yi >= dims.y || zi >= dims.z) return 0.0f;
        return vol[(static_cast<size_t>(zi) * dims.y + yi) * dims.x + xi];
    };

    const float c000 = at(x0, y0, z0);
    const float c100 = at(x1, y0, z0);
    const float c010 = at(x0, y1, z0);
    const float c110 = at(x1, y1, z0);
    const float c001 = at(x0, y0, z1);
    const float c101 = at(x1, y0, z1);
    const float c011 = at(x0, y1, z1);
    const float c111 = at(x1, y1, z1);

    const float c00 = c000 + (c100 - c000) * tx;
    const float c10 = c010 + (c110 - c010) * tx;
    const float c01 = c001 + (c101 - c001) * tx;
    const float c11 = c011 + (c111 - c011) * tx;
    const float c0 = c00 + (c10 - c00) * ty;
    const float c1 = c01 + (c11 - c01) * ty;
    return c0 + (c1 - c0) * tz;
}

// BEV发散坐标系下的射线追踪kernel
__global__ void rayTracingBEVKernel(
    float* bevDensity,
    float* bevCumulSp,
    float* bevIdd,
    float* bevRSigmaEff,
    float* rayWeights,
    int* beamFirstInside,
    int* firstStepOutside,
    int* firstPassive,
    DensityAndSpTracerParams params,
    FillIddAndSigmaParams iddParams,
    float* layerEnergy,
    cudaTextureObject_t imVolTex,
    const float* imVolData,
    cudaTextureObject_t densityTex,
    cudaTextureObject_t stoppingPowerTex,
    cudaTextureObject_t cumulIddTex,
    cudaTextureObject_t rRadiationLengthTex,
    int3 imVolDims
) {
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

    const unsigned int rayDimsX = iddParams.getRayDimsX();
    const unsigned int rayDimsY = iddParams.getRayDimsY();
    const unsigned int memStep = rayDimsY * rayDimsX;

    if (x >= rayDimsX || y >= rayDimsY) {
        return;
    }

    // This kernel currently only performs density / cumulative stopping power tracing
    // (IDD/sigma are handled in fillIddAndSigmaKernel). Keep unused parameters for
    // interface compatibility with the migrated pipeline.
    (void)bevIdd;
    (void)bevRSigmaEff;
    (void)rayWeights;
    (void)firstPassive;
    (void)layerEnergy;
    (void)cumulIddTex;
    (void)rRadiationLengthTex;
    (void)imVolDims;

    // Initialize tracing
    vec3f startPos = vec3f(params.getStart(x, y).x, params.getStart(x, y).y, params.getStart(x, y).z);
    vec3f pos = vec3f(startPos.x + HALF, startPos.y + HALF, startPos.z + HALF);
    vec3f step = vec3f(params.getInc(x, y).x, params.getInc(x, y).y, params.getInc(x, y).z);
    float stepLen = params.stepLen(x, y);

    // ---------------------------------------------------------------------
    // Reference-accurate BEV ray tracing state
    // ---------------------------------------------------------------------
    // Must match RayTraceDicom-main/src/kernel_wrapper.cu::fillBevDensityAndSp
    float cumulSp = 0.0f;
    float cumulHuPlus1000 = 0.0f;
    unsigned int beforeFirstInside = 0;
    unsigned int lastInside = 0;
    bool beforeFirstInsideFlag = true;

    unsigned int idx = y * rayDimsX + x;

    for (unsigned int i = 0; i < params.getSteps(); ++i) {
        float huPlus1000 = (imVolTex != 0)
            ? tex3D<float>(imVolTex, pos.x, pos.y, pos.z)
            : sample3DLinearBorder(imVolData, imVolDims, pos.x, pos.y, pos.z);
        cumulHuPlus1000 += huPlus1000;

        // Density and stopping power are *always* taken from the current HU sample (not cumulative HU)
        // NOTE: HU values in the reference are stored as HU+1000.
        bevDensity[idx] = tex1D<float>(densityTex, huPlus1000 * params.getDensityScale() + HALF);
        cumulSp += stepLen * tex1D<float>(stoppingPowerTex, huPlus1000 * params.getSpScale() + HALF);

        // Find BEV entry/exit steps using the reference cumulative threshold logic
        if (beforeFirstInsideFlag && cumulHuPlus1000 < 150.0f) {
            beforeFirstInside = i;
        } else {
            beforeFirstInsideFlag = false;
        }
        if (huPlus1000 > 150.0f) {
            lastInside = i;
        }

        bevCumulSp[idx] = cumulSp;

        // Advance to next step
        idx += memStep;
        pos.x += step.x;
        pos.y += step.y;
        pos.z += step.z;
    }

    // Step indices (0-based) in the RayTraceDicom convention.
    beamFirstInside[y * rayDimsX + x] = beforeFirstInside + 1;
    firstStepOutside[y * rayDimsX + x] = lastInside + 1;
}


// BEV到剂量网格的转换kernel
__global__ void bevToDoseGridKernel(
    float* doseGrid,             // final dose grid
    float* bevDose,              // input BEV dose array 
    TransferParamStructDiv3 params,
    int3 startIdx,
    int maxZ,
    uint3 doseDims,
    cudaTextureObject_t bevDoseTex
) {
    unsigned int x = startIdx.x + blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = startIdx.y + blockDim.y * blockIdx.y + threadIdx.y;

    if (x < doseDims.x && y < doseDims.y) {
        params.init(x, y);
        float *res = doseGrid + startIdx.z * doseDims.x * doseDims.y + y * doseDims.x + x;
        
        for (int z = startIdx.z; z <= maxZ; ++z) {
            vec3f pos = params.getFanIdx(z) + vec3f(HALF, HALF, HALF);
            float dose = tex3D<float>(bevDoseTex, pos.x, pos.y, pos.z);
            
            if (dose > 0.0f) {
                *res += dose;
            }
            res += doseDims.x * doseDims.y;
        }
    }
}

// 主机函数：执行完整的BEV射线追踪
int performBEVRayTracing(
    float* d_bevDensity,
    float* d_bevCumulSp,
    float* d_bevIdd,
    float* d_bevRSigmaEff,
    float* d_rayWeights,
    int* d_beamFirstInside,
    int* d_firstStepOutside,
    int* d_firstPassive,
    DensityAndSpTracerParams densityParams,
    FillIddAndSigmaParams iddParams,
    float* layerEnergy,
    cudaTextureObject_t imVolTex,
    const float* imVolData,
    cudaTextureObject_t densityTex,
    cudaTextureObject_t stoppingPowerTex,
    cudaTextureObject_t cumulIddTex,
    cudaTextureObject_t rRadiationLengthTex,
    int3 imVolDims,
    int gpuId
) {
    auto start = std::chrono::high_resolution_clock::now();
    cudaSetDevice(gpuId);
    
    // 设置网格和块大小
    dim3 blockSize(16, 16);
    dim3 gridSize((iddParams.getRayDimsX() + blockSize.x - 1) / blockSize.x,
                  (iddParams.getRayDimsY() + blockSize.y - 1) / blockSize.y);
    
    // 执行射线追踪
    rayTracingBEVKernel<<<gridSize, blockSize>>>(
        d_bevDensity,
        d_bevCumulSp,
        d_bevIdd,
        d_bevRSigmaEff,
        d_rayWeights,
        d_beamFirstInside,
        d_firstStepOutside,
        d_firstPassive,
        densityParams,
        iddParams,
        layerEnergy,
        imVolTex,
        imVolData,
        densityTex,
        stoppingPowerTex,
        cumulIddTex,
        rRadiationLengthTex,
        imVolDims
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Error in BEV ray tracing: %s\n", cudaGetErrorString(error));
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        if (rtdVerboseFineTiming()) printf("[TIMING] performBEVRayTracing_ERROR: %ld μs\n", duration.count());
        return 0;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    if (rtdVerboseFineTiming()) printf("[TIMING] performBEVRayTracing: %ld μs\n", duration.count());
    return 1;
}

// 主机函数：执行BEV到剂量网格的转换
int performBEVToDoseGridTransfer(
    float* d_doseGrid,
    float* d_bevDose,
    TransferParamStructDiv3 transferParams,
    int3 startIdx,
    int maxZ,
    uint3 doseDims,
    cudaTextureObject_t bevDoseTex,
    int gpuId
) {
    auto start = std::chrono::high_resolution_clock::now();
    cudaSetDevice(gpuId);
    
    // 设置网格和块大小
    dim3 blockSize(16, 16);
    dim3 gridSize((doseDims.x + blockSize.x - 1) / blockSize.x,
                  (doseDims.y + blockSize.y - 1) / blockSize.y);
    
    // 执行转换
    bevToDoseGridKernel<<<gridSize, blockSize>>>(
        d_doseGrid,
        d_bevDose,
        transferParams,
        startIdx,
        maxZ,
        doseDims,
        bevDoseTex
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Error in BEV to dose grid transfer: %s\n", cudaGetErrorString(error));
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        if (rtdVerboseFineTiming()) printf("[TIMING] performBEVToDoseGridTransfer_ERROR: %ld μs\n", duration.count());
        return 0;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    if (rtdVerboseFineTiming()) printf("[TIMING] performBEVToDoseGridTransfer: %ld μs\n", duration.count());
    return 1;
}
