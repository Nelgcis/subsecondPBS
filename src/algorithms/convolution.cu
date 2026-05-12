/**
 * \file
 * \brief Unified Convolution Implementation for Subspot Processing
 * 
 * This file combines subspot-to-CPB convolution and GPU convolution algorithms
 */

#include "../include/algorithms/convolution.h"
#include "../include/core/common.cuh"
#include "../include/core/Macro.cuh"
#include "../include/utils/debug_tools.h"
#include <cuda_runtime.h>
#include <texture_indirect_functions.h>
#include <chrono>

// ============================================================================
// Subspot Data Reading and Processing
// ============================================================================

// 从纹理读取subspot数据并存储到SubspotInfo数组
__global__ void readSubspotDataKernel(
    cudaTextureObject_t subspotData,    // 输入：subspot数据纹理
    SubspotInfo* subspotInfoArray,      // 输出：subspot信息数组
    int nsubspot,                       // subspot数量
    int layerIdx,                       // 能量层索引
    vec3f beamDirection,                // 束流主方向
    vec3f bmXDirection,                 // 束流X方向
    vec3f bmYDirection,                 // 束流Y方向
    vec3f sourcePosition,                // 源点位置
    float sad,                          // Source-to-axis distance
    float refPlaneZ                     // 参考平面Z坐标
) {
    int subspotIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (subspotIdx >= nsubspot) return;
    
    // 使用SubspotInfo的纹理构造函数
    subspotInfoArray[subspotIdx] = SubspotInfo(
        subspotData, subspotIdx, layerIdx,
        beamDirection, bmXDirection, bmYDirection,
        sourcePosition, sad, refPlaneZ
    );
}

// 计算subspot影响范围
__global__ void calculateSubspotRangeKernel(
    const SubspotInfo* subspotInfoArray,
    int nsubspot,
    vec3f cpbCorner,
    vec3f cpbResolution,
    vec3i cpbDims,
    int* subspotRanges  // [nsubspot * 4] -> (minX, maxX, minY, maxY)
) {
    int subspotIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (subspotIdx >= nsubspot) return;
    
    const SubspotInfo& subspot = subspotInfoArray[subspotIdx];
    if (!subspot.isValid) {
        subspotRanges[subspotIdx * 4 + 0] = 0;
        subspotRanges[subspotIdx * 4 + 1] = 0;
        subspotRanges[subspotIdx * 4 + 2] = 0;
        subspotRanges[subspotIdx * 4 + 3] = 0;
        return;
    }
    
    // 计算3-sigma截断范围
    float sigmaX = subspot.sigmaX;
    float sigmaY = subspot.sigmaY;
    float cutoff = SIGMA_CUTOFF;
    
    vec3f pos = subspot.position;
    float minX = pos.x - cutoff * sigmaX;
    float maxX = pos.x + cutoff * sigmaX;
    float minY = pos.y - cutoff * sigmaY;
    float maxY = pos.y + cutoff * sigmaY;
    
    // 转换为CPB网格索引
    int cpbMinX = max(0, (int)floorf((minX - cpbCorner.x) / cpbResolution.x));
    int cpbMaxX = min(cpbDims.x - 1, (int)ceilf((maxX - cpbCorner.x) / cpbResolution.x));
    int cpbMinY = max(0, (int)floorf((minY - cpbCorner.y) / cpbResolution.y));
    int cpbMaxY = min(cpbDims.y - 1, (int)ceilf((maxY - cpbCorner.y) / cpbResolution.y));
    
    // Debug output removed for performance
    
    subspotRanges[subspotIdx * 4 + 0] = cpbMinX;
    subspotRanges[subspotIdx * 4 + 1] = cpbMaxX;
    subspotRanges[subspotIdx * 4 + 2] = cpbMinY;
    subspotRanges[subspotIdx * 4 + 3] = cpbMaxY;
}

// ============================================================================
// Subspot to CPB Convolution
// ============================================================================

// 优化的subspot到CPB卷积kernel
__global__ void subspotToCPBConvolutionOptimizedKernel(
    const SubspotInfo* subspotInfoArray,
    const int* subspotRanges,
    int nsubspot,
    int layerIdx,
    vec3f cpbCorner,
    vec3f cpbResolution,
    vec3i cpbDims,
    float* cpbWeights
) {
    int subspotIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (subspotIdx >= nsubspot) return;
    
    const SubspotInfo& subspot = subspotInfoArray[subspotIdx];
    if (!subspot.isValid) return;
    
    // 获取影响范围
    int minX = subspotRanges[subspotIdx * 4 + 0];
    int maxX = subspotRanges[subspotIdx * 4 + 1];
    int minY = subspotRanges[subspotIdx * 4 + 2];
    int maxY = subspotRanges[subspotIdx * 4 + 3];
    
    // Debug output removed for performance
    
    int processedPoints = 0;
    for (int cpbY = minY; cpbY <= maxY; cpbY++) {
        for (int cpbX = minX; cpbX <= maxX; cpbX++) {
            // Debug output removed for performance
            // 计算CPB网格点位置
            vec3f cpbPos = vec3f(
                cpbCorner.x + (cpbX + 0.5f) * cpbResolution.x,
                cpbCorner.y + (cpbY + 0.5f) * cpbResolution.y,
                cpbCorner.z
            );
            
            // 计算高斯权重
            float dx = cpbPos.x - subspot.position.x;
            float dy = cpbPos.y - subspot.position.y;
            float sigmaX = subspot.sigmaX;
            float sigmaY = subspot.sigmaY;
            
            // 使用误差函数计算精确的高斯积分
            float erfX1 = erf((dx - 0.5f * cpbResolution.x) / (1.41421356f * sigmaX));
            float erfX2 = erf((dx + 0.5f * cpbResolution.x) / (1.41421356f * sigmaX));
            float erfY1 = erf((dy - 0.5f * cpbResolution.y) / (1.41421356f * sigmaY));
            float erfY2 = erf((dy + 0.5f * cpbResolution.y) / (1.41421356f * sigmaY));
            
            float weight = 0.25f * (erfX2 - erfX1) * (erfY2 - erfY1) * subspot.weight;
            
            // 边界检查：确保索引在有效范围内
            if (cpbX >= 0 && cpbX < cpbDims.x && cpbY >= 0 && cpbY < cpbDims.y) {
                // 原子累加到CPB权重
                int cpbIdx = layerIdx * cpbDims.x * cpbDims.y + cpbY * cpbDims.x + cpbX;
                atomicAdd(&cpbWeights[cpbIdx], weight);
            }
        }
    }
}

// GPU 2D Convolution

__global__ void gpuConvolution2DKernel(
    float* input,           // 输入数据
    float* output,          // 输出数据
    int width,              // 输入宽度
    int height,             // 输入高度
    float* kernel,          // 卷积核
    int kernelSize,         // 卷积核大小
    int padding             // 填充大小
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float sum = 0.0f;
    int halfKernel = kernelSize / 2;
    
    for (int ky = -halfKernel; ky <= halfKernel; ky++) {
        for (int kx = -halfKernel; kx <= halfKernel; kx++) {
            int ix = x + kx;
            int iy = y + ky;
            
            // 边界处理
            if (ix < 0 || ix >= width || iy < 0 || iy >= height) {
                if (padding == 0) continue; // 零填充
                // 可以添加其他填充模式
            }
            
            int inputIdx = iy * width + ix;
            int kernelIdx = (ky + halfKernel) * kernelSize + (kx + halfKernel);
            
            sum += input[inputIdx] * kernel[kernelIdx];
        }
    }
    
    output[y * width + x] = sum;
}

namespace {
constexpr float kConvSigmaCutoff = 3.0f;
}

__global__ void xConvGathResampGpu(
    float* const in,
    float* const out,
    float2* const sigma,
    const int inWidth,
    const int outWidth,
    const int height,
    const float pixelSp,
    const float inOutOffset,
    const float inOutDelta
) {
    const int idxY = blockDim.y * blockIdx.y + threadIdx.y;
    if (idxY >= height) return;

    const int outIdxX = blockDim.x * blockIdx.x + threadIdx.x;
    if (outIdxX >= outWidth) return;

    float res = 0.0f;
    const float sigmaEff = sigma[blockIdx.z].x / pixelSp;
    const float rSigmaEff = rsqrtf(2.0f) / sigmaEff;
    int currentInIdxX = int(ceilf((float(outIdxX) - (kConvSigmaCutoff * sigmaEff + HALF) - inOutOffset) / inOutDelta));
    float dist = currentInIdxX * inOutDelta + inOutOffset - float(outIdxX);
    while (dist < (kConvSigmaCutoff * sigmaEff + HALF)) {
        if (currentInIdxX >= 0 && currentInIdxX < inWidth) {
            res += HALF * (erf((dist + HALF) * rSigmaEff) - erf((dist - HALF) * rSigmaEff)) *
                   in[blockIdx.z * inWidth * height + idxY * inWidth + currentInIdxX];
        }
        ++currentInIdxX;
        dist = currentInIdxX * inOutDelta + inOutOffset - float(outIdxX);
    }
    out[blockIdx.z * outWidth * height + idxY * outWidth + outIdxX] = res;
}

__global__ void yConvGathResampGpu(
    float* const in,
    float* const out,
    float2* const sigma,
    const int width,
    const int inHeight,
    const int outHeight,
    const float pixelSp,
    const float inOutOffset,
    const float inOutDelta
) {
    const int idxX = blockDim.x * blockIdx.x + threadIdx.x;
    if (idxX >= width) return;

    const int outIdxY = blockDim.y * blockIdx.y + threadIdx.y;
    if (outIdxY >= outHeight) return;

    float res = 0.0f;
    const float sigmaEff = sigma[blockIdx.z].y / pixelSp;
    const float rSigmaEff = rsqrtf(2.0f) / sigmaEff;
    int currentInIdxY = int(ceilf((float(outIdxY) - (kConvSigmaCutoff * sigmaEff + HALF) - inOutOffset) / inOutDelta));
    float dist = currentInIdxY * inOutDelta + inOutOffset - float(outIdxY);
    while (dist < (kConvSigmaCutoff * sigmaEff + HALF)) {
        if (currentInIdxY >= 0 && currentInIdxY < inHeight) {
            res += HALF * (erf((dist + HALF) * rSigmaEff) - erf((dist - HALF) * rSigmaEff)) *
                   in[blockIdx.z * width * inHeight + currentInIdxY * width + idxX];
        }
        ++currentInIdxY;
        dist = currentInIdxY * inOutDelta + inOutOffset - float(outIdxY);
    }
    out[blockIdx.z * width * outHeight + outIdxY * width + idxX] = res;
}

// ============================================================================
// Ray Weight Initialization
// ============================================================================

// GPU kernel: 将CPB权重映射到ray权重（只处理指定层）
__global__ void mapCPBWeightsToRayWeightsKernel(
    const float* cpbWeights,               // input：CPB权重
    int cpbDimsX, int cpbDimsY, int cpbDimsZ,  // CPB维度
    float cpbCornerX, float cpbCornerY, float cpbCornerZ,  // CPB corner
    float cpbResolutionX, float cpbResolutionY, float cpbResolutionZ,  // CPB resolution
    float* rayWeights,                     // output：ray权重 (2D)
    int rayDimsX, int rayDimsY,            // ray网格维度
    float rayCornerX, float rayCornerY, float rayCornerZ,  // ray corner
    float rayResolutionX, float rayResolutionY, float rayResolutionZ,  // ray resolution
    int layerIdx,                          // 当前能量层索引（只处理这一层）
    float beamDirX, float beamDirY, float beamDirZ,        // (unused) beam direction
    float bmXDirX, float bmXDirY, float bmXDirZ,           // (unused) bmX direction
    float bmYDirX, float bmYDirY, float bmYDirZ,           // (unused) bmY direction
    float sourcePosX, float sourcePosY, float sourcePosZ,  // (unused) source position
    float sad,                             // (unused) SAD
    float refPlaneZ                        // (unused) reference plane Z
) {
    int rayY = blockIdx.y * blockDim.y + threadIdx.y;
    int rayX = blockIdx.x * blockDim.x + threadIdx.x;

    if (rayY >= rayDimsY || rayX >= rayDimsX) return;

    // Guard layer
    if (layerIdx < 0 || layerIdx >= cpbDimsZ) {
        rayWeights[rayY * rayDimsX + rayX] = 0.0f;
        return;
    }

    // Physical bounds of the ray cell in the (ray) plane coordinates
    const float x0 = rayCornerX + rayX * rayResolutionX;
    const float x1 = x0 + rayResolutionX;
    const float y0 = rayCornerY + rayY * rayResolutionY;
    const float y1 = y0 + rayResolutionY;

    // Find overlapping CPB cell index range
    int cpbMinX = (int)floorf((x0 - cpbCornerX) / cpbResolutionX);
    int cpbMaxX = (int)ceilf((x1 - cpbCornerX) / cpbResolutionX) - 1;
    int cpbMinY = (int)floorf((y0 - cpbCornerY) / cpbResolutionY);
    int cpbMaxY = (int)ceilf((y1 - cpbCornerY) / cpbResolutionY) - 1;

    // Clamp to CPB bounds
    if (cpbMinX < 0) cpbMinX = 0;
    if (cpbMinY < 0) cpbMinY = 0;
    if (cpbMaxX >= cpbDimsX) cpbMaxX = cpbDimsX - 1;
    if (cpbMaxY >= cpbDimsY) cpbMaxY = cpbDimsY - 1;

    const float cpbArea = cpbResolutionX * cpbResolutionY;
    float accum = 0.0f;

    if (cpbArea > 0.0f && cpbMinX <= cpbMaxX && cpbMinY <= cpbMaxY) {
        for (int cy = cpbMinY; cy <= cpbMaxY; ++cy) {
            const float cy0 = cpbCornerY + cy * cpbResolutionY;
            const float cy1 = cy0 + cpbResolutionY;
            const float oy = fmaxf(0.0f, fminf(y1, cy1) - fmaxf(y0, cy0));
            if (oy <= 0.0f) continue;

            for (int cx = cpbMinX; cx <= cpbMaxX; ++cx) {
                const float cx0 = cpbCornerX + cx * cpbResolutionX;
                const float cx1 = cx0 + cpbResolutionX;
                const float ox = fmaxf(0.0f, fminf(x1, cx1) - fmaxf(x0, cx0));
                if (ox <= 0.0f) continue;

                const float overlapFrac = (ox * oy) / cpbArea;
                const int cpbIdx = layerIdx * cpbDimsX * cpbDimsY + cy * cpbDimsX + cx;
                accum += cpbWeights[cpbIdx] * overlapFrac;
            }
        }
    }

    rayWeights[rayY * rayDimsX + rayX] = accum;
}


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
) {
    // NOTE: cpbWeights may be either a device pointer (cudaMalloc/managed) or a host pointer.
    // This function will write directly to device memory when possible, otherwise it will use a temporary device buffer and copy back.

    if (cpbWeights == nullptr) {
        fprintf(stderr, "Error: cpbWeights is null in performSubspotToCPBConvolution\n");
        return;
    }

    // Determine whether cpbWeights is a device-accessible pointer
    bool cpbOnDevice = false;
    cudaPointerAttributes attr;
    cudaError_t attrErr = cudaPointerGetAttributes(&attr, cpbWeights);
    if (attrErr == cudaSuccess) {
    #if CUDART_VERSION >= 11000
        cpbOnDevice = (attr.type == cudaMemoryTypeDevice) || (attr.type == cudaMemoryTypeManaged);
    #else
        cpbOnDevice = (attr.memoryType == cudaMemoryTypeDevice);
    #endif
    } else {
        // For pageable host pointers, cudaPointerGetAttributes may fail. Treat as host pointer.
        cudaGetLastError();
        cpbOnDevice = false;
    }

    // Allocate device memory for output if needed
    size_t cpbWeightsSize = static_cast<size_t>(cpbDims.x) * static_cast<size_t>(cpbDims.y) * static_cast<size_t>(cpbDims.z);
    float* d_cpbWeights = nullptr;
    if (cpbOnDevice) {
        d_cpbWeights = cpbWeights;
    } else {
        checkCudaErrors(cudaMalloc(&d_cpbWeights, cpbWeightsSize * sizeof(float)));
    }
    checkCudaErrors(cudaMemset(d_cpbWeights, 0, cpbWeightsSize * sizeof(float)));

    // Allocate temporary arrays
    SubspotInfo* d_subspotInfoArray;
    checkCudaErrors(cudaMalloc(&d_subspotInfoArray, maxSubspotsPerLayer * sizeof(SubspotInfo)));

    int* d_subspotRanges;
    checkCudaErrors(cudaMalloc(&d_subspotRanges, maxSubspotsPerLayer * 4 * sizeof(int)));

    // Launch configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (maxSubspotsPerLayer + threadsPerBlock - 1) / threadsPerBlock;

    if (rtdVerboseFineTiming()) {
        std::cout << "Starting convolution for " << numLayers << " layers with " << maxSubspotsPerLayer << " subspots per layer" << std::endl;
    }

    for (int layerIdx = 0; layerIdx < numLayers; layerIdx++) {
        auto startTime = std::chrono::high_resolution_clock::now();

        // 1) Read subspot data from texture
        readSubspotDataKernel<<<blocksPerGrid, threadsPerBlock>>>(
            subspotData, d_subspotInfoArray, maxSubspotsPerLayer, layerIdx,
            beamDirection, bmXDirection, bmYDirection, sourcePosition, sad, refPlaneZ
        );
        checkCudaErrors(cudaGetLastError());

        // 2) Compute subspot influence range
        calculateSubspotRangeKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_subspotInfoArray, maxSubspotsPerLayer, cpbCorner, cpbResolution, cpbDims, d_subspotRanges
        );
        checkCudaErrors(cudaGetLastError());

        // 3) Convolve subspots into CPB grid (accumulate into the correct layer slice)
        subspotToCPBConvolutionOptimizedKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_subspotInfoArray, d_subspotRanges, maxSubspotsPerLayer, layerIdx,
            cpbCorner, cpbResolution, cpbDims, d_cpbWeights
        );
        checkCudaErrors(cudaGetLastError());

        // NOTE: No per-layer cudaDeviceSynchronize here (was debug code); kernels are ordered on the default stream.

        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        if (rtdVerboseFineTiming()) {
            std::cout << "Layer " << layerIdx << " processed in " << duration.count() << " ms" << std::endl;
        }
    }

    // Ensure all kernels finished before returning/freeing memory
    checkCudaErrors(cudaDeviceSynchronize());

    // Copy result back only if cpbWeights is on host
    if (!cpbOnDevice) {
        checkCudaErrors(cudaMemcpy(cpbWeights, d_cpbWeights, cpbWeightsSize * sizeof(float), cudaMemcpyDeviceToHost));
        cudaFree(d_cpbWeights);
    }

    // Cleanup temporary arrays
    cudaFree(d_subspotInfoArray);
    cudaFree(d_subspotRanges);
}



void performGPUConvolution2D(
    float* input,
    float* output,
    int width,
    int height,
    float* kernel,
    int kernelSize,
    int padding
) {
    GPU_TIMER_START();
    
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    gpuConvolution2DKernel<<<gridSize, blockSize>>>(
        input, output, width, height, kernel, kernelSize, padding
    );
    checkCudaErrors(cudaDeviceSynchronize());
    
    GPU_TIMER_END("GPU 2D Convolution");
}

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
) {
    dim3 xConvBlock(32, 8);
    dim3 xConvGrid((rayGridDims.x + xConvBlock.x - 1) / xConvBlock.x,
                   (spotGridDims.y + xConvBlock.y - 1) / xConvBlock.y,
                   spotGridDims.z);
    dim3 yConvBlock(32, 8);
    dim3 yConvGrid((rayGridDims.x + yConvBlock.x - 1) / yConvBlock.x,
                   (rayGridDims.y + yConvBlock.y - 1) / yConvBlock.y,
                   rayGridDims.z);
    const float2 inOutDelta = make_float2(spotDelta.x / rayDelta.x, spotDelta.y / rayDelta.y);
    const float2 inOutOffset = make_float2((spotOffset.x - rayOffset.x) / rayDelta.x,
                                           (spotOffset.y - rayOffset.y) / rayDelta.y);
    xConvGathResampGpu<<<xConvGrid, xConvBlock>>>(
        spotWeights, convInterm, entrySigmas,
        spotGridDims.x, rayGridDims.x, spotGridDims.y,
        rayDelta.x * pxSpMult.x, inOutOffset.x, inOutDelta.x
    );
    checkCudaErrors(cudaGetLastError());
    yConvGathResampGpu<<<yConvGrid, yConvBlock>>>(
        convInterm, rayWeights, entrySigmas,
        rayGridDims.x, spotGridDims.y, rayGridDims.y,
        rayDelta.y * pxSpMult.y, inOutOffset.y, inOutDelta.y
    );
    checkCudaErrors(cudaGetLastError());
}

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
) {
    GPU_TIMER_START();

    // Validate input parameters
    if (cpbWeights == nullptr || rayWeights == nullptr) {
        fprintf(stderr, "CUDA error: Null pointer in performCPBToRayWeightMapping\n");
        exit(1);
    }
    
    if (cpbDims.x <= 0 || cpbDims.y <= 0 || cpbDims.z <= 0) {
        fprintf(stderr, "CUDA error: Invalid CPB dimensions (%d, %d, %d)\n", 
                cpbDims.x, cpbDims.y, cpbDims.z);
        exit(1);
    }
    
    if (rayDims.x <= 0 || rayDims.y <= 0) {
        fprintf(stderr, "CUDA error: Invalid ray dimensions (%d, %d)\n", 
                rayDims.x, rayDims.y);
        exit(1);
    }

    float* d_cpbWeights = nullptr;
    float* d_rayWeights = nullptr;

    size_t cpbWeightsSize = cpbDims.x * cpbDims.y * cpbDims.z * sizeof(float);
    size_t rayWeightsSize = rayDims.x * rayDims.y * sizeof(float); // 2D ray grid

    // Detect source pointer residency so device-to-device calls stay fully on GPU.
    cudaPointerAttributes cpbAttr;
    cudaError_t attrErr = cudaPointerGetAttributes(&cpbAttr, cpbWeights);
    bool cpbOnDevice = false;
    if (attrErr == cudaSuccess) {
#if CUDART_VERSION >= 11000
        cpbOnDevice = (cpbAttr.type == cudaMemoryTypeDevice) || (cpbAttr.type == cudaMemoryTypeManaged);
#else
        cpbOnDevice = (cpbAttr.memoryType == cudaMemoryTypeDevice);
#endif
    } else {
        cudaGetLastError();
    }

    cudaPointerAttributes rayAttr;
    cudaError_t rayAttrErr = cudaPointerGetAttributes(&rayAttr, rayWeights);
    bool rayOnDevice = false;
    if (rayAttrErr == cudaSuccess) {
#if CUDART_VERSION >= 11000
        rayOnDevice = (rayAttr.type == cudaMemoryTypeDevice) || (rayAttr.type == cudaMemoryTypeManaged);
#else
        rayOnDevice = (rayAttr.memoryType == cudaMemoryTypeDevice);
#endif
    } else {
        cudaGetLastError();
    }

    if (cpbOnDevice) {
        d_cpbWeights = cpbWeights;
    } else {
        checkCudaErrors(cudaMalloc(&d_cpbWeights, cpbWeightsSize));
        checkCudaErrors(cudaMemcpy(d_cpbWeights, cpbWeights, cpbWeightsSize, cudaMemcpyHostToDevice));
    }

    if (rayOnDevice) {
        d_rayWeights = rayWeights;
    } else {
        checkCudaErrors(cudaMalloc(&d_rayWeights, rayWeightsSize));
    }
    
    // Validate kernel launch parameters
    dim3 blockSize(16, 16);
    dim3 gridSize((rayDims.x + blockSize.x - 1) / blockSize.x,
                  (rayDims.y + blockSize.y - 1) / blockSize.y);
    
    // Check for valid grid and block sizes
    if (gridSize.x == 0 || gridSize.y == 0) {
        fprintf(stderr, "CUDA error: Invalid grid size (%d, %d) for rayDims (%d, %d)\n",
                gridSize.x, gridSize.y, rayDims.x, rayDims.y);
        if (!cpbOnDevice) cudaFree(d_cpbWeights);
        if (!rayOnDevice) cudaFree(d_rayWeights);
        exit(1);
    }
    
    // Check maximum grid size limits (typically 65535 for x and y)
    if (gridSize.x > 65535 || gridSize.y > 65535) {
        fprintf(stderr, "CUDA error: Grid size (%d, %d) exceeds maximum (65535, 65535)\n",
                gridSize.x, gridSize.y);
        if (!cpbOnDevice) cudaFree(d_cpbWeights);
        if (!rayOnDevice) cudaFree(d_rayWeights);
        exit(1);
    }
    
    // Validate kernel parameters before launch
    if (d_cpbWeights == nullptr || d_rayWeights == nullptr) {
        fprintf(stderr, "CUDA error: Null device pointer before kernel launch\n");
        if (!cpbOnDevice) cudaFree(d_cpbWeights);
        if (!rayOnDevice) cudaFree(d_rayWeights);
        exit(1);
    }
    
    // Validate floating point parameters
    if (std::isnan(sad) || std::isinf(sad) || sad <= 0.0f || sad > 10000.0f) {
        fprintf(stderr, "CUDA error: Invalid SAD value: %f\n", sad);
        cudaFree(d_cpbWeights);
        cudaFree(d_rayWeights);
        exit(1);
    }
    
    if (std::isnan(refPlaneZ) || std::isinf(refPlaneZ)) {
        fprintf(stderr, "CUDA error: Invalid refPlaneZ value: %f\n", refPlaneZ);
        cudaFree(d_cpbWeights);
        cudaFree(d_rayWeights);
        exit(1);
    }
    
    // Debug output removed for performance
    
    // Launch kernel with explicit error checking
    // Clear any previous errors before launch
    cudaGetLastError();
    
    // Expand struct parameters to avoid kernel launch issues
    mapCPBWeightsToRayWeightsKernel<<<gridSize, blockSize>>>(
        d_cpbWeights,
        cpbDims.x, cpbDims.y, cpbDims.z,
        cpbCorner.x, cpbCorner.y, cpbCorner.z,
        cpbResolution.x, cpbResolution.y, cpbResolution.z,
        d_rayWeights,
        rayDims.x, rayDims.y,
        rayCorner.x, rayCorner.y, rayCorner.z,
        rayResolution.x, rayResolution.y, rayResolution.z,
        layerIdx,  // 只处理当前层
        beamDirection.x, beamDirection.y, beamDirection.z,
        bmXDirection.x, bmXDirection.y, bmXDirection.z,
        bmYDirection.x, bmYDirection.y, bmYDirection.z,
        sourcePosition.x, sourcePosition.y, sourcePosition.z,
        sad, refPlaneZ
    );
    
    // Check for kernel launch errors immediately
    cudaError_t launchErr = cudaPeekAtLastError();
    if (launchErr != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error in mapCPBWeightsToRayWeightsKernel: %s\n",
                cudaGetErrorString(launchErr));
        fprintf(stderr, "  Parameters: gridSize=(%d,%d), blockSize=(%d,%d)\n",
                gridSize.x, gridSize.y, blockSize.x, blockSize.y);
        fprintf(stderr, "  rayDims=(%d,%d), cpbDims=(%d,%d,%d)\n",
                rayDims.x, rayDims.y, cpbDims.x, cpbDims.y, cpbDims.z);
        fprintf(stderr, "  beamDirection=(%.3f,%.3f,%.3f), sourcePosition=(%.3f,%.3f,%.3f)\n",
                beamDirection.x, beamDirection.y, beamDirection.z,
                sourcePosition.x, sourcePosition.y, sourcePosition.z);
        fprintf(stderr, "  sad=%.3f, refPlaneZ=%.3f\n", sad, refPlaneZ);
        
        // Check for invalid float values
        if (std::isnan(sad) || std::isinf(sad)) {
            fprintf(stderr, "  ERROR: Invalid SAD value detected!\n");
        }
        if (std::isnan(refPlaneZ) || std::isinf(refPlaneZ)) {
            fprintf(stderr, "  ERROR: Invalid refPlaneZ value detected!\n");
        }
        
        if (!cpbOnDevice) cudaFree(d_cpbWeights);
        if (!rayOnDevice) cudaFree(d_rayWeights);
        // Don't exit - return error code instead to allow program to continue
        GPU_TIMER_END("CPB to Ray Weight Mapping");
        return;
    }
    
    // Synchronize and check for execution errors
    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        fprintf(stderr, "CUDA kernel execution error in mapCPBWeightsToRayWeightsKernel: %s\n",
                cudaGetErrorString(syncErr));
        if (!cpbOnDevice) cudaFree(d_cpbWeights);
        if (!rayOnDevice) cudaFree(d_rayWeights);
        exit(1);
    }

    if (!rayOnDevice) {
        if (rayWeights == nullptr) {
            fprintf(stderr, "CUDA error: Null rayWeights pointer before copy\n");
            if (!cpbOnDevice) cudaFree(d_cpbWeights);
            cudaFree(d_rayWeights);
            exit(1);
        }
        checkCudaErrors(cudaMemcpy(rayWeights, d_rayWeights, rayWeightsSize, cudaMemcpyDeviceToHost));
    }

    if (!cpbOnDevice) checkCudaErrors(cudaFree(d_cpbWeights));
    if (!rayOnDevice) checkCudaErrors(cudaFree(d_rayWeights));
    
    GPU_TIMER_END("CPB to Ray Weight Mapping");
}
