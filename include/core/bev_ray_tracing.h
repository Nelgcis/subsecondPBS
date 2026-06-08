/**
 * \file
 * \brief BEV发散坐标系下的射线追踪头文件
 */

#ifndef RAY_TRACING_BEV_H
#define RAY_TRACING_BEV_H

#include "common.cuh"
#include "fill_idd_and_sigma_params.cuh"
#include "transfer_param_struct_div3.cuh"
#include <cuda_runtime.h>
#include <texture_types.h>

// 前向声明
struct DensityAndSpTracerParams;
struct FillIddAndSigmaParams;
struct TransferParamStructDiv3;

// BEV射线追踪kernel函数声明
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
);

// BEV到剂量网格转换kernel函数声明
__global__ void bevToDoseGridKernel(
    float* doseGrid,
    float* bevDose,
    TransferParamStructDiv3 params,
    int3 startIdx,
    int maxZ,
    uint3 doseDims,
    cudaTextureObject_t bevDoseTex
);

// 主机函数声明
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
    cudaTextureObject_t densityTex,
    cudaTextureObject_t stoppingPowerTex,
    cudaTextureObject_t cumulIddTex,
    cudaTextureObject_t rRadiationLengthTex,
    int3 imVolDims,
    int gpuId
);

int performBEVToDoseGridTransfer(
    float* d_doseGrid,
    float* d_bevDose,
    TransferParamStructDiv3 transferParams,
    int3 startIdx,
    int maxZ,
    uint3 doseDims,
    cudaTextureObject_t bevDoseTex,
    int gpuId
);

#endif // RAY_TRACING_BEV_H
