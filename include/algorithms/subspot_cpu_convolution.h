/**
 * \file
 * \brief CPU-based Subspot Convolution Header
 */

#ifndef SUBSPOT_CPU_CONVOLUTION_H
#define SUBSPOT_CPU_CONVOLUTION_H

#include "common.cuh"

#ifdef __cplusplus
extern "C" {
#endif

// CPU版本的subspot到CPB卷积
int subspotToCPBConvolutionCPU(
    float* h_subspotData,               // 输入：subspot数据 [numLayers * maxSubspotsPerLayer * 5]
    int numLayers,                      // 能量层数量
    int maxSubspotsPerLayer,            // 每层最大subspot数量
    vec3f cpbCorner,                    // CPB网格角点
    vec3f cpbResolution,                // CPB网格分辨率
    vec3i cpbDims,                      // CPB网格维度
    float* h_cpbWeights,                // 输出：CPB权重
    int gpuId                           // GPU设备ID (未使用)
);

#ifdef __cplusplus
}
#endif

#endif // SUBSPOT_CPU_CONVOLUTION_H


