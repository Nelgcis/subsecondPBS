/**
 * \file
 * \brief CPU-based Subspot Convolution Header
 */

#ifndef CPU_SUBSPOT_CONVOLUTION_H
#define CPU_SUBSPOT_CONVOLUTION_H

#include "common.cuh"
#include <chrono>

#ifdef __cplusplus
extern "C" {
#endif

// 主机函数：CPU版本的subspot到CPB卷积
int cpuSubspotToCPBConvolution(
    float* subspotData,          // 输入：subspot数据 [numLayers * maxSubspotsPerLayer * 5]
    int numLayers,              // 能量层数量
    int maxSubspotsPerLayer,    // 每层最大subspot数量
    vec3f cpbCorner,            // CPB网格角点
    vec3f cpbResolution,        // CPB网格分辨率
    vec3i cpbDims,              // CPB网格维度
    float sigmaCutoff,          // sigma截断值
    float* outputWeights        // 输出：CPB权重 [cpbDims.x * cpbDims.y * cpbDims.z]
);

#ifdef __cplusplus
}
#endif

#endif // CPU_SUBSPOT_CONVOLUTION_H


