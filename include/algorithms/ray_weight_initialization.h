/**
 * \file
 * \brief Ray Weight Initialization Header
 */

#ifndef RAY_WEIGHT_INITIALIZATION_H
#define RAY_WEIGHT_INITIALIZATION_H

#include "common.cuh"
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

// 主要的ray weight初始化函数
int initializeRayWeightsFromSubspotData(
    std::vector<float>& rayWeights,    // 输出：ray权重
    vec3i rayDims,                     // ray网格维度
    vec3f beamDirection,               // 束流方向
    vec3f bmXDirection,                // 束流X方向
    vec3f bmYDirection,                // 束流Y方向
    vec3f sourcePosition,              // 源点位置
    float sad,                         // Source-to-axis distance
    float refPlaneZ                    // 参考平面Z坐标
);

#ifdef __cplusplus
}
#endif

#endif // RAY_WEIGHT_INITIALIZATION_H


