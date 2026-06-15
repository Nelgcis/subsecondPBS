/**
 * \file
 * \brief ROI Range Calculation Header
 */

#ifndef ROI_RANGE_CALCULATION_H
#define ROI_RANGE_CALCULATION_H

#include "common.cuh"
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// 主机函数：计算ROI范围
int calculateROIRange(
    vec3i* h_roiIndexArray,      // 输入：ROI索引数组
    int roiCount,                // ROI数量
    vec3f doseGridCorner,        // 剂量网格角点
    vec3f doseGridResolution,    // 剂量网格分辨率
    vec3f beamDirection,         // 束流方向
    vec3f sourcePosition,        // 源点位置
    float sad,                   // Source-to-axis distance
    float margin,                // ROI边距
    vec3f* h_roiMinBounds,       // 输出：ROI最小边界
    vec3f* h_roiMaxBounds,       // 输出：ROI最大边界
    bool* h_isValidROI,          // 输出：ROI是否有效
    int gpuId                    // GPU设备ID
);

#ifdef __cplusplus
}
#endif

#endif // ROI_RANGE_CALCULATION_H


