/**
 * \file
 * \brief CPU-only Ray Weight Initialization Header
 */

#ifndef RAY_WEIGHT_INITIALIZATION_CPU_H
#define RAY_WEIGHT_INITIALIZATION_CPU_H

#include <vector>
#include <cmath>

// CPU版本的vec3f结构体
struct vec3f {
    float x, y, z;
    vec3f(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}
};

// CPU版本的vec3i结构体
struct vec3i {
    int x, y, z;
    vec3i(int x = 0, int y = 0, int z = 0) : x(x), y(y), z(z) {}
};

// CPU版本的make_vec3f函数
inline vec3f make_vec3f(float x, float y, float z) { return vec3f(x, y, z); }

// CPU版本的make_vec3i函数
inline vec3i make_vec3i(int x, int y, int z) { return vec3i(x, y, z); }

// 常量定义
#define WEIGHT_CUTOFF 1e-6f
#define SIGMA_CUTOFF_DEFAULT 3.0f

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

#endif // RAY_WEIGHT_INITIALIZATION_CPU_H


