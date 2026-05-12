/**
 * \file
 * \brief Transfer parameter structure for divergent coordinate system
 */

#ifndef TRANSFER_PARAM_STRUCT_DIV3_CUH
#define TRANSFER_PARAM_STRUCT_DIV3_CUH

#include "common.cuh"
#include "Macro.cuh"

// Forward declarations
struct Float3ToFanTransform;
struct Float3IdxTransform;
struct Float3AffineTransform;
struct Matrix3x3;

// Transfer parameter structure for divergent coordinate system
struct TransferParamStructDiv3 {
    vec3f globalOffset;
    vec3f coefOffset;
    vec3f coefIdxI;
    vec3f coefIdxJ;
    vec3f inc;
    vec3f start;
    vec2f normDist;
    
    __host__ __device__ TransferParamStructDiv3() {}
    
    // Constructor from Float3ToFanTransform (simplified - will be implemented separately)
    __host__ __device__ TransferParamStructDiv3(
        const Float3ToFanTransform& imIdxToFanIdx
    );
    
    __host__ __device__ void init(const int idxI, const int idxJ) {
        start = vec3f(float(idxI)) * coefIdxI + vec3f(float(idxJ)) * coefIdxJ + coefOffset;
    }
    
    __host__ __device__ vec3f getFanIdx(const int idxK) const {
        vec3f result = start + vec3f(float(idxK)) * inc;
        result.x *= 1 + result.z / (normDist.x - result.z);
        result.y *= 1 + result.z / (normDist.y - result.z);
        result += globalOffset;
        return result;
    }
};

#endif // TRANSFER_PARAM_STRUCT_DIV3_CUH
