/**
 * \file
 * \brief Float3IdxTransform (ported from RayTraceDicom-main)
 *
 * Index -> coordinate transform: out = in * delta + offset (element-wise).
 */

#ifndef FLOAT3_IDX_TRANSFORM_H
#define FLOAT3_IDX_TRANSFORM_H

#include "common.cuh"
#include "Macro.cuh"

/**
 * \brief Transform indices (i,j,k) to a 3D coordinate system.
 */
struct Float3IdxTransform {
    vec3f delta;  ///< element-wise scaling
    vec3f offset; ///< translation

    CUDA_CALLABLE_MEMBER Float3IdxTransform()
        : delta(1.0f, 1.0f, 1.0f), offset(0.0f, 0.0f, 0.0f) {}

    CUDA_CALLABLE_MEMBER Float3IdxTransform(const vec3f dIn, const vec3f oIn)
        : delta(dIn), offset(oIn) {}

    CUDA_CALLABLE_MEMBER vec3f getDelta() const { return delta; }
    CUDA_CALLABLE_MEMBER vec3f getOffset() const { return offset; }

    CUDA_CALLABLE_MEMBER vec3f transformPoint(const vec3f in) const {
        return in * delta + offset;
    }

    CUDA_CALLABLE_MEMBER Float3IdxTransform inverse() const {
        const vec3f invDelta(1.0f / delta.x, 1.0f / delta.y, 1.0f / delta.z);
        const vec3f invOffset(-offset.x * invDelta.x, -offset.y * invDelta.y, -offset.z * invDelta.z);
        return Float3IdxTransform(invDelta, invOffset);
    }

    CUDA_CALLABLE_MEMBER Float3IdxTransform shiftOffset(const vec3f shift) const {
        return Float3IdxTransform(delta, offset + shift);
    }

    // Mirrors reference helper: convert 1-based to 0-based conventions.
    CUDA_CALLABLE_MEMBER void oneBasedToZeroBased(const bool toIdx) {
        if (toIdx) {
            offset -= vec3f(1.0f, 1.0f, 1.0f);
        } else {
            offset = offset + delta;
        }
    }
};

#endif // FLOAT3_IDX_TRANSFORM_H
