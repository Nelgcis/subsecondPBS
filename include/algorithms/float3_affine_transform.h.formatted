/**
 * \file
 * \brief Float3AffineTransform (ported from RayTraceDicom-main)
 *
 * Affine transform: out = M * in + offset.
 */

#ifndef FLOAT3_AFFINE_TRANSFORM_H
#define FLOAT3_AFFINE_TRANSFORM_H

#include "common.cuh"
#include "Macro.cuh"
#include "matrix_3x3.h"

// Small helper mirroring reference helper_float3.cuh
CUDA_CALLABLE_MEMBER inline float sum_vec3f(const vec3f& a) { return a.x + a.y + a.z; }

/**
 * \brief Affine transform in 3D.
 */
struct Float3AffineTransform {
    Matrix3x3 m; ///< Linear part
    vec3f v;     ///< Translation

    CUDA_CALLABLE_MEMBER Float3AffineTransform() : m(1.0f, 1.0f, 1.0f), v(0.0f, 0.0f, 0.0f) {}

    CUDA_CALLABLE_MEMBER Float3AffineTransform(const Matrix3x3 mIn, const vec3f vIn) : m(mIn), v(vIn) {}

    CUDA_CALLABLE_MEMBER vec3f transformPoint(const vec3f a) const {
        return m * a + v;
    }

    CUDA_CALLABLE_MEMBER vec3f transformVector(const vec3f a) const {
        return m * a;
    }

    CUDA_CALLABLE_MEMBER Float3AffineTransform inverse() const {
        const Matrix3x3 mInv = m.inverse();
        return Float3AffineTransform(mInv, mInv * (v * (-1.0f)));
    }

    // Mirrors reference helper: convert 1-based to 0-based conventions.
    CUDA_CALLABLE_MEMBER void oneBasedToZeroBased(const bool toIdx) {
        if (toIdx) {
            v -= vec3f(1.0f, 1.0f, 1.0f);
        } else {
            v += make_vec3f(sum_vec3f(m.row0()), sum_vec3f(m.row1()), sum_vec3f(m.row2()));
        }
    }

    CUDA_CALLABLE_MEMBER Matrix3x3 getMatrix() const { return m; }
    CUDA_CALLABLE_MEMBER vec3f getOffset() const { return v; }
};

/**
 * \brief Concatenate two affine transforms (apply first, then second).
 */
CUDA_CALLABLE_MEMBER inline Float3AffineTransform concat(const Float3AffineTransform first,
                                                         const Float3AffineTransform second) {
    const Matrix3x3 outM = second.getMatrix() * first.getMatrix();
    const vec3f outV = second.getMatrix() * first.getOffset() + second.getOffset();
    return Float3AffineTransform(outM, outV);
}

#endif // FLOAT3_AFFINE_TRANSFORM_H
