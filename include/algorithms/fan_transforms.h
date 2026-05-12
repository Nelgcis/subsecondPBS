/**
 * \file
 * \brief Fan (divergent) coordinate transforms
 *
 * Ported from RayTraceDicom-main:
 *   - Float3FromFanTransform
 *   - Float3ToFanTransform
 *
 * The reference uses CUDA float3/float2 + helper_math. Here we use vec3f/vec2f
 * from include/core/common.cuh.
 */

#ifndef FAN_TRANSFORMS_H
#define FAN_TRANSFORMS_H

#include "common.cuh"
#include "Macro.cuh"
#include "float3_idx_transform.h"
#include "float3_affine_transform.h"

struct Float3ToFanTransform;

/**
 * \brief Transform from fan-index space (ray grid indices) to image-index space.
 */
struct Float3FromFanTransform {
    Float3IdxTransform fITF;        ///< fanIdxToFan
    Float3AffineTransform gTII;     ///< gantryToImIdx
    vec2f dist;                    ///< source distance (x,y)

    CUDA_CALLABLE_MEMBER Float3FromFanTransform() : fITF(), gTII(), dist(0.0f, 0.0f) {}

    CUDA_CALLABLE_MEMBER Float3FromFanTransform(const Float3IdxTransform fanIdxToFan,
                                                const vec2f sourceDist,
                                                const Float3AffineTransform gantryToImIdx)
        : fITF(fanIdxToFan), gTII(gantryToImIdx), dist(sourceDist) {}

    CUDA_CALLABLE_MEMBER vec3f transformPoint(const vec3f fanIdx) const {
        // fanIdx -> fan (gantry) coordinates
        vec3f interm = fITF.transformPoint(fanIdx);

        // Apply divergence scaling (z points away from beam direction)
        interm.x *= (1.0f - interm.z / dist.x);
        interm.y *= (1.0f - interm.z / dist.y);

        // gantry -> image index
        return gTII.transformPoint(interm);
    }

    CUDA_CALLABLE_MEMBER Float3ToFanTransform inverse() const;

    CUDA_CALLABLE_MEMBER Float3ToFanTransform invertAndShift(const vec3f shift) const;

    CUDA_CALLABLE_MEMBER Float3IdxTransform getFanIdxToFan() const { return fITF; }
    CUDA_CALLABLE_MEMBER Float3AffineTransform getGantryToImIdx() const { return gTII; }
    CUDA_CALLABLE_MEMBER vec2f getSourceDist() const { return dist; }
};

/**
 * \brief Transform from image-index space to fan-index space.
 */
struct Float3ToFanTransform {
    Float3IdxTransform fTFI;        ///< fanToFanIdx
    Float3AffineTransform iITG;     ///< imIdxToGantry
    vec2f dist;                    ///< source distance (x,y)

    CUDA_CALLABLE_MEMBER Float3ToFanTransform() : fTFI(), iITG(), dist(0.0f, 0.0f) {}

    CUDA_CALLABLE_MEMBER Float3ToFanTransform(const Float3AffineTransform imIdxToGantry,
                                              const vec2f sourceDist,
                                              const Float3IdxTransform fanToFanIdx)
        : fTFI(fanToFanIdx), iITG(imIdxToGantry), dist(sourceDist) {}

    CUDA_CALLABLE_MEMBER vec3f transformPoint(const vec3f imIdx) const {
        // image index -> gantry coords
        vec3f interm = iITG.transformPoint(imIdx);

        // Apply inverse divergence scaling
        interm.x /= (1.0f - interm.z / dist.x);
        interm.y /= (1.0f - interm.z / dist.y);

        // gantry coords -> fan index
        return fTFI.transformPoint(interm);
    }

    CUDA_CALLABLE_MEMBER Float3FromFanTransform inverse() const {
        // Inverse matches reference: fanIdxToFan = fTFI.inverse(), gantryToImIdx = iITG.inverse()
        return Float3FromFanTransform(fTFI.inverse(), dist, iITG.inverse());
    }

    CUDA_CALLABLE_MEMBER Float3IdxTransform getFanToFanIdx() const { return fTFI; }
    CUDA_CALLABLE_MEMBER Float3AffineTransform getImIdxToGantry() const { return iITG; }
    CUDA_CALLABLE_MEMBER vec2f getSourceDist() const { return dist; }
};

// -----------------------------------------------------------------------------
// Inline definitions that require both types
// -----------------------------------------------------------------------------
CUDA_CALLABLE_MEMBER inline Float3ToFanTransform Float3FromFanTransform::inverse() const {
    return Float3ToFanTransform(gTII.inverse(), dist, fITF.inverse());
}

CUDA_CALLABLE_MEMBER inline Float3ToFanTransform Float3FromFanTransform::invertAndShift(const vec3f shift) const {
    const Float3IdxTransform fTFI = fITF.inverse().shiftOffset(shift);
    return Float3ToFanTransform(gTII.inverse(), dist, fTFI);
}

#endif // FAN_TRANSFORMS_H
