/**
 * \file
 * \brief Ray tracing parameter structs (ported from RayTraceDicom-main)
 *
 * IMPORTANT:
 * - The *reference* implementation performs ray tracing in a divergent (fan)
 *   coordinate system ("z points away from beam direction") and constructs
 *   start/increment/stepLen from a Float3FromFanTransform.
 * - This header restores the same behavior.
 */

#ifndef RAY_TRACING_H
#define RAY_TRACING_H


#include <cuda_runtime.h>
#include "common.cuh"
#include "Macro.cuh"
#include "fan_transforms.h"
#include "matrix_3x3.h"

/**
 * \brief Parameters required by the BEV (fan) ray tracer.
 *
 * Ported from RayTraceDicom-main/src/density_and_sp_tracer_params.*
 */
struct DensityAndSpTracerParams {
private:
    float densityScale;
    float spScale;
    unsigned int steps;

    vec3f coefOffset;
    vec3f coefIdxI;
    vec3f coefIdxJ;
    vec3f transl;

    vec3f corner;
    vec3f delta;
    vec2f dist;

public:
    CUDA_CALLABLE_MEMBER DensityAndSpTracerParams()
        : densityScale(0.0f), spScale(0.0f), steps(0),
          coefOffset(0.0f, 0.0f, 0.0f), coefIdxI(0.0f, 0.0f, 0.0f), coefIdxJ(0.0f, 0.0f, 0.0f),
          transl(0.0f, 0.0f, 0.0f), corner(0.0f, 0.0f, 0.0f), delta(1.0f, 1.0f, 1.0f), dist(1.0f, 1.0f) {}

    CUDA_CALLABLE_MEMBER DensityAndSpTracerParams(const float densityScaleFact,
                                                  const float spScaleFact,
                                                  const unsigned int tracerSteps,
                                                  const Float3FromFanTransform fanIdxToImIdx)
        : densityScale(densityScaleFact), spScale(spScaleFact), steps(tracerSteps) {
        dist = fanIdxToImIdx.getSourceDist();
        corner = fanIdxToImIdx.getFanIdxToFan().getOffset();
        delta = fanIdxToImIdx.getFanIdxToFan().getDelta();

        const Matrix3x3 tTransp = fanIdxToImIdx.getGantryToImIdx().getMatrix().transpose();
        coefOffset = tTransp.row2()
                  - tTransp.row0() * (corner.x / dist.x)
                  - tTransp.row1() * (corner.y / dist.y);
        coefIdxI = tTransp.row0() * delta.x;
        coefIdxJ = tTransp.row1() * delta.y;

        transl = fanIdxToImIdx.getGantryToImIdx().getOffset()
               + tTransp.row2() * corner.z
               + tTransp.row0() * (corner.x * (1.0f - corner.z / dist.x))
               + tTransp.row1() * (corner.y * (1.0f - corner.z / dist.y));
    }

    CUDA_CALLABLE_MEMBER vec3f getStart(const unsigned int idxI, const unsigned int idxJ) const {
        return coefIdxI * (float(idxI) * (1.0f - corner.z / dist.x))
             + coefIdxJ * (float(idxJ) * (1.0f - corner.z / dist.y))
             + transl;
    }

    CUDA_CALLABLE_MEMBER vec3f getInc(const unsigned int idxI, const unsigned int idxJ) const {
        return (coefOffset
              - coefIdxI * (float(idxI) / dist.x)
              - coefIdxJ * (float(idxJ) / dist.y))
             * delta.z;
    }

    CUDA_CALLABLE_MEMBER float stepLen(const unsigned int idxI, const unsigned int idxJ) const {
        const float deltaX = (corner.x + float(idxI) * delta.x) / dist.x;
        const float deltaY = (corner.y + float(idxJ) * delta.y) / dist.y;
        return fabsf(delta.z) * sqrtf(1.0f + deltaX * deltaX + deltaY * deltaY);
    }

    CUDA_CALLABLE_MEMBER unsigned int getSteps() const { return steps; }
    CUDA_CALLABLE_MEMBER float getDensityScale() const { return densityScale; }
    CUDA_CALLABLE_MEMBER float getSpScale() const { return spScale; }
};

// (The actual ray tracing kernel is implemented in src/core/bev_ray_tracing.cu)

#endif // RAY_TRACING_H
