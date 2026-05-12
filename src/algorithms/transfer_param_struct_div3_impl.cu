/**
 * \file
 * \brief TransferParamStructDiv3 implementation from Float3ToFanTransform
 */

#include "transfer_param_struct_div3.cuh"
#include "Macro.cuh"
#include "fan_transforms.h"
#include "matrix_3x3.h"

// The implementation is copied from the reference RayTraceDicom project.
// It constructs a compact parameterization of the inverse (doseIdx -> fanIdx)
// mapping used by primTransfDiv.
__host__ __device__ TransferParamStructDiv3::TransferParamStructDiv3(
    const Float3ToFanTransform& imIdxToFanIdx
) {
    // Transpose of the linear part of (imIdx -> gantry) so that we can compute
    // dot-products with (i,j,k) efficiently.
    const Matrix3x3 tTransp = imIdxToFanIdx.getImIdxToGantry().getMatrix().transpose();
    const vec3f delta = imIdxToFanIdx.getFanToFanIdx().getDelta();

    coefIdxI = tTransp.row0() * delta;
    coefIdxJ = tTransp.row1() * delta;
    coefOffset = imIdxToFanIdx.getImIdxToGantry().getOffset() * delta;
    globalOffset = imIdxToFanIdx.getFanToFanIdx().getOffset();
    inc = tTransp.row2() * delta;
    start = vec3f(0.0f, 0.0f, 0.0f);

    // normDist rescales z into the divergence correction in getFanIdx().
    normDist = make_vec2f(
        delta.z * imIdxToFanIdx.getSourceDist().x,
        delta.z * imIdxToFanIdx.getSourceDist().y
    );
}
