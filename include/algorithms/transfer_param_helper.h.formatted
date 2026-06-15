/**
 * \file
 * \brief Helper function declarations for TransferParamStructDiv3
 */

#ifndef TRANSFER_PARAM_HELPER_H
#define TRANSFER_PARAM_HELPER_H

#include "transfer_param_struct_div3.cuh"
#include "common.cuh"
#include "Macro.cuh"

/**
 * \brief Create TransferParamStructDiv3 from simplified beam parameters
 * This is a simplified version that constructs the transform directly from ray spacing and source distance
 */
__host__ TransferParamStructDiv3 createTransferParamFromBeamParams(
    const vec3f raySpacing,      // Spacing between rays at origin
    const vec3f rayOffset,       // Offset of ray grid
    const vec2f sourceDist,       // Source distance (SAD)
    const vec3f doseVolSpacing,  // Dose volume spacing
    const vec3f doseVolOrigin,    // Dose volume origin
    const vec3f shift             // Shift to account for padding and beamFirstInside
);

#endif // TRANSFER_PARAM_HELPER_H

