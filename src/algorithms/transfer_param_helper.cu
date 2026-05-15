/**
 * \file
 * \brief Helper function to create TransferParamStructDiv3 from beam parameters
 */

#include "../include/algorithms/transfer_param_helper.h"

/**
 * \brief Create TransferParamStructDiv3 from simplified beam parameters
 * This is a simplified version that constructs the transform directly from ray spacing and source distance
 */
__host__ TransferParamStructDiv3 createTransferParamFromBeamParams(
    const vec3f raySpacing,      // Spacing between rays at origin
    const vec3f rayOffset,       // Offset of ray grid
    const vec2f sourceDist,      // Source distance (SAD)
    const vec3f doseVolSpacing,  // Dose volume spacing
    const vec3f doseVolOrigin,   // Dose volume origin
    const vec3f shift            // Shift to account for padding and beamFirstInside
) {
    TransferParamStructDiv3 params;
    
    // Simplified transform: assumes beam direction is along Z axis
    // This matches the simplified beam setup in the wrapper
    // For a proper implementation, we would need full Float3ToFanTransform
    
    // Calculate coefficients based on ray spacing and dose volume spacing
    // The transform maps from dose volume indices to BEV fan indices
    // BEV fan indices are in units of ray spacing, so we need to convert dose volume indices to ray indices
    // raySpacing is the spacing between rays in BEV coordinate system (CPB resolution)
    // doseVolSpacing is the spacing between dose volume voxels
    
    // coefIdxI: maps dose volume X index to BEV X index (in ray spacing units)
    // Since ray grid has spacing raySpacing.x and dose grid has spacing doseVolSpacing.x,
    // we need to scale by the ratio
    params.coefIdxI = make_vec3f(
        doseVolSpacing.x / raySpacing.x,  // Convert dose index to ray index
        0.0f,
        0.0f
    );
    params.coefIdxJ = make_vec3f(
        0.0f,
        doseVolSpacing.y / raySpacing.y,  // Convert dose index to ray index
        0.0f
    );
    
    // coefOffset: offset in BEV coordinate system (ray indices)
    // This represents the BEV coordinate for dose volume index (0, 0)
    // BEV coordinate system origin is at CPB corner (rayOffset)
    // Dose volume index (0, 0) corresponds to dose volume origin (doseVolOrigin)
    // So we need to find the BEV coordinate for dose volume origin
    // BEV coordinate = (doseVolOrigin - rayOffset) / raySpacing
    // But since BEV origin is at rayOffset, this gives us the offset
    params.coefOffset = make_vec3f(
        (doseVolOrigin.x - rayOffset.x) / raySpacing.x,  // BEV X coordinate for dose index 0
        (doseVolOrigin.y - rayOffset.y) / raySpacing.y,  // BEV Y coordinate for dose index 0
        (doseVolOrigin.z - rayOffset.z) / doseVolSpacing.z  // Z uses dose volume spacing
    );
    
    // Increment per Z step in BEV coordinate system
    // Z step is in dose volume indices, need to convert to BEV Z indices
    params.inc = make_vec3f(
        0.0f,
        0.0f,
        doseVolSpacing.z / raySpacing.z  // Convert dose Z step to BEV Z step
    );
    
    // Global offset accounts for padding
    params.globalOffset = shift;
    
    // Normalized distance for fan beam divergence
    // Based on original RayTracedicom: normDist = delta.z * sourceDist
    // where delta.z is the step size in BEV coordinate system (ray spacing in Z direction)
    // For divergent beam, we need to account for the divergence factor
    // In the simplified case, delta.z should be the ray spacing in Z direction
    // But since we're stepping through dose volume Z, we use doseVolSpacing.z
    float deltaZ = doseVolSpacing.z;  // Step size in dose volume Z direction (matches ray spacing.z)
    params.normDist = make_vec2f(
        deltaZ * sourceDist.x,
        deltaZ * sourceDist.y
    );
    
    return params;
}

