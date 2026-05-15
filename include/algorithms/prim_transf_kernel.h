/**
 * \file
 * \brief primTransfDiv kernel declaration
 */

#ifndef PRIM_TRANSF_KERNEL_H
#define PRIM_TRANSF_KERNEL_H

#include "common.cuh"
#include "transfer_param_struct_div3.cuh"
#include "Macro.cuh"

/**
 * \brief Kernel transferring from BEV index to dose index
 * \param result where the resulting 3D array is stored, preallocated, not owned, linearized
 * \param params the settings of the transfer kernel
 * \param startIdx the starting 3D point (indices)
 * \param maxZ the maximum index in Z
 * \param doseDims the 3D dimensions of the dose matrix
 * \param bevPrimDoseTex 3D dose texture matrix
 */
__global__ void primTransfDiv(
    float* const result,
    TransferParamStructDiv3 params,
    const vec3i startIdx,
    const int maxZ,
    const vec3i doseDims,
    cudaTextureObject_t bevPrimDoseTex
);

__global__ void nucTransfDiv(
    float* const result,
    TransferParamStructDiv3 params,
    const vec3i startIdx,
    const int maxZ,
    const vec3i doseDims,
    cudaTextureObject_t bevNucDoseTex
);

#endif // PRIM_TRANSF_KERNEL_H
