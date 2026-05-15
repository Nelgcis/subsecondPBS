/**
 * \file
 * \brief primTransfDiv kernel implementation
 */

#include "prim_transf_kernel.h"
#include "transfer_param_struct_div3.cuh"
#include "common.cuh"
#include "Macro.cuh"

__global__ void primTransfDiv(
    float* const result,
    TransferParamStructDiv3 params,
    const vec3i startIdx,
    const int maxZ,
    const vec3i doseDims,
    cudaTextureObject_t bevPrimDoseTex
) {
    unsigned int x = startIdx.x + blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = startIdx.y + blockDim.y * blockIdx.y + threadIdx.y;

    if (x < doseDims.x && y < doseDims.y) {
        params.init(x, y); // Initialize object with current index position
        const int sliceStride = doseDims.x * doseDims.y;
        float* res = result + startIdx.z * sliceStride + y * doseDims.x + x;
        
        for (int z = startIdx.z; z <= maxZ; ++z) {
            vec3f pos = params.getFanIdx(z) + make_vec3f(HALF, HALF, HALF); // Compensate for voxel value sitting at centre of voxel
            
            // Check bounds
            if (pos.x < 0.0f || pos.y < 0.0f || pos.z < 0.0f) {
            res += sliceStride;
            continue;
            }
            
            float tmp = tex3D<float>(bevPrimDoseTex, pos.x, pos.y, pos.z);
            
            if (tmp > 0.0f) { // Only write to global memory if non-zero
                *res += tmp;
            }
            res += sliceStride;
        }
    }
}

__global__ void nucTransfDiv(
    float* const result,
    TransferParamStructDiv3 params,
    const vec3i startIdx,
    const int maxZ,
    const vec3i doseDims,
    cudaTextureObject_t bevNucDoseTex
) {
    unsigned int x = startIdx.x + blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = startIdx.y + blockDim.y * blockIdx.y + threadIdx.y;

    if (x < doseDims.x && y < doseDims.y) {
        params.init(x, y);
        const int sliceStride = doseDims.x * doseDims.y;
        float* res = result + startIdx.z * sliceStride + y * doseDims.x + x;

        for (int z = startIdx.z; z <= maxZ; ++z) {
            vec3f pos = params.getFanIdx(z) + make_vec3f(HALF, HALF, HALF);

            if (pos.x < 0.0f || pos.y < 0.0f || pos.z < 0.0f) {
                res += sliceStride;
                continue;
            }

            float tmp = tex3D<float>(bevNucDoseTex, pos.x, pos.y, pos.z);

            if (tmp > 0.0f) {
                *res += tmp;
            }
            res += sliceStride;
        }
    }
}
