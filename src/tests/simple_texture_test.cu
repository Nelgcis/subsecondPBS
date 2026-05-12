/**
 * \file
 * \brief Simple Texture Access Test Kernel
 */

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <texture_types.h>

// Simple kernel to test texture access
__global__ void testTextureAccessKernel(cudaTextureObject_t texture, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Test different texture coordinates
    float val1 = tex3D<float>(texture, 0.0f, 0.0f, 0.0f);
    float val2 = tex3D<float>(texture, 1.0f, 0.0f, 0.0f);
    float val3 = tex3D<float>(texture, 0.0f, 1.0f, 0.0f);
    
    if (idx == 0) {
        printf("GPU: tex3D(0,0,0)=%.6f, tex3D(1,0,0)=%.6f, tex3D(0,1,0)=%.6f\n", val1, val2, val3);
    }
    
    output[idx] = val1;
}

int main() {
    std::cout << "=== Simple Texture Access Test ===" << std::endl;
    
    // Create simple test data: 2x2x2 array
    std::vector<float> testData = {
        // Layer 0
        1.0f, 2.0f, 3.0f, 4.0f,  // Channel 0: [1,2,3,4]
        5.0f, 6.0f, 7.0f, 8.0f,  // Channel 1: [5,6,7,8]
        // Layer 1  
        9.0f, 10.0f, 11.0f, 12.0f, // Channel 0: [9,10,11,12]
        13.0f, 14.0f, 15.0f, 16.0f // Channel 1: [13,14,15,16]
    };
    
    std::cout << "Test data layout:" << std::endl;
    std::cout << "Layer 0, Channel 0: [1,2,3,4]" << std::endl;
    std::cout << "Layer 0, Channel 1: [5,6,7,8]" << std::endl;
    std::cout << "Layer 1, Channel 0: [9,10,11,12]" << std::endl;
    std::cout << "Layer 1, Channel 1: [13,14,15,16]" << std::endl;
    
    // Create 3D texture: width=4, height=2, depth=2
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray* devArray;
    cudaExtent extent = make_cudaExtent(4, 2, 2);
    cudaMalloc3DArray(&devArray, &channelDesc, extent);
    
    // Copy data to device
    float* devData;
    cudaMalloc(&devData, testData.size() * sizeof(float));
    cudaMemcpy(devData, testData.data(), testData.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    // Copy to 3D array
    cudaMemcpy3DParms copyParams = {};
    copyParams.srcPtr = make_cudaPitchedPtr(devData, 4 * sizeof(float), 4, 2);
    copyParams.dstArray = devArray;
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyDeviceToDevice;
    cudaMemcpy3D(&copyParams);
    
    // Create texture object
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = devArray;
    
    cudaTextureDesc texDesc = {};
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    
    cudaTextureObject_t texture;
    cudaCreateTextureObject(&texture, &resDesc, &texDesc, NULL);
    
    // Test texture access
    float* d_output;
    cudaMalloc(&d_output, sizeof(float));
    
    testTextureAccessKernel<<<1, 1>>>(texture, d_output, 1);
    cudaDeviceSynchronize();
    
    // Copy result back
    float result;
    cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "Host result: " << result << std::endl;
    
    // Cleanup
    cudaFree(d_output);
    cudaDestroyTextureObject(texture);
    cudaFreeArray(devArray);
    cudaFree(devData);
    
    std::cout << "\n=== Test Complete ===" << std::endl;
    return 0;
}