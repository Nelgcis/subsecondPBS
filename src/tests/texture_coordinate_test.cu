/**
 * \file
 * \brief Texture Coordinate Mapping Test
 * 
 * This file tests the correct texture coordinate mapping for subspot data
 */

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <texture_types.h>

// Test kernel to verify texture coordinate mapping
__global__ void testTextureCoordinatesKernel(cudaTextureObject_t texture, float* output, int width, int height, int depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height * depth) return;
    
    // Calculate 3D coordinates from linear index
    int x = idx % width;
    int y = (idx / width) % height;
    int z = idx / (width * height);
    
    // Test texture access with correct coordinate mapping (0.5 offset for pixel center)
    float val1 = tex3D<float>(texture, float(x) + 0.5f, float(y) + 0.5f, float(z) + 0.5f);
    float val2 = tex3D<float>(texture, float(x), float(y), float(z));
    
    if (idx < 10) { // Only print first 10 values
        printf("GPU: idx=%d, coords=(%d,%d,%d), val1=%.6f, val2=%.6f\n", idx, x, y, z, val1, val2);
    }
    
    output[idx] = val1;
}

int main() {
    std::cout << "=== Texture Coordinate Mapping Test ===" << std::endl;
    
    // Create test data with known pattern
    int width = 5;   // channels (deltaX, deltaY, weight, sigmaX, sigmaY)
    int height = 3;  // subspots per layer
    int depth = 2;   // layers
    
    std::vector<float> testData(width * height * depth);
    
    // Fill with known pattern: value = x + y*10 + z*100
    for (int z = 0; z < depth; z++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = z * width * height + y * width + x;
                testData[idx] = x + y * 10.0f + z * 100.0f;
            }
        }
    }
    
    std::cout << "Test data pattern (value = x + y*10 + z*100):" << std::endl;
    for (int z = 0; z < depth; z++) {
        std::cout << "Layer " << z << ":" << std::endl;
        for (int y = 0; y < height; y++) {
            std::cout << "  Subspot " << y << ": ";
            for (int x = 0; x < width; x++) {
                int idx = z * width * height + y * width + x;
                std::cout << testData[idx] << " ";
            }
            std::cout << std::endl;
        }
    }
    
    // Create 3D texture
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray* devArray;
    cudaExtent extent = make_cudaExtent(width, height, depth);
    cudaMalloc3DArray(&devArray, &channelDesc, extent);
    
    // Copy data directly to 3D array
    cudaMemcpy3DParms copyParams = {};
    copyParams.srcPtr = make_cudaPitchedPtr(testData.data(), width * sizeof(float), width, height);
    copyParams.dstArray = devArray;
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyHostToDevice;
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
    cudaMalloc(&d_output, testData.size() * sizeof(float));
    
    testTextureCoordinatesKernel<<<1, testData.size()>>>(texture, d_output, width, height, depth);
    cudaDeviceSynchronize();
    
    // Copy results back
    std::vector<float> results(testData.size());
    cudaMemcpy(results.data(), d_output, testData.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "\nTexture access results:" << std::endl;
    for (int z = 0; z < depth; z++) {
        std::cout << "Layer " << z << ":" << std::endl;
        for (int y = 0; y < height; y++) {
            std::cout << "  Subspot " << y << ": ";
            for (int x = 0; x < width; x++) {
                int idx = z * width * height + y * width + x;
                std::cout << results[idx] << " ";
            }
            std::cout << std::endl;
        }
    }
    
    // Cleanup
    cudaFree(d_output);
    cudaDestroyTextureObject(texture);
    cudaFreeArray(devArray);
    
    std::cout << "\n=== Test Complete ===" << std::endl;
    return 0;
}
