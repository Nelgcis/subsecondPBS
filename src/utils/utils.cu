/**
 * \file
 * \brief Utility Functions Implementation for RTD Integration
 */

#include "../include/utils/utils.h"
#include <cuda_runtime.h>
#include <texture_types.h>
#include "../include/utils/debug_tools.h"
#include "../include/utils/texture_ultra_optimized.h"
#include "../include/utils/advanced_memory_texture.h"
#include <iostream>

static cudaTextureObject_t create3DTextureLegacy(const float* data, const int3& dims,
                                                cudaTextureFilterMode filterMode,
                                                cudaTextureAddressMode addressMode);
static bool resolveArrayUploadKind(const void* data, cudaMemcpyKind& kind);
static cudaTextureObject_t createTextureObjectFromArray(cudaArray_t devArray,
                                                        int ndims,
                                                        cudaTextureFilterMode filterMode,
                                                        cudaTextureAddressMode addressMode);

static bool resolveArrayUploadKind(const void* data, cudaMemcpyKind& kind) {
    if (data == nullptr) {
        if (rtdVerboseFineTiming()) {
            std::cerr << "[RTD] Texture creation received a null data pointer." << std::endl;
        }
        return false;
    }

    cudaPointerAttributes attrs = {};
    const cudaError_t attrErr = cudaPointerGetAttributes(&attrs, data);
    if (attrErr == cudaSuccess) {
#if CUDART_VERSION >= 10000
        if (attrs.type == cudaMemoryTypeDevice || attrs.type == cudaMemoryTypeManaged) {
            kind = cudaMemcpyDefault;
            return true;
        }
        kind = cudaMemcpyHostToDevice;
        return true;
#else
        if (attrs.memoryType == cudaMemoryTypeDevice) {
            kind = cudaMemcpyDeviceToDevice;
            return true;
        }
        kind = cudaMemcpyHostToDevice;
        return true;
#endif
    }

    if (attrErr == cudaErrorInvalidValue || attrErr == cudaErrorInvalidDevicePointer) {
        cudaGetLastError();
        kind = cudaMemcpyHostToDevice;
        return true;
    }

    cudaGetLastError();
    if (rtdVerboseFineTiming()) {
        std::cerr << "[RTD] cudaPointerGetAttributes failed during texture creation: "
                  << cudaGetErrorString(attrErr) << std::endl;
    }
    return false;
}

static cudaTextureObject_t createTextureObjectFromArray(cudaArray_t devArray,
                                                        int ndims,
                                                        cudaTextureFilterMode filterMode,
                                                        cudaTextureAddressMode addressMode) {
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = devArray;

    cudaTextureDesc texDesc = {};
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;
    texDesc.filterMode = filterMode;
    if (ndims > 0) texDesc.addressMode[0] = addressMode;
    if (ndims > 1) texDesc.addressMode[1] = addressMode;
    if (ndims > 2) texDesc.addressMode[2] = addressMode;

    cudaTextureObject_t texObj = 0;
    const cudaError_t err = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
    if (err != cudaSuccess) {
        printf("Error: cudaCreateTextureObject failed: %s\n", cudaGetErrorString(err));
        return 0;
    }
    return texObj;
}

// Texture creation utilities
cudaTextureObject_t create3DTexture(const float* data, const int3& dims,
                                   cudaTextureFilterMode filterMode,
                                   cudaTextureAddressMode addressMode) {
    // Try the optimized path first, then fall back to the conservative legacy path.
    cudaTextureObject_t tex = create3DTextureUltraOptimized(data, dims, filterMode, addressMode);
    if (tex != 0) {
        return tex;
    }

    cudaGetLastError();
    if (rtdVerboseFineTiming()) {
        std::cerr << "[RTD] create3DTextureUltraOptimized failed, falling back to legacy texture creation." << std::endl;
    }
    return create3DTextureLegacy(data, dims, filterMode, addressMode);
}

// Legacy version for backward compatibility
cudaTextureObject_t create3DTextureLegacy(const float* data, const int3& dims,
                                        cudaTextureFilterMode filterMode,
                                        cudaTextureAddressMode addressMode) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // 优化1: 使用CUDA 12.1的优化内存分配
    cudaChannelFormatDesc floatChannelDesc = cudaCreateChannelDesc<float>();
    
    cudaArray* devArray = nullptr;
    cudaExtent extent = make_cudaExtent(dims.x, dims.y, dims.z);
    cudaError_t err = cudaMalloc3DArray(&devArray, &floatChannelDesc, extent);
    if (err != cudaSuccess) {
        printf("Error: cudaMalloc3DArray failed: %s\n", cudaGetErrorString(err));
        return 0;
    }

    cudaMemcpyKind copyKind;
    if (!resolveArrayUploadKind(data, copyKind)) {
        cudaFreeArray(devArray);
        return 0;
    }
    
    // 优化2: 使用异步内存拷贝和错误检查
    cudaMemcpy3DParms copyParams = {};
    copyParams.srcPtr = make_cudaPitchedPtr((void*)data, extent.width*sizeof(float), extent.width, extent.height);
    copyParams.dstArray = devArray;
    copyParams.extent = extent;
    copyParams.kind = copyKind;
    
    err = cudaMemcpy3D(&copyParams);
    if (err != cudaSuccess) {
        printf("Error: cudaMemcpy3D failed: %s\n", cudaGetErrorString(err));
        cudaFreeArray(devArray);
        return 0;
    }
    
    // 优化3: 使用更高效的纹理描述符设置
    cudaTextureObject_t texObj = createTextureObjectFromArray(devArray, 3, filterMode, addressMode);
    if (texObj == 0) {
        cudaFreeArray(devArray);
        return 0;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    if (rtdVerboseFineTiming()) printf("[TIMING] create3DTexture (optimized): %ld μs\n", duration.count());
    return texObj;
}

cudaTextureObject_t create2DTexture(const float* data, const int2& dims,
                                   cudaTextureFilterMode filterMode,
                                   cudaTextureAddressMode addressMode) {
    auto start = std::chrono::high_resolution_clock::now();
    
    cudaChannelFormatDesc floatChannelDesc = cudaCreateChannelDesc<float>();
    
    cudaArray* devArray = nullptr;
    cudaError_t err = cudaMallocArray(&devArray, &floatChannelDesc, dims.x, dims.y);
    if (err != cudaSuccess) {
        printf("Error: cudaMallocArray failed: %s\n", cudaGetErrorString(err));
        return 0;
    }

    cudaMemcpyKind copyKind;
    if (!resolveArrayUploadKind(data, copyKind)) {
        cudaFreeArray(devArray);
        return 0;
    }
    
    // Use cudaMemcpy3D instead of deprecated cudaMemcpyToArray
    cudaMemcpy3DParms copyParams = {};
    copyParams.srcPtr = make_cudaPitchedPtr((void*)data, dims.x * sizeof(float), dims.x, dims.y);
    copyParams.dstArray = devArray;
    copyParams.extent = make_cudaExtent(dims.x, dims.y, 1);
    copyParams.kind = copyKind;
    err = cudaMemcpy3D(&copyParams);
    if (err != cudaSuccess) {
        printf("Error: cudaMemcpy3D failed: %s\n", cudaGetErrorString(err));
        cudaFreeArray(devArray);
        return 0;
    }

    cudaTextureObject_t texObj = createTextureObjectFromArray(devArray, 2, filterMode, addressMode);
    if (texObj == 0) {
        cudaFreeArray(devArray);
        return 0;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    if (rtdVerboseFineTiming()) printf("[TIMING] create2DTexture: %ld μs\n", duration.count());
    return texObj;
}

cudaTextureObject_t create1DTexture(const float* data, int size,
                                   cudaTextureFilterMode filterMode,
                                   cudaTextureAddressMode addressMode) {
    auto start = std::chrono::high_resolution_clock::now();
    
    cudaChannelFormatDesc floatChannelDesc = cudaCreateChannelDesc<float>();
    
    cudaArray* devArray = nullptr;
    cudaError_t err = cudaMallocArray(&devArray, &floatChannelDesc, size);
    if (err != cudaSuccess) {
        printf("Error: cudaMallocArray failed: %s\n", cudaGetErrorString(err));
        return 0;
    }

    cudaMemcpyKind copyKind;
    if (!resolveArrayUploadKind(data, copyKind)) {
        cudaFreeArray(devArray);
        return 0;
    }
    
    // Use cudaMemcpy3D instead of deprecated cudaMemcpyToArray
    cudaMemcpy3DParms copyParams = {};
    copyParams.srcPtr = make_cudaPitchedPtr((void*)data, size * sizeof(float), size, 1);
    copyParams.dstArray = devArray;
    copyParams.extent = make_cudaExtent(size, 1, 1);
    copyParams.kind = copyKind;
    err = cudaMemcpy3D(&copyParams);
    if (err != cudaSuccess) {
        printf("Error: cudaMemcpy3D failed: %s\n", cudaGetErrorString(err));
        cudaFreeArray(devArray);
        return 0;
    }

    cudaTextureObject_t texObj = createTextureObjectFromArray(devArray, 1, filterMode, addressMode);
    if (texObj == 0) {
        cudaFreeArray(devArray);
        return 0;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    if (rtdVerboseFineTiming()) printf("[TIMING] create1DTexture: %ld μs\n", duration.count());
    return texObj;
}

void destroyTextureObjectAndArray(cudaTextureObject_t texObj) {
    if (texObj == 0) return;

    cudaResourceDesc resDesc = {};
    cudaArray_t devArray = nullptr;
    const cudaError_t resErr = cudaGetTextureObjectResourceDesc(&resDesc, texObj);
    if (resErr == cudaSuccess && resDesc.resType == cudaResourceTypeArray) {
        devArray = resDesc.res.array.array;
    } else if (resErr != cudaSuccess && rtdVerboseFineTiming()) {
        std::cerr << "[RTD] cudaGetTextureObjectResourceDesc failed during texture cleanup: "
                  << cudaGetErrorString(resErr) << std::endl;
    }

    const cudaError_t destroyErr = cudaDestroyTextureObject(texObj);
    if (destroyErr != cudaSuccess && rtdVerboseFineTiming()) {
        std::cerr << "[RTD] cudaDestroyTextureObject failed: "
                  << cudaGetErrorString(destroyErr) << std::endl;
    }

    if (devArray != nullptr) {
        const cudaError_t freeErr = cudaFreeArray(devArray);
        if (freeErr != cudaSuccess && rtdVerboseFineTiming()) {
            std::cerr << "[RTD] cudaFreeArray failed during texture cleanup: "
                      << cudaGetErrorString(freeErr) << std::endl;
        }
    }
}

// Memory management utilities
void* allocateDeviceMemory(size_t size) {
    auto start = std::chrono::high_resolution_clock::now();
    void* ptr;
    cudaMalloc(&ptr, size);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    if (rtdVerboseFineTiming()) printf("[TIMING] allocateDeviceMemory: %ld μs\n", duration.count());
    return ptr;
}

void freeDeviceMemory(void* ptr) {
    auto start = std::chrono::high_resolution_clock::now();
    cudaFree(ptr);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    if (rtdVerboseFineTiming()) printf("[TIMING] freeDeviceMemory: %ld μs\n", duration.count());
}

void copyToDevice(void* dst, const void* src, size_t size) {
    auto start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    if (rtdVerboseFineTiming()) printf("[TIMING] copyToDevice: %ld μs\n", duration.count());
}

void copyFromDevice(void* dst, const void* src, size_t size) {
    auto start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    if (rtdVerboseFineTiming()) printf("[TIMING] copyFromDevice: %ld μs\n", duration.count());
}

void copyToHost(void* dst, const void* src, size_t size) {
    auto start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    if (rtdVerboseFineTiming()) printf("[TIMING] copyToHost: %ld μs\n", duration.count());
}

// Grid and block size calculation utilities
dim3 calculateGridSize(const dim3& blockSize, const int3& problemSize) {
    return dim3((problemSize.x + blockSize.x - 1) / blockSize.x,
                (problemSize.y + blockSize.y - 1) / blockSize.y,
                (problemSize.z + blockSize.z - 1) / blockSize.z);
}

dim3 calculateGridSize(const dim3& blockSize, const int2& problemSize) {
    return dim3((problemSize.x + blockSize.x - 1) / blockSize.x,
                (problemSize.y + blockSize.y - 1) / blockSize.y);
}

// Debug utilities
void printDeviceInfo() {
    auto start = std::chrono::high_resolution_clock::now();
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    if (rtdVerboseFineTiming()) {
        std::cout << "CUDA Device Information:" << std::endl;
        std::cout << "Number of devices: " << deviceCount << std::endl;
    }
    /*
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        std::cout << "Device " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Global memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
    }
    */
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    if (rtdVerboseFineTiming()) printf("[TIMING] printDeviceInfo: %ld μs\n", duration.count());
}

void printMemoryInfo() {
    auto start = std::chrono::high_resolution_clock::now();
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    
    if (rtdVerboseFineTiming()) {
        std::cout << "GPU Memory Information:" << std::endl;
        std::cout << "  Total: " << total / (1024*1024) << " MB" << std::endl;
        std::cout << "  Free: " << free / (1024*1024) << " MB" << std::endl;
        std::cout << "  Used: " << (total - free) / (1024*1024) << " MB" << std::endl;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    if (rtdVerboseFineTiming()) printf("[TIMING] printMemoryInfo: %ld μs\n", duration.count());
}

void checkCudaError(const char* message) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << message << " - " << cudaGetErrorString(error) << std::endl;
    }
}
