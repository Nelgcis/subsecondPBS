/**
 * \file
 * \brief CUDA 12.1 Advanced Memory Pool and Pre-compiled Texture Headers
 */

#ifndef ADVANCED_MEMORY_TEXTURE_H
#define ADVANCED_MEMORY_TEXTURE_H

#include <string>
#include <cuda_runtime.h>
#include <texture_types.h>

// Advanced GPU Memory Pool
void initializeAdvancedMemoryPool();
void cleanupAdvancedMemoryPool();

// Pre-compiled Texture Manager
void initializePrecompiledTextureManager();
void cleanupPrecompiledTextureManager();

// LUT texture access
cudaTextureObject_t getLUTTexture(const std::string& filename);

// Pre-compiled texture creation
cudaTextureObject_t createPrecompiledTexture(const float* data, const int3& dims,
                                            cudaTextureFilterMode filterMode,
                                            cudaTextureAddressMode addressMode);

#endif // ADVANCED_MEMORY_TEXTURE_H
