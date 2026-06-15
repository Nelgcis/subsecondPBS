/**
 * \file
 * \brief CUDA 12.1 Ultra-Optimized Texture Creation Headers
 */

#ifndef TEXTURE_ULTRA_OPTIMIZED_H
#define TEXTURE_ULTRA_OPTIMIZED_H

#include <cuda_runtime.h>
#include <texture_types.h>

// Ultra-optimized texture creation functions
cudaTextureObject_t create3DTextureUltraOptimized(const float* data, const int3& dims,
                                                 cudaTextureFilterMode filterMode,
                                                 cudaTextureAddressMode addressMode);

// CUDA unified memory version
cudaTextureObject_t create3DTextureUnifiedMemory(const float* data, const int3& dims,
                                                cudaTextureFilterMode filterMode,
                                                cudaTextureAddressMode addressMode);

#endif // TEXTURE_ULTRA_OPTIMIZED_H
