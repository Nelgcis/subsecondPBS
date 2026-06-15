/**
 * \file
 * \brief CUDA 12.1 Advanced Texture Pool and Cache System
 * 
 * This file implements:
 * - Texture memory pool pre-allocation
 * - Asynchronous memory operations
 * - CUDA unified memory support
 * - Texture object caching mechanism
 */

#include <cuda_runtime.h>
#include <texture_types.h>
#include <chrono>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <mutex>
#include "../include/utils/debug_tools.h"

// Texture pool configuration
struct TexturePoolConfig {
    size_t maxPoolSize = 1024 * 1024 * 1024; // 1GB pool
    size_t maxCacheSize = 100; // Max cached texture objects
    bool useUnifiedMemory = true;
    bool enableAsyncOps = true;
};

// Texture pool entry
struct TexturePoolEntry {
    cudaArray* array;
    int3 dims;
    cudaExtent extent;
    bool inUse;
    std::chrono::steady_clock::time_point lastUsed;
};

// Texture cache entry
struct TextureCacheEntry {
    cudaTextureObject_t texture;
    int3 dims;
    cudaTextureFilterMode filterMode;
    cudaTextureAddressMode addressMode;
    std::chrono::steady_clock::time_point lastUsed;
};

class TexturePoolManager {
private:
    std::vector<TexturePoolEntry> pool_;
    std::unordered_map<std::string, TextureCacheEntry> cache_;
    std::mutex poolMutex_;
    std::mutex cacheMutex_;
    TexturePoolConfig config_;
    cudaStream_t asyncStream_;
    
    // Statistics
    size_t poolHits_ = 0;
    size_t poolMisses_ = 0;
    size_t cacheHits_ = 0;
    size_t cacheMisses_ = 0;
    
public:
    TexturePoolManager() {
        // Create async stream for asynchronous operations
        cudaStreamCreate(&asyncStream_);
        
        // Pre-allocate some common sizes
        preAllocateCommonSizes();
    }
    
    ~TexturePoolManager() {
        cleanup();
        cudaStreamDestroy(asyncStream_);
    }
    
    // Pre-allocate common texture sizes
    void preAllocateCommonSizes() {
        std::vector<int3> commonSizes = {
            {64, 64, 64},   // Small volume
            {128, 128, 128}, // Medium volume
            {256, 256, 256}, // Large volume
            {512, 512, 64},  // Wide volume
            {64, 512, 512}   // Tall volume
        };
        
        for (const auto& dims : commonSizes) {
            preAllocateEntry(dims);
        }
    }
    
    // Pre-allocate a texture pool entry
    bool preAllocateEntry(const int3& dims) {
        std::lock_guard<std::mutex> lock(poolMutex_);
        
        cudaChannelFormatDesc floatChannelDesc = cudaCreateChannelDesc<float>();
        cudaExtent extent = make_cudaExtent(dims.x, dims.y, dims.z);
        
        cudaArray* array;
        cudaError_t err = cudaMalloc3DArray(&array, &floatChannelDesc, extent);
        if (err != cudaSuccess) {
            printf("Warning: Failed to pre-allocate texture pool entry: %s\n", cudaGetErrorString(err));
            return false;
        }
        
        TexturePoolEntry entry;
        entry.array = array;
        entry.dims = dims;
        entry.extent = extent;
        entry.inUse = false;
        entry.lastUsed = std::chrono::steady_clock::now();
        
        pool_.push_back(entry);
        return true;
    }
    
    // Get a texture pool entry
    TexturePoolEntry* getPoolEntry(const int3& dims) {
        std::lock_guard<std::mutex> lock(poolMutex_);
        
        // Look for exact match
        for (auto& entry : pool_) {
            if (!entry.inUse && 
                entry.dims.x == dims.x && 
                entry.dims.y == dims.y && 
                entry.dims.z == dims.z) {
                entry.inUse = true;
                entry.lastUsed = std::chrono::steady_clock::now();
                poolHits_++;
                return &entry;
            }
        }
        
        // Look for compatible size (larger)
        for (auto& entry : pool_) {
            if (!entry.inUse && 
                entry.dims.x >= dims.x && 
                entry.dims.y >= dims.y && 
                entry.dims.z >= dims.z) {
                entry.inUse = true;
                entry.lastUsed = std::chrono::steady_clock::now();
                poolHits_++;
                return &entry;
            }
        }
        
        poolMisses_++;
        return nullptr;
    }
    
    // Return a texture pool entry
    void returnPoolEntry(TexturePoolEntry* entry) {
        if (entry) {
            std::lock_guard<std::mutex> lock(poolMutex_);
            entry->inUse = false;
            entry->lastUsed = std::chrono::steady_clock::now();
        }
    }
    
    // Generate cache key
    std::string generateCacheKey(const int3& dims, cudaTextureFilterMode filterMode, cudaTextureAddressMode addressMode) {
        return std::to_string(dims.x) + "x" + std::to_string(dims.y) + "x" + std::to_string(dims.z) + 
               "_" + std::to_string(filterMode) + "_" + std::to_string(addressMode);
    }
    
    // Get cached texture
    cudaTextureObject_t getCachedTexture(const int3& dims, cudaTextureFilterMode filterMode, cudaTextureAddressMode addressMode) {
        std::lock_guard<std::mutex> lock(cacheMutex_);
        
        std::string key = generateCacheKey(dims, filterMode, addressMode);
        auto it = cache_.find(key);
        
        if (it != cache_.end()) {
            it->second.lastUsed = std::chrono::steady_clock::now();
            cacheHits_++;
            return it->second.texture;
        }
        
        cacheMisses_++;
        return 0; // Not found
    }
    
    // Cache texture
    void cacheTexture(const int3& dims, cudaTextureFilterMode filterMode, cudaTextureAddressMode addressMode, cudaTextureObject_t texture) {
        std::lock_guard<std::mutex> lock(cacheMutex_);
        
        // Check cache size limit
        if (cache_.size() >= config_.maxCacheSize) {
            // Remove oldest entry
            auto oldest = cache_.begin();
            for (auto it = cache_.begin(); it != cache_.end(); ++it) {
                if (it->second.lastUsed < oldest->second.lastUsed) {
                    oldest = it;
                }
            }
            
            // Destroy old texture
            cudaDestroyTextureObject(oldest->second.texture);
            cache_.erase(oldest);
        }
        
        std::string key = generateCacheKey(dims, filterMode, addressMode);
        TextureCacheEntry entry;
        entry.texture = texture;
        entry.dims = dims;
        entry.filterMode = filterMode;
        entry.addressMode = addressMode;
        entry.lastUsed = std::chrono::steady_clock::now();
        
        cache_[key] = entry;
    }
    
    // Async memory copy
    cudaError_t asyncMemcpy3D(cudaArray* dstArray, const float* srcData, const int3& dims) {
        cudaExtent extent = make_cudaExtent(dims.x, dims.y, dims.z);
        cudaMemcpy3DParms copyParams = {};
        copyParams.srcPtr = make_cudaPitchedPtr((void*)srcData, extent.width*sizeof(float), extent.width, extent.height);
        copyParams.dstArray = dstArray;
        copyParams.extent = extent;
        copyParams.kind = cudaMemcpyHostToDevice;
        
        return cudaMemcpy3DAsync(&copyParams, asyncStream_);
    }
    
    // Wait for async operations
    void waitForAsyncOps() {
        cudaStreamSynchronize(asyncStream_);
    }
    
    // Cleanup
    void cleanup() {
        std::lock_guard<std::mutex> poolLock(poolMutex_);
        std::lock_guard<std::mutex> cacheLock(cacheMutex_);
        
        // Cleanup cache
        for (auto& entry : cache_) {
            cudaDestroyTextureObject(entry.second.texture);
        }
        cache_.clear();
        
        // Cleanup pool
        for (auto& entry : pool_) {
            cudaFreeArray(entry.array);
        }
        pool_.clear();
    }
    
    // Print statistics
    void printStats() {
        std::lock_guard<std::mutex> poolLock(poolMutex_);
        std::lock_guard<std::mutex> cacheLock(cacheMutex_);
        
        printf("=== Texture Pool Statistics ===\n");
        printf("Pool entries: %zu\n", pool_.size());
        printf("Pool hits: %zu, misses: %zu (%.2f%% hit rate)\n", 
               poolHits_, poolMisses_, 
               poolHits_ + poolMisses_ > 0 ? 100.0 * poolHits_ / (poolHits_ + poolMisses_) : 0.0);
        
        printf("Cache entries: %zu\n", cache_.size());
        printf("Cache hits: %zu, misses: %zu (%.2f%% hit rate)\n", 
               cacheHits_, cacheMisses_,
               cacheHits_ + cacheMisses_ > 0 ? 100.0 * cacheHits_ / (cacheHits_ + cacheMisses_) : 0.0);
    }
};

// Global texture pool manager
static TexturePoolManager* g_texturePool = nullptr;

// Initialize texture pool
void initializeTexturePool() {
    if (g_texturePool == nullptr) {
        g_texturePool = new TexturePoolManager();
        if (rtdVerboseFineTiming()) printf("[TIMING] Texture pool initialized\n");
    }
}

// Cleanup texture pool
void cleanupTexturePool() {
    if (g_texturePool != nullptr) {
        g_texturePool->printStats();
        delete g_texturePool;
        g_texturePool = nullptr;
        if (rtdVerboseFineTiming()) printf("[TIMING] Texture pool cleaned up\n");
    }
}

// Get texture pool manager
TexturePoolManager* getTexturePool() {
    if (g_texturePool == nullptr) {
        initializeTexturePool();
    }
    return g_texturePool;
}
