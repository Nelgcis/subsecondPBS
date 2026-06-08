/**
 * \file
 * \brief CUDA 12.1 Advanced GPU Memory Pool Manager
 * 
 * This file implements:
 * - Advanced GPU memory pool with hierarchical allocation
 * - Memory fragmentation prevention
 * - Automatic memory defragmentation
 * - Memory usage statistics and monitoring
 * - CUDA 12.1 memory pool features
 */

#include <cuda_runtime.h>
#include <texture_types.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <algorithm>

// Memory pool configuration
struct MemoryPoolConfig {
    size_t initialPoolSize = 512 * 1024 * 1024; // 512MB initial pool
    size_t maxPoolSize = 2ULL * 1024 * 1024 * 1024; // 2GB max pool (使用ULL避免溢出)
    size_t minAllocationSize = 1024; // 1KB minimum allocation
    size_t maxAllocationSize = 256 * 1024 * 1024; // 256MB maximum allocation
    bool enableDefragmentation = true;
    bool enableMemoryCompression = true;
    float fragmentationThreshold = 0.3f; // 30% fragmentation threshold
};

// Memory block structure
struct MemoryBlock {
    void* ptr;
    size_t size;
    bool isFree;
    std::chrono::steady_clock::time_point lastUsed;
    int allocationCount;
    MemoryBlock* next;
    MemoryBlock* prev;
};

// Memory pool statistics
struct MemoryPoolStats {
    size_t totalAllocated = 0;
    size_t totalFree = 0;
    size_t totalFragmented = 0;
    size_t allocationCount = 0;
    size_t deallocationCount = 0;
    size_t defragmentationCount = 0;
    double fragmentationRatio = 0.0;
    std::chrono::steady_clock::time_point lastDefragmentation;
};

class AdvancedGPUMemoryPool {
private:
    std::vector<MemoryBlock> memoryBlocks_;
    std::unordered_map<void*, MemoryBlock*> ptrToBlock_;
    std::mutex poolMutex_;
    MemoryPoolConfig config_;
    MemoryPoolStats stats_;
    
    // CUDA 12.1 memory pool features
    cudaMemPool_t memPool_;
    bool useCudaMemPool_;
    
    // Memory alignment for optimal performance
    static constexpr size_t MEMORY_ALIGNMENT = 256; // 256-byte alignment for CUDA 12.1
    
public:
    AdvancedGPUMemoryPool() : useCudaMemPool_(false) {
        initializeMemoryPool();
    }
    
    ~AdvancedGPUMemoryPool() {
        cleanupMemoryPool();
    }
    
    // Initialize CUDA 12.1 memory pool
    void initializeMemoryPool() {
        // Prefer the device default memory pool (valid on modern CUDA)
        int dev = 0;
        cudaGetDevice(&dev);
        cudaError_t err = cudaDeviceGetDefaultMemPool(&memPool_, dev);
        if (err == cudaSuccess) {
            useCudaMemPool_ = true;
            printf("[MEMORY_POOL] Using CUDA device default memory pool\n");

            // Set memory pool attributes for optimal performance
            cudaMemPoolAttr attr = cudaMemPoolAttrReleaseThreshold;
            uint64_t threshold = 0; // No threshold for immediate release
            cudaMemPoolSetAttribute(memPool_, attr, &threshold);
        } else {
            useCudaMemPool_ = false;
            printf("[MEMORY_POOL] CUDA memory pool not available, using custom pool\n");
            printf("Error: %s\n", cudaGetErrorString(err));
        }
        
        // Initialize custom memory pool as fallback
        initializeCustomPool();
    }
    
    // Initialize custom memory pool
    void initializeCustomPool() {
        std::lock_guard<std::mutex> lock(poolMutex_);
        
        // Allocate initial memory pool
        void* poolPtr;
        cudaError_t err = cudaMalloc(&poolPtr, config_.initialPoolSize);
        if (err != cudaSuccess) {
            printf("Error: Failed to allocate initial memory pool: %s\n", cudaGetErrorString(err));
            return;
        }
        
        // Create initial memory block
        MemoryBlock initialBlock;
        initialBlock.ptr = poolPtr;
        initialBlock.size = config_.initialPoolSize;
        initialBlock.isFree = true;
        initialBlock.lastUsed = std::chrono::steady_clock::now();
        initialBlock.allocationCount = 0;
        initialBlock.next = nullptr;
        initialBlock.prev = nullptr;
        
        memoryBlocks_.push_back(initialBlock);
        ptrToBlock_[poolPtr] = &memoryBlocks_.back();
        
        stats_.totalAllocated = config_.initialPoolSize;
        stats_.totalFree = config_.initialPoolSize;
        
        printf("[MEMORY_POOL] Custom memory pool initialized: %zu MB\n", config_.initialPoolSize / (1024*1024));
    }
    
    // Allocate memory with advanced features
    void* allocate(size_t size) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Align size for optimal performance
        size = alignSize(size);
        
        if (useCudaMemPool_) {
            return allocateWithCudaPool(size, start);
        } else {
            return allocateWithCustomPool(size, start);
        }
    }
    
    // Allocate using CUDA 12.1 memory pool
    void* allocateWithCudaPool(size_t size, std::chrono::high_resolution_clock::time_point start) {
        void* ptr;
        cudaError_t err = cudaMallocFromPoolAsync(&ptr, size, memPool_, 0);
        if (err != cudaSuccess) {
            printf("Error: CUDA memory pool allocation failed: %s\n", cudaGetErrorString(err));
            return nullptr;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::lock_guard<std::mutex> lock(poolMutex_);
        stats_.allocationCount++;
        stats_.totalAllocated += size;
        
        printf("[MEMORY_POOL] CUDA pool allocation: %zu bytes in %ld μs\n", size, duration.count());
        return ptr;
    }
    
    // Allocate using custom memory pool
    void* allocateWithCustomPool(size_t size, std::chrono::high_resolution_clock::time_point start) {
        std::lock_guard<std::mutex> lock(poolMutex_);
        
        // Find best fit free block
        MemoryBlock* bestBlock = findBestFit(size);
        if (!bestBlock) {
            // Try to expand pool
            if (!expandPool(size)) {
                printf("Error: Failed to allocate %zu bytes - pool exhausted\n", size);
                return nullptr;
            }
            bestBlock = findBestFit(size);
        }
        
        if (!bestBlock) {
            printf("Error: No suitable memory block found for %zu bytes\n", size);
            return nullptr;
        }
        
        // Split block if necessary
        void* allocatedPtr = splitBlock(bestBlock, size);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        stats_.allocationCount++;
        stats_.totalFree -= size;
        
        printf("[MEMORY_POOL] Custom pool allocation: %zu bytes in %ld μs\n", size, duration.count());
        return allocatedPtr;
    }
    
    // Deallocate memory
    void deallocate(void* ptr) {
        if (!ptr) return;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        if (useCudaMemPool_) {
            deallocateWithCudaPool(ptr, start);
        } else {
            deallocateWithCustomPool(ptr, start);
        }
    }
    
    // Deallocate using CUDA 12.1 memory pool
    void deallocateWithCudaPool(void* ptr, std::chrono::high_resolution_clock::time_point start) {
        cudaError_t err = cudaFreeAsync(ptr, 0);
        if (err != cudaSuccess) {
            printf("Error: CUDA memory pool deallocation failed: %s\n", cudaGetErrorString(err));
            return;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::lock_guard<std::mutex> lock(poolMutex_);
        stats_.deallocationCount++;
        
        printf("[MEMORY_POOL] CUDA pool deallocation in %ld μs\n", duration.count());
    }
    
    // Deallocate using custom memory pool
    void deallocateWithCustomPool(void* ptr, std::chrono::high_resolution_clock::time_point start) {
        std::lock_guard<std::mutex> lock(poolMutex_);
        
        auto it = ptrToBlock_.find(ptr);
        if (it == ptrToBlock_.end()) {
            printf("Error: Attempting to deallocate unknown pointer\n");
            return;
        }
        
        MemoryBlock* block = it->second;
        block->isFree = true;
        block->lastUsed = std::chrono::steady_clock::now();
        stats_.totalFree += block->size;
        stats_.deallocationCount++;
        
        // Try to merge with adjacent free blocks
        mergeAdjacentBlocks(block);
        
        // Check if defragmentation is needed
        if (config_.enableDefragmentation && shouldDefragment()) {
            defragmentMemory();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        printf("[MEMORY_POOL] Custom pool deallocation in %ld μs\n", duration.count());
    }
    
    // Find best fit memory block
    MemoryBlock* findBestFit(size_t size) {
        MemoryBlock* bestBlock = nullptr;
        size_t bestSize = SIZE_MAX;
        
        for (auto& block : memoryBlocks_) {
            if (block.isFree && block.size >= size && block.size < bestSize) {
                bestBlock = &block;
                bestSize = block.size;
            }
        }
        
        return bestBlock;
    }
    
    // Split memory block
    void* splitBlock(MemoryBlock* block, size_t size) {
        if (block->size <= size + sizeof(MemoryBlock)) {
            // Block is too small to split
            block->isFree = false;
            block->allocationCount++;
            return block->ptr;
        }
        
        // Create new free block for remaining space
        MemoryBlock newBlock;
        newBlock.ptr = static_cast<char*>(block->ptr) + size;
        newBlock.size = block->size - size;
        newBlock.isFree = true;
        newBlock.lastUsed = std::chrono::steady_clock::now();
        newBlock.allocationCount = 0;
        newBlock.next = block->next;
        newBlock.prev = block;
        
        // Update original block
        block->size = size;
        block->isFree = false;
        block->allocationCount++;
        block->next = &memoryBlocks_.back();
        
        // Add new block to pool
        memoryBlocks_.push_back(newBlock);
        ptrToBlock_[newBlock.ptr] = &memoryBlocks_.back();
        
        return block->ptr;
    }
    
    // Merge adjacent free blocks
    void mergeAdjacentBlocks(MemoryBlock* block) {
        // Merge with next block if it's free
        if (block->next && block->next->isFree) {
            block->size += block->next->size;
            block->next = block->next->next;
            if (block->next) {
                block->next->prev = block;
            }
        }
        
        // Merge with previous block if it's free
        if (block->prev && block->prev->isFree) {
            block->prev->size += block->size;
            block->prev->next = block->next;
            if (block->next) {
                block->next->prev = block->prev;
            }
        }
    }
    
    // Check if defragmentation is needed
    bool shouldDefragment() {
        if (stats_.totalAllocated == 0) return false;
        
        stats_.fragmentationRatio = static_cast<double>(stats_.totalFragmented) / stats_.totalAllocated;
        return stats_.fragmentationRatio > config_.fragmentationThreshold;
    }
    
    // Defragment memory pool
    void defragmentMemory() {
        printf("[MEMORY_POOL] Starting memory defragmentation...\n");
        
        // Sort blocks by address
        std::sort(memoryBlocks_.begin(), memoryBlocks_.end(), 
                  [](const MemoryBlock& a, const MemoryBlock& b) {
                      return a.ptr < b.ptr;
                  });
        
        // Merge all free blocks
        for (size_t i = 0; i < memoryBlocks_.size() - 1; ++i) {
            if (memoryBlocks_[i].isFree && memoryBlocks_[i + 1].isFree) {
                memoryBlocks_[i].size += memoryBlocks_[i + 1].size;
                memoryBlocks_.erase(memoryBlocks_.begin() + i + 1);
                --i;
            }
        }
        
        stats_.defragmentationCount++;
        stats_.lastDefragmentation = std::chrono::steady_clock::now();
        
        printf("[MEMORY_POOL] Defragmentation completed\n");
    }
    
    // Expand memory pool
    bool expandPool(size_t additionalSize) {
        size_t expansionSize = std::max(additionalSize, config_.initialPoolSize);
        if (stats_.totalAllocated + expansionSize > config_.maxPoolSize) {
            return false;
        }
        
        void* newPtr;
        cudaError_t err = cudaMalloc(&newPtr, expansionSize);
        if (err != cudaSuccess) {
            printf("Error: Failed to expand memory pool: %s\n", cudaGetErrorString(err));
            return false;
        }
        
        // Add new block to pool
        MemoryBlock newBlock;
        newBlock.ptr = newPtr;
        newBlock.size = expansionSize;
        newBlock.isFree = true;
        newBlock.lastUsed = std::chrono::steady_clock::now();
        newBlock.allocationCount = 0;
        newBlock.next = nullptr;
        newBlock.prev = nullptr;
        
        memoryBlocks_.push_back(newBlock);
        ptrToBlock_[newPtr] = &memoryBlocks_.back();
        
        stats_.totalAllocated += expansionSize;
        stats_.totalFree += expansionSize;
        
        printf("[MEMORY_POOL] Pool expanded by %zu MB\n", expansionSize / (1024*1024));
        return true;
    }
    
    // Align size for optimal performance
    size_t alignSize(size_t size) {
        return (size + MEMORY_ALIGNMENT - 1) & ~(MEMORY_ALIGNMENT - 1);
    }
    
    // Print memory pool statistics
    void printStats() {
        std::lock_guard<std::mutex> lock(poolMutex_);
        
        printf("\n=== Advanced GPU Memory Pool Statistics ===\n");
        printf("Pool Type: %s\n", useCudaMemPool_ ? "CUDA 12.1 Memory Pool" : "Custom Memory Pool");
        printf("Total Allocated: %zu MB\n", stats_.totalAllocated / (1024*1024));
        printf("Total Free: %zu MB\n", stats_.totalFree / (1024*1024));
        printf("Fragmentation Ratio: %.2f%%\n", stats_.fragmentationRatio * 100);
        printf("Allocations: %zu\n", stats_.allocationCount);
        printf("Deallocations: %zu\n", stats_.deallocationCount);
        printf("Defragmentations: %zu\n", stats_.defragmentationCount);
        printf("Memory Blocks: %zu\n", memoryBlocks_.size());
        printf("==========================================\n\n");
    }
    
    // Cleanup memory pool
    void cleanupMemoryPool() {
        std::lock_guard<std::mutex> lock(poolMutex_);
        
        if (useCudaMemPool_) {
            cudaMemPoolDestroy(memPool_);
        } else {
            // Free all allocated memory blocks
            for (const auto& block : memoryBlocks_) {
                cudaFree(block.ptr);
            }
        }
        
        memoryBlocks_.clear();
        ptrToBlock_.clear();
        
        printf("[MEMORY_POOL] Memory pool cleaned up\n");
    }
};

// Global memory pool manager
static AdvancedGPUMemoryPool* g_memoryPool = nullptr;

// Initialize memory pool
void initializeAdvancedMemoryPool() {
    if (g_memoryPool == nullptr) {
        g_memoryPool = new AdvancedGPUMemoryPool();
        printf("[MEMORY_POOL] Advanced GPU memory pool initialized\n");
    }
}

// Get memory pool manager
AdvancedGPUMemoryPool* getAdvancedMemoryPool() {
    if (g_memoryPool == nullptr) {
        initializeAdvancedMemoryPool();
    }
    return g_memoryPool;
}

// Cleanup memory pool
void cleanupAdvancedMemoryPool() {
    if (g_memoryPool != nullptr) {
        g_memoryPool->printStats();
        delete g_memoryPool;
        g_memoryPool = nullptr;
        printf("[MEMORY_POOL] Advanced GPU memory pool cleaned up\n");
    }
}
