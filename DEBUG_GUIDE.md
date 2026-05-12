# RayTraceDicom集成项目 - 调试指南

## 调试环境设置

### 1. 构建配置

首先需要更新CMakeLists.txt以包含新的源文件：

```cmake
# 更新CUDA源文件列表
set(CUDA_SOURCES
    src/dose_calculation.cu          # 主要剂量计算
    src/bev_ray_tracing.cu          # BEV射线追踪
    src/superposition_enhanced.cu    # 优化的superposition
    src/gpu_convolution_2d.cu       # GPU卷积
    src/idd_sigma_calculation.cu    # IDD和sigma计算
    src/superposition_kernels.cu    # Superposition核函数
    src/utils.cu                    # 工具函数
    src/raytracedicom_wrapper.cu    # 包装器
)

# 添加调试标志
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++14;-g;-G;-lineinfo)
```

### 2. 调试标志说明

- `-g`: 生成调试信息
- `-G`: 生成设备代码调试信息
- `-lineinfo`: 生成行号信息
- `-O0`: 关闭优化（调试时推荐）

## 调试方法

### 1. CUDA错误检查

在代码中添加CUDA错误检查：

```cpp
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// 使用示例
CUDA_CHECK(cudaMalloc((void**)&d_data, size));
CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
```

### 2. 核函数调试

在核函数中添加调试输出：

```cpp
__global__ void debugKernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx == 0) {
        printf("Debug: Kernel launched with %d blocks, %d threads per block\n", 
               gridDim.x, blockDim.x);
    }
    
    if (idx < size) {
        printf("Debug: Thread %d processing data[%d] = %f\n", 
               idx, idx, data[idx]);
    }
}
```

### 3. 内存调试

检查GPU内存使用情况：

```cpp
void checkGPUMemory(const char* location) {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    printf("GPU Memory at %s: Free = %.2f MB, Total = %.2f MB\n", 
           location, free / (1024.0 * 1024.0), total / (1024.0 * 1024.0));
}
```

### 4. 数据验证

验证数据传输的正确性：

```cpp
void validateData(float* h_data, float* d_data, int size, const char* name) {
    std::vector<float> h_copy(size);
    cudaMemcpy(h_copy.data(), d_data, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    bool valid = true;
    for (int i = 0; i < size; i++) {
        if (abs(h_data[i] - h_copy[i]) > 1e-6f) {
            printf("Data validation failed for %s at index %d: %f != %f\n", 
                   name, i, h_data[i], h_copy[i]);
            valid = false;
            break;
        }
    }
    
    if (valid) {
        printf("Data validation passed for %s\n", name);
    }
}
```

## 调试工具

### 1. 创建调试测试程序

```cpp
// src/debug_test.cpp
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "common.cuh"

int main() {
    // 初始化CUDA
    cudaSetDevice(0);
    
    // 测试数据
    const int size = 1024;
    std::vector<float> h_data(size, 1.0f);
    float* d_data;
    
    // 分配GPU内存
    cudaMalloc(&d_data, size * sizeof(float));
    
    // 传输数据
    cudaMemcpy(d_data, h_data.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    
    // 测试核函数
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    
    // 这里调用你的核函数进行测试
    
    // 验证结果
    std::vector<float> h_result(size);
    cudaMemcpy(h_result.data(), d_data, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 清理
    cudaFree(d_data);
    
    return 0;
}
```

### 2. 性能分析工具

使用NVIDIA Nsight Compute进行性能分析：

```bash
# 安装Nsight Compute
# 然后使用以下命令分析性能
ncu --set full -o profile_output ./test_raytracedicom
```

### 3. 内存检查工具

使用cuda-memcheck检查内存错误：

```bash
cuda-memcheck ./test_raytracedicom
```

## 常见调试问题

### 1. 核函数启动失败

```cpp
// 检查核函数启动
cudaError_t error = cudaGetLastError();
if (error != cudaSuccess) {
    printf("Kernel launch failed: %s\n", cudaGetErrorString(error));
}
```

### 2. 内存访问越界

```cpp
// 在核函数中添加边界检查
__global__ void safeKernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= size) {
        printf("Warning: Thread %d accessing out of bounds (size = %d)\n", idx, size);
        return;
    }
    
    // 安全的数据访问
    data[idx] = idx;
}
```

### 3. 纹理绑定问题

```cpp
// 检查纹理对象创建
cudaTextureObject_t texObj;
cudaError_t error = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
if (error != cudaSuccess) {
    printf("Texture creation failed: %s\n", cudaGetErrorString(error));
}
```

## 调试步骤

### 1. 编译调试版本

```bash
mkdir build_debug
cd build_debug
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j4
```

### 2. 运行调试

```bash
# 使用GDB调试
gdb ./bin/test_raytracedicom

# 或者使用CUDA-GDB
cuda-gdb ./bin/test_raytracedicom
```

### 3. 设置断点

```bash
# 在GDB中设置断点
(gdb) break main
(gdb) break cudaFinalDose
(gdb) run
```

### 4. 检查变量

```bash
# 在GDB中检查变量
(gdb) print variable_name
(gdb) info locals
(gdb) info args
```

## 调试技巧

### 1. 分步调试

将复杂函数分解为小步骤：

```cpp
void debugStepByStep() {
    printf("Step 1: Initialize data\n");
    // 初始化代码
    
    printf("Step 2: Allocate GPU memory\n");
    // 内存分配代码
    
    printf("Step 3: Launch kernel\n");
    // 核函数启动代码
    
    printf("Step 4: Copy results back\n");
    // 结果复制代码
}
```

### 2. 条件调试

使用条件编译进行调试：

```cpp
#ifdef DEBUG
    printf("Debug: Variable value = %f\n", value);
    checkGPUMemory("After allocation");
#endif
```

### 3. 日志记录

创建日志系统：

```cpp
class DebugLogger {
public:
    static void log(const char* message) {
        printf("[DEBUG] %s\n", message);
    }
    
    static void logError(const char* message) {
        printf("[ERROR] %s\n", message);
    }
    
    static void logMemory(const char* location) {
        checkGPUMemory(location);
    }
};
```

## 调试检查清单

- [ ] 检查CUDA错误
- [ ] 验证内存分配
- [ ] 检查数据传输
- [ ] 验证核函数参数
- [ ] 检查纹理绑定
- [ ] 验证结果正确性
- [ ] 检查内存泄漏
- [ ] 验证性能指标

## 调试工具推荐

1. **NVIDIA Nsight Compute** - 性能分析
2. **NVIDIA Nsight Systems** - 系统分析
3. **CUDA-GDB** - GPU调试器
4. **cuda-memcheck** - 内存检查
5. **Valgrind** - 内存泄漏检测
6. **GDB** - 标准调试器

## 总结

调试CUDA项目需要：
1. 正确的构建配置
2. 完善的错误检查
3. 合适的调试工具
4. 系统性的调试方法
5. 耐心和细致的态度

记住：调试是一个迭代过程，需要逐步缩小问题范围，最终找到根本原因。


