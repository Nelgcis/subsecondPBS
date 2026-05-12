# RayTraceDicom集成项目 - 调试配置

## 调试环境设置

### 1. 构建调试版本

```bash
# Windows
debug.bat

# Linux/Mac
./debug.sh
```

### 2. 手动构建

```bash
# 创建调试构建目录
mkdir build_debug
cd build_debug

# 配置CMake
cmake -DCMAKE_BUILD_TYPE=Debug ..

# 编译
make -j4
```

### 3. 运行调试

```bash
# 运行调试测试
./bin/debug_test

# 运行主测试
./bin/test_raytracedicom

# 使用cuda-memcheck检查内存错误
cuda-memcheck ./bin/test_raytracedicom

# 使用GDB调试
gdb ./bin/test_raytracedicom

# 使用CUDA-GDB调试
cuda-gdb ./bin/test_raytracedicom
```

## 调试技巧

### 1. 常见问题排查

- **编译错误**: 检查CUDA版本和编译器兼容性
- **运行时错误**: 使用cuda-memcheck检查内存访问
- **性能问题**: 使用NVIDIA Nsight Compute分析
- **逻辑错误**: 使用GDB设置断点调试

### 2. 调试输出

在代码中添加调试输出：

```cpp
#include "debug_tools.h"

// 检查GPU内存
DebugTools::checkGPUMemory("After allocation");

// 验证数据
DebugTools::validateData(h_data, d_data, size, "test_data");

// 检查核函数启动
DebugTools::checkKernelLaunch("myKernel");

// 同步设备
DebugTools::syncDevice("After kernel launch");
```

### 3. 性能分析

```bash
# 使用NVIDIA Nsight Compute
ncu --set full -o profile_output ./bin/test_raytracedicom

# 使用NVIDIA Nsight Systems
nsys profile ./bin/test_raytracedicom
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

## 常见错误和解决方案

### 1. CUDA错误

```cpp
// 检查CUDA错误
cudaError_t error = cudaGetLastError();
if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
}
```

### 2. 内存访问错误

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

## 调试工具推荐

1. **NVIDIA Nsight Compute** - 性能分析
2. **NVIDIA Nsight Systems** - 系统分析
3. **CUDA-GDB** - GPU调试器
4. **cuda-memcheck** - 内存检查
5. **Valgrind** - 内存泄漏检测
6. **GDB** - 标准调试器

## 调试步骤

1. **编译调试版本**: 使用Debug构建类型
2. **运行调试测试**: 使用debug_test程序
3. **检查错误**: 使用cuda-memcheck
4. **设置断点**: 使用GDB或CUDA-GDB
5. **分析性能**: 使用NVIDIA Nsight工具
6. **验证结果**: 检查输出正确性

## 调试配置文件

### .gdbinit
```
set print pretty on
set print array on
set print array-indexes on
set print elements 100
```

### .cuda-gdbinit
```
set cuda memcheck on
set cuda memcheck on
set cuda memcheck on
```

## 总结

调试CUDA项目需要：
1. 正确的构建配置
2. 完善的错误检查
3. 合适的调试工具
4. 系统性的调试方法
5. 耐心和细致的态度

记住：调试是一个迭代过程，需要逐步缩小问题范围，最终找到根本原因。


