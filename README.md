# RayTraceDicom Integration Project

## 项目概述

这是一个完整的RTD（Ray Trace DICOM）项目集成，实现了基于GPU的质子治疗剂量计算。项目将原始的RayTraceDicom代码重新组织为模块化的C++/CUDA接口，支持subspot到CPB（Convolution Pencil Beam）卷积、BEV（Beam's Eye View）射线追踪和优化的superposition算法。

## 核心功能

### 1. 剂量计算流程 (dose_calculation.cu)

**主要函数**: `cudaFinalDose`

**计算步骤**:
1. **ROI范围计算**: 根据ROI索引计算3D边界范围，添加margin，检查束流方向与ROI体积的交集
2. **Subspot信息读取**: 从subspotData纹理读取每个subspot的deltaX、deltaY、weight、sigmaX、sigmaY和能量信息
3. **CPB网格构建**: 创建3D纹理存储CPB权重，x、y维度为ROI范围，z维度为能量层
4. **Subspot卷积**: 使用GPU卷积将subspot权重卷积到CPB网格
5. **Sigma纹理计算**: 计算参考平面上每个点的sigmaX和sigmaY值
6. **剂量累加**: 将CPB权重累加到最终剂量网格

**关键核函数**:
- `calculateROIRangeKernel`: ROI范围计算和束流交集检查
- `calculateSigmaTexturesKernel`: Sigma纹理计算
- `addCPBWeightsToDose`: CPB权重累加到最终剂量

### 2. BEV射线追踪 (bev_ray_tracing.cu)

**主要函数**: `rayTracingBEVKernel`

**功能**: 在BEV发散坐标系下进行射线追踪，计算密度、累积停止功率、IDD和有效sigma

**计算步骤**:
1. **射线初始化**: 从源点出发，计算射线方向和步长
2. **密度计算**: 使用LUT计算体素密度
3. **停止功率计算**: 累积停止功率值
4. **IDD计算**: 基于差分方程计算积分深度剂量
5. **有效Sigma计算**: 考虑辐射长度的sigma计算

### 3. 优化的Superposition算法 (superposition_enhanced.cu)

**主要函数**: `kernelSuperpositionCarbon`

**功能**: 使用优化的高斯核superposition算法，支持tile半径计算和批处理

**优化特性**:
- **Tile半径计算**: 动态计算每个tile的superposition半径
- **批处理优化**: 按半径组织tiles进行批处理
- **共享内存**: 使用共享内存提高访问效率
- **快速erf近似**: 使用优化的误差函数近似

**关键核函数**:
- `calculateTileRadiusKernel`: 计算tile半径
- `kernelSuperpositionCarbon`: 优化的superposition核函数

### 4. GPU卷积 (gpu_convolution_2d.cu)

**主要函数**: `subspotToCPBConvolutionKernel`

**功能**: 将subspot权重卷积到CPB网格，使用高斯积分和误差函数

**特性**:
- **高斯积分**: 使用误差函数计算精确的高斯权重积分
- **共享内存**: 优化内存访问模式
- **协作线程**: 使用协作线程处理大卷积核

### 5. IDD和Sigma计算 (idd_sigma_calculation.cu)

**主要函数**: `fillIddAndSigmaKernel`

**功能**: 计算积分深度剂量和有效sigma值

**计算步骤**:
1. **IDD计算**: 基于能量和深度计算IDD值
2. **Sigma计算**: 考虑辐射长度的有效sigma计算
3. **LUT查找**: 使用查找表进行快速计算

## 数据结构

### 主要类定义 (common.cuh)

#### Grid类
- **功能**: 表示3D网格结构
- **成员**: corner, resolution, dims, upperCorner
- **方法**: worldToGrid, gridToWorld, getLinearIndex

#### Source类
- **功能**: 表示射线源
- **成员**: sourcePos, isoPos, vsDirX, vSAD
- **方法**: 构造函数和访问方法

#### Beam类
- **功能**: 表示束流参数
- **成员**: source, bmdir, bmxdir, ene, transCutoff, longitudalCutoff
- **方法**: 构造函数和参数设置

#### SubspotInfo类
- **功能**: 存储subspot信息
- **成员**: deltaX, deltaY, weight, sigmaX, sigmaY, position, direction
- **方法**: 从纹理读取数据的构造函数

#### CPBGrid类
- **功能**: 表示CPB网格
- **成员**: dims, corner, resolution, data
- **方法**: 网格操作和访问方法

### 宏定义 (Macro.cuh)

#### 物理常量
- `WATERDENSITY`: 水密度 (1.0 g/cm³)
- `MP`: 质子质量 (938.272046 MeV)
- `XW`: 水的辐射长度 (36.514 cm)
- `PI`: 圆周率

#### 计算参数
- `WEIGHT_CUTOFF`: 权重截断值 (0.001f)
- `SIGMA_CUTOFF`: Sigma截断值 (3.0f)
- `MAX_SUPERP_RADIUS`: 最大superposition半径 (32)
- `SUPERP_TILE_X/Y`: Superposition tile尺寸

#### 算法参数
- `MAXSTEP`: 最大步长 (0.32 cm)
- `MAXENERGYRATIO`: 最大能量衰减比 (0.2)
- `BATCHSIZE`: 批处理大小 (65536)

## 文件结构

```
raytracedicom_integration/
├── include/                          # 头文件目录
│   ├── common.cuh                    # 通用数据结构和类定义
│   ├── Macro.cuh                     # 宏定义和常量
│   ├── dose_calculation.h            # 剂量计算头文件
│   ├── bev_ray_tracing.h             # BEV射线追踪头文件
│   ├── superposition_enhanced.h      # 优化superposition头文件
│   ├── gpu_convolution_2d.cuh        # GPU卷积头文件
│   ├── idd_sigma_calculation.cuh     # IDD计算头文件
│   ├── superposition_kernels.h       # Superposition核函数头文件
│   ├── ray_tracing.h                 # 射线追踪头文件
│   ├── raytracedicom_integration.h   # 主集成头文件
│   ├── utils.h                       # 工具函数头文件
│   ├── transformation_func.cuh       # 变换函数头文件
│   └── debug_tools.h                 # 调试工具头文件
├── src/                              # 源代码目录
│   ├── dose_calculation.cu           # 主要剂量计算实现
│   ├── bev_ray_tracing.cu            # BEV射线追踪实现
│   ├── superposition_enhanced.cu     # 优化superposition实现
│   ├── gpu_convolution_2d.cu         # GPU卷积实现
│   ├── idd_sigma_calculation.cu      # IDD计算实现
│   ├── superposition_kernels.cu      # Superposition核函数实现
│   ├── ray_tracing.cu                # 射线追踪实现
│   ├── raytracedicom_wrapper.cu      # 主包装函数实现
│   ├── utils.cu                      # 工具函数实现
│   ├── cpb_composition.cu            # CPB组合实现
│   ├── superposition.cu              # 基础superposition实现
│   ├── test_raytracedicom.cpp        # 主测试应用程序
│   ├── test_cpb_convolution.cpp      # CPB卷积测试
│   └── debug_test.cpp                # 调试测试程序
├── tables/                           # 查找表目录
│   ├── density_Schneider2000_adj.txt # 密度查找表
│   ├── HU_to_SP_H&N_adj.txt          # HU到停止功率转换表
│   ├── nuclear_weights_and_sigmas_*.txt # 核权重和sigma表
│   ├── proton_cumul_ddd_data.txt     # 质子累积DDD数据
│   └── radiation_length_*.txt        # 辐射长度表
├── build/                            # 构建目录（自动创建）
├── bin/                              # 可执行文件目录（自动创建）
├── CMakeLists.txt                    # CMake构建文件
├── Makefile                          # Make构建文件
├── quick_test.sh                     # Linux/Mac快速测试脚本
├── quick_test.bat                    # Windows快速测试脚本
├── test.sh                           # 简单测试脚本
├── debug.sh                          # Linux/Mac调试脚本
├── debug.bat                         # Windows调试脚本
└── README.md                         # 本文档
```

## 计算流程

### 1. 初始化阶段
1. **数据加载**: 加载CT数据、ROI信息、subspot数据
2. **网格设置**: 设置剂量网格、CPB网格参数
3. **纹理绑定**: 绑定各种查找表纹理

### 2. ROI处理阶段
1. **范围计算**: 计算ROI的3D边界范围
2. **交集检查**: 验证束流方向与ROI体积的交集
3. **Margin添加**: 为ROI添加可配置的边距

### 3. Subspot处理阶段
1. **信息读取**: 从纹理读取subspot的deltaX、deltaY、weight、sigmaX、sigmaY
2. **位置计算**: 计算subspot在参考平面的3D位置
3. **有效性检查**: 基于权重和sigma阈值验证subspot有效性

### 4. CPB卷积阶段
1. **网格构建**: 创建3D CPB网格
2. **权重卷积**: 将subspot权重卷积到CPB网格
3. **高斯积分**: 使用误差函数计算精确的高斯权重积分

### 5. 射线追踪阶段
1. **BEV坐标**: 构建BEV发散坐标系
2. **密度计算**: 使用LUT计算体素密度
3. **停止功率**: 累积停止功率值
4. **IDD计算**: 基于差分方程计算积分深度剂量

### 6. Superposition阶段
1. **Tile计算**: 计算每个tile的superposition半径
2. **批处理**: 按半径组织tiles进行批处理
3. **核函数**: 应用高斯核到剂量分布

### 7. 输出阶段
1. **剂量累加**: 将各层剂量累加到最终网格
2. **结果输出**: 输出最终剂量分布

## 编译和运行

### 编译要求
- CUDA Toolkit 11.0+
- CMake 3.10+
- C++14支持
- NVIDIA GPU (Compute Capability 6.0+)

### 编译步骤
```bash
# 创建构建目录
mkdir build && cd build

# 配置CMake
cmake ..

# 编译
make -j4

# 运行测试
./bin/test_raytracedicom
```

### 快速测试
```bash
# Linux/Mac
./quick_test.sh

# Windows
quick_test.bat
```

## 调试和优化

### 调试工具
- `debug_test.cpp`: 单独的调试测试程序
- `debug_tools.h`: 调试工具和宏定义
- `cuda-memcheck`: CUDA内存检查工具
- `cuda-gdb`: CUDA调试器

### 性能优化
- **共享内存**: 使用共享内存优化内存访问
- **批处理**: 按半径组织tiles进行批处理
- **纹理内存**: 使用纹理内存加速查找表访问
- **协作线程**: 使用协作线程处理大卷积核

## 注意事项

1. **内存管理**: 确保正确分配和释放GPU内存
2. **错误检查**: 使用`checkCudaErrors`检查CUDA API调用
3. **数值稳定性**: 注意浮点数精度和溢出问题
4. **线程同步**: 确保核函数执行完成后再进行下一步
5. **纹理绑定**: 确保纹理对象正确绑定和释放

## 扩展功能

### 计划中的改进
1. **完整射线追踪**: 实现完整的BEV射线追踪参数设置
2. **深度剂量**: 考虑深度剂量分布的计算
3. **多GPU支持**: 支持多GPU并行计算
4. **实时可视化**: 添加实时剂量分布可视化
5. **参数优化**: 进一步优化算法参数

### 自定义扩展
- 添加新的查找表
- 实现新的superposition算法
- 支持不同的粒子类型
- 添加剂量验证功能

## 参考文献

1. da Silva, J., et al. "Sub-second pencil beam dose calculation on GPU for adaptive proton therapy." Physics in Medicine & Biology 60.12 (2015): 4777.
2. Hueso-González, F., et al. "Fast Monte Carlo simulation for proton therapy dose calculation on GPU." Journal of Parallel and Distributed Computing 85 (2015): 1-12.
3. Schneider, U., et al. "The influence of CT image quality on Monte Carlo dose calculation accuracy." Medical Physics 27.7 (2000): 1521-1529.# subsecondPBS
