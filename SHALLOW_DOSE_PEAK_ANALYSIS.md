# 浅层峰值问题分析

**问题**: BEV剂量分布在z=1处有明显峰值，z=1的剂量是z=2的2倍，这在物理上不合理。

---

## 🔍 问题诊断

### 观测的数据

```
Layer 0 (120 MeV):
  z=1 sum = 3.067e-07 ← 最大值（峰）
  z=2 sum = 1.534e-07 ← 下降到一半
  z=3-5: sum ≈ 1.534e-07 (平坦)

Layer 1 (160 MeV):
  z=1 sum = 2.503e-07 ← 最大值（峰）
  z=2 sum = 1.253e-07 ← 下降到一半

Layer 2 (180 MeV):
  z=1 sum = 2.310e-07 ← 最大值（峰）
  z=2 sum = 1.158e-07 ← 下降到一半
```

**模式**: 每层都在z=1处有峰，z=2之后平坦

### 几何参数

```
射线追踪范围:
  gantry Z bounds: [-32 mm, 0 mm]  (负向，向后)
  startZ = 0 mm
  stepLen = 1.0 mm
  steps = 33

Bragg峰深度:
  120 MeV: 103.75 mm
  160 MeV: 173.92 mm
  180 MeV: 214.05 mm

关键观察:
  ❌ 射线追踪只覆盖 -32 到 0 mm
  ❌ Bragg峰在 100+ mm 处
  ❌ 射线路径与Bragg峰之间相差 100+ mm！
```

### BEV到最终剂量的变换

```
BEV剂量体 (576×576×32)
  ↓ [primTransfDiv 变换核]
最终剂量体 (512×512×32)

变换参数:
  normDist = (-1000, -1000)
  globalOffset = (232, 344, -1)
  startIdx = (0, 0, 1)
  maxZ = 31
```

---

## 🤔 问题的可能原因

### 假说1：射线权重分布不对（概率：40%）

在z=1处可能有射线权重的堆积，导致该层接收更多射线。

**检验**:
```
观测: [RAY_W] Layer 0 sum=9.959792e-01, max=2.733946e-04, nnz(>0)=13541/262144
但没有看到按z分层的射线权重分布
```

### 假说2：超立体化（卷积）的边界效应（概率：35%）

CPB到射线权重的映射或超立体化卷积可能在边界处有特殊处理，导致z=1处异常。

**检验**:
```
Layer 0 processed in 91 ms  ← 其他层只需 0 ms
Layer 1 processed in 0 ms
Layer 2 processed in 0 ms
```

### 假说3：IDD查表的边界采样问题（概率：15%）

在浅层（接近起点），IDD查表可能采样到边界值或特殊值。

**检验**:
```
gantry Z bounds: z=[-32, 0]
startZ = 0
如果射线从z=0开始，第一步可能有特殊处理
```

### 假说4：坐标变换错误（概率：10%）

BEV到最终剂量的坐标变换可能对z=0-1有特殊处理。

```
startIdx = (0, 0, 1)  ← 注意！从z=1开始，跳过z=0
这是否故意跳过第一层？
```

---

## 📊 详细分析

### 关键观察1：z=0 总是空的

```
所有三层的结果:
z=0 sum = 0.000000e+00  (完全空)
z=1 sum = (最大值)      (峰值)
z=2-31 sum = (递减-平坦)
```

这表明：
- ✓ z=0 被故意跳过（见 startIdx=(0,0,1)）
- ✓ z=1 是实际的第一层计算结果
- ? 为什么z=1有峰？

### 关键观察2：z=1到z=2的下降

```
z=1: 3.067e-07  (100%)
z=2: 1.534e-07  (50%)
z=3: 1.535e-07  (50%)
```

这种**急剧的一次性下降**然后平坦的模式，暗示：
- 可能不是Bragg峰（那会是尖峰）
- 可能是**初始条件或边界效应**
- 可能是**射线权重分布不均**

### 关键观察3：与Bragg峰深度的矛盾

```
Z范围: [-32, 0] mm ← 这是射线在体内的深度
Bragg峰: 103-214 mm ← 远在后面

矛盾:
❌ 射线还没进入剂量体多少就被追踪停止了
❌ Bragg峰完全在追踪范围之外
❌ 看到的z=1峰不可能是Bragg峰
```

---

## 🔧 问题定位

### 最可能的原因：射线初始化

问题很可能在**BEV射线追踪的初始化阶段**：

1. **射线从z=0 (gantry坐标系中的体表)开始**
2. **第一步沉积到z=1处** (BEV坐标系中)
3. **射线进入体后逐步吸收能量**
4. **初始步可能有特殊处理，导致z=1有额外的沉积**

### 代码位置

需要检查以下文件：

1. **bev_ray_tracing.cu** - 射线初始化和第一步计算
   - 射线是从何处开始
   - 第一步的处理是否特殊

2. **idd_sigma.cu** - 剂量计算核函数
   - `if (stepNo == params.getFirstStep())` 是否有特殊处理

3. **raytracedicom_wrapper.cu** - 参数初始化
   - `startIdx` 的设置
   - 初始步的参数

4. **superposition.cu** - 超立体化卷积
   - z=1处是否有边界处理

---

## 🎯 验证步骤

### 1. 检查射线权重的z分布

添加调试代码在**raytracedicom_wrapper.cu**中，在BEV射线权重计算后：

```cpp
// After ray weight calculation
{
    // Analyze ray weight distribution by z-slice
    std::vector<float> rayWeightByZ(32, 0.0f);
    std::vector<int> countByZ(32, 0);
    
    for (int z = 0; z < 32; z++) {
        float sumRayWeight = 0.0f;
        int count = 0;
        for (int y = 0; y < 262144; y++) {  // All rays in this z-layer
            if (rayWeightLayer0[y*32 + z] > 1e-15f) {
                sumRayWeight += rayWeightLayer0[y*32 + z];
                count++;
            }
        }
        rayWeightByZ[z] = sumRayWeight;
        countByZ[z] = count;
    }
    
    std::cout << "[DEBUG] Ray weight distribution by z-slice:" << std::endl;
    for (int z = 0; z < 32; z++) {
        std::cout << "  z=" << z << ": sum=" << rayWeightByZ[z] 
                  << " count=" << countByZ[z] << std::endl;
    }
}
```

### 2. 检查IDD值的z分布

添加调试代码在**idd_sigma.cu**中的`fillIddAndSigma`核函数：

```cuda
// After IDD lookup, sum by z
if (stepNo == 0) {
    printf("IDD z-slice sample (step 0): z=%d, IDD=%.3e\n", /* z-index */, bevIdd[idx]);
}
```

### 3. 追踪第一步的剂量沉积

添加打印：
```cpp
if (stepNo == params.getFirstStep()) {
    printf("Step 0 dose contribution: voxel=%d, dose=%.3e\n", idx, res);
}
```

---

## 💡 可能的物理解释

即使在不是Bragg峰的浅层，z=1的峰也可以解释：

### 情况1：射线能量损失最快（最可能）
- 在浅层（低速质子），能量损失率（dE/dx）最高
- 高能量损失 → 高IDD值 → 高剂量
- 向深处，质子减速后进入相对论区域，能量损失率下降
- 导致出现一个**非Bragg峰的剂量峰**

### 情况2：射线密度分布
- 放射野可能在表面（z=0-1）处射线密度更高
- 随深度扩散

### 情况3：初始化伪影
- 射线初始化处理特殊
- 导致第一层有额外贡献

---

## 🚀 建议的修复验证

### 立即做（无风险）

1. **在test中添加z分层的输出**
   ```cpp
   // 在wrapper_integration_test.cu中
   // 按z输出每层的总剂量和平均剂量
   std::map<int, float> doseByZ;
   for (int z = 0; z < 32; z++) {
       float sum = 0.0f;
       for (int y = 0; y < 512; y++) {
           for (int x = 0; x < 512; x++) {
               sum += doseVolData[z*512*512 + y*512 + x];
           }
       }
       doseByZ[z] = sum;
   }
   
   std::cout << "\nFinal dose distribution by z-slice:" << std::endl;
   for (auto& p : doseByZ) {
       std::cout << "  z=" << p.first << ": sum=" << p.second << std::endl;
   }
   ```

2. **用合成数据而不是参考LUT运行**
   ```bash
   cd build
   ./bin/wrapper_integration_test  # 不设置 RTD_TEST_USE_REFERENCE_LUT
   # 查看是否仍有z=1峰
   ```

3. **检查是否与CT数据有关**
   - 看z=1处的CT HU值是否特殊

### 进阶调试

添加GPU内核中的逐步调试输出，追踪从IDD查表到最终剂量的每一步

---

## 📋 总结

| 方面 | 发现 |
|------|------|
| **现象** | z=1处剂量是z=2的2倍 |
| **物理合理性** | ❌ 不合理（距Bragg峰太远） |
| **可能原因** | 射线权重、初始化、或IDD采样 |
| **代码问题** | 可能，但难以从输出日志确定 |
| **建议** | 添加详细z分层调试输出 |

