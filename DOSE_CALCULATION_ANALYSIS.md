# 小剂量问题分析报告
## Wrapper Integration Test 剂量计算 (120, 160, 180 MeV) 只有 1E-5 量级

---

## 问题描述

Wrapper integration test 中计算的质子剂量只有 **1E-5 量级**，明显偏小。需要确认：
1. 是否遵循了raytracedicom的完整算法
2. IDD查找过程是否正确
3. 是否存在单位混淆（cm vs mm）

---

## 核心算法分析

### 1. 剂量计算的关键公式 (源自 idd_sigma.cu)

```cuda
// IDD查找
const float depthIdx = cumulSp * params.getEnergyScaleFact() + HALF;
const float energyIdx = params.getEnergyIdx() + HALF;
cumulDose = tex2D<float>(cumulIddTex, depthIdx, energyIdx);

// 剂量增量 = rayWeight * (cumulDose差) / mass
const float mass = density * params.stepVol(stepNo);
if (mass > 1e-2f) {
    res = rayWeight * (cumulDose - cumulDoseOld) / mass;
}
```

这是关键一步！**剂量 = rayWeight × (IDD差) / mass**

---

## 关键参数检查清单

### 2. proton_cumul_ddd_data.txt 文件结构

从文件头可见：
```
1024 147  (nEnergySamples=1024, nEnergies=147)
[energiesPerU - 147个值，范围62-320 MeV]
[peakDepths - 147个值，范围30-320]
[scaleFacts - 147个值，非常小的数值~1E-5量级]
[ciddMatrix - 1024*147个IDD值，也是~1E-5量级]
```

**关键发现：**
- **scaleFacts 和 ciddMatrix 的值都非常小（~1E-5）**
- 这可能导致最终剂量也是 1E-5 量级

---

## 问题根源分析

### 3. raytracedicom_wrapper.cu中的能量深度单位处理

位置：[src/core/raytracedicom_wrapper.cu](src/core/raytracedicom_wrapper.cu#L580-L650)

```cpp
// 关键参数：energyDepthToMm
const float stepLength_input = ctResolution.z;
const float lenToMm = (stepLength_input > 0.0f && stepLength_input < 0.3f) ? 10.0f : 1.0f;

// 能量缩放因子的计算
float energyDepthToMm = lenToMm;
energyScaleFact = energyScaleFact_table / energyDepthToMm;
```

**单位转换链：**
1. CT分辨率可能是cm (0.1cm = 1mm)
2. `lenToMm = 10.0f` (cm → mm)
3. LUT中的 peakDepth 和 scaleFact 已经假设是某个单位
4. `energyScaleFact = energyScaleFact_table / energyDepthToMm`

---

## 可能的问题

### 4. 三大怀疑点

#### **问题A: IDD矩阵值太小**

LUT中的 `ciddMatrix` 值如果本身就是 ~1E-5，那么：
```
最终剂量 = rayWeight * (1E-5差) / mass ≈ 1E-5 ~ 1E-6
```

需要检查：
- ciddMatrix 的单位是什么？（应该是Gy、cGy还是其他？）
- 是否需要乘以某个归一化因子？

#### **问题B: scaleFact 计算或应用错误**

在 [src/core/raytracedicom_wrapper.cu#L1056-L1064](src/core/raytracedicom_wrapper.cu#L1056-L1064)：
```cpp
const float energyScaleFact_table = energyScaleFact;
// ...
energyScaleFact = energyScaleFact_table / energyDepthToMm;
```

如果 `energyScaleFact_table` 本身就是 ~1E-5（从LUT读出），再除以 `energyDepthToMm`，会导致更小的值。

#### **问题C: 质量计算错误**

```cpp
const float mass = density * params.stepVol(stepNo);
```

如果 `stepVol` 计算有误（例如被误缩放了1000倍），会反映到分母中，使剂量变小。

---

## wrapper_integration_test.cu 中的可能问题

### 5. 合成LUT的问题 (第136-170行)

```cuda
energy.ciddMatrix.resize(energy.nEnergySamples * energy.nEnergies);
for (int e = 0; e < energy.nEnergies; ++e) {
    float cumul = 0.0f;
    const float peakDepth = energy.peakDepths[e];
    const float sigma = 3.0f;  // "mm"
    for (int s = 0; s < energy.nEnergySamples; ++s) {
        const float depth = static_cast<float>(s);  // "mm"
        const float idd = expf(-0.5f * (depth - peakDepth) * (depth - peakDepth) / (sigma * sigma));
        cumul += idd;
        energy.ciddMatrix[e * energy.nEnergySamples + s] = cumul * 10.0f;  // 缩放因子10.0
    }
}
```

**问题：** 
- 合成的IDD值被乘以 `10.0f`，但缩放因子是什么单位？
- `peakDepths` 初始化为 `{10.0f, 12.0f, 14.0f}` (太小了！应该是30-200mm)
- `scaleFacts` 初始化为 `{1.0f, 1.0f, 1.0f}` (可能错误)

---

## 从参考LUT中读取时的问题

### 6. loadEnergyDataFromReferenceTables 函数 (第197-210行)

当使用 `RTD_TEST_USE_REFERENCE_LUT=1` 时，从 `tables/proton_cumul_ddd_data.txt` 读取参数。

**需要检查的问题：**
1. **LUT单位是否明确？**
   - energiesPerU: 单位是MeV吗？
   - peakDepths: 单位是mm吗？  
   - scaleFacts: 这些值应该是什么量级？
   - ciddMatrix: 单位是什么？(Gy, cGy, 相对剂量？)

2. **scaleFacts 的含义**
   
   从代码看：
   ```cuda
   const float depthIdx = cumulSp * params.getEnergyScaleFact() + HALF;
   ```
   
   这意味着：`depthIdx = WEPL(mm) * scaleFact`
   
   如果 WEPL 典型范围是 0-300mm，scaleFact 应该在 1-3.5 范围内才能得到合理的texture索引。
   
   **但LUT中的scaleFact是 ~1E-5！** 这会导致：
   ```
   depthIdx = 100mm * 1E-5 = 1E-3 (远小于样本数1024)
   ```
   这是错误的！

---

## 结论与建议

### 7. 最可能的原因：scaleFact 和 ciddMatrix 的量纲问题

**问题排序（概率从高到低）：**

1. **最可能 [80%]：** 
   - LUT中的 `ciddMatrix` 和 `scaleFacts` 的单位问题
   - ciddMatrix 可能是相对剂量（0-1范围），需要乘以某个归一化常数
   - scaleFacts 的实际含义与使用方式不符

2. **次可能 [15%]：**
   - test中合成的IDD数据参数设置不当
   - peakDepths 太小（10-14mm）导致查表时索引不在Bragg峰区域

3. **可能性较小 [5%]：**
   - 单位转换链（cm/mm混淆）中某处错误
   - 质量计算或归一化系数应用错误

---

##  建议的修复步骤

### 8. 立即检查

1. **打印LUT数据统计：**
   ```cpp
   std::cout << "ciddMatrix[0]=" << energy.ciddMatrix[0] << std::endl;
   std::cout << "ciddMatrix[max]=" << energyData->ciddMatrix.back() << std::endl;
   std::cout << "scaleFact[0]=" << energy.scaleFacts[0] << std::endl;
   ```

2. **检查IDD查表的实际indices：**
   ```cuda
   float depthIdx = cumulSp * params.getEnergyScaleFact() + HALF;
   // 打印 depthIdx 的实际范围（应该是 0-1024）
   ```

3. **比对raytracedicom源代码中的算法实现**

---

## 附录：关键代码位置

| 文件 | 行号 | 内容 |
|------|------|------|
| [wrapper_integration_test.cu](src/tests/wrapper_integration_test.cu) | 136-170 | 合成IDD数据 |
| [wrapper_integration_test.cu](src/tests/wrapper_integration_test.cu) | 197-210 | 读取参考LUT |
| [idd_sigma.cu](src/algorithms/idd_sigma.cu) | 90-130 | 核心剂量计算 |
| [raytracedicom_wrapper.cu](src/core/raytracedicom_wrapper.cu) | 580-660 | 能量单位处理 |
| [raytracedicom_wrapper.cu](src/core/raytracedicom_wrapper.cu) | 1020-1140 | 参数插值和scaleFact计算 |

