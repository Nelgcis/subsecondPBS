# RayTraceDicom LUT单位调查报告

**调查日期**: 2026-02-06  
**结论**: ciddMatrix的物理单位**不明确**，但代码中**没有任何缩放因子**

---

## 📋 调查方法与发现

### 1. 源代码分析

#### energy_struct.h 中的定义
```cpp
std::vector<float> ciddMatrix;  ///< 2D (lineared) matrix of cumulative integral 
                                 ///  dose as function of energy and depth?
```
**观察**: 
- 注释说"cumulative integral dose"，但后面有问号
- **没有说明物理单位**
- **没有说明是绝对值还是相对值**

#### energy_reader.cpp 中的加载
```cpp
eStr.ciddMatrix.resize(eStr.nEnergySamples*eStr.nEnergies);
for (int i=0; i<eStr.nEnergySamples*eStr.nEnergies; ++i) {
    fileReader >> eStr.ciddMatrix[i];
}
```
**观察**:
- **直接读取，没有任何缩放**
- **没有单位转换**
- **没有后处理**

#### kernel_wrapper.cu 中的纹理绑定
```cpp
cudaErrchk(cudaMemcpyToArray(devCumulIddArr, 0, 0, &iddData.ciddMatrix[0], 
                             iddData.nEnergySamples*iddData.nEnergies*sizeof(float), 
                             cudaMemcpyHostToDevice));

cumulIddTex.normalized = false;  // 关键：NOT normalized，存储绝对值
```
**观察**:
- `normalized = false` 意味着纹理存储的是**绝对数值**，不是0-1范围的归一化值
- **这确认了ciddMatrix直接被使用，没有归一化**

#### myKernel.cpp 中的使用
```cpp
cumulDose = tex2D(cumulIddTex, 
                  cumulSp*params.getEnergyScaleFact() + HALF, 
                  params.getEnergyIdx() + HALF);

if (mass > 1e-2f) {
    res = rayWeight * (cumulDose - cumulDoseOld) / mass;
}
```
**观察**:
- **直接使用纹理值，没有乘以任何缩放因子**
- **在计算中也没有×100或×1000**
- **这与我们的当前实现完全相同**

### 2. 代码搜索结果

搜索关键词：`ciddMatrix.*\*`, `dose.*scale`, `dose.*factor`, `dose.*100`, 等

**结果**:
- ❌ 没有找到任何 `ciddMatrix * 100` 或 `ciddMatrix * 1000`
- ❌ 没有找到任何 `DOSE_SCALE` 或 `DOSE_FACTOR` 定义
- ❌ 没有找到任何后期乘以倍数的代码
- ✅ 确认能找到 `res = rayWeight * (cumulDose - cumulDoseOld) / mass` 

**结论**: RayTraceDicom源代码中**确实没有缩放因子**

### 3. 文档搜索

- ❌ energy_struct.h: 没有单位说明
- ❌ README.md: 没有LUT格式说明
- ❌ energy_reader.cpp: 没有注释说明单位
- ❌ myKernel.cpp: 只说"cumulative depth-dose profile"

**搜索关键词**: "PDD", "percent depth dose", "relative dose", "normalized", "Bragg"

**结果**: 没有找到关于IDD数据的单位或含义的明确说明

### 4. LUT数据本身

```
proton_cumul_ddd_data.txt:
  nEnergies: 147
  nEnergySamples: 1024
  ciddMatrix: [0, 3.40e-05]
  Mean: 1.20e-05

数据特征:
  - 最大值 3.4e-05 非常小
  - 不像是绝对剂量(Gy)，太小了
  - 可能是相对值或百分比
```

---

## ⚠️ 关键问题

### 问题1：LUT数据是否是RayTraceDicom原始的？

目前不清楚：
- proton_cumul_ddd_data.txt 是从何处获取的
- 是否是原始RayTraceDicom项目的LUT
- 还是项目中使用的修改版本

### 问题2：为什么RayTraceDicom中也没有缩放因子？

三种可能：

**假说A**：RayTraceDicom原本就输出这么小的值
- 这样的话，它的输出也是1e-5 Gy级别
- 用户需要自行乘以倍数才能使用
- **证据**：代码中没有任何倍数因子

**假说B**：有另外的后处理步骤（不在这些源文件中）
- 可能在主程序(main.cpp)或输出模块中应用
- **证据**：找不到main.cpp，无法验证

**假说C**：这个LUT数据本身就不是用于final dose的
- 可能是中间步骤的数据
- 或者是相对值，需要在使用时乘以某个参考值
- **证据**：数值量级提示这可能是相对值

---

##  🔍 医学物理视角分析

### IDD数据通常的含义

在医学物理中，IDD(Integrated Depth Dose)通常有两种表示：

1. **绝对值形式**: Dose in Gy as function of depth
   - 范围：0-100 Gy（对于治疗束）
   - 或 0-1 Gy（对于标准化参考）

2. **相对值形式**: % depth dose (PDD)
   - 范围：0-100%
   - 相对于Bragg峰处的剂量

### 当前数据的分析

```
观测: ciddMatrix ∈ [0, 3.4e-05]

如果是绝对值(Gy):
  → 3.4e-05 Gy = 34 µGy = 0.034 mGy
  → 这太小了，不像治疗束 ❌

如果是相对值(0-1范围的百分比):
  → 3.4e-05 相当于 0.0034% (百万分之34)
  → 这太小了，如果是相对值也不合理 ❌

如果是相对值需要乘以参考值:
  → ciddMatrix[i] * ReferenceValue = final dose
  → ReferenceValue 可能是 100-1000
  → 这样就合理了 ✓
```

---

## 🤔 现状总结

| 项目 | 结论 |
|------|------|
| RayTraceDicom源代码 | **确实没有缩放因子** |
| 当前wrapper实现 | **完全一致** |
| LUT数据量级 | **太小(1e-5)** |
| 物理合理性 | **输出不合理(12 µGy)** |
| 原因确认 | **困难-源代码中无线索** |

---

## 📊 关键对比

### RayTraceDicom 中的相关代码

```
energy_struct.h:
  ↓
energy_reader.cpp (直接读取，无缩放)
  ↓
kernel_wrapper.cu (upload到GPU，normalized=false)
  ↓
myKernel.cpp (直接使用：res = rayWeight * cumulDose / mass)
  ↓
输出: 小值 (~1e-5 Gy)
```

### 当前实现 中的相关代码

```
include/utils/energy_struct.h:
  ↓
src/utils/energy_reader.cpp (直接读取，无缩放)
  ↓
src/utils/precompiled_texture_manager.cu (upload到GPU)
  ↓
src/algorithms/idd_sigma.cu (直接使用：res = rayWeight * cumulDose / mass)
  ↓
输出: 小值 (~1e-5 Gy)
```

**结论**：两者完全相同，没有差异

---

## 可能的原因推断

### 原因1：这可能是原始设计（概率：30%）
- RayTraceDicom可能一直都是这样
- 用户在使用时手动应用缩放
- 代码中没有记录这个操作

### 原因2：LUT数据源有问题（概率：40%）
- proton_cumul_ddd_data.txt 可能被错误地缩放或修改过
- 原始RayTraceDicom使用的可能是不同的LUT
- 导致数据量级不同

### 原因3：有隐藏的后处理步骤（概率：20%）
- 可能在输出DICOM或其他文件时应用
- 可能在上层调用程序中应用
- 代码中不可见

### 原因4：这是相对值，需要参考值乘法（概率：10%）
- ciddMatrix可能是相对于某个参考值
- 使用时应该：final_dose = ciddMatrix * reference_dose
- 但这会很不寻常

---

## ✅ 建议的下一步

### 立即可做（无风险）

1. **查找LUT的原始来源**
   ```bash
   grep -r "proton_cumul_ddd" . --include="*.txt" --include="*.md"
   grep -r "RayTraceDicom" . --include="*.txt" --include="*.md"
   ```

2. **查找项目文档**
   - 是否有项目报告、技术文档、论文？
   - 是否有使用说明提及剂量单位？

3. **比对医学物理参考**
   - 联系原始author (Joakim da Silva, Fernando Hueso-González)
   - 查看他们的论文中是否有说明

### 如果无法确认（谨慎修复）

根据物理合理性，**很可能需要乘以倍数因子**：

**安全的做法**：
1. 在代码中添加配置常数：
   ```cpp
   const float CIDD_NORMALIZATION_FACTOR = 1.0f;  // 或 100.0f 或 1000.0f
   ```

2. 在读取或使用时应用：
   ```cpp
   res = CIDD_NORMALIZATION_FACTOR * rayWeight * (cumulDose - cumulDoseOld) / mass;
   ```

3. 添加详细注释：
   ```cpp
   // TODO: Verify the physical units of ciddMatrix
   // Current: outputs dose in [1e-5 Gy range]
   // May need CIDD_NORMALIZATION_FACTOR = 100 or 1000
   ```

4. 通过能与已知数据对比来验证

---

## 结论

根据对RayTraceDicom源代码的深入调查：

**❌ RayTraceDicom中确实没有缩放因子**
- energy_reader.cpp：直接读取
- kernel_wrapper.cu：直接上传到GPU
- myKernel.cpp：直接使用

**❌ 当前wrapper实现完全一致**
- 没有遗漏任何地方
- 代码是正确的

**⚠️ 输出的剂量值确实太小**
- 12.29 µGy 不符合物理预期
- 原始RayTraceDicom也会有相同问题

**🤔 原因不明**
- 源代码中无法找到线索
- 可能需要查找：
  1. 原始论文（da Silva et al.）
  2. 项目更完整的文档
  3. 主程序或输出模块
  4. 原始LUT的来源和定义

**建议**：在应用任何修复之前，应该先通过以下途径确认：
1. 查找RayTraceDicom的原始使用说明
2. 对比已知的proton therapy数据
3. 联系原始作者

