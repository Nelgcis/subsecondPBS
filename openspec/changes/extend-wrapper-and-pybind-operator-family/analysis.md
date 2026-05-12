## 报告：`NUCLEAR_CORR` 在 RayTraceDicom 中的位置、作用与 primary/halo 差异

我先说明一下范围：我这边没有成功直接读取到你上传 zip 的本地内容，因此以下报告是基于同名公开仓库 `ferdymercury/RayTraceDicom` 的当前 `main` 结构分析；如果你手头 zip 有本地改动，文件路径或行号可能略有差异，但宏相关的计算链路应基本一致。该项目是 CUDA 质子剂量计算代码，`NUCLEAR_CORR` 在这里用于给 primary 剂量路径加入核相互作用产生的 halo 修正。([GitHub][1])

---

# 1. 总体结论

`NUCLEAR_CORR` 不是简单地“在 primary 剂量后面额外加一坨 halo”。代码里启用该宏后，primary core 本身也会被改写：原本完整的 primary depth-dose 增量会按 `nucWeight` 拆分成两部分：

```cpp
primary_core = (1 - nucWeight) * primary_increment
halo         = nucWeight * primary_increment
```

所以它的物理/数值语义更接近：

```text
总剂量 = 被核权重削减后的 primary core + 额外用更宽横向核卷积处理的 nuclear halo
```

也就是说，halo 是 primary 剂量的一种补充，但为了避免重复计量，启用 `NUCLEAR_CORR` 后 primary 的局部贡献会先乘以 `(1 - nucWeight)`。核心拆分发生在 `fillIddAndSigma()` 内。([GitHub][2])

---

# 2. `NUCLEAR_CORR` 的宏定义与模式

## 2.1 编译入口：`CMakeLists.txt`

`NUCLEAR_CORR` 是 CMake 缓存字符串选项，允许值为：

```text
SOUKUP
FLUKA
GAUSS_FIT
OFF
```

CMake 中还先定义了：

```cpp
SOUKUP    = 0
FLUKA     = 1
GAUSS_FIT = 2
```

然后根据用户选择，把 `NUCLEAR_CORR` 定义成对应的整数宏：

```cpp
-DNUCLEAR_CORR=SOUKUP
-DNUCLEAR_CORR=FLUKA
-DNUCLEAR_CORR=GAUSS_FIT
```

如果为 `OFF`，则不定义 `NUCLEAR_CORR`，走纯 primary 路径。([GitHub][3])

---

# 3. 涉及文件总览

| 文件                                      | 对应 primary 的位置/函数                                                           | `NUCLEAR_CORR` 增加的内容                                            | 作用                                                  |
| --------------------------------------- | --------------------------------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------- |
| `CMakeLists.txt`                        | 整体编译配置                                                                      | `NUCLEAR_CORR=SOUKUP/FLUKA/GAUSS_FIT`                           | 选择 halo correction 模式                               |
| `src/energy_struct.h`                   | `EnergyStruct` 中 primary LUT 数据结构                                           | `nucWeightMatrix`、`nucSqSigmaMatrix`                            | 保存核权重和 halo 额外横向方差                                  |
| `src/energy_reader.cpp`                 | 读取 primary 能量/深度剂量 LUT                                                      | 读取 `nuclear_weights_and_sigmas_*.txt`                           | 为每个能量和深度采样点加载 halo 参数                               |
| `src/kernel_wrapper.cuh`                | primary kernel 声明、`primTransfDiv()`、非核修正版 `fillIddAndSigma()`               | `extendAndPadd()`、`nucTransfDiv()`、带核参数的 `fillIddAndSigma()` 重载 | 声明 halo 的 BEV map 扩展、ID算子和体素回填流程                    |
| `src/kernel_wrapper.cu`                 | 主 GPU pipeline：ray tracing、primary IDD/sigma、KS superposition、dose transfer | 绝大多数 halo 实现逻辑都在这里                                              | 负责 halo 内存、纹理、剂量拆分、核 sigma、halo superposition 和最终加和 |
| `src/fill_idd_and_sigma_params.cuh/.cu` | `fillIddAndSigma()` 的参数对象                                                   | `spotDist`、`nucMemStep`、`entrySigmaSq` 等参数被 halo 路径使用           | 给 halo 的归一化、索引步进、入口束斑宽度修正提供参数                       |
| `LUTs/nuclear_weights_and_sigmas_*.txt` | 对应 primary 的 `proton_cumul_ddd_data.txt`                                    | 三套 nuclear correction LUT                                       | 提供核权重 `nucWeight` 和核 halo 额外方差 `nucSqSigma`         |

`EnergyStruct` 中 primary 本来保存的是能量采样、峰深、scale factor、cIDD matrix、density/SP/radiation length 等；启用 `NUCLEAR_CORR` 后，额外增加 `nucWeightMatrix` 和 `nucSqSigmaMatrix` 两个矩阵。([GitHub][4])

---

# 4. 数据读取层：primary LUT 与 halo LUT 的关系

## 4.1 primary 原始数据

primary 的核心数据来自类似：

```text
LUTs/proton_cumul_ddd_data.txt
```

它提供每个能量、深度采样点的 cumulative depth dose，即后续 `fillIddAndSigma()` 中用：

```cpp
cumulDose - cumulDoseOld
```

得到当前 depth step 的剂量增量。

## 4.2 halo 额外数据

启用 `NUCLEAR_CORR` 后，`energy_reader.cpp` 会根据宏模式选择不同的 nuclear LUT：

| 模式          | 文件                                           |
| ----------- | -------------------------------------------- |
| `SOUKUP`    | `LUTs/nuclear_weights_and_sigmas_Soukup.txt` |
| `FLUKA`     | `LUTs/nuclear_weights_and_sigmas_Fluka.txt`  |
| `GAUSS_FIT` | `LUTs/nuclear_weights_and_sigmas_fit.txt`    |

读取时并不是盲目加载，而是会检查 nuclear LUT 与 primary LUT 的结构是否一致，包括采样点数、能量数、peak depth、scale factor 等。这说明 halo 的深度维度和能量维度必须严格对齐 primary 的 cIDD 数据。之后代码把每个采样点对应的 `nucWeight` 和 `nucSqSigma` 写入 `eStr.nucWeightMatrix` 和 `eStr.nucSqSigmaMatrix`。([GitHub][5])

---

# 5. GPU 数据准备：纹理、数组和 grid 的差异

## 5.1 primary 原有路径

primary 的 GPU 路径主要包括：

1. 把 primary cIDD、density、stopping power 等数据放入 GPU texture。
2. 对 spot weights 做 2D convolution，生成 primary ray weights。
3. 计算每个 BEV step 的 primary IDD 和 effective sigma。
4. 做 kernel superposition。
5. 把 BEV dose 转回 CT/dose volume。

## 5.2 `NUCLEAR_CORR` 增加的 GPU 数据

启用宏后，`kernel_wrapper.cu` 会额外：

```cpp
devNucWeightArr
devNucSqSigmaArr
nucWeightTex
nucSqSigmaTex
```

也就是把 nuclear weight 和 nuclear sigma square 绑定到 CUDA texture，供 `fillIddAndSigma()` 采样。([GitHub][2])

还会额外分配：

```cpp
devNucRayWeights
devNucIdd
devNucRSigmaEff
devNucSpotIdx
devTileNucRadCtrs
devBevNucDose
```

这些分别对应 halo 的 ray weight、depth dose、effective reciprocal sigma、spot 索引映射、tile 半径统计和 BEV nuclear dose buffer。([GitHub][2])

---

# 6. primary grid 与 halo grid 的区别

这是 `NUCLEAR_CORR` 最重要的结构差异之一。

## 6.1 primary：使用扩大后的 ray grid

primary 会根据 spot grid 和卷积截断半径扩展计算网格：

```cpp
primRayDims
convIntermN
rayWeightsN
rayIddN
```

其中 primary 的 ray grid 会被 `CONV_SIGMA_CUTOFF * maxSpotSigmas` 扩大，用于包含入口束斑的横向扩展。随后通过：

```cpp
gpuConvolution2D(...)
```

把 spot weights 卷积成 `devPrimRayWeights`。([GitHub][2])

## 6.2 halo：使用 padded spot grid，不做 primary 那种预卷积

halo 的 `nucRayDims` 基本来自原始 `spotGridDims`，只是为了适配 kernel superposition tile，会在 x/y 方向 pad 到 `superpTileX/Y` 的整数倍：

```cpp
nucRayDims = roundTo(spotGridDims.x/y, superpTileX/Y)
```

注释里明确说是为了让 nuclear PB map 可以使用同一套 KS kernel function。halo 权重不是通过 `gpuConvolution2D()` 得到，而是通过：

```cpp
extendAndPadd(devSpotWeights, devNucRayWeights, spotGridDims)
```

把 spot weight 复制/扩展并 pad 成 nuclear PB map。([GitHub][2])

**含义：**

| 项目                     | primary                               | halo                          |
| ---------------------- | ------------------------------------- | ----------------------------- |
| 横向 grid                | 扩大后的 computational ray grid           | 原始 spot grid pad 到 tile 整数倍   |
| 权重生成                   | `gpuConvolution2D()` 预卷积 spot weights | `extendAndPadd()` 复制并补零       |
| 是否每个 primary ray 都有 PB | 是                                     | 不是，只有对应 spot 的位置才有 nuclear PB |
| 后续 superposition       | 使用 primary IDD/sigma                  | 使用 nuclear IDD/sigma          |

---

# 7. ray tracing：halo 复用 primary 的 ray tracing 结果

代码中 `fillBevDensityAndSp()` 对 density 和 stopping power 的 ray tracing 只在 primary ray grid 上做一次。halo 并没有再做一套完整的独立 ray tracing，而是通过 `devNucSpotIdx` 建立从 primary ray index 到 nuclear PB index 的映射：

```cpp
devNucSpotIdx
```

默认值为 `-1`，只有原始 spot 对应的 primary ray 位置会映射到 nuclear index。这样 `fillIddAndSigma()` 在遍历 primary ray 时，如果该 ray 对应一个 nuclear PB，才会写 halo 的 `nucIdd` 和 `nucRSigmaEff`。([GitHub][2])

---

# 8. 核心插入点：`fillIddAndSigma()`

## 8.1 文件与函数

主要文件：

```text
src/kernel_wrapper.cu
src/kernel_wrapper.cuh
```

primary 对应函数：

```cpp
fillIddAndSigma(...)
```

`kernel_wrapper.cuh` 中有两个版本声明：

1. 普通 primary 版本：没有 nuclear 参数。
2. `#ifdef NUCLEAR_CORR` 版本：额外接收 `bevNucIdd`、`bevNucRSigmaEff`、`nucRayWeights`、`nucIdcs`、`nucWeightTex`、`nucSqSigmaTex` 等。([GitHub][6])

---

## 8.2 primary 在无 `NUCLEAR_CORR` 时的公式

未启用宏时，当前 step 的 primary dose 为：

```cpp
res = rayWeight * (cumulDose - cumulDoseOld) / mass;
```

也就是完整的 primary cumulative DDD 增量乘以 ray weight，再除以当前体素质量因子。([GitHub][2])

---

## 8.3 启用 `NUCLEAR_CORR` 后 primary 被削减

启用宏后，primary 的同一段变为：

```cpp
res = (1.0f - nucWeight)
    * rayWeight
    * (cumulDose - cumulDoseOld)
    / mass;
```

这就是 nuclear correction 对 primary 的第一处关键改动：primary core 被乘了 `(1 - nucWeight)`。([GitHub][2])

---

## 8.4 halo dose 的新增公式

同一处还会额外计算 nuclear halo：

```cpp
nucRes = nucWeight
       * nucRayWeight
       * (cumulDose - cumulDoseOld)
       / (mass * spotDist * spotDist);
```

其中：

* `nucWeight` 来自 nuclear LUT。
* `nucRayWeight` 来自 `devNucRayWeights`，也就是 pad 后的 spot weight。
* `spotDist * spotDist` 是横向 spacing 相关的面积归一化。
* `mass` 与 primary 一样参与剂量归一化。([GitHub][2])

所以，halo 和 primary 共享同一个 depth-dose 增量：

```cpp
cumulDose - cumulDoseOld
```

但权重、横向 spread 和归一化处理不同。

---

# 9. sigma / 横向扩散处理的异同

## 9.1 primary sigma

primary 的 effective reciprocal sigma 大致来自：

```cpp
rSigmaEff = voxelWidth / (sqrt(2) * (sqrt(sigmaSq) + sigmaDelta))
```

其中 `sigmaSq` 来自多重库仑散射、发散等计算，`sigmaDelta` 是经验修正项。

## 9.2 halo sigma

halo 使用的是：

```cpp
nucRSigmaEff =
    0.5f * spotDist * (vx + vy)
    / (sqrt(2) * sqrt(sigmaSq + nucSqSigma + entrySigmaSq));
```

关键差异：

| 项目             | primary           | halo                    |
| -------------- | ----------------- | ----------------------- |
| 基础散射           | `sigmaSq`         | `sigmaSq`               |
| nuclear 额外横向方差 | 无                 | `+ nucSqSigma`          |
| 入口束斑方差         | primary 已在预卷积中体现  | 显式加 `+ entrySigmaSq`    |
| 经验 delta       | 使用 `sigmaDelta`   | halo 公式中没有 `sigmaDelta` |
| 横向尺度           | voxel/ray spacing | 额外乘 `spotDist`          |
| 输出             | `bevRSigmaEff`    | `bevNucRSigmaEff`       |

这说明 halo 的横向扩散不是复用 primary 的 primary sigma，而是在 primary 散射基础上叠加 nuclear LUT 提供的额外横向方差，并且补上 entry spot size。([GitHub][2])

---

# 10. 各 nuclear correction 模式的额外修正

在 `fillIddAndSigma()` 附近，不同模式使用不同的 `eRefSq` 和 `sigmaDelta`：

| 模式          |                             nuclear LUT |  `eRefSq` | `sigmaDelta` | 额外修正                                                                   |
| ----------- | --------------------------------------: | --------: | -----------: | ---------------------------------------------------------------------- |
| `SOUKUP`    | `nuclear_weights_and_sigmas_Soukup.txt` | `190.44f` |       `0.0f` | 使用 nuclear weight/sigma；post-Bragg empirical correction 仍启用            |
| `FLUKA`     |  `nuclear_weights_and_sigmas_Fluka.txt` | `216.09f` |      `0.08f` | 使用 nuclear weight/sigma；post-Bragg empirical correction 仍启用            |
| `GAUSS_FIT` |    `nuclear_weights_and_sigmas_fit.txt` | `169.00f` |      `0.06f` | `entrySigmas.x/y *= 0.97f`；并禁用某个 post-Bragg empirical sigma correction |
| `OFF`       |                                       无 | `198.81f` |      `0.21f` | 纯 primary，不拆分 halo                                                     |

`GAUSS_FIT` 的 `*0.97` 发生在每个 layer 的 `entrySigmas` 计算之后、spot weight convolution 和 `fillIddAndSigma()` 之前：

```cpp
entrySigmas[layerNo].x = 0.97f * entrySigmas[layerNo].x;
entrySigmas[layerNo].y = 0.97f * entrySigmas[layerNo].y;
```

这会同时影响 primary 的 spot weight convolution，以及传给 halo 的 `entrySigmaSq`。另外，代码中有一个近 Bragg peak 后的经验修正：

```cpp
sigmaSq -= 1.5f * (incScat + incDiv) * density;
```

这个修正只有在非 `GAUSS_FIT` 时启用；对于 `NUCLEAR_CORR == GAUSS_FIT` 被排除。([GitHub][2])

---

# 11. superposition 阶段：primary 与 halo 用同一套 KS 框架，但数组独立

## 11.1 primary

primary 的 KS superposition 使用：

```cpp
devPrimIdd
devPrimRSigmaEff
devBevPrimDose
devPrimInOutIdcs
devTilePrimRadCtrs
```

先用 `tileRadCalc()` 根据 `devPrimRSigmaEff` 计算每个 tile 的卷积半径，然后用 `kernelSuperposition<rad>()` 把 primary IDD map 转成 BEV primary dose。([GitHub][6])

## 11.2 halo

halo 走同一个 template kernel，但输入和输出全部换成 nuclear 版本：

```cpp
devNucIdd
devNucRSigmaEff
devBevNucDose
devNucInOutIdcs
devTileNucRadCtrs
```

也就是说，superposition 算法本身没有单独为 halo 写一套物理模型；差异已经体现在前一步生成的 `nucIdd` 和 `nucRSigmaEff` 中。([GitHub][2])

---

# 12. dose transfer / 体素回填：primary 与 halo 分别转回 dose grid 后相加

## 12.1 primary transfer

primary 使用：

```cpp
primTransfDiv(...)
```

从：

```cpp
bevPrimDoseTex
```

采样，并把正值累加到输出 dose。

## 12.2 halo transfer

启用宏后额外使用：

```cpp
nucTransfDiv(...)
```

从：

```cpp
bevNucDoseTex
```

采样，也把正值累加到同一个输出 dose。

`kernel_wrapper.cuh` 中可以看到 `primTransfDiv()` 和 `nucTransfDiv()` 是并列声明的两个函数，参数结构非常相似，只是 halo 读取的是 nuclear BEV dose texture。([GitHub][6])

最终结果的含义就是：

```text
输出 dose = primary BEV dose transfer + nuclear halo BEV dose transfer
```

但要记住：这里的 primary 已经不是未修正 primary，而是 `(1 - nucWeight)` 之后的 primary core。

---

# 13. 按 primary 计算环节总结：哪些 primary 需要额外考虑 halo

| primary 环节        | 是否受 `NUCLEAR_CORR` 影响 | halo 额外处理                                          | 与 primary 的异同                                                  |
| ----------------- | --------------------- | -------------------------------------------------- | -------------------------------------------------------------- |
| 编译配置              | 是                     | 选择 `SOUKUP/FLUKA/GAUSS_FIT` 模式                     | primary OFF 时没有 halo；启用后编译不同代码路径                               |
| 能量/LUT 读取         | 是                     | 读取 nuclear weight 和 nuclear sigma LUT              | nuclear LUT 必须与 primary cIDD 采样结构一致                            |
| EnergyStruct 数据结构 | 是                     | 新增 `nucWeightMatrix`、`nucSqSigmaMatrix`            | primary 保存 cIDD/density/SP；halo 保存权重比例和额外 sigma²               |
| GPU texture       | 是                     | 新增 `nucWeightTex`、`nucSqSigmaTex`                  | primary texture 给 cIDD/density/SP；halo texture 给核权重和核扩散        |
| ray grid          | 是                     | halo 用 padded spot grid                            | primary grid 会因卷积截断扩大；halo grid 主要是 spot grid pad              |
| spot weight       | 是                     | `extendAndPadd()` 生成 `devNucRayWeights`            | primary 用 `gpuConvolution2D()` 预卷积；halo 不走该预卷积                 |
| ray tracing       | 间接受影响                 | halo 通过 `devNucSpotIdx` 复用 primary ray tracing     | halo 不单独 ray trace 全网格                                         |
| depth dose / IDD  | 是，核心位置                | `nucWeight` 拆分 primary 与 halo                      | primary 乘 `(1-nucWeight)`；halo 乘 `nucWeight`                   |
| sigma             | 是                     | halo 加 `nucSqSigma + entrySigmaSq`                 | primary 用 primary sigma 和 `sigmaDelta`；halo 用更宽的 nuclear sigma |
| KS superposition  | 是                     | halo 单独 `tileRadCalc()` + `kernelSuperposition()`  | 算法相同，输入数组不同                                                    |
| dose transfer     | 是                     | `nucTransfDiv()` 加到最终 dose                         | primary/halo 分别采样 BEV dose texture 后累加                         |
| `GAUSS_FIT` 模式    | 是                     | `entrySigmas *= 0.97`，禁用某 post-BP sigma correction | 这是 GAUSS_FIT 专属额外修正                                            |

---

# 14. 关键代码逻辑摘要

最核心的逻辑可以概括为下面这段伪代码：

```cpp
// 原 primary depth-dose 增量
deltaIDD = cumulDose - cumulDoseOld;

// 无 NUCLEAR_CORR
primary = rayWeight * deltaIDD / mass;

// 有 NUCLEAR_CORR
nucWeight = tex(nucWeightTex, energy, depth);

primary = (1 - nucWeight)
        * rayWeight
        * deltaIDD
        / mass;

halo = nucWeight
     * nucRayWeight
     * deltaIDD
     / (mass * spotDist * spotDist);

primarySigma = f(sigmaSq, sigmaDelta);

haloSigma = f(
    sigmaSq
  + nucSqSigma
  + entrySigmaSq
);
```

这正好对应你说的“halo 作为 primary 剂量的补充”，但实现上必须强调：它不是直接追加完整 primary 的 halo，而是先从 primary 中按 `nucWeight` 分出去，再以 halo 的横向模型重新加回来。([GitHub][2])

---

# 15. 一个建议核对点

我在当前公开 `main` 中看到，`FillIddAndSigmaParams` 里有 `nucMemStep` 字段和 `getNucMemStep()`，`fillIddAndSigma()` 中 nuclear index 会使用这个步长推进；但在调用构造参数时，`spotDistInRays` 后面传入的参数显示为 `0`。如果该位置确实对应 `nucMemoryStep`，那么这看起来值得重点核对，因为 nuclear depth slice 的内存步长通常应当与 nuclear ray plane size 相关，而不是 0。这个点我建议你在本地 zip 版本里再确认一下是否已经修过，或者构造函数实参与字段顺序是否有其它重载/宏条件影响。([GitHub][7])

---

# 16. 最终判断

`NUCLEAR_CORR` 覆盖的 primary 计算主要有三大类：

1. **primary depth dose 的数值拆分**
   最核心：`fillIddAndSigma()` 中 primary 从完整贡献变为 `(1 - nucWeight)` 贡献，同时新增 `nucWeight` 对应的 halo 贡献。

2. **primary 横向扩散模型的补充**
   halo 不复用 primary 的预卷积横向模型，而是用 `nucSqSigma + entrySigmaSq` 构造更宽的 nuclear halo sigma。

3. **primary BEV dose pipeline 的并行扩展**
   halo 拥有独立的 IDD、sigma、tile radius、BEV dose 和 transfer kernel，但 superposition 框架与 primary 共享同一套模板逻辑。

一句话总结：

> `NUCLEAR_CORR` 把原本 primary 的剂量核分解为“削减后的 primary core + 单独横向展宽的 nuclear halo”，其中 `SOUKUP/FLUKA/GAUSS_FIT` 决定使用哪套 nuclear weight/sigma LUT 和对应经验修正；`GAUSS_FIT` 额外包含 `entrySigmas * 0.97` 以及关闭某个 post-Bragg sigma 修正。

[1]: https://github.com/ferdymercury/RayTraceDicom?utm_source=chatgpt.com "GitHub - ferdymercury/RayTraceDicom: Sub-second pencil beam dose ..."
[2]: https://github.com/ferdymercury/RayTraceDicom/blob/main/src/kernel_wrapper.cu "RayTraceDicom/src/kernel_wrapper.cu at main · ferdymercury/RayTraceDicom · GitHub"
[3]: https://raw.githubusercontent.com/ferdymercury/RayTraceDicom/master/CMakeLists.txt "raw.githubusercontent.com"
[4]: https://github.com/ferdymercury/RayTraceDicom/blob/main/src/energy_struct.h "RayTraceDicom/src/energy_struct.h at main · ferdymercury/RayTraceDicom · GitHub"
[5]: https://github.com/ferdymercury/RayTraceDicom/blob/main/src/energy_reader.cpp "RayTraceDicom/src/energy_reader.cpp at main · ferdymercury/RayTraceDicom · GitHub"
[6]: https://github.com/ferdymercury/RayTraceDicom/blob/main/src/kernel_wrapper.cuh "RayTraceDicom/src/kernel_wrapper.cuh at main · ferdymercury/RayTraceDicom · GitHub"
[7]: https://github.com/ferdymercury/RayTraceDicom/blob/main/src/fill_idd_and_sigma_params.cuh "RayTraceDicom/src/fill_idd_and_sigma_params.cuh at main · ferdymercury/RayTraceDicom · GitHub"

---

## 2026-04-29 本地 patch10 严格迁移审计补充（对照本地 RTD-main 源码）

范围说明：本补充基于本仓内 `patch10_mod_20260410/RayTraceDicom-main (1)/RayTraceDicom-main` 与 `patch10_mod_20260410/include` + `patch10_mod_20260410/src` 的逐步对照，检查标准采用“主路径不允许省略、不允许硬赋值”。

### 审计结论（补充）

当前 patch10 仍不满足“完全依照 RTD-main 主路径迁移”的严格标准，主要有三类阻塞项：

1. **运行时核修正路径省略**
   `subsecondWrapper` 对 `nuclear_correction=true` 仍直接抛出 “not wired yet” 级别错误，未进入已编译宏分支执行链。此项属于明确省略，不是参数细节差异。

2. **主路径物理量存在硬赋值**
   在 live IDD/sigma 参数构造中仍可见 `iddParams.rRlScale = 1.0f` 等固定值路径；而 RTD-main 主链语义是使用能量表驱动缩放（`iddData.rRlScaleFact` 语义家族）。

3. **主路径几何/覆盖仍有 fallback 常量**
   CPB/ray 网格仍存在兼容分支常量回退（例如 `doseResolution * 0.5` 级别 spacing fallback，以及投影覆盖不可用时回退到整 dose box 尺度），这与“严格按 upstream 主路径数据驱动”不一致。

### 对本 change 的约束含义

- Section 9 的 halo 迁移任务必须把上述 3 项视为**一等阻塞**，否则后续 halo-on 与 upstream 的差异来源会混淆（分不清是缺核分支、硬赋值、还是几何回退导致）。
- 在这些阻塞项关闭前，本 change 不应宣称“已完成严格 RTD-main 算法迁移”，最多只能声明“主链可运行但带兼容回退”。
