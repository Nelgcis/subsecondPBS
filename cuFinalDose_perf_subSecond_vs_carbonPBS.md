# 为什么 subSecond(RTD) cuFinalDose 比 carbonPBS 慢 ~3x（274mm grid）

日期：2026-06-15
实测（同一 274mm 边长 grid，逐 layer）：
- **subSecond (RTD WEQ 模式)**：~0.59–0.87 s/layer（均值 ~0.74s）
- **carbonPBS**：~0.23–0.29 s/layer（均值 ~0.25s）
- 倍率：**~2.9x**

**重要前提**：这两组都跑在 **WEQ 模式**——两边都吃 `cuPrepareWEQ` 预算好的 `rayweq`，都没有在 dose 阶段重新追踪 CT。所以差距**不来自 ray tracing**，而来自 **rayweq 之后那段「怎么把剂量铺到 grid 上」的架构差异**。

代码坐标：
- carbonPBS：`tps/tps_service_25B/carbonPBS/{cudaCalDose.cpp, deviceCalDose.cu}`
- subSecond：`subSecond/src/core/raytracedicom_wrapper.cu` + `src/algorithms/{convolution,superposition,complete_superposition,prim_transf_kernel}.cu`

---

## 1. 两套架构本质对比

### carbonPBS：单 kernel，ROI-直接沉积（稀疏）
`cudaFinalDose`（cudaCalDose.cpp:1322-1522）每个 layer：
- 建几张 texture（rayweq/idd/profile/subspot），**全是查表，无 CT 重建**；
- 一个 kernel `calFinalDoseKernel`（deviceCalDose.cu:477-585）；
- 线程网格 = **`nBeam × nROI`**（gtx=spot, gty=ROI 体素）；
- 每个 (spot, ROI体素) 线程：查 WEQ 深度 + 解析高斯权重 → **直接 `atomicAdd` 到 ROI 体素**（deviceCalDose.cu:577，`absId` 由 `roiIndex` 算）。
- **没有 BEV 中间体、没有 superposition、没有 scatter-back、没有坐标回变换。** 全程在 spot→ROI 体素这一跳里完成。

代价 ≈ `O(nBeam × nROI)`。274mm grid 的 ROI 通常远稀疏于整盒（几万体素 vs 整盒上千万）。

### subSecond/RTD：多级 BEV 流水线（稠密中间体）
`subsecondWrapper`（raytracedicom_wrapper.cu:4015+）每个 beam call，每个 layer 要走：
1. **spot→ray 可分离 2D 卷积**：`xConvGathResampGpu` + `yConvGathResampGpu`（convolution.cu:501-511）；
2. **superposition**（IDD ⊗ 高斯 → BEV 剂量体）：`tileRadCalcDoseGated` + `kernelSuperposition<R>`，**按半径桶 R=0..32 分多次 launch**（complete_superposition.cu:544,587）——单 layer **~几十次 kernel launch**；
3. **建 BEV 剂量 3D texture**（raytracedicom_wrapper.cu:7258，**每 layer 都新建/绑定一次**）；
4. **scatter-back**：`primTransfDiv`（prim_transf_kernel.cu:11）把 BEV fan 通过纹理采样 + 坐标回变换累加进 dose grid 的投影包围盒（raytracedicom_wrapper.cu:7356-7463）。

代价 = 卷积 + superposition（BEV fan 体）+ **scatter-back（投影包围盒 × 深度，最坏接近 `O(274² × depth)`）** + 每 layer 若干次纹理创建和数十次 kernel 启动开销。

---

## 2. 慢在哪：按影响排序

### (A) 架构性：BEV 流水线天然比 ROI-直接重 ★最根本
RTD 的「优势」superposition 是把**横向散射写成 beam 空间里的卷积**——物理上优雅、对大野/稠密野摊薄得好。但它要付两笔 carbonPBS 完全不付的钱：
- **构建并卷积一个稠密 BEV 中间体**（fan × depth 的完整体），即使最终只有稀疏 ROI 要剂量；
- **再把 BEV 体 scatter-back 回 dose grid**（`primTransfDiv`，一次额外的坐标变换 + 纹理采样写回）。

carbonPBS 直接 spot→ROI 体素一跳到位，**没有中间体、没有回变换**。当 ROI 相对整盒稀疏时（274mm grid 正是如此），`nBeam×nROI` 的工作量小于「建满 BEV + scatter 回投影盒」。**superposition 只有在野足够大/稠密、且 BEV 能被大量体素复用时才回本；这里回本不了，回变换 + 每层纹理重建把红利吃掉了。**

### (B) 验证属实：WEQ 模式下仍无条件上传 274³ CT texture（纯浪费）★
`raytracedicom_wrapper.cu:4092` 的 `create3DTexture(ctForTexture, ctDims=274³)` 在 `hasWeqVolume`（:4839）判断**之前**、**无条件执行**。
- WEQ 模式下喂进来的 CT 是 pybind 伪造的 ROI mask（274³ ≈ 6.7M 体素 × 4B ≈ **~26MB**，若按 full padded 更大），上传 + 纹理绑定后，`fillBevFromWeqVolumeKernel` **根本不读它**。
- 即纯属每个 beam call 白扔一次 274³ 纹理上传 + 可能的 `convertCtToHUPlus1000` 全体扫描（:4081-4090）。carbonPBS 无此项。

### (C) 每层固定开销：数十次 kernel launch + 每层纹理创建
- superposition 的半径桶 R=0..32 多次 launch（complete_superposition.cu:544）：单层就有 ~33 次内核启动，纯启动延迟 × layer 数累积。
- 每层 `create3DTexture(devBevPrimDose, BEV size)`（:7258）：纹理 array 分配/绑定的固定开销 × layer。
- carbonPBS 单层只有 **1 个** `calFinalDoseKernel`。
- launch/纹理这类固定开销与 ROI 稀疏度无关，是 RTD 的「常数项」劣势。

### (D) scatter-back 包围盒可能逼近全 grid
`primTransfDiv` 覆盖的是 BEV 投影到 dose grid 的包围盒 `startIdx→maxIdx`（:7339-7367），`transferBoxVoxels = Nx·Ny·Nz`。野一大/一斜，这个盒接近 `274² × depth`，是稠密遍历；carbonPBS 那边永远只碰 ROI 列表里的体素。

### 不是原因（已排除）
- **per-layer malloc/free**：RTD 大缓冲在 beam 循环外一次性分配、层间只 `cudaMemset` 复用（:5542-5583），不是每层重分配。✗
- **full-grid D2H 回拷**：两边都只在 call 结束回拷一次整盒，属打平项；RTD 那些 274³ 的 `copyToHost` 都 gated 在 `transferAudit`/`fineTiming` 下，verbose=0 不触发。✗
- **CT/LUT 纹理 per-layer 重建**：CT+4 张 LUT 纹理是 per beam-call 建一次（:4092-4132），非每层。但见 (B)，在 WEQ 模式下这次 CT 上传本身就是浪费。

---

## 3. 直接回答你的疑问

> 「RTD 原型改编过来的优势不是 BEV 里的 superposition 吗，为什么反而耗时？」

superposition 的优势是**算法表达**（横向散射 = beam 空间卷积，对大野摊销好、精度可控），**不是绝对速度**。它的速度优势要满足两个条件才兑现：
1. 野足够大/稠密，BEV 中间体被足够多 ROI 体素复用；
2. BEV→dose grid 的 scatter-back 和每层纹理/卷积固定开销，相对计算量可忽略。

在 274mm grid + 稀疏 ROI 的这组计划里，两个条件都不满足：ROI 稀疏 → carbonPBS 的 `O(nBeam×nROI)` 本就小；而 RTD 仍要付「建稠密 BEV + ~33 次 superposition launch + 每层建 BEV 纹理 + scatter-back 回变换 + 白扔一次 274³ CT 纹理」的全套固定成本。于是优雅的架构反而更慢。

---

## 4. 如果要追平 carbonPBS（优化方向，按性价比）

1. **(B) 立刻可做**：WEQ 模式下跳过 `create3DTexture(CT)` 与 `convertCtToHUPlus1000`——在 :4092 前加 `if (!willUseWeqVolume)` 守卫。零数值风险，省每 beam 一次 274³ 上传。
2. **(C)**：减少 superposition 的半径桶 launch 数（合并空桶 / 单 kernel 内分支），并把每层 BEV 剂量纹理改成线性内存采样或纹理对象复用。
3. **(D)**：scatter-back 限制在 ROI 投影包围盒（而非 BEV 全投影盒），或对稀疏 ROI 回退到「按 ROI 体素 gather」而非「按包围盒 scatter」。
4. **(A) 结构性**：对稀疏 ROI 计划提供一条「ROI-direct」快路（类似 carbonPBS），野大时才切 BEV superposition——two-path 按 `nROI/gridVolume` 自动选择。

注：本组数据是 WEQ 模式。之前讨论的「改成 CT 直追」会再叠加一次真 CT 的 `rayTracingBEVKernel`，**不会让它更快**，只会把 (B) 那次纹理上传变成有用功——性能优化和 CT-直追是两件独立的事，别混在一起做。

---

# 追加分析（2026-06-15）：三个具体问题

## Q1. 能否「只用注释」关掉 CT 上传，方便以后复用？

**不能只注释那一行，但可以用一个 `if` 守卫做到「关掉 + 易复用」。** 原因是 `imVolTex` 不是死代码，它在 CT-直追路径里是被读的。

### 现状依赖关系
- `imVolTex = create3DTexture(ctForTexture, ctDims, ...)`：`raytracedicom_wrapper.cu:4092`
- 谁读它：**只有** `rayTracingBEVKernel`（:5221，把 `imVolTex, devCtLinear` 当参数）。
- WEQ 模式走的是 `fillBevFromWeqVolumeKernel`（:5205），**不碰 `imVolTex`**。
- 销毁：:7839 `destroyTextureObjectAndArray(imVolTex)`；:4107/4116/4126/4137/4158 是各 LUT 纹理创建失败时的清理分支里也会 destroy 它。

### 如果直接注释 :4092
会炸/出错：
1. :4094 `if (imVolTex == 0)` 会判 fallback，进而 `devCtLinear` 上传（:4098-4099）——反而又上传一遍 CT 线性内存，**没省到，还改了语义**；
2. :4107 等失败清理分支 `destroyTextureObjectAndArray(imVolTex)` 传未初始化变量；
3. 若哪天切回 CT-直追，:5221 拿到的是 0 纹理 → 静默零剂量。

所以「注释一行」既省不了（fallback 会补上传），又埋雷。

### 正确做法（仍然很轻，且天然可复用）
在 :4092 前先确定本次是否真要 CT，再守卫纹理创建与 fallback：
```cpp
// 本次是否有任何 beam 会走 CT 直追（= 没有 WEQ body）
bool needCtTexture = false;
for (size_t b = 0; b < numBeams; ++b) {
    const auto& wh = getActiveWeqHeader(beamSettings[b]);
    const bool hasWeqVol = (wh.size() >= 9) && (beamSettings[b].waterEquivalence.size() > 9);
    if (!hasWeqVol) { needCtTexture = true; break; }
}

cudaTextureObject_t imVolTex = 0;
float* devCtLinear = nullptr;
if (needCtTexture) {
    // 原 :4073-4100 的 CT 类型识别 + 转换 + create3DTexture + fallback 整段搬进来
    ...
}
```
然后把 :7839/:4107… 的 destroy 改成 `if (imVolTex) destroy...`（本就有 `imVolTex==0` 语义，安全）。

- **复用性**：未来开 CT-直追时，`needCtTexture` 自动变 true，纹理回来，无需再改代码——比"注释/反注释"更省心。
- **收益**：WEQ 模式每个 beam call 省掉 1 次 274³ 纹理上传 + 可能的 `convertCtToHUPlus1000` 全体扫描（:4081-4090，密度→HU+1000 的整盒遍历），**零数值风险**（WEQ kernel 本来就不读 CT）。
- **改动量**：~15 行，集中在 :4069-4100 与几处 destroy 守卫。

> 结论：Q1 = 不能纯注释（会触发 fallback 反而上传 + 未初始化销毁），但用一个编译期不变的 `if (needCtTexture)` 守卫即可，且比注释更利于以后复用。

---

## Q2. 每层 33 次 superposition launch，能否用 CUDA 12.1 并行/压缩？

**能压，而且有三档力度。当前实现把它们全串行在同一个 stream 上，还夹了一次 host 往返同步，这才是固定开销的大头。**

### 现状（complete_superposition.cu）
- 33 次 `LAUNCH_SUPERP_KERNEL(0..32)`（:764+），宏展开后全部 `<<<..., 0, stream>>>` 用**同一个 stream**（:544）→ 彼此串行。
- 每次只有 `batchedTileRadCtrs[R] > 0` 才真正 launch（:543），且已有 `MIN_TILES_IN_BATCH` 的桶合并（:742-750）——所以**实际 launch 数通常远小于 33**（典型几条，看 `layerMaxSuperpR`）。真正 33 的只是源码里的宏展开条数。
- **真正的串行瓶颈在前面**：:587 `tileRadCalcDoseGated` 之后 :589 `cudaStreamSynchronize` + :592-597 把 `tileRadCtrs` D2H 拷回 host，host 上算 `batchedTileRadCtrs`，再决定 launch 哪些桶。**这是一次强制的 device→host→device 往返**，每层一次，把 GPU 流水线打断。

### 可用的 CUDA 12.x 手段（按性价比）
1. **多 stream 并发（最简单）**：不同半径桶的 `kernelSuperposition<R>` 都写 `devBevDose`（:545 末参），但它们处理的是**不同 tile 集合**（`devInOutIdcs` 分桶），输出 voxel 不重叠 → 可以安全放到 N 个 stream 并发。把 `stream` 换成 round-robin 的 `stream[r % K]`，末尾 event 汇合。小 launch 之间的尾延迟可被填掉。**改动小，收益中。**

2. **CUDA Graph 捕获（最契合这里）**：每层的 kernel 序列（tileRadCalc → 若干 superposition 桶 → padding/transfer）**拓扑固定、只是参数和 launch 个数变**。用 `cudaGraph` 捕获一次、后续层 `cudaGraphLaunch` 重放，可把「每层数十次 launch 的 CPU 启动延迟」摊销到近零。难点：launch 个数依赖 `batchedTileRadCtrs`（运行时才知），需要么 (a) 固定捕获全部 33 桶、空桶用 0-grid（CUDA 允许 0-block launch 直接返回），换取图可复用；么 (b) 用 conditional graph nodes（CUDA 12.4+）。**改动中，收益高**——尤其层数多时。

3. **干掉那次 host 往返（最高价值的单点）**：把 `batchedTileRadCtrs` 的归约（:707-750）做成一个**device 端 kernel**，直接产出每桶的 grid 配置写进 device 内存，再用 **device-side launch / `cudaLaunchKernelEx` + 已知上界**，或干脆把 superposition 改成「单 kernel 内按桶 grid-stride」消除分桶 launch。这样 :589/:597 两次 `cudaStreamSynchronize` 就能去掉，GPU 不再每层停顿等 host。**改动较大，收益最高**，因为同步停顿往往比 launch 延迟更伤。

4. **`tileRadCalcDoseGated` 本身**：它已是「跳过零剂量 tile」的优化版（commit 14ce2d4，见 MEMORY）。这块不用动。

### 建议落地顺序
先做 (1) 多 stream（半天工作量，验证并发是否真填得满）→ 再评估 (3) 去 host 往返（收益最大）→ 层数多且形状稳定时上 (2) CUDA Graph。

> 注意：33 这个数字本身不是问题（空桶不 launch），**问题是同一 stream 串行 + 每层一次 host 同步往返**。CUDA 12.1 的 stream/graph 正是冲着这两点。

---

## Q3. subSecond 完全没用到 ROI 吗？carbonPBS 在哪用 roiIdx？

### subSecond：ROI 传进来了，但「几乎只用于 debug 打印」，**不参与任何计算裁剪**
- pybind 把 `roiIdx` 转成 `roiLinearIndices` 存进 `beamSettings`（raytracedicom_pybind.cpp:1617 `to_roi_linear_indices`）。
- wrapper 里唯一的实质使用：:4517 `computeDoseRoiBounds(beam.roiLinearIndices, ...)` 算出 `roiMinIdx/roiMaxIdx`。
- 但这个结果**只在 :4757-4760 的 `if (hasDoseRoi)` 里被打印**（fineTiming 诊断），`grep roiMinIdx/roiMaxIdx` 全文件只有 4515-4517 定义 + 4757-4760 打印两处。
- 注释 :4519-4521 自己写明："ROI bounds are used elsewhere for dose masking, but cannot restrict the CPB grid"——但实际上 `prim_transf_kernel.cu` 里**搜不到任何 roi/ROI**（scatter-back 不读 ROI mask），所谓 "dose masking" 在当前 BEV 路径并没有落地。
- 结论：**subSecond 的 BEV 流水线计算量与 ROI 稀疏度完全无关**。CPB/ray 网格覆盖整个 dose volume 投影（:4519-4522 强制 full volume），superposition 铺满 BEV fan，scatter-back 铺满投影包围盒——ROI 再小也不省。**这正是 §2(A) 它比 carbonPBS 慢的根因之一。**

### carbonPBS：roiIdx 是 kernel 的**主循环维度**，直接决定工作量
`deviceCalDose.cu`，dose 沉积 kernel `calFinalDoseKernel`（:477-585）：
- 线程映射（:514-528）：`gtx = spot 索引`（`while gtx < num_beam`），`gty = ROI 体素索引`（`while gty < num_roi`）——**整个线程网格 = `num_beam × num_roi`**。
- 每个 (spot, ROI体素) 线程：
  - :50-52 由 `roiIndex[gty]` 直接算出该 ROI 体素的物理坐标 `pos`；
  - :54/91 `absId = roiIndex[gty].x*dims.y*dims.z + ...` 把剂量写回**该 ROI 体素的线性地址**；
  - :577（RBE 变体 :112-115）`atomicAdd(dose + absId, ...)`。
- 即 **carbonPBS 只在 ROI 列表里的体素上做功并写结果**，整盒里 ROI 之外的体素一次都不碰。
- 另一处同构使用：:159-166（另一个 kernel 的 `voxelIndex < num_roi` grid-stride）、:281-283、:410-412 也都是 `roiIndex[...]` 取 ROI 体素坐标。

### 一句话对比
| | ROI 是否进 kernel 主循环 | 工作量随 ROI 稀疏度 |
|---|---|---|
| carbonPBS | 是（`gty < num_roi`，写 `roiIndex[gty]` 地址） | **线性缩减**（ROI 越稀疏越快） |
| subSecond | 否（只算个 bounds 然后打印） | **与 ROI 无关**（恒等于 full-volume BEV 流水线） |

> 这就是为什么 274mm grid + 稀疏 ROI 上 carbonPBS 占大便宜：它的成本 ∝ `nBeam×nROI`，而 subSecond 的成本 ∝ BEV fan 体 + 投影包围盒，**无视 ROI**。若要让 subSecond 在稀疏 ROI 上追平，最彻底的就是 §4(A) 的「ROI-direct 快路」或 §4(D) 的「scatter-back 限制到 ROI」。
