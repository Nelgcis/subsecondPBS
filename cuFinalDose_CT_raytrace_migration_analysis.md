# cuFinalDose 改造分析：从「传入 WEQ」改为「像 RTD 一样从 CT 直接 ray tracing」

日期：2026-06-15
分析对象：
- TPS 侧：`~/CASHIM_HL/tps/tps_service_25B/content/dose_operator/{dosecal.py, multiBeamGroups.py}`
- 计算引擎侧：`~/CASHIM_HL/subsecond/raytracedicom_pybind_stage/subSecond/src/{bindings/raytracedicom_pybind.cpp, core/raytracedicom_wrapper.cu}`
- 已编译产物：`tps_service_25B/content/tools/cudaPKG/cudaCalDoseRTD.cpython-36m-...so`（由上面 subSecond 源码编译而来）

---

## 0. 结论速览（TL;DR）

**好改，而且改动量比想象的小。** 因为 RTD 引擎里**「从 CT 自己 ray trace」的整条路径其实已经存在并且是默认实现**（`rayTracingBEVKernel`，吃 CT 3D texture），现在的 WEQ 路径（`fillBevFromWeqVolumeKernel`）反而是叠加在上面的一个"用预算好的 WEQ 替换 CT 追踪"的分支。

当前之所以走 WEQ，是因为：
1. Python 端 `cuFinalDose(...)` 把 `rayweq`（`cuPrepareWEQ` 的产物）塞进 `beam.water_equivalence`；
2. pybind `cu_final_dose_py` 根本**不接收真实 CT**，而是用 ROI mask 伪造一个 dummy CT（`make_ct_from_roi_mask`，全是 0/1000）；
3. wrapper 里 `hasWeqVolume == true` → 走 `fillBevFromWeqVolumeKernel`，dummy CT 被忽略。

要改成 RTD 原生 CT 路径，本质是三件事：
- **(A) 把真实 CT 体数据传进去**（`self.doseGrid.data`，已经在 `dosecal.py` 里 resample 好了）；
- **(B) 不再传 / 传空 `water_equivalence`**，让 `hasWeqVolume == false`，从而落到 `rayTracingBEVKernel`；
- **(C) 处理几何来源**：现在 ray 网格的 corner/resolution/steps/起始平面在 WEQ 路径里来自 `weqHeader`（rayweq[0:9]），去掉 WEQ 后需要改走 raySpacing + CT extent 推导（wrapper 里这套 fallback 已写好，但需要 `beam.raySpacing` 有效）。

风险点集中在 **(C) 几何一致性** 和 **CT 单位/类型识别**，不在算法内核本身（内核是同一套 IDD/σ/卷积）。

---

## 1. 当前数据流（WEQ 模式）逐层拆解

### 1.1 Python: `dosecal.py :: caldose_raytrace_all()`（calType in {Dose,QA,Scale,DoseRecalculation}）

关键行：
- `dosecal.py:1573-1594` 构造 `rayweq`（shape `9 + nInterpX*nInterpY*nMaxStep`，`nMaxStep=10000`），调用 `cuPrepareWEQ(...)` 在 GPU 上把 `self.doseGrid.data`（已 resample 的 CT 密度/SPR 体）沿每条 interp 射线累积成 WEQ 体，并把 9 元 header 写进 `rayweq[0:9]`（`-ylim,1,2ylim+1,-xlim,1,2xlim+1` + 投影起点等）。
- `dosecal.py:1726` `beam["water_equivalence"] = rayweq`（注意：这一支其实没直接用到，真正传参是位置参数）。
- `dosecal.py:1814-1844` 调用 `cuFinalDose(finalDose, rayweq, roiIdx, all_energies, sourcePos, tmpBeamDir, bmxdir, bmydir, corner, resolution, dims, longitudalCutoff, enelist, idddata, ..., idbeamxy, numParticlesPerBeam, sad, 0.00005, beamParaPos, 0, spotSpacingX=, spotSpacingZ=)`。
  - **注意：这里第二个位置参数就是 `rayweq`，没有传 CT。**

`self.doseGrid.data` 来源：`dosecal.py:142 self.doseGrid.data, _ = resample(self.doseGrid, ct)` —— 已经是按 dose grid 几何重采样好的 CT（密度/SPR 加权），即 RTD 想要的体数据已经在手里。

### 1.2 pybind: `cudaCalDoseRTD.cuFinalDose` → `cu_final_dose_py`（`raytracedicom_pybind.cpp:2388-2472`）

- 入参签名里**没有 ct**（见 `:2388-2421`）。
- `:2430-2433`：用 `roiIdx` 反推 `roi_linear`，再 `make_ct_from_roi_mask(dims, roi_linear)` —— **伪造一个 CT**：ROI 体素填 1000，其余填 0。`make_dummy_ct_from_dims`（全 1000）是 ROI 为空时的退化。
  - `raytracedicom_pybind.cpp:2095-2117`，注释明说 "Compatibility-only placeholder. The wrapper consumes WEQ for transport when available."
- 转调 `run_carbonpbs_final_dose(finalDose, ct_data(伪造), rayweq, ...)`。

### 1.3 pybind: `run_carbonpbs_final_dose`（`:1924-2093`）

- `:1986-1987` 把伪造 CT 转 float32 contiguous。
- `:1993` `build_carbonpbs_context(rayweq, ...)`：
  - `:1585` `ctx.beamSettings.waterEquivalence = flatten(rayweq)`；
  - `:1594-1596` `rayWeqHeader = waterEquivalence[0:9]`；
  - `:1606` `raySpacing = (header[7], header[4])`（**ray 间距来自 WEQ header！**）；
  - `:1612` `steps = round(header[2])`；
  - `:1639-1640` `spotPositions = idbeamxy`，`spotPositionsAreIndices = true`。
- `:2071-2086` 调 `subsecondWrapper(伪造CT, dims, res, corner, dose..., &ctx.beamSettings, 1, &ctx.energyData, gpuId, nuclear_correction, verbose)`。

### 1.4 引擎: `subsecondWrapper`（`raytracedicom_wrapper.cu:4015+`）

- `:4069-4100` **无论如何都会**把传入 CT 建成 3D texture `imVolTex`（现在喂的是伪造 CT，被浪费）。
- `:4570-4571` `weqHeader = getActiveWeqHeader(beam)`；`hasWeqHeader = size>=9`（当前 true）。
- 几何分支（决定 `cpbCorner/cpbResolution/rayDims/cpbDims`）：
  - `:4578` 若 `hasRawSpotLattice && raySpacing>0` → 用 raw spot lattice 对齐；
  - `:4637` else if `hasWeqHeader` → 用 `weqHeader[3..8]` 当 ray 网格几何（**当前走这里**）；
  - `:4710` else legacy CPB（需要显式 `raySpacing`）。
- `:4839` `hasWeqVolume = hasWeqHeader && waterEquivalence.size() > 9`（当前 true，body 非空）。
- `:4840` `stepLength_mm`：WEQ 用 `weqHeader[1]`，否则用 `ctResolution.z`。
- `:4850-4895` WEQ 模式下把 BEV 起始平面 align 到 `weqHeader[0]`。
- `:4896` `tracerSteps`：WEQ 用 `weqHeader[2]`，否则用 CT extent `ceil((maxGZ-minGZ)/step)`。
- **核心分叉 `:5095-5225`：**
  - `if (hasWeqVolume)` → `fillBevFromWeqVolumeKernel`（`:5205`）：从上传的 WEQ body 采样填 BEV 密度/累积 SP。**CT texture 完全没用。**
  - `else` → `rayTracingBEVKernel`（`:5217`）：**吃 `imVolTex`（CT 3D texture）+ densityTex/stoppingPowerTex/cumulIddTex/rRadiationLengthTex 真正沿 BEV 射线追踪 CT。这就是 RTD 原生路径。**
- 之后的 σ/IDD/卷积/superposition 流程两条路完全共用。

> **关键洞察**：要"像 RTD 一样从 CT 追踪"，工程上等价于"让 `hasWeqVolume` 为 false 并喂真 CT"，剩下的内核已经现成。

---

## 2. 改造方案

有两种实现层级，建议 **方案 B（引擎内开关，最干净）**，方案 A 可作为快速验证。

### 方案 A（最小改动，纯 Python 侧验证）— 不推荐长期用
利用引擎里已有的 `raytracedicom_wrapper`（`:2138`，**它本来就吃真 CT + beam dict**）这个 entry，绕开 `cuFinalDose`。
- 在 `dosecal.py` 里把 `cuFinalDose(...)` 换成 `raytracedicom_wrapper(ct=self.doseGrid.data, ct_dims, ct_res, ct_corner, dose_dims, dose_res, dose_corner, beam=dict(...), energy=dict(...), ...)`。
- 问题：`raytracedicom_wrapper_py` 的 beam dict 仍可能带 `water_equivalence`（`:2248`），且需要手工组装 energy dict / beam dict（spotPositions 单位、raySpacing、subspot LUT 布局都得对齐）。组装成本高、易错。
- 结论：能验证 CT 路径数值是否正确，但不适合作为正式接口。

### 方案 B（推荐）— 给 `cuFinalDose` 增加「CT 直追」开关 + 传真 CT
**目标**：保持 `cuFinalDose(...)` 这个 TPS 调用契约，新增一个 `ct_data`（和/或 `use_ct_raytrace=True`），当启用时：
1. pybind 用真 CT 替换 `make_ct_from_roi_mask`；
2. **不**把 `rayweq` 的 body 放进 `waterEquivalence`（只保留 9 元 header 或干脆传 `None`，让几何走 raySpacing 分支）；
3. wrapper 端 `hasWeqVolume` 自然为 false → 落 `rayTracingBEVKernel`。

需要决定 ray 网格几何来源（见 §3 风险）。

---

## 3. 逐文件 / 逐函数改动清单

### 3.1 引擎侧 `src/bindings/raytracedicom_pybind.cpp`

| 位置 | 现状 | 需要的改动 |
|---|---|---|
| `cu_final_dose_py` 签名 `:2388-2421` | 无 `ct_data`，有 `rayweq` | 新增 `py::array ct_data`（建议放在 `rayweq` 之后，或新增 `py::object ct_data = py::none()`）+ 新增 `bool use_ct_raytrace = false`（或用 `ct_data is not None` 隐式判定）。 |
| `:2430-2433` 伪造 CT | `make_ct_from_roi_mask` | 当 `use_ct_raytrace`：改用传入的真实 `ct_data`（校验 size==dims 乘积、float32）。否则保持伪造（向后兼容）。 |
| `run_carbonpbs_final_dose` 签名 `:1924-1957` | 已有 `ct_data` 形参（!） | 已经接收 `ct_data`，无需改签名，只需让上游把真 CT 传进来。 |
| `build_carbonpbs_context` `:1509,1585-1612` | 无条件 `waterEquivalence = flatten(rayweq)` | 当 CT 直追：**只取 `rayweq[0:9]` 作为 `rayWeqHeader`，但把 `waterEquivalence` 截断到长度 9（或清空 body）**，使 wrapper 端 `waterEquivalence.size() > 9` 为 false → `hasWeqVolume=false`。`raySpacing` 仍可来自 header[7],header[4]（保留几何），或改由新参数传入。 |
| `m.def("cuFinalDose",...)` `:2804-2846` + `"calcDose"` `:2652` + `"cuCalDose"` `:2754` | 三个 alias 都绑到 `cu_final_dose_py` | 三处都要补 `py::arg("ct_data")=py::none()`、`py::arg("use_ct_raytrace")=false`，保持默认值以向后兼容旧调用。 |

> 注意：`run_carbonpbs_final_dose` 已经有 `ct_data` 形参且 `:2059-2062` 已校验 `ct size == dims`，说明这条接 CT 的"管子"本就预留好了，只是 `cu_final_dose_py` 一直用伪造 CT 填它。这进一步说明**改动是顺着既有设计走，不是逆向改架构**。

### 3.2 引擎侧 `src/core/raytracedicom_wrapper.cu`

理想情况下**无需改 kernel**，但需确认/微调几何分支：

| 位置 | 关注点 | 可能的改动 |
|---|---|---|
| `:4570-4571` `hasWeqHeader` | 去掉 body 后 header 仍在（size==9），`hasWeqHeader` 仍 true | 行为 OK：仍可用 header 做几何，但 §4 要确认 align 逻辑。若想完全脱离 WEQ header，需让几何走 `:4578` raw-spot-lattice 分支（要求 `hasRawSpotLattice && raySpacing>0`）。 |
| `:4578 / 4637 / 4710` 几何三分支 | CT 直追时希望走 raw-spot-lattice 或 CT-extent | 确保 `beam.raySpacing>0` 且 `buildRawSpotLattice` 成功（依赖 `idbeamxy`+spotDelta）。若都不满足会落 `:4710` legacy 并**抛错**（需要显式 raySpacing）。 |
| `:4839` `hasWeqVolume` | `= hasWeqHeader && size>9` | body 清空后自动 false ✅ —— 这是触发 CT 路径的关键，无需改代码，只要上游不塞 body。 |
| `:4896-4898` `tracerSteps` | 非 WEQ 时 `= ceil((maxGZ-minGZ)/step)+1` 来自 CT extent | 确认 `stepLength_mm` 用 `ctResolution.z`（`:4840` else 分支）合理；步数会比 WEQ 的 header[2] 大，影响显存/耗时。 |
| `:5216-5225` `rayTracingBEVKernel` | 吃 CT texture | 无需改，但要确认 `imVolTex` 是真 CT（依赖 3.1 传真 CT）。 |
| `:4069-4090` CT 单位识别 `parseCtInputTypeEnv/autoDetectCtType` | 真 CT 会被 auto-detect 成 HU/Density/SPR | **重要**：`self.doseGrid.data` 是 SPR/密度加权体（值 ~[0,3]），auto-detect `:3902` 会判成 `Density`，然后 `convertCtToHUPlus1000` 转换。需确认转换链与 cuPrepareWEQ 当年用的密度→WEPL 映射一致，否则数值会偏。可用环境变量 `RTD_CT_INPUT_TYPE=DENSITY`（或 SPR）强制。 |

### 3.3 TPS 侧 `content/dose_operator/dosecal.py`

| 位置 | 现状 | 改动 |
|---|---|---|
| `:1579-1594` `cuPrepareWEQ(...)` | 计算整条 WEQ 体（耗时项之一，见 perf_logger） | CT 直追模式下**可整段跳过**（省掉一次 GPU 全体 ray trace）。但 §4 注意：当前 align/几何依赖 `rayweq[0:9]` header；若引擎几何改走 raySpacing，可只保留构造 header 的轻量代码或彻底删除。 |
| `:1814-1844` `cuFinalDose(finalDose, rayweq, ...)` | 传 rayweq | 增加 `ct_data=self.doseGrid.data.astype(np.float32).flatten(order=?)` 与 `use_ct_raytrace=True`；`rayweq` 可传仅含 9 元 header 的轻量数组或 `None`。需确认 CT 展平顺序（order）与 `dims`/`corner`/`resolution` 约定一致（dosecal 其他地方用 `order="F"`，但 pybind 把 CT 当 flat buffer + C-style forcecast，**展平顺序必须和 wrapper 内 `idx = x + y*dimx + z*dimx*dimy` 的约定对齐**——这是头号易错点）。 |
| `:1762,1852,1904,1983` 计时 CSV | 记 cuFinalDose 耗时 | 可加一列标记 ct_raytrace vs weq，便于对比性能（本来项目目标就是 sub-second）。 |
| import `:52-54` | `from ...cudaCalDoseRTD import cuFinalDose` | 不变（接口名保留）。需重编译 .so 后接口才有新参数。 |

### 3.4 TPS 侧 `content/dose_operator/multiBeamGroups.py`
- `:83 self.dc = DOSECAL(...)`，`:561/:756` 用到 `self.dc.doseGrid.data`（SPR 体）。
- **基本无需改**：它只是 DOSECAL 的上层编排，CT 体已经在 `self.dc.doseGrid.data`。若想从 multiBeam 层下发"是否 CT 直追"的开关，可加一个配置位透传给 `caldose_raytrace_all`。
- 唯一要确认：`beam_group['patient_pixels_resample'] = self.dc.doseGrid.data`（`:756`）与传给 cuFinalDose 的 CT 是否同一份、同一展平约定。

### 3.5 其他受影响调用点（需回归，不一定改）
- `pencilBeamScanning.py:34 from ...cudaCalDose1 import cuFinalDose` —— **注意这是另一个 `cuFinalDose`（来自 `cudaCalDose1`，老 CarbonPBS），不是 RTD 的那个**。改 RTD 接口不影响它，但要确认运行时实际走哪条。
- `rbecal.py:682 cuFinalDoseAndRBEMap(...)` 和 `dosecal.py:1758 dose_type=="bio"` 分支：bio 剂量由 RBE 路径算，当前**直接 return 不调 cuFinalDose**（`:1773`）。CT 直追若也要覆盖 bio 路径，需同步改 `rbecal.py`（`cuFinalDoseAndRBEMap` 来自 `cudaCalDose1`，又是另一套 .so）。**先只做 physical dose（dose_type!="bio"）路径，范围可控。**
- `dose_optimizing.py:176 from ...newcudaPKG.cudaCalDose import cuPrepareWEQ` —— 优化器也用 WEQ；本次只动 final dose，不要动优化路径（`cuCalDose3`/`cuCalDoseNorm` 仍吃 rayweq，见 `dosecal.py:1874,1931`）。

---

## 4. 主要风险与必须验证项

1. **几何对齐（最高风险）**：当前 ray 网格 corner/res/steps/起始平面全部由 `rayweq[0:9]` header 提供（`wrapper.cu:4637-4707, 4850-4898`）。去掉 WEQ body 但保留 header 最省事；若连 header 也去掉，必须保证 `hasRawSpotLattice`（`buildRawSpotLattice` 依赖 `idbeamxy`+`spotDelta`）或显式 `raySpacing` 成立，否则 `:4710` legacy 分支会抛 "Legacy CPB path requires explicit beam.raySpacing"。**建议第一步：保留 9 元 header，仅清空 body**，几何不变、只把"密度来源"从 WEQ 切到 CT texture，变量最少。

2. **CT 单位/类型**：`doseGrid.data` 是密度/SPR 加权体（~[0,3]），auto-detect 会判 `Density` 并做 `convertCtToHUPlus1000`。必须验证此转换 + 引擎内 `densityTex/stoppingPowerTex` 查表，与原 `cuPrepareWEQ` 的 WEPL 累积在数值上一致（同一套 density→SP 映射）。用 `RTD_CT_INPUT_TYPE` 显式锁定，避免 auto 抖动。

3. **CT 展平内存序**：Python 的 `doseGrid.data`（numpy，可能 F-order）→ pybind `c_style|forcecast` → wrapper 线性索引。`dims/corner/resolution` 必须与 CT buffer 的轴序严格对应。**强烈建议写一个 CT round-trip 校验**（在引擎里按已知体素取值打印）。

4. **性能（本来的目标）**：CT 直追的 `tracerSteps` 来自 CT extent（`ceil((maxGZ-minGZ)/ctRes.z)`），通常 > WEQ header[2]，BEV 体更大 → 可能更慢。但省掉了 `cuPrepareWEQ` 全体追踪。需对 sub-second 目标重新测时（参考 MEMORY: cuFinalDose 现 ~23s，根因未定）。CT 直追同时也是**排查 23s 回归的诊断手段**：如果 CT 路径明显更快，说明慢在 WEQ 相关环节。

5. **bio/RBE 路径未覆盖**：`dose_type=="bio"` 当前不走 cuFinalDose。首版只做 physical dose，bio 留待 `rbecal.py` 单独评估。

6. **三个接口 alias**：`calcDose`/`cuCalDose`/`cuFinalDose` 都绑同一函数，新增参数必须都给默认值，确保旧 .pyc/旧调用不报 "unexpected keyword"。

---

## 5. 建议的落地顺序（增量、可回退）

1. **引擎**：`cu_final_dose_py` 增加 `ct_data=None, use_ct_raytrace=False` 两个带默认值的参数；`use_ct_raytrace` 时用真 CT，且在 `build_carbonpbs_context` 里把 `waterEquivalence` 截到 9（保留 header，清 body）。三个 `m.def` 同步加 arg。重编译 .so。
2. **数值校验**：先用一个已知 plan，分别跑 (a) 旧 WEQ 路径 (b) 新 CT 直追，比较 finalDose（用 `RTD_CT_INPUT_TYPE=DENSITY` 锁单位）。先不删 `cuPrepareWEQ`。
3. **dosecal.py**：仅在 physical dose 分支加 `ct_data=self.doseGrid.data(对齐展平), use_ct_raytrace=cfg`，开关默认关。验证通过后再考虑跳过 `cuPrepareWEQ`。
4. **性能对比**：记 CSV，确认 CT 直追相对 WEQ 的时间，决定是否作为默认。
5. （可选）几何彻底脱离 WEQ header：改走 raw-spot-lattice / raySpacing，删 `cuPrepareWEQ`。这步风险最高，放最后。

---

## 6. 关键代码坐标速查

- WEQ 构造：`dosecal.py:1573-1594`
- cuFinalDose 调用：`dosecal.py:1814-1844`
- pybind 入口（无 CT）：`raytracedicom_pybind.cpp:2388-2472`（`cu_final_dose_py`）
- 伪造 CT：`raytracedicom_pybind.cpp:2095-2117, 2430-2433`
- context 构造（WEQ→beamSettings）：`raytracedicom_pybind.cpp:1509-1612`
- run_carbonpbs_final_dose（已有 ct_data 形参）：`raytracedicom_pybind.cpp:1924-2093`
- subsecondWrapper 调用：`raytracedicom_pybind.cpp:2071-2086`
- CT texture 构建（永远执行）：`raytracedicom_wrapper.cu:4069-4100`
- 几何三分支：`raytracedicom_wrapper.cu:4578 / 4637 / 4710`
- `hasWeqVolume` 判定：`raytracedicom_wrapper.cu:4839`
- **核心分叉 WEQ-fill vs CT-raytrace**：`raytracedicom_wrapper.cu:5095-5225`
- CT 单位识别/转换：`raytracedicom_wrapper.cu:3851-3974, 4069-4090`
- 模块绑定（3 个 alias）：`raytracedicom_pybind.cpp:2652, 2754, 2804`
