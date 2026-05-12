/**
 * @file diagnostic_integration_example.cpp
 * @brief 展示如何将诊断代码集成到现有测试中
 * 
 * 这是一个集成示例，说明如何在 wrapper_integration_test.cu 中添加诊断功能
 */

/*
============================================================================
在 wrapper_integration_test.cu 中的修改步骤：
============================================================================

第1步：在文件顶部包含诊断头文件

    #include "test_zslice_diagnosis.cu"

第2步：在 main() 函数的 RTD Wrapper 部分添加诊断调用

    // 现有代码：
    std::cout << "\nCalling subsecondWrapper..." << std::endl;
    subsecond::RayTraceDicomWrapper wrapper;
    std::vector<float> doseData = wrapper.computeDose(
        imageVol, energyLayers, raySteps, scanParams);
    std::cout << "Wrapper completed successfully!" << std::endl;
    
    // 添加诊断代码（在这里）：
    std::cout << "\n[DEBUG] Starting z-slice diagnosis..." << std::endl;
    
    // 诊断需要的GPU指针：
    // - 射线权重 (layer 0): d_rayWeightLayer0, dimensions [262144, 32]
    // - IDD数据 (layer 0): d_iddLayer0, dimensions [262144, 33]
    // - BEV剂量: d_doseBEV, dimensions [576, 576, 32]
    
    diagnoseZSlicePeak(
        d_rayWeightLayer0,      // GPU pointer to ray weights
        d_iddLayer0,            // GPU pointer to IDD values
        d_doseBEV,              // GPU pointer to BEV dose
        33                      // steps parameter
    );
    
    std::cout << "[DEBUG] Diagnosis complete" << std::endl;

============================================================================
预期输出格式：
============================================================================

================================================================================
  1. RAY WEIGHT DISTRIBUTION BY Z-SLICE
================================================================================
Z     Sum RayWeight      Max RayWeight      Count       % of Max    
----------------------------------------------------------------------
0     0.000000e+00              -               0          0.0%
1     5.234000e-02              -           13541        100.0%
2     2.617000e-02              -           12813         50.0%
3     2.612000e-02              -           12800         49.9%
...

[ANALYSIS] Ray weight observation:
  z=0: 0 (expected: 0 if skipped)
  z=1: 5.234e-02 (peak? should match dose peak?)
  z=2: 2.617e-02 (compare with z=1)
  z=1/z=2 ratio: 2.0

================================================================================
  2. IDD DISTRIBUTION BY Z-SLICE
================================================================================
...

[ANALYSIS] IDD observation:
  z=0: sum=0.000000e+00 max=0.000000e+00
  z=1: sum=1.234e-06 max=1.234e-11 (peak?)
  z=2: sum=5.678e-07 max=5.678e-12
  z=1/z=2 IDD ratio: 2.17

================================================================================
  3. FINAL DOSE DISTRIBUTION BY Z-SLICE (BEV coordinates)
================================================================================
...

[ANALYSIS] Dose observation:
  Total dose: 1.229089e-05
  z=0: sum=0.000000e+00 (should be 0 if skipped)
  z=1: sum=3.067029e-07 (peak!)
  z=2: sum=1.534471e-07
  z=1/z=2 dose ratio: 2.0

================================================================================
DIAGNOSTIC SUMMARY
================================================================================

[KEY FINDINGS]
  1. Ray weight distribution: ✓ Shows z=1 > z=2 (similar to dose peak)
  2. IDD distribution: ✗ Does NOT show z=1 peak (likely not IDD lookup issue)

[HYPOTHESES]
  If ray weight shows z=1 peak:
    → Problem: CPB to ray weight mapping concentrates rays at z=1
    → Check: raytracedicom_wrapper.cu, CPB projection logic
  
  If IDD shows z=1 peak:
    → Problem: IDD lookup returns higher values at z=1
    → Check: Energy-to-index interpolation, boundary conditions
  
  If neither ray weight nor IDD shows z=1 peak:
    → Problem: Likely in superposition convolution or coordinate transform
    → Check: primTransfDiv kernel, BEV to dose coordinate mapping

[SPECIAL NOTE] z=0 behavior:
  z=0 sum: 0.000000e+00
  → Confirmed: z=0 is empty (startIdx=(0,0,1) skips it)

============================================================================
诊断解读指南：
============================================================================

情景1：射线权重在z=1有峰
  ┌─────────────────────────────────────────────────────────────────┐
  │ 结论：问题在CPB投影或射线权重映射                               │
  │ 位置：raytracedicom_wrapper.cu，CPB到射线权重的映射             │
  │ 原因：可能：                                                    │
  │   • CPB卷积核在边界处有偏差                                      │
  │   • 射线权重初始化时第一层被特殊处理                            │
  │ 修复：检查射线权重计算中是否有缩放或权重重分配的问题            │
  └─────────────────────────────────────────────────────────────────┘

情景2：IDD在z=1有峰
  ┌─────────────────────────────────────────────────────────────────┐
  │ 结论：问题在IDD查表或能量到深度的映射                           │
  │ 位置：idd_sigma.cu，IDD查表逻辑                                 │
  │ 原因：可能：                                                    │
  │   • 浅层能量损失率(dE/dx)异常高                                  │
  │   • IDD插值边界处理有问题                                        │
  │ 修复：检查能量索引计算和IDD插值的数值稳定性                     │
  └─────────────────────────────────────────────────────────────────┘

情景3：两者都没有z=1峰
  ┌─────────────────────────────────────────────────────────────────┐
  │ 结论：问题在超立体化卷积或坐标变换                              │
  │ 位置：superposition.cu 或 primTransfDiv kernel                  │
  │ 原因：可能：                                                    │
  │   • 超立体化卷积在z=0边界处理特殊                               │
  │   • BEV到最终剂量的坐标变换有符号错误                           │
  │   • primTransfDiv kernel中startIdx=(0,0,1)导致重新排列          │
  │ 修复：在超立体化和变换代码中添加详细的调试输出                  │
  └─────────────────────────────────────────────────────────────────┘

============================================================================
如何获取GPU指针：
============================================================================

在 wrapper_integration_test.cu 中，这些指针通常在以下位置获得：

1. 射线权重 (rayWeightLayer0):
   
   // 在 subsecond::RayTraceDiomWrapper::computeDose() 中
   // 或在测试中使用内部API获取
   const float* d_rayWeightLayer0 = /* from CPB mapping kernel output */;

2. IDD值:
   
   // 在 fillIddAndSigma() 核函数的输出
   const float* d_iddLayer0 = /* from IDD lookup kernel output */;

3. BEV剂量 (doseBEV):
   
   // 在 tile-based superposition 之后
   const float* d_doseBEV = /* padded BEV dose array [576,576,32] */;

如果这些指针不能直接访问，需要修改RayTraceDicomWrapper来暴露这些中间结果。

============================================================================
*/

// 示例C++代码：如何从不同的编译单元调用诊断函数

namespace subsecond {

class RayTraceDiomWrapperWithDiagnostics : public RayTraceDiomWrapper {
public:
    // 添加诊断方法
    void diagnoseZSlicePeak(
        const float* rayWeightBEV,
        const float* iddBEV,
        const float* doseBEV,
        int steps)
    {
        // 这会调用 test_zslice_diagnosis.cu 中的函数
        // 需要在CMakeLists.txt中链接该文件
    }
};

}  // namespace subsecond
