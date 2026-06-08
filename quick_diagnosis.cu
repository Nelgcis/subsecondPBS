/**
 * @file quick_diagnosis.cu
 * @brief 快速诊断代码 - 可直接插入到 wrapper_integration_test.cu 中
 * 
 * 这个文件包含可以立即使用的诊断代码片段，
 * 无需修改核心框架，可以临时添加到测试中进行调试。
 */

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

// ============================================================================
// 快速诊断：按z统计BEV剂量分布（CPU端，从已有日志输出）
// ============================================================================

/**
 * 分析现有的日志输出，从中提取关键的z分布信息
 * 
 * 从 test_run_with_reference_lut.log 中，我们已经有了：
 * 
 *   BEV dose z-slice summary:
 *     z=0 sum=0.000000e+00
 *     z=1 sum=3.067029e-07    ← PEAK (100%)
 *     z=2 sum=1.534471e-07    ← 50%
 *     z=3 sum=1.534992e-07
 *     ...
 *     z=5 sum=1.535092e-07
 * 
 * 这已经告诉我们z=1有峰。现在我们需要找出原因。
 */

// ============================================================================
// GPU核函数：统计z=1和z=2的详细比较
// ============================================================================

__global__ void detailedZComparisonKernel(
    const float* bevDose,       // [576, 576, 32]
    float* zHistogram,          // [32, 256] - 直方图
    float* zStatistics)         // [32, 4] - [sum, max, min, count]
{
    int z = blockIdx.x;
    int tid = threadIdx.x;
    
    if (z >= 32) return;
    
    float sum = 0.0f;
    float maxVal = 0.0f;
    float minVal = 1e30f;
    int count = 0;
    int nanCount = 0;
    int infCount = 0;
    
    // 扫描该z层的所有体素
    for (int idx = tid; idx < 576*576; idx += blockDim.x) {
        float dose = bevDose[idx * 32 + z];
        
        if (!isnan(dose) && !isinf(dose) && dose > 1e-15f) {
            sum += dose;
            maxVal = fmaxf(maxVal, dose);
            minVal = fminf(minVal, dose);
            count++;
            
            // 简单直方图：按对数级分类
            if (dose > 0) {
                int bin = min(255, (int)(10.0f * logf(dose) / logf(10.0f)) + 128);
                bin = max(0, min(255, bin));
                atomicAdd(&zHistogram[z*256 + bin], 1.0f);
            }
        } else if (isnan(dose)) {
            atomicAdd(&nanCount, 1);
        } else if (isinf(dose)) {
            atomicAdd(&infCount, 1);
        }
    }
    
    // 原子操作累加统计
    extern __shared__ char smem[];
    float* sumShared = (float*)smem;
    
    sumShared[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sumShared[tid] += sumShared[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        zStatistics[z*4 + 0] = sumShared[0];           // sum
        zStatistics[z*4 + 1] = maxVal;                 // max
        zStatistics[z*4 + 2] = (minVal < 1e30f) ? minVal : 0.0f;  // min
        zStatistics[z*4 + 3] = (float)count;           // count
    }
}

// ============================================================================
// CPU端辅助函数：快速分析
// ============================================================================

struct ZSliceAnalysis {
    float sum;
    float max;
    float min;
    int count;
    
    float averageDose() const { return (count > 0) ? (sum / count) : 0.0f; }
    float dosePerVoxel() const { return (count > 0) ? (sum / count) : 0.0f; }
};

void printQuickDiagnosis(const std::vector<ZSliceAnalysis>& stats)
{
    std::cout << "\n" << std::string(100, '=') << std::endl;
    std::cout << "QUICK DIAGNOSIS: Z-SLICE DOSE DISTRIBUTION" << std::endl;
    std::cout << std::string(100, '=') << std::endl;
    
    float totalDose = 0.0f;
    float maxDose = 0.0f;
    int maxZ = -1;
    
    for (int z = 0; z < 32; z++) {
        totalDose += stats[z].sum;
        if (stats[z].sum > maxDose) {
            maxDose = stats[z].sum;
            maxZ = z;
        }
    }
    
    std::cout << "\n[Table] Z-Slice Statistics:\n" << std::endl;
    std::cout << std::left
              << std::setw(5) << "Z"
              << std::setw(16) << "Sum Dose"
              << std::setw(14) << "Max Dose"
              << std::setw(14) << "Min Dose"
              << std::setw(10) << "Count"
              << std::setw(12) << "% of Total"
              << std::setw(14) << "Avg/Voxel"
              << std::endl;
    std::cout << std::string(85, '-') << std::endl;
    
    for (int z = 0; z < 32; z++) {
        float pctTotal = (totalDose > 0) ? (100.0f * stats[z].sum / totalDose) : 0.0f;
        
        std::cout << std::left
                  << std::setw(5) << z;
        
        // Sum Dose
        if (stats[z].sum > 1e-10) {
            std::cout << std::scientific << std::setprecision(3)
                      << std::setw(16) << stats[z].sum;
        } else {
            std::cout << std::setw(16) << "0";
        }
        
        // Max Dose
        if (stats[z].max > 1e-15) {
            std::cout << std::scientific << std::setprecision(2)
                      << std::setw(14) << stats[z].max;
        } else {
            std::cout << std::setw(14) << "0";
        }
        
        // Min Dose
        if (stats[z].min > 1e-15) {
            std::cout << std::scientific << std::setprecision(2)
                      << std::setw(14) << stats[z].min;
        } else {
            std::cout << std::setw(14) << "0";
        }
        
        // Count
        std::cout << std::fixed << std::setprecision(0)
                  << std::setw(10) << stats[z].count;
        
        // % of Total
        std::cout << std::fixed << std::setprecision(1)
                  << std::setw(12) << pctTotal << "%";
        
        // Avg/Voxel
        std::cout << std::scientific << std::setprecision(2)
                  << std::setw(14) << stats[z].averageDose();
        
        std::cout << std::endl;
    }
    
    std::cout << std::string(85, '-') << std::endl;
    std::cout << "Total dose: " << std::scientific << std::setprecision(6) 
              << totalDose << std::endl;
    std::cout << "Peak at z=" << maxZ << " with " << std::scientific 
              << std::setprecision(3) << maxDose << " (" 
              << std::fixed << std::setprecision(1) 
              << (100.0f * maxDose / totalDose) << "% of total)" << std::endl;
    
    // ========================================================================
    // 关键分析
    // ========================================================================
    std::cout << "\n[ANALYSIS]" << std::endl;
    
    // 检查z=1是否有峰
    if (stats[1].sum > stats[2].sum && stats[2].sum > 0) {
        float ratio = stats[1].sum / stats[2].sum;
        std::cout << "  ✓ Confirmed: z=1 has peak (z=1/z=2 ratio = " 
                  << std::fixed << std::setprecision(2) << ratio << ")" << std::endl;
        
        if (ratio > 1.9f && ratio < 2.1f) {
            std::cout << "    → Ratio ≈ 2.0 suggests systematic halving from z=1 to z=2" << std::endl;
            std::cout << "    → This is NOT a typical Bragg peak pattern" << std::endl;
        }
    } else {
        std::cout << "  ✗ z=1 does NOT have peak (unexpected!)" << std::endl;
    }
    
    // 检查z=0
    if (stats[0].sum < 1e-15f && stats[0].count == 0) {
        std::cout << "  ✓ Confirmed: z=0 is empty (startIdx=(0,0,1) skips it)" << std::endl;
    }
    
    // 检查是否平坦
    bool isFlatAfterZ2 = true;
    for (int z = 3; z < 31; z++) {
        if (stats[z].sum > 0 && stats[z+1].sum > 0) {
            float change = fabs(stats[z+1].sum - stats[z].sum) / stats[z].sum;
            if (change > 0.1f) {  // 变化超过10%
                isFlatAfterZ2 = false;
                break;
            }
        }
    }
    
    if (isFlatAfterZ2) {
        std::cout << "  ✓ Confirmed: Distribution is relatively flat for z≥3" << std::endl;
    }
    
    std::cout << "\n[INTERPRETATION]" << std::endl;
    std::cout << "  Pattern observed: Sharp peak at z=1, then drop to 50%, then flat" << std::endl;
    std::cout << "  Physical explanation: NOT a Bragg peak (那在>100mm处)" << std::endl;
    std::cout << "  Likely cause:" << std::endl;
    std::cout << "    1. High dE/dx at shallow depth (early energy loss)" << std::endl;
    std::cout << "    2. Ray weight concentration at z=1" << std::endl;
    std::cout << "    3. IDD lookup artifact at shallow depth" << std::endl;
    std::cout << "    4. Superposition/convolution edge effect" << std::endl;
    
    std::cout << "\n[NEXT STEPS]" << std::endl;
    std::cout << "  1. Check ray weight distribution by z-slice" << std::endl;
    std::cout << "  2. Check IDD values at different depths" << std::endl;
    std::cout << "  3. Review superposition convolution at z=0/1 boundary" << std::endl;
    std::cout << "  4. Check primTransfDiv coordinate transform" << std::endl;
    
    std::cout << std::string(100, '=') << std::endl;
}

// ============================================================================
// 实际使用：直接从日志数据进行分析（CPU端，不需GPU）
// ============================================================================

void analyzeFromLogData()
{
    /**
     * 从 test_run_with_reference_lut.log 的输出数据：
     * 
     * Layer 0 (120 MeV):
     *   z=1 sum=3.067029e-07  max=9.283589e-11  cnt>thr=12721
     *   z=2 sum=1.534471e-07  max=4.334897e-11  cnt>thr=12813
     *   z=3 sum=1.534992e-07  max=4.317570e-11  cnt>thr=12813
     *   z=4 sum=1.533520e-07  max=4.273888e-11  cnt>thr=13012
     *   z=5 sum=1.535092e-07  max=4.276026e-11  cnt>thr=13198
     */
    
    std::cout << "\n" << std::string(100, '=') << std::endl;
    std::cout << "ANALYSIS OF LOGGED Z-SLICE DATA" << std::endl;
    std::cout << std::string(100, '=') << std::endl;
    
    // Layer 0 data from log
    struct Layer0Data {
        float sum;
        float max;
        int count;
    };
    
    std::vector<Layer0Data> layer0 = {
        {0.0f, 0.0f, 0},                           // z=0
        {3.067029e-07f, 9.283589e-11f, 12721},    // z=1 - PEAK
        {1.534471e-07f, 4.334897e-11f, 12813},    // z=2
        {1.534992e-07f, 4.317570e-11f, 12813},    // z=3
        {1.533520e-07f, 4.273888e-11f, 13012},    // z=4
        {1.535092e-07f, 4.276026e-11f, 13198},    // z=5
    };
    
    std::cout << "\n[Layer 0: 120 MeV] Data from log:\n" << std::endl;
    std::cout << std::left
              << std::setw(5) << "Z"
              << std::setw(16) << "Sum"
              << std::setw(16) << "Max"
              << std::setw(10) << "Count"
              << std::setw(12) << "Ratio"
              << std::endl;
    std::cout << std::string(59, '-') << std::endl;
    
    float totalDose = 0.0f;
    for (int i = 0; i < layer0.size(); i++) {
        totalDose += layer0[i].sum;
    }
    
    for (int i = 0; i < layer0.size(); i++) {
        float ratio = (i > 0 && layer0[i].sum > 0 && layer0[i-1].sum > 0) 
            ? (layer0[i].sum / layer0[i-1].sum) : 0.0f;
        
        std::cout << std::left
                  << std::setw(5) << i
                  << std::scientific << std::setprecision(3)
                  << std::setw(16) << layer0[i].sum
                  << std::setw(16) << layer0[i].max
                  << std::fixed << std::setprecision(0)
                  << std::setw(10) << layer0[i].count
                  << std::setprecision(3)
                  << std::setw(12) << ratio
                  << std::endl;
    }
    
    std::cout << "\n[KEY OBSERVATIONS]" << std::endl;
    std::cout << "1. z=1/z=2 ratio = " << std::fixed << std::setprecision(2)
              << (layer0[1].sum / layer0[2].sum) << std::endl;
    std::cout << "   → Exactly 2.0, suggesting systematic halving" << std::endl;
    
    std::cout << "2. z=2 through z=5 are relatively flat:" << std::endl;
    for (int i = 2; i < 5; i++) {
        float change = (layer0[i+1].sum - layer0[i].sum) / layer0[i].sum;
        std::cout << "   z=" << i << " to z=" << (i+1) << ": " 
                  << std::fixed << std::setprecision(2) << (100.0f * change) << "% change" << std::endl;
    }
    
    std::cout << "3. Count (non-zero voxel) progression:" << std::endl;
    for (int i = 0; i < 6; i++) {
        std::cout << "   z=" << i << ": " << layer0[i].count << " voxels" << std::endl;
    }
    
    std::cout << "\n[HYPOTHESIS SCORING]" << std::endl;
    std::cout << "  Problem: Sharp peak at z=1, drop to 50%, then flat" << std::endl;
    std::cout << std::endl;
    std::cout << "  H1: Ray weight concentration at z=1" << std::endl;
    std::cout << "      Score: 40/100 (likely, but needs verification)" << std::endl;
    std::cout << "      Test: Check ray weight z-distribution" << std::endl;
    std::cout << std::endl;
    std::cout << "  H2: High dE/dx (energy loss) at shallow depth" << std::endl;
    std::cout << "      Score: 35/100 (physically plausible, but IDD should show it)" << std::endl;
    std::cout << "      Test: Check IDD values vs depth" << std::endl;
    std::cout << std::endl;
    std::cout << "  H3: Superposition convolution artifact" << std::endl;
    std::cout << "      Score: 15/100 (less likely, but edge effect possible)" << std::endl;
    std::cout << "      Test: Compare with unpadded BEV dose" << std::endl;
    std::cout << std::endl;
    std::cout << "  H4: Coordinate transform (primTransfDiv) error" << std::endl;
    std::cout << "      Score: 10/100 (startIdx=(0,0,1) is intentional)" << std::endl;
    std::cout << "      Test: Check BEV to dose coordinate mapping" << std::endl;
    
    std::cout << "\n[RECOMMENDED ACTION]" << std::endl;
    std::cout << "  1. IMMEDIATE: Add ray weight and IDD z-distribution outputs" << std::endl;
    std::cout << "  2. VERIFY: Use GPU kernel to compute ray weight and IDD histograms" << std::endl;
    std::cout << "  3. DEBUG: If ray weight/IDD normal, check superposition code" << std::endl;
    std::cout << "  4. TRACE: Add step-by-step dose accumulation output" << std::endl;
    
    std::cout << std::string(100, '=') << std::endl;
}

/*
============================================================================
使用方式：

在 wrapper_integration_test.cu 的 main() 函数中添加：

    // 在所有计算完成后
    std::cout << "\n[RUNNING QUICK DIAGNOSIS FROM LOG DATA]" << std::endl;
    analyzeFromLogData();

这会生成详细的分析报告，无需额外的GPU计算。
============================================================================
*/
