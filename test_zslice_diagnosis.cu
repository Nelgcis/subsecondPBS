/**
 * @file test_zslice_diagnosis.cu
 * @brief 诊断z=1浅层峰值问题的测试程序
 * 
 * 问题现象：
 *   - BEV dose分布在z=1处有峰值：3.067e-07
 *   - z=2处下降到一半：1.534e-07
 *   - z=3-31处保持平坦
 * 
 * 诊断目标：
 *   1. 检查射线权重是否在z=1处堆积
 *   2. 检查IDD值是否在z=1处异常高
 *   3. 追踪每一步的剂量贡献
 *   4. 检查startIdx=(0,0,1)跳过z=0的影响
 */

#include <iostream>
#include <vector>
#include <cstring>
#include <iomanip>
#include <cuda_runtime.h>

// ============================================================================
// 核函数：按z层统计射线权重分布
// ============================================================================

__global__ void analyzeRayWeightByZKernel(
    const float* rayWeightBEV,      // [262144, 32]
    float* zSliceSum,               // [32] - 输出：每个z的总权重
    int* zSliceCount)               // [32] - 输出：每个z的非零权重数量
{
    int z = blockIdx.x;
    int tid = threadIdx.x;
    
    if (z >= 32) return;
    
    float sum = 0.0f;
    int count = 0;
    
    // 每个线程处理一部分射线
    for (int y = tid; y < 262144; y += blockDim.x) {
        float w = rayWeightBEV[y * 32 + z];
        if (w > 1e-15f) {
            sum += w;
            count++;
        }
    }
    
    // 线程间求和
    extern __shared__ char smem[];
    float* sumShared = (float*)smem;
    int* countShared = (int*)(sumShared + blockDim.x);
    
    sumShared[tid] = sum;
    countShared[tid] = count;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sumShared[tid] += sumShared[tid + s];
            countShared[tid] += countShared[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        zSliceSum[z] = sumShared[0];
        zSliceCount[z] = countShared[0];
    }
}

// ============================================================================
// 核函数：按z层统计IDD分布
// ============================================================================

__global__ void analyzeIDDByZKernel(
    const float* iddBEV,            // [262144, 33] - 每个射线每步的IDD
    int steps,
    float* zSliceSum,               // [32] - 输出：每个z的IDD总和
    float* zSliceMax,               // [32] - 输出：每个z的IDD最大值
    int* zSliceCount)               // [32] - 输出：每个z的非零IDD数量
{
    int z = blockIdx.x;
    int tid = threadIdx.x;
    
    if (z >= 32) return;
    
    float sum = 0.0f;
    float maxVal = 0.0f;
    int count = 0;
    
    // 每个线程处理一部分射线
    for (int y = tid; y < 262144; y += blockDim.x) {
        if (z < steps) {
            float idd = iddBEV[y * steps + z];
            if (idd > 1e-15f) {
                sum += idd;
                maxVal = fmaxf(maxVal, idd);
                count++;
            }
        }
    }
    
    // 线程间求和
    extern __shared__ char smem[];
    float* sumShared = (float*)smem;
    float* maxShared = (float*)(sumShared + blockDim.x);
    int* countShared = (int*)(maxShared + blockDim.x);
    
    sumShared[tid] = sum;
    maxShared[tid] = maxVal;
    countShared[tid] = count;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sumShared[tid] += sumShared[tid + s];
            maxShared[tid] = fmaxf(maxShared[tid], maxShared[tid + s]);
            countShared[tid] += countShared[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        zSliceSum[z] = sumShared[0];
        zSliceMax[z] = maxShared[0];
        zSliceCount[z] = countShared[0];
    }
}

// ============================================================================
// 核函数：追踪每层的剂量贡献
// ============================================================================

__global__ void analyzeDoseByStepKernel(
    const float* doseBEV,           // [576, 576, 32]
    float* zSliceSum,               // [32] - 输出：每个z的总剂量
    float* zSliceMax,               // [32] - 输出：每个z的最大剂量
    int* zSliceCount)               // [32] - 输出：每个z的非零剂量数
{
    int z = blockIdx.x;
    int tid = threadIdx.x;
    
    if (z >= 32) return;
    
    float sum = 0.0f;
    float maxVal = 0.0f;
    int count = 0;
    
    // 每个线程处理该z层的一部分体素
    for (int idx = tid; idx < 576*576; idx += blockDim.x) {
        float dose = doseBEV[idx * 32 + z];
        if (dose > 1e-15f) {
            sum += dose;
            maxVal = fmaxf(maxVal, dose);
            count++;
        }
    }
    
    // 线程间求和
    extern __shared__ char smem[];
    float* sumShared = (float*)smem;
    float* maxShared = (float*)(sumShared + blockDim.x);
    int* countShared = (int*)(maxShared + blockDim.x);
    
    sumShared[tid] = sum;
    maxShared[tid] = maxVal;
    countShared[tid] = count;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sumShared[tid] += sumShared[tid + s];
            maxShared[tid] = fmaxf(maxShared[tid], maxShared[tid + s]);
            countShared[tid] += countShared[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        zSliceSum[z] = sumShared[0];
        zSliceMax[z] = maxShared[0];
        zSliceCount[z] = countShared[0];
    }
}

// ============================================================================
// CPU 辅助函数
// ============================================================================

void printAnalysisHeader(const std::string& title)
{
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << std::string(80, '=') << std::endl;
}

void printZSliceHeader(const std::string& metric)
{
    std::cout << std::left
              << std::setw(6) << "Z"
              << std::setw(18) << ("Sum " + metric)
              << std::setw(18) << ("Max " + metric)
              << std::setw(12) << "Count"
              << std::setw(12) << "% of Max"
              << std::endl;
    std::cout << std::string(66, '-') << std::endl;
}

// ============================================================================
// 主诊断函数
// ============================================================================

void diagnoseZSlicePeak(
    const float* rayWeightBEV,  // [262144, 32]
    const float* iddBEV,         // [262144, 33]
    const float* doseBEV,        // [576, 576, 32]
    int steps)
{
    cudaError_t err;
    
    // 分配GPU内存存储结果
    float *d_zSum, *d_zMax;
    int *d_zCount;
    
    cudaMalloc(&d_zSum, 32 * sizeof(float));
    cudaMalloc(&d_zMax, 32 * sizeof(float));
    cudaMalloc(&d_zCount, 32 * sizeof(int));
    
    // CPU结果缓冲区
    std::vector<float> h_zSum(32), h_zMax(32);
    std::vector<int> h_zCount(32);
    
    int blockSize = 256;
    int smemSize = blockSize * (sizeof(float) + sizeof(float) + sizeof(int));
    
    // ========================================================================
    // 诊断1：射线权重分布
    // ========================================================================
    printAnalysisHeader("1. RAY WEIGHT DISTRIBUTION BY Z-SLICE");
    
    cudaMemset(d_zSum, 0, 32 * sizeof(float));
    cudaMemset(d_zCount, 0, 32 * sizeof(int));
    
    analyzeRayWeightByZKernel<<<32, blockSize, blockSize * sizeof(float) + blockSize * sizeof(int)>>>(
        rayWeightBEV, d_zSum, d_zCount);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Ray weight kernel error: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    cudaMemcpy(h_zSum.data(), d_zSum, 32 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_zCount.data(), d_zCount, 32 * sizeof(int), cudaMemcpyDeviceToHost);
    
    float maxRayWeight = *std::max_element(h_zSum.begin(), h_zSum.end());
    
    printZSliceHeader("RayWeight");
    for (int z = 0; z < 32; z++) {
        float pctOfMax = (maxRayWeight > 0) ? (100.0f * h_zSum[z] / maxRayWeight) : 0.0f;
        std::cout << std::left
                  << std::setw(6) << z
                  << std::setw(18) << std::scientific << h_zSum[z]
                  << std::setw(18) << "-"
                  << std::setw(12) << h_zCount[z]
                  << std::setw(12) << std::fixed << std::setprecision(1) << pctOfMax << "%"
                  << std::endl;
    }
    
    // 关键观察
    std::cout << "\n[ANALYSIS] Ray weight observation:" << std::endl;
    std::cout << "  z=0: " << h_zSum[0] << " (expected: 0 if skipped)" << std::endl;
    std::cout << "  z=1: " << h_zSum[1] << " (peak? should match dose peak?)" << std::endl;
    std::cout << "  z=2: " << h_zSum[2] << " (compare with z=1)" << std::endl;
    if (h_zSum[1] > 0 && h_zSum[2] > 0) {
        std::cout << "  z=1/z=2 ratio: " << (h_zSum[1] / h_zSum[2]) << std::endl;
    }
    
    // ========================================================================
    // 诊断2：IDD分布
    // ========================================================================
    printAnalysisHeader("2. IDD DISTRIBUTION BY Z-SLICE");
    
    cudaMemset(d_zSum, 0, 32 * sizeof(float));
    cudaMemset(d_zMax, 0, 32 * sizeof(float));
    cudaMemset(d_zCount, 0, 32 * sizeof(int));
    
    analyzeIDDByZKernel<<<32, blockSize, smemSize>>>(
        iddBEV, steps, d_zSum, d_zMax, d_zCount);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "IDD kernel error: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    cudaMemcpy(h_zSum.data(), d_zSum, 32 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_zMax.data(), d_zMax, 32 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_zCount.data(), d_zCount, 32 * sizeof(int), cudaMemcpyDeviceToHost);
    
    float maxIDD = *std::max_element(h_zMax.begin(), h_zMax.end());
    
    printZSliceHeader("IDD");
    for (int z = 0; z < 32; z++) {
        float pctOfMax = (maxIDD > 0) ? (100.0f * h_zMax[z] / maxIDD) : 0.0f;
        std::cout << std::left
                  << std::setw(6) << z
                  << std::scientific << std::setprecision(3)
                  << std::setw(18) << h_zSum[z]
                  << std::setw(18) << h_zMax[z]
                  << std::setw(12) << h_zCount[z]
                  << std::fixed << std::setprecision(1)
                  << std::setw(12) << pctOfMax << "%"
                  << std::endl;
    }
    
    // 关键观察
    std::cout << "\n[ANALYSIS] IDD observation:" << std::endl;
    std::cout << "  z=0: sum=" << h_zSum[0] << " max=" << h_zMax[0] << std::endl;
    std::cout << "  z=1: sum=" << h_zSum[1] << " max=" << h_zMax[1] << " (peak?)" << std::endl;
    std::cout << "  z=2: sum=" << h_zSum[2] << " max=" << h_zMax[2] << std::endl;
    if (h_zSum[1] > 0 && h_zSum[2] > 0) {
        std::cout << "  z=1/z=2 IDD ratio: " << (h_zSum[1] / h_zSum[2]) << std::endl;
    }
    
    // ========================================================================
    // 诊断3：最终剂量分布
    // ========================================================================
    printAnalysisHeader("3. FINAL DOSE DISTRIBUTION BY Z-SLICE (BEV coordinates)");
    
    cudaMemset(d_zSum, 0, 32 * sizeof(float));
    cudaMemset(d_zMax, 0, 32 * sizeof(float));
    cudaMemset(d_zCount, 0, 32 * sizeof(int));
    
    analyzeDozeByStepKernel<<<32, blockSize, smemSize>>>(
        doseBEV, d_zSum, d_zMax, d_zCount);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Dose kernel error: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    cudaMemcpy(h_zSum.data(), d_zSum, 32 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_zMax.data(), d_zMax, 32 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_zCount.data(), d_zCount, 32 * sizeof(int), cudaMemcpyDeviceToHost);
    
    float maxDose = *std::max_element(h_zMax.begin(), h_zMax.end());
    float totalDose = 0.0f;
    for (int i = 0; i < 32; i++) totalDose += h_zSum[i];
    
    printZSliceHeader("Dose");
    for (int z = 0; z < 32; z++) {
        float pctOfMax = (maxDose > 0) ? (100.0f * h_zSum[z] / maxDose) : 0.0f;
        float pctOfTotal = (totalDose > 0) ? (100.0f * h_zSum[z] / totalDose) : 0.0f;
        std::cout << std::left
                  << std::setw(6) << z
                  << std::scientific << std::setprecision(3)
                  << std::setw(18) << h_zSum[z]
                  << std::setw(18) << h_zMax[z]
                  << std::fixed << std::setprecision(0)
                  << std::setw(12) << h_zCount[z]
                  << std::setprecision(1)
                  << std::setw(12) << pctOfMax << "%"
                  << std::endl;
    }
    
    // 关键观察
    std::cout << "\n[ANALYSIS] Dose observation:" << std::endl;
    std::cout << "  Total dose: " << std::scientific << totalDose << std::endl;
    std::cout << "  z=0: sum=" << h_zSum[0] << " (should be 0 if skipped)" << std::endl;
    std::cout << "  z=1: sum=" << h_zSum[1] << " (peak!)" << std::endl;
    std::cout << "  z=2: sum=" << h_zSum[2] << std::endl;
    std::cout << "  z=1/z=2 dose ratio: " << std::fixed << (h_zSum[2] > 0 ? h_zSum[1] / h_zSum[2] : 0) << std::endl;
    
    // ========================================================================
    // 诊断总结
    // ========================================================================
    printAnalysisHeader("DIAGNOSTIC SUMMARY");
    
    std::cout << "\n[KEY FINDINGS]" << std::endl;
    std::cout << "  1. Ray weight distribution: ";
    if (h_zSum[1] > h_zSum[2]) {
        std::cout << "✓ Shows z=1 > z=2 (similar to dose peak)" << std::endl;
    } else {
        std::cout << "✗ Does NOT show z=1 peak (likely not ray weight issue)" << std::endl;
    }
    
    std::cout << "  2. IDD distribution: ";
    if (h_zMax[1] > h_zMax[2]) {
        std::cout << "✓ Shows z=1 > z=2 (could explain dose peak)" << std::endl;
    } else {
        std::cout << "✗ Does NOT show z=1 peak (likely not IDD lookup issue)" << std::endl;
    }
    
    std::cout << "\n[HYPOTHESES]" << std::endl;
    std::cout << "  If ray weight shows z=1 peak:" << std::endl;
    std::cout << "    → Problem: CPB to ray weight mapping concentrates rays at z=1" << std::endl;
    std::cout << "    → Check: raytracedicom_wrapper.cu, CPB projection logic" << std::endl;
    
    std::cout << "  If IDD shows z=1 peak:" << std::endl;
    std::cout << "    → Problem: IDD lookup returns higher values at z=1" << std::endl;
    std::cout << "    → Check: Energy-to-index interpolation, boundary conditions" << std::endl;
    
    std::cout << "  If neither ray weight nor IDD shows z=1 peak:" << std::endl;
    std::cout << "    → Problem: Likely in superposition convolution or coordinate transform" << std::endl;
    std::cout << "    → Check: primTransfDiv kernel, BEV to dose coordinate mapping" << std::endl;
    
    std::cout << "\n[SPECIAL NOTE] z=0 behavior:" << std::endl;
    std::cout << "  z=0 sum: " << h_zSum[0] << std::endl;
    if (h_zSum[0] < 1e-15f) {
        std::cout << "  → Confirmed: z=0 is empty (startIdx=(0,0,1) skips it)" << std::endl;
    }
    
    // 清理
    cudaFree(d_zSum);
    cudaFree(d_zMax);
    cudaFree(d_zCount);
}

// ============================================================================
// 使用示例（集成到主程序）
// ============================================================================

/*
在 wrapper_integration_test.cu 中的测试流程中添加：

    // 在所有三层处理完毕后，执行诊断
    std::cout << "\n[DIAGNOSTIC] Running z-slice diagnosis..." << std::endl;
    
    diagnoseZSlicePeak(
        d_rayWeightLayer0,  // GPU指针
        d_iddLayer0,        // GPU指针  
        d_doseBEV,          // GPU指针
        params.steps        // 33
    );
*/

#endif // __cplusplus
