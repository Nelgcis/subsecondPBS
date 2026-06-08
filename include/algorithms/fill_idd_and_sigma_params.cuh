/**
 * \file
 * \brief IDD and Sigma calculation parameters
 */

#ifndef FILL_IDD_AND_SIGMA_PARAMS_CUH
#define FILL_IDD_AND_SIGMA_PARAMS_CUH

#include "common.cuh"
#include "Macro.cuh"

// DensityAndSpTracerParams已在ray_tracing.h中定义

// IDD and Sigma calculation parameters
struct FillIddAndSigmaParams {
    float energyIdx;
    float energyScaleFact;
    float peakDepth;
    float rangeStopDepth;
    float rRlScale;
    float spotDist;
    unsigned int nucMemStep;
    unsigned int first;
    unsigned int afterLast;
    float entrySigmaSq;
    float stepLength;
    float sigmaSqAirLin;
    float sigmaSqAirQuad;
    vec3f dist;
    vec3f corner;
    vec3f delta;
    float volConst;
    float volLin;
    float volSq;
    unsigned int rayDimsX;
    unsigned int rayDimsY;
    
    __host__ __device__ FillIddAndSigmaParams() : energyIdx(0), energyScaleFact(1.0f), peakDepth(0),
                                                 rangeStopDepth(0), rRlScale(1.0f), spotDist(0), nucMemStep(0),
                                                 first(0), afterLast(0), entrySigmaSq(0), stepLength(0),
                                                 sigmaSqAirLin(0), sigmaSqAirQuad(0), volConst(0), volLin(0), volSq(0),
                                                 rayDimsX(0), rayDimsY(0) {}
    
    __host__ __device__ unsigned int getFirstStep() const { return first; }
    __host__ __device__ unsigned int getAfterLastStep() const { return afterLast; }
    __host__ __device__ float getEnergyIdx() const { return energyIdx; }
    __host__ __device__ float getEnergyScaleFact() const { return energyScaleFact; }
    __host__ __device__ float getPeakDepth() const { return peakDepth; }
    __host__ __device__ float getRangeStopDepth() const {
        return (rangeStopDepth > 0.0f) ? rangeStopDepth : (peakDepth * BP_DEPTH_CUTOFF);
    }
    __host__ __device__ float getEntrySigmaSq() const { return entrySigmaSq; }
    __host__ __device__ float getSpotDist() const { return spotDist; }
    __host__ __device__ unsigned int getNucMemStep() const { return nucMemStep; }
    __host__ __device__ float getStepLength() const { return stepLength; }
    __host__ __device__ float getSigmaSqAirLin() const { return sigmaSqAirLin; }
    __host__ __device__ float getSigmaSqAirQuad() const { return sigmaSqAirQuad; }
    __host__ __device__ float getRRlScale() const { return rRlScale; }
    __host__ __device__ unsigned int getRayDimsX() const { return rayDimsX; }
    __host__ __device__ unsigned int getRayDimsY() const { return rayDimsY; }
    __host__ __device__ vec2f voxelWidth(const unsigned int idxK) const {
        return vec2f(delta.x * (1.0f-(corner.z+idxK*delta.z)/dist.x), 
                    delta.y * (1.0f-(corner.z+idxK*delta.z)/dist.y));
    }
    __host__ __device__ float stepVol(const unsigned int k) const { 
        return volConst + k*volLin + k*k*volSq; 
    }
    __host__ __device__ void initStepAndAirDiv() {
        float relStepLenSq = 1.0f;
        vec2f sigmaSqCoefs = sigmaSqAirCoefs(peakDepth);
        sigmaSqAirQuad = sigmaSqCoefs.x * relStepLenSq * delta.z * delta.z;
        float zDist = corner.z;
        sigmaSqAirLin = 2.0f*sigmaSqCoefs.x*relStepLenSq*delta.z*zDist + sigmaSqCoefs.y*delta.z;
        stepLength = abs(delta.z);
    }
    __host__ __device__ vec2f sigmaSqAirCoefs(const float r0) const {
        return vec2f(0.00270f / (r0 - 4.50f), -4.39f / (r0 - 3.86f));
    }
    
    // Additional methods needed by bev_ray_tracing.cu
    __host__ __device__ unsigned int getFirstInside() const { return first; }
    __host__ __device__ unsigned int getFirstOutside() const { return afterLast; }
    
    // 获取初始能量 - 需要从外部传入layerEnergy数组
    __host__ __device__ float getInitialEnergy(const float* layerEnergy) const { 
        if (layerEnergy && energyIdx >= 0 && energyIdx < 1000) { // 假设最大1000层
            return layerEnergy[(int)energyIdx] * energyScaleFact;
        }
        return 0.0f;
    }
    
    // 计算有效sigma - 基于Soukup等人2005年的方法
    __host__ __device__ float calculateEffectiveSigma(float depth) const { 
        // 基于深度的sigma计算：sigma^2 = sigma0^2 + sigma_air^2 + sigma_multiple^2
        float sigmaAirSq = sigmaSqAirLin * depth + sigmaSqAirQuad * depth * depth;
        float sigmaMultipleSq = 0.0f; // 多重散射项，需要根据材料计算
        
        return sqrtf(entrySigmaSq + sigmaAirSq + sigmaMultipleSq);
    }
    
    // 检查被动散射 - 基于能量和深度阈值
    __host__ __device__ bool isPassiveScattering(float currentEnergy, float thresholdEnergy = 50.0f) const { 
        // 当能量低于阈值时认为是被动散射
        return currentEnergy < thresholdEnergy;
    }
};

#endif // FILL_IDD_AND_SIGMA_PARAMS_CUH
