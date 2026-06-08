// Forward declarations for missing functions
extern "C" {
    int initializeRayWeightsFromSubspotDataGPU(
        float* rayWeights,
        vec3i rayDims,
        vec3f beamDirection,
        vec3f bmXDirection,
        vec3f bmYDirection,
        vec3f sourcePosition,
        float sad,
        float refPlaneZ
    );
    
    // Forward declare structs
    struct FillIddAndSigmaParams;
    struct RTDEnergyStruct;
    
    FillIddAndSigmaParams createIddParams(int energyIdx, float peakDepth, const RTDEnergyStruct* energyData);
    
    void launchSuperpositionKernels(
        float* devRayIdd,
        float* devRayRSigmaEff,
        float* devBevPrimDose,
        int rayDimsX,
        int rayDimsY
    );
    
    void primTransfDivKernel(
        float* devBevPrimDose,
        float* devDoseVol,
        int2 rayDims,
        int3 doseVolDims,
        float3 doseVolSpacing,
        float3 doseVolOrigin,
        float3 beamDirection,
        float3 bmXDirection,
        float3 bmYDirection,
        float3 sourcePosition,
        float sad
    );
    
    void simpleDoseAccumulationKernel(
        float* doseVol,
        float* bevPrimDose,
        int2 rayDims,
        int3 doseVolDims,
        float3 doseVolSpacing,
        float3 doseVolOrigin,
        float3 spotOffset,
        float3 gantryToImMatrix,
        float3 gantryToDoseMatrix,
        float3 gantryToDoseOffset,
        float sad
    );
    
}

// Forward declaration for performCPBToRayWeightMapping
void performCPBToRayWeightMapping(
    float* cpbWeights,
    vec3i cpbDims,
    vec3f cpbCorner,
    vec3f cpbResolution,
    float* rayWeights,
    vec3i rayDims,
    vec3f rayCorner,
    vec3f rayResolution,
    int layerIdx,  // Current energy layer index
    vec3f beamDirection,
    vec3f bmXDirection,
    vec3f bmYDirection,
    vec3f sourcePosition,
    float sad,
    float refPlaneZ
);

// Forward declaration for SuperpositionParams
struct SuperpositionParams {
    int radius;
    float cutoff;
    // Add other parameters as needed
};

// Enhanced superposition function declaration
void performEnhancedSuperposition(
    float* inDose,
    float* inRSigmaEff,
    float* outDose,
    int inDosePitch,
    int rayDimsX,
    int rayDimsY,
    int numLayers,
    int startZ
);

// Basic superposition function declaration
void performKernelSuperposition(
    float* inDose,
    float* inRSigmaEff,
    float* outDose,
    int inDosePitch,
    int rayDimsX,
    int rayDimsY,
    int radius
);
