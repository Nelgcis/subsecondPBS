/**
 * \file
 * \brief RTD Integration - Main Header
 * 
 * Complete integration of RTD project with all core components
 */

#ifndef RAYTRACEDICOM_INTEGRATION_H
#define RAYTRACEDICOM_INTEGRATION_H

#include <vector>
#include <cuda_runtime.h>
#include <texture_types.h>
#include <texture_indirect_functions.h>
#include <device_launch_parameters.h>

// Forward declarations
struct float2;
struct float3;
struct int3;
struct uint3;


#define P_INV 0.5649718f // 1/p, p=1.77
#define E_COEF 8.639415f // (10*alpha)^(-1/p), alpha=2.2e-3
#define SQRT2 1.41421356f // sqrt(2.0f)
#define E_REF_SQ 198.81f // 14.1^2, E_s^2
#define SIGMA_DELTA 0.21f
#define HALF 0.5f
#define RAY_WEIGHT_CUTOFF 1e-6f
struct RTDBeamSettings {
    std::vector<float> energies;           // Energy for each layer
    std::vector<float2> spotSigmas;        // (sigmax, sigmay) at iso in air for each energy layer
    float2 raySpacing;                     // Spacing between adjacent raytracing rays at iso
    unsigned int steps;                    // Number of raytracing steps
    float2 sourceDist;                     // Source to iso distance in x and y
    float3 spotOffset;                     // Spot offset in gantry coordinates
    float3 spotDelta;                      // Spot spacing in gantry coordinates
    float3 gantryToImOffset;              // Gantry to image transform offset
    float3 gantryToImMatrix;              // Gantry to image transform matrix (3x3 linear part)
    float3 gantryToDoseOffset;             // Gantry to dose transform offset
    float3 gantryToDoseMatrix;             // Gantry to dose transform matrix (3x3 linear part)

    // ------------------------------------------------------------------------
    // CarbonPBS-compatible inputs (wrapper must NOT synthesize these internally)
    // ------------------------------------------------------------------------
    // subspotData layout follows CarbonPBS create3DTexture usage:
    //   width = 5 channels  (deltaX, deltaY, weight, sigmaX, sigmaY)
    //   height = maxSubspotsPerLayer
    //   depth  = numLayers (energy layers)
    int maxSubspotsPerLayer = 0;
    std::vector<float> subspotData;        // Flattened [numLayers][maxSubspotsPerLayer][5]

    // Original CarbonPBS spot lattice inputs. These are required to reconstruct
    // the exact RTD spot/ray lattice instead of re-projecting from the dose box.
    std::vector<int> layerSpotCounts;      // layer_info, one entry per energy layer
    std::vector<float> spotPositions;      // idbeamxy flattened as [N][2] on ref plane
    bool spotPositionsAreIndices = false;  // true when spotPositions are WEQ-lattice indices that need header decoding
    std::vector<float> spotWeights;        // number_particle / weight per spot, length N
    std::vector<float> spotBeamDirections; // beam_dir flattened as [N][3], optional but preferred
    std::vector<float> layerLongitudinalCutoffs; // CarbonPBS longitudal_cutoff reduced to one value per energy layer

    // Preferred direct import of CarbonPBS water_equivalence.
    // Header layout follows [depth_start, depth_step, depth_n,
    //                        y_start,     y_step,     y_n,
    //                        x_start,     x_step,     x_n].
    std::vector<float> waterEquivalence;

    // Backward-compatible alias for the first 9 header values of water_equivalence.
    std::vector<float> rayWeqHeader;

    // CarbonPBS lateral profile model exported by dosecal.py.
    // profileData layout: [numLayers][profileDepthSamples][profileChannels]
    // where channels are typically nGauss weights, nGauss sigmas, and one
    // overall-weight channel.
    std::vector<float> profileEnergies;   // CarbonPBS energy_list axis for profile/beamPara tables
    std::vector<float> profileData;
    std::vector<float> profileSetting;    // [depth0, depthStep, depthN]
    std::vector<float> beamParaData;      // [numLayers][3] -> (2*sigmaIso^2, rtheta, theta2)
    float beamParaPos = 0.0f;

    // Beam geometry on the reference plane (gantry local coordinate system)
    // These are required for CPB/spot processing and ray tracing.
    float3 beamDirection   = {0.0f, 0.0f, 1.0f};
    float3 bmXDirection    = {1.0f, 0.0f, 0.0f};
    float3 bmYDirection    = {0.0f, 1.0f, 0.0f};
    float3 sourcePosition  = {0.0f, 0.0f, 0.0f};
    float  sad             = 0.0f;         // Source-to-axis distance (same unit as grid)
    float  refPlaneZ       = 0.0f;         // Reference plane Z (same unit as grid)

    // Optional ROI mask in linear dose-grid indexing. When provided, the wrapper
    // uses its bounding box projection to tighten the CPB/ray grid while still
    // accumulating dose into the full output dose volume.
    std::vector<int> roiLinearIndices;
};

struct RTDEnergyStruct {
    int nEnergySamples = 0;               // Number of energy bins
    int nEnergies = 0;                    // Number of energies
    std::vector<float> energiesPerU;      // Energy per bin
    std::vector<float> peakDepths;        // Proton penetration depth per bin
    std::vector<float> scaleFacts;        // Scaling factor per bin
    std::vector<float> ciddMatrix;        // 2D matrix of cumulative integral dose
    
    int nDensitySamples = 0;              // Number of density bins
    float densityScaleFact = 0.0f;        // Density scaling factor
    std::vector<float> densityVector;     // Densities for each HU
    
    int nSpSamples = 0;                   // Number of stopping power bins
    float spScaleFact = 0.0f;             // Scaling factor for stopping power
    std::vector<float> spVector;          // Stopping power for each HU
    
    int nRRlSamples = 0;                  // Number of radiation length bins
    float rRlScaleFact = 0.0f;            // Radiation length scaling factor
    std::vector<float> rRlVector;         // Radiation length for each HU

    // Optional public halo / nuclear-correction payload aligned to the active
    // primary energy-depth axis used by ciddMatrix/peakDepths/scaleFacts.
    int nNucEnergySamples = 0;
    int nNucEnergies = 0;
    std::vector<float> nucEnergiesPerU;
    std::vector<float> nucPeakDepths;
    std::vector<float> nucScaleFacts;
    std::vector<float> nucWeightMatrix;
    std::vector<float> nucSqSigmaMatrix;
    bool nuclearTablesAlignedToPrimaryAxis = false;
    bool nuclearTablesResampledFromReference = false;
};

// Main wrapper function declaration
// verbose: 0=no output, 1=fine timing (all details), 2=summary (total time only)
extern "C" void subsecondWrapper(
    const float* ctData, const int3& ctDims, const float3& ctResolution, const float3& ctCorner,
    float* doseData, const int3& doseDims, const float3& doseResolution, const float3& doseCorner,
    const RTDBeamSettings* beamSettings, size_t numBeams,
    const RTDEnergyStruct* energyData,
    int gpuId = 0, bool nuclearCorrection = false, int verbose = 0
);

// Helper functions
extern "C" RTDBeamSettings* createRTDBeamSettings();
extern "C" void destroyRTDBeamSettings(RTDBeamSettings* beam);
extern "C" RTDEnergyStruct* createRTDEnergyStruct();
extern "C" void destroyRTDEnergyStruct(RTDEnergyStruct* energy);

#endif // RAYTRACEDICOM_INTEGRATION_H
