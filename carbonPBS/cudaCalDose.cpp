#include <iostream>
#include "pybind11/include/pybind11/pybind11.h"
#include "pybind11/include/pybind11/numpy.h"
#include "pybind11/include/pybind11/stl.h"
#include <cuda_runtime.h>
#include "math.h"
#include "common/common.cuh"
#include "common/CUDABuffer.cuh"
#include "thrust/sort.h"
#include "thrust/host_vector.h"
#include "thrust/functional.h"
#include "thrust/execution_policy.h"
#include "thrust/reduce.h"
#include <chrono>
#include "common/common_utils.h"

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

#define MAXIMUM_SPOTS_ONE_BATCH 4096
void setROI(vec2i* d_roiFlag, vec3i* d_roiInd, vec3f*, Grid grid, int size);
void RBEMap(int num, Grid doseGrid, vec3i* roiIndex, float* alphamap, float* betamap, float* rbemap, float*, float*,
            float* phyDose,
            int, int gpuid);
void calSqrt(float *first, float *last, float *result);

void calFluenceMapAlphaBeta(float* dalpha,
                            float* dbeta,
                            float* ddosecube,
                            float* ddose,
                            float* weights,
                            float*,
                            float*,
                            vec3i* roiIndex,
                            vec3f* beamDirect,
                            vec3f* dsource,
                            int* drowIndex,
                            int* colPtr,
                            int* deneId,
                            vec3f rayweqSetting,
                            int nSpots,
                            Grid doseGrid,
                            float rs,
                            float estDimension,
                            cudaTextureObject_t rayweqData,
                            cudaTextureObject_t letData,
                            cudaTextureObject_t lqData,
                            int gpuid);

void calFinalDoseAndMKM(float* dose,
                        float* Z1Dmix,
                        vec3f* d_beamDirect,
                        int    num_beam,
                        vec3f  source,
                        vec3f  bmxdir,
                        vec3f  bmydir,
                        vec3i* roiIndex,
                        int    num_roi,
                        Grid   doseGrid,
                        cudaTextureObject_t rayweqData,
                        cudaTextureObject_t iddData,
                        cudaTextureObject_t profileData,
                        cudaTextureObject_t subspotData,
                        cudaTextureObject_t Z1DijData,
                        vec3f  rayweqSetting,
                        vec3f  iddDepth,
                        vec3f  profileDepth,
                        float* d_idbeamxy,
                        int*   d_numParticles,
                        int    nsubspot,
                        int    nGauss,
                        int    eneIdx,
                        int,
                        float  beamParaPos,
                        float  longitudalCutoff,
                        float  transCutoff,
                        float  rtheta,
                        float  theta2,
                        float  r2,
                        float  estDimension,
                        float  sad,
                        float  rs,
                        float  energy,
                        int    gpuid);

void mkmRBEMap(int num,
               Grid doseGrid,
               vec3i* roiIndex,
               float* rbemap1,
               float* rbemap2,
               float* Z1Dmix,
               float* phyDose,
               vec2f alpha0AndBeta0,
               int,
               int gpuid);

void calFinalDose(float* dose,
                  vec3f* d_beamDirect,
                  int num_beam,
                  vec3f source,
                  vec3f bmxdir,
                  vec3f bmydir,
                  vec3i* roiIndex,
                  int num_roi,
                  Grid doseGrid,
                  cudaTextureObject_t rayweqData,
                  cudaTextureObject_t iddData,
                  cudaTextureObject_t profileData,
                  cudaTextureObject_t subspotData,
                  vec3f rayweqSetting,
                  vec3f iddDepth,
                  vec3f profileDepth,
                  float* d_idbeamxy,
                  int* d_numParticles,
                  int nsubspot,
                  int nGauss,
                  int eneIdx,
                  int beamOffset,
                  float beamParaPos,
                  float longitudalCutoff,
                  float transCutoff,
                  float rtheta,
                  float theta2,
                  float r2,
                  float sad,
                  int gpuid);

void calFinalDoseAndRBE(float* dose,
                        float*,
                        float*,
                        float* dalpha,
                        float* dbeta,
                        vec3f* d_beamDirect,
                        int num_beam,
                        vec3f source,
                        vec3f bmxdir,
                        vec3f bmydir,
                        vec3i* roiIndex,
                        int num_roi,
                        Grid doseGrid,
                        cudaTextureObject_t rayweqData,
                        cudaTextureObject_t iddData,
                        cudaTextureObject_t profileData,
                        cudaTextureObject_t subspotData,
                        cudaTextureObject_t letData,
                        cudaTextureObject_t lqData,
                        vec3f rayweqSetting,
                        vec3f iddDepth,
                        vec3f profileDepth,
                        float* d_idbeamxy,
                        int* d_numParticles,
                        int nsubspot,
                        int nGauss,
                        int eneIdx,
                        int,
                        float beamParaPos,
                        float longitudalCutoff,
                        float transCutoff,
                        float rtheta,
                        float theta2,
                        float r2,
                        float estDimension,
                        float sad,
                        float rs,
                        int gpuid);

int rotate3DArray(Grid& new_g,
                  const Grid& old_g,
                  float* h_Array_new,
                  float* h_Array_old,
                  float* d_rot_forward,
                  int,
                  int gpuId);

void calDoseSumSquares(float *doseNorm,
                vec3f *d_beamDirect,
                int num_beam,
                vec3f source,
                vec3f bmxdir,
                vec3f bmydir,
                vec3i *roiIndex,
                int num_roi,
                Grid doseGrid,
                cudaTextureObject_t rayweqData,
                cudaTextureObject_t iddData,
                cudaTextureObject_t profileData,
                cudaTextureObject_t subspotData,
                vec3f rayweqSetting,
                vec3f iddDepth,
                vec3f profileDepth,
                float *d_idbeamxy,
                int nsubspot,
                int nGauss,
                int eneIdx,
                int beamOffset,
                float beamParaPos,
                float longitudalCutoff,
                float transCutoff,
                float rtheta,
                float theta2,
                float r2,
                float sad,
                int gpuid);
int calOneLayerDoseOld(float* dose,
                       int* cscIndices,
                       int* colindices,
                       int* cscIndptr,
                       int totalnnz,
                       int* sumNNZ,
                       vec3f* d_beamDirect,
                       int num_beam,
                       vec3f source,
                       vec3f bmxdir,
                       vec3f bmydir,
                       vec3i* roiIndex,
                       int num_roi,
                       Grid doseGrid,
                       cudaTextureObject_t rayweqData,
                       cudaTextureObject_t iddData,
                       cudaTextureObject_t profileData,
                       cudaTextureObject_t subspotData,
                       vec3f rayweqSetting,
                       vec3f iddDepth,
                       vec3f profileDepth,
                       float* d_idbeamxy,
                       int nsubspot,
                       int nGauss,
                       int eneIdx,
                       int beamOffset,
                       float beamParaPos,
                       float longitudalCutoff,
                       float transCutoff,
                       float,
                       float,
                       float,
                       float sad,
                       int gpuid);

int calOneLayerDoseOld_(float *dose,
					   int *cscIndices,
					   int *colindices,
					   int *cscIndptr,
						 float* d_idd_per_depth, //suplementary
					   int totalnnz,
					   int *sumNNZ,
					   vec3f *d_beamDirect,
					   int num_beam,
					   vec3f source,
					   vec3f bmxdir,
					   vec3f bmydir,
					   vec3i *roiIndex,
					   int num_roi,
					   Grid doseGrid,
					   cudaTextureObject_t rayweqData,
					   cudaTextureObject_t iddData,
					   cudaTextureObject_t profileData,
					   cudaTextureObject_t subspotData,
					   vec3f rayweqSetting,
					   vec3f iddDepth,
					   vec3f profileDepth,
					   float *d_idbeamxy,
					   int nsubspot,
					   int nGauss,
					   int eneIdx,
					   int beamOffset,
					   float beamParaPos,
					   float longitudalCutoff,
					   float transCutoff,
					   float rtheta,
					   float theta2,
					   float r2,
					   float sad,
					   int gpuid);


int binarySearchEneIdx(float ene, float* eneList, int nEne)
{
    int l = 0;
    int r = nEne - 1;
    int mid = 0;
    while (l <= r)
    {
        mid = (l + r) / 2;
        if (eneList[mid] > ene) r = mid - 1;
        else if (eneList[mid] < ene) l = mid + 1;
        else { return mid; }
    }
    return mid;
}

// void calWEQ(pybind11::array ctdata,
//             pybind11::array corner,
//             pybind11::array resolution,
//             pybind11::array dims,
//             pybind11::array source,
//             pybind11::array bmdir,
//             pybind11::array weq,
//             pybind11::array hitPos,
//             pybind11::array hitDis,
//             int nBeam)
// {
//     printf("2024/07/31\n");
//     auto h_ctdata = pybind11::cast<pybind11::array_t<float>>(ctdata).request();
//     auto h_source = pybind11::cast<pybind11::array_t<float>>(source).request();
//     auto h_bmdir = pybind11::cast<pybind11::array_t<float>>(bmdir).request();
//     auto h_corner = pybind11::cast<pybind11::array_t<float>>(corner).request();
//     auto h_resolution = pybind11::cast<pybind11::array_t<float>>(resolution).request();
//     auto h_weq = pybind11::cast<pybind11::array_t<float>>(weq).request();
//     auto h_hitPos = pybind11::cast<pybind11::array_t<float>>(hitPos).request();
//     auto h_dims = pybind11::cast<pybind11::array_t<int>>(dims).request();
//     auto h_hitDis = pybind11::cast<pybind11::array_t<float>>(hitDis).request();

//     Grid doseGrid;
//     memcpy(&(doseGrid.corner), ((vec3f*)h_corner.ptr), sizeof(vec3f));
//     memcpy(&(doseGrid.resolution), ((vec3f*)h_resolution.ptr), sizeof(vec3f));
//     memcpy(&(doseGrid.dims), ((vec3i*)h_dims.ptr), sizeof(vec3i));

//     vector<Beam> allBeams;
//     allBeams.resize(nBeam);

//     for (int i = 0; i < nBeam; i++)
//     {
//         allBeams[i].source.x = ((float*)h_source.ptr)[i];
//         allBeams[i].source.y = ((float*)h_source.ptr)[nBeam + i];
//         allBeams[i].source.z = ((float*)h_source.ptr)[2 * nBeam + i];

//         allBeams[i].bmdir.x = ((float*)h_bmdir.ptr)[i];
//         allBeams[i].bmdir.y = ((float*)h_bmdir.ptr)[nBeam + i];
//         allBeams[i].bmdir.z = ((float*)h_bmdir.ptr)[2 * nBeam + i];
//     }

    // CarbonPBS cp(doseGrid, allBeams, allBeams.size(), 0.000f,
    //              min(min(doseGrid.resolution.x, doseGrid.resolution.y), doseGrid.resolution.z));

    // cp.createScene((float*)h_ctdata.ptr);
    // cp.calWeq();
    //    printf("size:%d________________________\n", cp.weq.size());

    // memcpy((float*)h_weq.ptr, cp.weq.data(), sizeof(float) * cp.weq.size());
    // memcpy((float*)h_hitPos.ptr, (float*)cp.hitPos.data(), sizeof(vec3f) * cp.hitPos.size());
    // memcpy((float*)h_hitDis.ptr, (float*)cp.dis.data(), sizeof(float) * cp.dis.size());
// }
EXPORT
int cudaCalDoseNorm(pybind11::array doseNorm, 
                 pybind11::array weqData,
                 pybind11::array roiIndex,
                 pybind11::array sourceEne, // shape N*1
                 pybind11::array source, // shape N*3, start consider x/y difference
                 pybind11::array bmdir, // shape N*3
                 pybind11::array bmxdir,
                 pybind11::array bmydir,
                 pybind11::array corner,
                 pybind11::array resolution,
                 pybind11::array dims,
                 pybind11::array longitudalCutoff,
                 pybind11::array enelist, // energy list of machine(机器能打的所有能量)
                 pybind11::array idddata,
                 pybind11::array iddsetting,
                 pybind11::array profiledata,
                 pybind11::array profilesetting,
                 pybind11::array beamparadata, // shape K*3
                 pybind11::array subSpotData, // sub spots parameters(子束分解sigma数据)
                 pybind11::array layerInfo, // how many spots for each layer
                 pybind11::array layerEnergy, // energy for each layer
                 pybind11::array idbeamxy, // 子束在V平面上的位置
                 float sad,
                 float cutoff,
                 float beamParaPos,
                 int gpuId
)
{
    int returnFlag = 1;
    checkCudaErrors(cudaSetDevice(gpuId));

    auto h_doseNorm = pybind11::cast<pybind11::array_t<float>>(doseNorm).request();

    auto h_sourceEne = pybind11::cast<pybind11::array_t<float>>(sourceEne).request();
    auto h_source = pybind11::cast<pybind11::array_t<float>>(source).request();
    auto h_bmdir = pybind11::cast<pybind11::array_t<float>>(bmdir).request();
    auto h_bmxdir = pybind11::cast<pybind11::array_t<float>>(bmxdir).request();
    auto h_bmydir = pybind11::cast<pybind11::array_t<float>>(bmydir).request();

    auto h_corner = pybind11::cast<pybind11::array_t<float>>(corner).request();
    auto h_resolution = pybind11::cast<pybind11::array_t<float>>(resolution).request();
    auto h_longitudalCutoff = pybind11::cast<pybind11::array_t<float>>(longitudalCutoff).request();
    auto h_dims = pybind11::cast<pybind11::array_t<int>>(dims).request();

    auto h_enelist = pybind11::cast<pybind11::array_t<float>>(enelist).request();
    auto h_idddata = pybind11::cast<pybind11::array_t<float>>(idddata).request();
    auto h_iddsetting = pybind11::cast<pybind11::array_t<float>>(iddsetting).request();
    auto h_profiledata = pybind11::cast<pybind11::array_t<float>>(profiledata).request();
    auto h_profilesetting = pybind11::cast<pybind11::array_t<float>>(profilesetting).request();
    auto h_beamparadata = pybind11::cast<pybind11::array_t<float>>(beamparadata).request();
    auto h_subspotdata = pybind11::cast<pybind11::array_t<float>>(subSpotData).request();
    auto h_layer_info = pybind11::cast<pybind11::array_t<int>>(layerInfo).request();
    auto h_layer_energy = pybind11::cast<pybind11::array_t<float>>(layerEnergy).request();
    auto h_weq = pybind11::cast<pybind11::array_t<float>>(weqData).request();
    auto h_roiIndex = pybind11::cast<pybind11::array_t<int>>(roiIndex).request();
    auto h_idbeamxy = pybind11::cast<pybind11::array_t<float>>(idbeamxy).request();

    int nBeam = h_sourceEne.size;
    int nEne = h_enelist.size;
    int nRoi = h_roiIndex.size / 3;

    int beamOffset = 0;
    int maximumLayerSize = thrust::reduce((int*)h_layer_info.ptr, (int*)h_layer_info.ptr + layerInfo.size(), -1,
                                          thrust::maximum<int>());

    int nsubspot = h_subspotdata.shape[1];
    int nProfilePara = h_profiledata.shape[2];
    int nGauss = (nProfilePara - 1) / 2;

    vec3f bmxDir = vec3f(((float*)h_bmxdir.ptr)[0], ((float*)h_bmxdir.ptr)[1], ((float*)h_bmxdir.ptr)[2]);
    vec3f bmyDir = vec3f(((float*)h_bmydir.ptr)[0], ((float*)h_bmydir.ptr)[1], ((float*)h_bmydir.ptr)[2]);
    vec3f sourcePos = vec3f(((float*)h_source.ptr)[0], ((float*)h_source.ptr)[1], ((float*)h_source.ptr)[2]);

    Grid ctGrid;
    vec3f iddDepth, profileDepth, rayweqSetting;

    memcpy(&(ctGrid.corner), ((vec3f*)h_corner.ptr), sizeof(vec3f));
    memcpy(&(ctGrid.resolution), ((vec3f*)h_resolution.ptr), sizeof(vec3f));
    memcpy(&(ctGrid.dims), ((vec3i*)h_dims.ptr), sizeof(vec3i));
    memcpy(&iddDepth, (float*)h_iddsetting.ptr, sizeof(vec3f));
    memcpy(&profileDepth, (float*)h_profilesetting.ptr, sizeof(vec3f));
    memcpy(&rayweqSetting, ((float*)h_weq.ptr), sizeof(vec3f));

    int nStep = ((float*)h_weq.ptr)[2];
    int ny = ((float*)h_weq.ptr)[5];
    int nx = ((float*)h_weq.ptr)[8];

    int gridSize = ctGrid.dims.x * ctGrid.dims.y * ctGrid.dims.z;

    cudaArray_t iddArray;
    cudaTextureObject_t iddObj = 0;
    create2DTexture((float*)h_idddata.ptr, &iddArray, &iddObj, weq::LINEAR,
                    weq::R, h_idddata.shape[1], h_idddata.shape[0], gpuId);

    cudaArray_t profileArray;
    cudaTextureObject_t profileObj = 0;
    create3DTexture((float*)h_profiledata.ptr, &profileArray, &profileObj, weq::LINEAR,
                    weq::R, h_profiledata.shape[2], h_profiledata.shape[1], h_profiledata.shape[0], gpuId);

    cudaArray_t subSpotDataArray;
    cudaTextureObject_t subSpotDataObj = 0;
    create3DTexture((float*)h_subspotdata.ptr, &subSpotDataArray, &subSpotDataObj, weq::POINTS,
                    weq::R, h_subspotdata.shape[2], h_subspotdata.shape[1], h_subspotdata.shape[0], gpuId);

    cudaArray_t rayweqArray;
    cudaTextureObject_t rayweqObj = 0;
    create3DTexture((float*)h_weq.ptr + 9, &rayweqArray, &rayweqObj, weq::LINEAR,
                    weq::R, nStep, ny, nx, gpuId);

    float* d_doseNorm;
    vec3i* d_roiInd;
    vec3f* d_bmzDir;
    float* d_idbeamxy;

    checkCudaErrors(cudaMalloc((void **)&d_doseNorm, sizeof(float) * nRoi));
    checkCudaErrors(cudaMalloc((void **)&d_roiInd, sizeof(vec3i) * h_roiIndex.size / 3));
    checkCudaErrors(cudaMalloc((void **)&d_bmzDir, sizeof(vec3f) * maximumLayerSize));
    checkCudaErrors(cudaMalloc((void **)&d_idbeamxy, sizeof(float) * maximumLayerSize * 2));
    checkCudaErrors(cudaMemcpy(d_roiInd, (vec3i *)h_roiIndex.ptr, sizeof(vec3i) * h_roiIndex.size / 3, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemset(d_doseNorm, 0, sizeof(float) * nRoi));

    for (int i = 0; i < layerInfo.size(); i++) {
        int layerSize = ((int*)h_layer_info.ptr)[i];
        int energyIdx = binarySearchEneIdx(((float*)h_layer_energy.ptr)[i], ((float*)h_enelist.ptr), nEne);

        float rtheta, theta2, longitudalCutoff, r2;
        r2     = ((float*)h_beamparadata.ptr)[energyIdx * 3];
        rtheta = ((float*)h_beamparadata.ptr)[energyIdx * 3 + 1];
        theta2 = ((float*)h_beamparadata.ptr)[energyIdx * 3 + 2];
        longitudalCutoff = *((float*)h_longitudalCutoff.ptr + beamOffset);

        // checkCudaErrors(cudaMemset(d_bmzDir, 0, sizeof(vec3f) * maximumLayerSize));
        checkCudaErrors(cudaMemcpy(d_bmzDir, 
            (vec3f *)h_bmdir.ptr + beamOffset, 
            sizeof(vec3f) * layerSize, 
            cudaMemcpyHostToDevice));

        checkCudaErrors(cudaMemcpy(d_idbeamxy, 
            (float *)h_idbeamxy.ptr + beamOffset * 2, 
            sizeof(float) * 2 * layerSize,
            cudaMemcpyHostToDevice));

        calDoseSumSquares(d_doseNorm,
                d_bmzDir,
                layerSize,
                sourcePos,
                bmxDir,
                bmyDir,
                d_roiInd,
                nRoi,
                ctGrid,
                rayweqObj,
                iddObj,
                profileObj,
                subSpotDataObj,
                rayweqSetting,
                iddDepth,
                profileDepth,
                d_idbeamxy,
                nsubspot,
                nGauss,
                energyIdx,
                beamOffset,
                beamParaPos,
                longitudalCutoff,
                cutoff,
                rtheta,
                theta2,
                r2,
                sad,
                gpuId);

        beamOffset += layerSize;
    }
    calSqrt(d_doseNorm, d_doseNorm + nRoi, d_doseNorm);
    checkCudaErrors(cudaMemcpy((float*)h_doseNorm.ptr, d_doseNorm, sizeof(float) * nRoi, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_doseNorm));
    checkCudaErrors(cudaFree(d_roiInd));
    checkCudaErrors(cudaFree(d_bmzDir));
    checkCudaErrors(cudaFree(d_idbeamxy));

    checkCudaErrors(cudaFreeArray(iddArray));
    checkCudaErrors(cudaFreeArray(profileArray));
    checkCudaErrors(cudaFreeArray(subSpotDataArray));
    checkCudaErrors(cudaFreeArray(rayweqArray));

    checkCudaErrors(cudaDestroyTextureObject(iddObj));
    checkCudaErrors(cudaDestroyTextureObject(profileObj));
    checkCudaErrors(cudaDestroyTextureObject(subSpotDataObj));
    checkCudaErrors(cudaDestroyTextureObject(rayweqObj));
    return returnFlag;
}
EXPORT
void cudaCalFluenceRBE(pybind11::array out_alpha,
                       pybind11::array out_beta,
                       pybind11::array out_rbe1,
                       pybind11::array out_rbe2,
                       pybind11::array in_phyDose,
                       pybind11::array roiIndex,
                       pybind11::array counter,
                       pybind11::array corner,
                       pybind11::array resolution,
                       pybind11::array dims,
                       int nField,
                       int gpuId) {
    checkCudaErrors(cudaSetDevice(gpuId));
    auto h_out_alpha  = pybind11::cast<pybind11::array_t<float>>(out_alpha).request();
    auto h_out_beta   = pybind11::cast<pybind11::array_t<float>>(out_beta).request();
    auto h_out_rbe1   = pybind11::cast<pybind11::array_t<float>>(out_rbe1).request();
    auto h_out_rbe2   = pybind11::cast<pybind11::array_t<float>>(out_rbe2).request();
    auto h_counter    = pybind11::cast<pybind11::array_t<float>>(counter).request();
    auto h_in_phyDose = pybind11::cast<pybind11::array_t<float>>(in_phyDose).request();
    auto h_in_roi     = pybind11::cast<pybind11::array_t<int>>(roiIndex).request();

    auto h_corner     = pybind11::cast<pybind11::array_t<float>>(corner).request();
    auto h_resolution = pybind11::cast<pybind11::array_t<float>>(resolution).request();
    auto h_dims       = pybind11::cast<pybind11::array_t<int>>(dims).request();

    float *d_alpha, *d_beta, *d_rbe1, *d_rbe2, *d_counter, *d_physDose;
    vec3i *d_roiInd;

    Grid ctGrid;
    memcpy(&(ctGrid.corner), ((vec3f*)h_corner.ptr), sizeof(vec3f));
    memcpy(&(ctGrid.resolution), ((vec3f*)h_resolution.ptr), sizeof(vec3f));
    memcpy(&(ctGrid.dims), ((vec3i*)h_dims.ptr), sizeof(vec3i));
    int gridSize = ctGrid.dims.x * ctGrid.dims.y * ctGrid.dims.z;
    int nRoi  = h_in_roi.size / 3;

    checkCudaErrors(cudaMalloc((void**)&d_alpha,       sizeof(float) * ；gridSize));
    checkCudaErrors(cudaMalloc((void**)&d_beta,        sizeof(float) * gridSize));         //store LET
    checkCudaErrors(cudaMalloc((void**)&d_rbe1,        sizeof(float) * gridSize));
    checkCudaErrors(cudaMalloc((void**)&d_rbe2,        sizeof(float) * gridSize));
    checkCudaErrors(cudaMalloc((void**)&d_counter,     sizeof(float) * gridSize));
    checkCudaErrors(cudaMalloc((void**)&d_physDose,    sizeof(float) * gridSize * nField));
    checkCudaErrors(cudaMalloc((void**)&d_roiInd,      sizeof(vec3i) * (nRoi)));

    checkCudaErrors(cudaMemcpy(d_alpha,    (float *)h_out_alpha.ptr, sizeof(float) * gridSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_beta,     (float *)h_out_beta.ptr, sizeof(float) * gridSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_rbe1,     (float *)h_out_rbe1.ptr, sizeof(float) * gridSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_rbe2,     (float *)h_out_rbe2.ptr, sizeof(float) * gridSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_counter,  (float *)h_counter.ptr, sizeof(float) * gridSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_physDose, (float *)h_in_phyDose.ptr, sizeof(float) * gridSize * nField, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_roiInd,   (vec3i *)h_in_roi.ptr, sizeof(vec3i) * nRoi, cudaMemcpyHostToDevice));

    RBEMap(nRoi, ctGrid, d_roiInd, d_alpha, d_beta, d_rbe1, d_rbe2, d_counter, d_physDose, nField, gpuId);

    checkCudaErrors(cudaMemcpy((float *)h_out_rbe1.ptr, d_rbe1, sizeof(float) * gridSize, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy((float *)h_out_rbe2.ptr, d_rbe2, sizeof(float) * gridSize, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_alpha));
    checkCudaErrors(cudaFree(d_beta));
    checkCudaErrors(cudaFree(d_rbe1));
    checkCudaErrors(cudaFree(d_rbe2));
    checkCudaErrors(cudaFree(d_counter));
    checkCudaErrors(cudaFree(d_physDose));
    checkCudaErrors(cudaFree(d_roiInd));
}

void cudaCalFluenceMapAlphaBeta(pybind11::array out_finalDose,
                         pybind11::array out_alpha,
                         pybind11::array out_beta,
                         pybind11::array out_LET,
                         pybind11::array in_nnzDose,
                         pybind11::array in_rowIndex,
                         pybind11::array in_colPtr,
                         pybind11::array in_weights,
                         pybind11::array weqData,
                         pybind11::array roiIndex,
                         pybind11::array sourceEneId,     // shape N*1
                         pybind11::array source,          // shape N*3, start consider x/y difference
                         pybind11::array bmdir,           // shape N*3
                         pybind11::array corner,
                         pybind11::array resolution,
                         pybind11::array dims,
                         pybind11::array letData,
                         pybind11::array lqData,
                         unsigned long NNZ,
                         float rs,
                         int gpuId
)
{
    printf("CalFluenceMapRBE  20250521 %d\n", gpuId);
    checkCudaErrors(cudaSetDevice(gpuId));
    auto h_out_final  = pybind11::cast<pybind11::array_t<float>>(out_finalDose).request();
    auto h_out_alpha  = pybind11::cast<pybind11::array_t<float>>(out_alpha).request();
    auto h_out_beta   = pybind11::cast<pybind11::array_t<float>>(out_beta).request();
    auto h_out_let    = pybind11::cast<pybind11::array_t<float>>(out_LET).request();

    auto h_in_cscdata = pybind11::cast<pybind11::array_t<float>>(in_nnzDose).request();
    auto h_in_rowind  = pybind11::cast<pybind11::array_t<int>>(in_rowIndex).request();
    auto h_in_colptr  = pybind11::cast<pybind11::array_t<int>>(in_colPtr).request();

    auto h_in_weights = pybind11::cast<pybind11::array_t<float>>(in_weights).request();
    auto h_in_weq     = pybind11::cast<pybind11::array_t<float>>(weqData).request();
    auto h_in_roi     = pybind11::cast<pybind11::array_t<int>>(roiIndex).request();
    auto h_in_eneid   = pybind11::cast<pybind11::array_t<int>>(sourceEneId).request();
    auto h_in_source  = pybind11::cast<pybind11::array_t<float>>(source).request();
    auto h_in_bmdir   = pybind11::cast<pybind11::array_t<float>>(bmdir).request();

    auto h_corner     = pybind11::cast<pybind11::array_t<float>>(corner).request();
    auto h_resolution = pybind11::cast<pybind11::array_t<float>>(resolution).request();
    auto h_dims       = pybind11::cast<pybind11::array_t<int>>(dims).request();

    auto h_let        = pybind11::cast<pybind11::array_t<float>>(letData).request();
    auto h_lq         = pybind11::cast<pybind11::array_t<float>>(lqData).request();

    float *d_final, *d_rbe2, *d_nnzValue, *d_weights, *dalpha, *dbeta, *d_counter;
    int   *d_eneId, *d_rowIndex, *d_colPtr;
    vec3i *d_roi;
    vec3f *d_source, *d_bmdir;

    int nRoi  = h_in_roi.size / 3;
    int nBeam = h_in_eneid.size;
    vec3f rayweqSetting;
    Grid ctGrid;

    memcpy(&(ctGrid.corner), ((vec3f*)h_corner.ptr), sizeof(vec3f));
    memcpy(&(ctGrid.resolution), ((vec3f*)h_resolution.ptr), sizeof(vec3f));
    memcpy(&(ctGrid.dims), ((vec3i*)h_dims.ptr), sizeof(vec3i));
    memcpy(&rayweqSetting, ((float*)h_in_weq.ptr), sizeof(vec3f));

    float estDimension = cbrtf(ctGrid.resolution.x * ctGrid.resolution.y * ctGrid.resolution.z);

    int gridSize = ctGrid.dims.x * ctGrid.dims.y * ctGrid.dims.z;

    cudaArray_t rayweqArray;
    cudaTextureObject_t rayweqObj = 0;

    int nStep = ((float*)h_in_weq.ptr)[2];
    create2DTexture((float*)h_in_weq.ptr + 9, &rayweqArray, &rayweqObj, weq::LINEAR, weq::R, nStep, nBeam, gpuId);

    cudaArray_t lqDataArray;
    cudaTextureObject_t lqDataObj = 0;
    create3DTexture((float*)h_lq.ptr, &lqDataArray, &lqDataObj, weq::LINEAR,
                    weq::R, h_lq.shape[2], h_lq.shape[1], h_lq.shape[0], gpuId);

    cudaArray_t letArray;
    cudaTextureObject_t letObj = 0;
    create2DTexture((float *)h_let.ptr, &letArray, &letObj, weq::LINEAR, weq::R, h_let.shape[1], h_let.shape[0], gpuId);

    checkCudaErrors(cudaMalloc((void**)&d_final,      sizeof(float) * gridSize));
    checkCudaErrors(cudaMalloc((void**)&d_rbe2,        sizeof(float) * gridSize));    //store LET
    checkCudaErrors(cudaMalloc((void**)&d_counter,     sizeof(float) * gridSize));
    checkCudaErrors(cudaMalloc((void**)&dalpha,        sizeof(float) * gridSize));
    checkCudaErrors(cudaMalloc((void**)&dbeta,        sizeof(float) * gridSize));

    checkCudaErrors(cudaMalloc((void**)&d_nnzValue,   sizeof(float) * NNZ));
    checkCudaErrors(cudaMalloc((void**)&d_rowIndex,   sizeof(int) * NNZ));
    checkCudaErrors(cudaMalloc((void**)&d_colPtr,   sizeof(int) * (nBeam + 1)));
    checkCudaErrors(cudaMalloc((void**)&d_weights,   sizeof(float) * (nBeam)));
    checkCudaErrors(cudaMalloc((void**)&d_eneId,   sizeof(int) * (nBeam)));
    checkCudaErrors(cudaMalloc((void**)&d_source,   sizeof(vec3f) * (nBeam)));
    checkCudaErrors(cudaMalloc((void**)&d_bmdir,   sizeof(vec3f) * (nBeam)));
    checkCudaErrors(cudaMalloc((void**)&d_roi,   sizeof(vec3i) * (nRoi)));

    checkCudaErrors(cudaMemcpy(d_nnzValue, (float *)h_in_cscdata.ptr, sizeof(float)*NNZ, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_rowIndex, (int *)h_in_rowind.ptr, sizeof(int)*NNZ, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_colPtr, (int *)h_in_colptr.ptr, sizeof(int)*(nBeam + 1), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dalpha, (float *)h_out_alpha.ptr, sizeof(float)*(gridSize), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dbeta, (float *)h_out_beta.ptr, sizeof(float)*(gridSize), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_final, (float *)h_out_final.ptr, sizeof(float)*(gridSize), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_rbe2, (float *)h_out_let.ptr, sizeof(float)*(gridSize), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_weights, (float *)h_in_weights.ptr, sizeof(float)*(nBeam), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_roi, (vec3i *)h_in_roi.ptr, sizeof(vec3i)*(nRoi), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_source, (vec3f *)h_in_source.ptr, sizeof(vec3f)*(nBeam), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_bmdir, (vec3f *)h_in_bmdir.ptr, sizeof(vec3f)*(nBeam), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_eneId, (int *)h_in_eneid.ptr, sizeof(int)*(nBeam), cudaMemcpyHostToDevice));

    calFluenceMapAlphaBeta(dalpha, dbeta, d_final, d_nnzValue, d_weights, d_rbe2, d_counter, d_roi,
                           d_bmdir, d_source, d_rowIndex,
                           d_colPtr, d_eneId, rayweqSetting, nBeam, ctGrid, rs, estDimension, rayweqObj, letObj,
                           lqDataObj, gpuId);

    checkCudaErrors(cudaMemcpy((float *)h_out_final.ptr, d_final, sizeof(float) * gridSize, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy((float *)h_out_let.ptr, d_rbe2, sizeof(float) * gridSize, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy((float *)h_out_alpha.ptr, dalpha, sizeof(float) * gridSize, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy((float *)h_out_beta.ptr, dbeta, sizeof(float) * gridSize, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(dalpha));
    checkCudaErrors(cudaFree(dbeta));
    checkCudaErrors(cudaFree(d_final));
    checkCudaErrors(cudaFree(d_nnzValue));
    checkCudaErrors(cudaFree(d_rowIndex));
    checkCudaErrors(cudaFree(d_colPtr));
    checkCudaErrors(cudaFree(d_weights));
    checkCudaErrors(cudaFree(d_roi));
    checkCudaErrors(cudaFree(d_source));
    checkCudaErrors(cudaFree(d_bmdir));
    checkCudaErrors(cudaFree(d_rbe2));
    checkCudaErrors(cudaFree(d_counter));
    checkCudaErrors(cudaFree(d_eneId));
}

int cudaCalDose3(pybind11::array out_cscdata,
                 pybind11::array out_cscptr,
                 pybind11::array out_cscrowind,
                 pybind11::array weqData,
                 pybind11::array roiIndex,
                 pybind11::array sourceEne, // shape N*1
                 pybind11::array source, // shape N*3, start consider x/y difference
                 pybind11::array bmdir, // shape N*3
                 pybind11::array bmxdir,
                 pybind11::array bmydir,
                 pybind11::array corner,
                 pybind11::array resolution,
                 pybind11::array dims,
                 pybind11::array longitudalCutoff,
                 pybind11::array enelist,
                 pybind11::array idddata,
                 pybind11::array iddsetting,
                 pybind11::array profiledata,
                 pybind11::array profilesetting,
                 pybind11::array beamparadata, // shape K*3
                 pybind11::array subSpotData,
                 pybind11::array layerInfo, // how many spots for each layer
                 pybind11::array layerEnergy, // energy for each layer
                 pybind11::array outNNZ,
                 pybind11::array idbeamxy,
                 float sad,
                 float cutoff,
                 float beamParaPos,
                 long long pythonNNZSize,
                 int gpuId
)
{
    printf("2024/09/21PM!!!!\n");

    checkCudaErrors(cudaSetDevice(gpuId));
    auto h_out_cscdata1 = pybind11::cast<pybind11::array_t<float>>(out_cscdata).request();
    auto h_out_cscptr = pybind11::cast<pybind11::array_t<int>>(out_cscptr).request();
    auto h_out_cscrowind1 = pybind11::cast<pybind11::array_t<int>>(out_cscrowind).request();

    auto h_sourceEne = pybind11::cast<pybind11::array_t<float>>(sourceEne).request();
    auto h_source = pybind11::cast<pybind11::array_t<float>>(source).request();
    auto h_bmdir = pybind11::cast<pybind11::array_t<float>>(bmdir).request();
    auto h_bmxdir = pybind11::cast<pybind11::array_t<float>>(bmxdir).request();
    auto h_bmydir = pybind11::cast<pybind11::array_t<float>>(bmydir).request();

    auto h_corner = pybind11::cast<pybind11::array_t<float>>(corner).request();
    auto h_resolution = pybind11::cast<pybind11::array_t<float>>(resolution).request();
    auto h_longitudalCutoff = pybind11::cast<pybind11::array_t<float>>(longitudalCutoff).request();
    auto h_dims = pybind11::cast<pybind11::array_t<int>>(dims).request();

    auto h_enelist = pybind11::cast<pybind11::array_t<float>>(enelist).request();
    auto h_idddata = pybind11::cast<pybind11::array_t<float>>(idddata).request();
    auto h_iddsetting = pybind11::cast<pybind11::array_t<float>>(iddsetting).request();
    auto h_profiledata = pybind11::cast<pybind11::array_t<float>>(profiledata).request();
    auto h_profilesetting = pybind11::cast<pybind11::array_t<float>>(profilesetting).request();
    auto h_beamparadata = pybind11::cast<pybind11::array_t<float>>(beamparadata).request();
    auto h_subspotdata = pybind11::cast<pybind11::array_t<float>>(subSpotData).request();
    auto h_layer_info = pybind11::cast<pybind11::array_t<int>>(layerInfo).request();
    auto h_layer_energy = pybind11::cast<pybind11::array_t<float>>(layerEnergy).request();
    auto h_weq = pybind11::cast<pybind11::array_t<float>>(weqData).request();
    auto h_roiIndex = pybind11::cast<pybind11::array_t<int>>(roiIndex).request();
    auto h_outNNZ = pybind11::cast<pybind11::array_t<uint64_t>>(outNNZ).request();
    auto h_idbeamxy = pybind11::cast<pybind11::array_t<float>>(idbeamxy).request();

    int nBeam = h_sourceEne.size;
    int nEne = h_enelist.size;
    int nRoi = h_roiIndex.size / 3;


    int beamOffset = 0;
    uint64_t nnzOffset = 0;
    int maximumLayerSize = thrust::reduce((int*)h_layer_info.ptr, (int*)h_layer_info.ptr + layerInfo.size(), -1,
                                          thrust::maximum<int>());

    int nsubspot = h_subspotdata.shape[1];
    int nProfilePara = h_profiledata.shape[2];
    int nGauss = (nProfilePara - 1) / 2;

    vec3f bmxDir = vec3f(((float*)h_bmxdir.ptr)[0], ((float*)h_bmxdir.ptr)[1], ((float*)h_bmxdir.ptr)[2]);
    vec3f bmyDir = vec3f(((float*)h_bmydir.ptr)[0], ((float*)h_bmydir.ptr)[1], ((float*)h_bmydir.ptr)[2]);
    vec3f sourcePos = vec3f(((float*)h_source.ptr)[0], ((float*)h_source.ptr)[1], ((float*)h_source.ptr)[2]);

    Grid ctGrid;
    vec3f iddDepth, profileDepth, rayweqSetting;

    memcpy(&(ctGrid.corner), ((vec3f*)h_corner.ptr), sizeof(vec3f));
    memcpy(&(ctGrid.resolution), ((vec3f*)h_resolution.ptr), sizeof(vec3f));
    memcpy(&iddDepth, (float*)h_iddsetting.ptr, sizeof(vec3f));
    memcpy(&profileDepth, (float*)h_profilesetting.ptr, sizeof(vec3f));
    memcpy(&(ctGrid.dims), ((vec3i*)h_dims.ptr), sizeof(vec3i));
    memcpy(&rayweqSetting, ((float*)h_weq.ptr), sizeof(vec3f));

    int nStep = ((float*)h_weq.ptr)[2];
    int ny = ((float*)h_weq.ptr)[5];
    int nx = ((float*)h_weq.ptr)[8];

    cudaArray_t iddArray;
    cudaTextureObject_t iddObj = 0;
    create2DTexture((float*)h_idddata.ptr, &iddArray, &iddObj, weq::LINEAR,
                    weq::R, h_idddata.shape[1], h_idddata.shape[0], gpuId);

    cudaArray_t profileArray;
    cudaTextureObject_t profileObj = 0;
    create3DTexture((float*)h_profiledata.ptr, &profileArray, &profileObj, weq::LINEAR,
                    weq::R, h_profiledata.shape[2], h_profiledata.shape[1], h_profiledata.shape[0], gpuId);

    cudaArray_t subSpotDataArray;
    cudaTextureObject_t subSpotDataObj = 0;
    create3DTexture((float*)h_subspotdata.ptr, &subSpotDataArray, &subSpotDataObj, weq::POINTS,
                    weq::R, h_subspotdata.shape[2], h_subspotdata.shape[1], h_subspotdata.shape[0], gpuId);

    cudaArray_t rayweqArray;
    cudaTextureObject_t rayweqObj = 0;
    create3DTexture((float*)h_weq.ptr + 9, &rayweqArray, &rayweqObj, weq::LINEAR,
                    weq::R, nStep, ny, nx, gpuId);

    int*   d_sumNNZ;
    int*   d_tmpRowIndex;
    int*   d_tmpColIndex;
    int*   d_tmpCscPtr;
    vec3i* d_roiInd;
    vec3f* d_bmzDir;
    float* d_tmpValues;
    float* d_idbeamxy;

    float sparseRatio = 1.0f;
    size_t nnzSizeThisLayer = size_t((float)maximumLayerSize * (float)nRoi * sparseRatio);

    if (nnzSizeThisLayer * 4 >= 4294967295)
        nnzSizeThisLayer = size_t(4294967295 / 4);

    checkCudaErrors(cudaMalloc((void **)&d_roiInd, sizeof(vec3i) * h_roiIndex.size / 3));
    checkCudaErrors(cudaMalloc((void **)&d_sumNNZ, sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_tmpCscPtr, sizeof(int) * (maximumLayerSize + 1)));
    checkCudaErrors(cudaMalloc((void **)&d_bmzDir, sizeof(vec3f) * maximumLayerSize));
    checkCudaErrors(cudaMalloc((void **)&d_idbeamxy, sizeof(float) * maximumLayerSize * 2));

    printf("%zu\n", nnzSizeThisLayer);

    checkCudaErrors(cudaMalloc((void **)&d_tmpValues, sizeof(float) * nnzSizeThisLayer));
    checkCudaErrors(cudaMalloc((void **)&d_tmpRowIndex, sizeof(int) * nnzSizeThisLayer));
    checkCudaErrors(cudaMalloc((void **)&d_tmpColIndex, sizeof(int) * nnzSizeThisLayer));

    checkCudaErrors(
        cudaMemcpy(d_roiInd, (vec3i *)h_roiIndex.ptr, sizeof(vec3i) * h_roiIndex.size / 3, cudaMemcpyHostToDevice));


    int returnFlag = 1;
    int spots_simulated = 0;

    for (int i = 0; i < layerInfo.size(); i++)
    {
        int layerSize = ((int*)h_layer_info.ptr)[i];
        int energyIdx = binarySearchEneIdx(((float*)h_layer_energy.ptr)[i], ((float*)h_enelist.ptr), nEne);

        float rtheta, theta2, longitudalCutoff, r2;
        r2     = ((float*)h_beamparadata.ptr)[energyIdx * 3];
        rtheta = ((float*)h_beamparadata.ptr)[energyIdx * 3 + 1];
        theta2 = ((float*)h_beamparadata.ptr)[energyIdx * 3 + 2];
        longitudalCutoff = *((float*)h_longitudalCutoff.ptr + beamOffset);

        int spots_left_this_layer = layerSize;
        while (spots_left_this_layer > 0)
        {
            int spot_size_this_batch = 0;
            if (spots_left_this_layer >= MAXIMUM_SPOTS_ONE_BATCH)
            {
                spot_size_this_batch = MAXIMUM_SPOTS_ONE_BATCH;
            }
            else
            {
                spot_size_this_batch = spots_left_this_layer;
            }
            checkCudaErrors(cudaMemset(d_tmpCscPtr, 0, sizeof(int) * (maximumLayerSize + 1)));
            checkCudaErrors(cudaMemset(d_bmzDir, 0, sizeof(vec3f) * maximumLayerSize));
            checkCudaErrors(cudaMemset(d_tmpColIndex, 0, sizeof(int) * nnzSizeThisLayer));
            checkCudaErrors(cudaMemset(d_tmpRowIndex, 0, sizeof(int) * nnzSizeThisLayer));
            checkCudaErrors(cudaMemset(d_tmpValues, 0, sizeof(float) * nnzSizeThisLayer));
            checkCudaErrors(cudaMemset(d_sumNNZ, 0, sizeof(int))); // reset sumNNZ value to 0

            checkCudaErrors(cudaMemcpy(d_bmzDir,
                (vec3f *)h_bmdir.ptr + beamOffset,
                sizeof(vec3f) * spot_size_this_batch,
                cudaMemcpyHostToDevice));

            checkCudaErrors(
                cudaMemcpy(d_idbeamxy, (float *)h_idbeamxy.ptr + beamOffset * 2, sizeof(float) * 2 *
                    spot_size_this_batch,
                    cudaMemcpyHostToDevice));

            int nnz = calOneLayerDoseOld(d_tmpValues,
                                         d_tmpRowIndex,
                                         d_tmpColIndex,
                                         d_tmpCscPtr,
                                         nnzSizeThisLayer,
                                         d_sumNNZ,
                                         d_bmzDir,
                                         spot_size_this_batch,
                                         sourcePos,
                                         bmxDir,
                                         bmyDir,
                                         d_roiInd,
                                         nRoi,
                                         ctGrid,
                                         rayweqObj,
                                         iddObj,
                                         profileObj,
                                         subSpotDataObj,
                                         rayweqSetting,
                                         iddDepth,
                                         profileDepth,
                                         d_idbeamxy,
                                         nsubspot,
                                         nGauss,
                                         energyIdx,
                                         beamOffset,
                                         beamParaPos,
                                         longitudalCutoff,
                                         cutoff,
                                         rtheta,
                                         theta2,
                                         r2,
                                         sad,
                                         gpuId
            );
            spots_simulated += spot_size_this_batch;
            printf("spots simulated = %d ", spots_simulated);

            if (nnz > nnzSizeThisLayer)
            {
                printf("nnz larger than nnz size this batch....!\n");
                throw -1;
            }
            if (nnzOffset + nnz < pythonNNZSize)
            {
                checkCudaErrors(
                    cudaMemcpy((float *)h_out_cscdata1.ptr + nnzOffset, d_tmpValues, sizeof(float) * nnz,
                        cudaMemcpyDeviceToHost));
                checkCudaErrors(
                    cudaMemcpy((int *)h_out_cscrowind1.ptr + nnzOffset, d_tmpRowIndex, sizeof(int) * nnz,
                        cudaMemcpyDeviceToHost));
                checkCudaErrors(
                    cudaMemcpy((int *)h_out_cscptr.ptr + 1 + beamOffset, d_tmpCscPtr + 1, sizeof(int) * spot_size_this_batch,
                        cudaMemcpyDeviceToHost));
            }
            else
            {
                returnFlag = -1;
                break;
            }
            beamOffset += spot_size_this_batch;
            nnzOffset += nnz;
            spots_left_this_layer -= spot_size_this_batch;
            printf("layer:%d spots left this layer:%d ", i, spots_left_this_layer);
            printf("nnz offset:%lu nnz size:%u\n", nnzOffset, nnz);
        }
    }
    checkCudaErrors(cudaFree(d_roiInd));
    checkCudaErrors(cudaFree(d_sumNNZ));
    checkCudaErrors(cudaFree(d_bmzDir));
    checkCudaErrors(cudaFree(d_tmpValues));
    checkCudaErrors(cudaFree(d_tmpRowIndex));
    checkCudaErrors(cudaFree(d_tmpColIndex));
    checkCudaErrors(cudaFree(d_tmpCscPtr));
    checkCudaErrors(cudaFree(d_idbeamxy));

    checkCudaErrors(cudaFreeArray(iddArray));
    checkCudaErrors(cudaFreeArray(profileArray));
    checkCudaErrors(cudaFreeArray(subSpotDataArray));
    checkCudaErrors(cudaFreeArray(rayweqArray));

    checkCudaErrors(cudaDestroyTextureObject(iddObj));
    checkCudaErrors(cudaDestroyTextureObject(profileObj));
    checkCudaErrors(cudaDestroyTextureObject(subSpotDataObj));
    checkCudaErrors(cudaDestroyTextureObject(rayweqObj));
    //  delete[] h_cscPtr;
    ((uint64_t*)(h_outNNZ.ptr))[0] = nnzOffset;
    return returnFlag;
}

EXPORT
int cudaCalDose3_(pybind11::array out_cscdata,
                pybind11::array out_cscptr,
                pybind11::array out_cscrowind,
                pybind11::array idd_per_depth,//suplementary 
                pybind11::array weqData,
                pybind11::array roiIndex,
                pybind11::array sourceEne, // shape N*1
                pybind11::array source, // shape N*3, start consider x/y difference
                pybind11::array bmdir, // shape N*3
                pybind11::array bmxdir,
                pybind11::array bmydir,
                pybind11::array corner,
                pybind11::array resolution,
                pybind11::array dims,
                pybind11::array longitudalCutoff,
                pybind11::array enelist,
                pybind11::array idddata,
                pybind11::array iddsetting,
                pybind11::array profiledata,
                pybind11::array profilesetting,
                pybind11::array beamparadata, // shape K*3
                pybind11::array subSpotData,
                pybind11::array layerInfo, // how many spots for each layer
                pybind11::array layerEnergy, // energy for each layer
                pybind11::array outNNZ,
                pybind11::array idbeamxy,
                float sad,
                float cutoff,
                float beamParaPos,
                long long pythonNNZSize,
                int gpuId
)
{
    printf("2024/09/21PM!!!!\n");

    checkCudaErrors(cudaSetDevice(gpuId));
    auto h_out_cscdata1 = pybind11::cast<pybind11::array_t<float>>(out_cscdata).request();
    auto h_out_cscptr = pybind11::cast<pybind11::array_t<int>>(out_cscptr).request();
    auto h_out_cscrowind1 = pybind11::cast<pybind11::array_t<int>>(out_cscrowind).request();

    auto h_sourceEne = pybind11::cast<pybind11::array_t<float>>(sourceEne).request();
    auto h_source = pybind11::cast<pybind11::array_t<float>>(source).request();
    auto h_bmdir = pybind11::cast<pybind11::array_t<float>>(bmdir).request();
    auto h_bmxdir = pybind11::cast<pybind11::array_t<float>>(bmxdir).request();
    auto h_bmydir = pybind11::cast<pybind11::array_t<float>>(bmydir).request();

    auto h_corner = pybind11::cast<pybind11::array_t<float>>(corner).request();
    auto h_resolution = pybind11::cast<pybind11::array_t<float>>(resolution).request();
    auto h_longitudalCutoff = pybind11::cast<pybind11::array_t<float>>(longitudalCutoff).request();
    auto h_dims = pybind11::cast<pybind11::array_t<int>>(dims).request();

    auto h_enelist = pybind11::cast<pybind11::array_t<float>>(enelist).request();
    auto h_idddata = pybind11::cast<pybind11::array_t<float>>(idddata).request();
    auto h_iddsetting = pybind11::cast<pybind11::array_t<float>>(iddsetting).request();
    auto h_profiledata = pybind11::cast<pybind11::array_t<float>>(profiledata).request();
    auto h_profilesetting = pybind11::cast<pybind11::array_t<float>>(profilesetting).request();
    auto h_beamparadata = pybind11::cast<pybind11::array_t<float>>(beamparadata).request();
    auto h_subspotdata = pybind11::cast<pybind11::array_t<float>>(subSpotData).request();
    auto h_layer_info = pybind11::cast<pybind11::array_t<int>>(layerInfo).request();
    auto h_layer_energy = pybind11::cast<pybind11::array_t<float>>(layerEnergy).request();
    auto h_weq = pybind11::cast<pybind11::array_t<float>>(weqData).request();
    auto h_roiIndex = pybind11::cast<pybind11::array_t<int>>(roiIndex).request();
    auto h_outNNZ = pybind11::cast<pybind11::array_t<uint64_t>>(outNNZ).request();
    auto h_idbeamxy = pybind11::cast<pybind11::array_t<float>>(idbeamxy).request();

    int nBeam = h_sourceEne.size; // 
    int nEne = h_enelist.size;
    int nRoi = h_roiIndex.size / 3;


    int beamOffset = 0;
    uint64_t nnzOffset = 0;
    int maximumLayerSize = thrust::reduce((int*)h_layer_info.ptr, (int*)h_layer_info.ptr + layerInfo.size(), -1,
                                          thrust::maximum<int>());

    int nsubspot = h_subspotdata.shape[1];
    int nProfilePara = h_profiledata.shape[2];
    int nGauss = (nProfilePara - 1) / 2;

    vec3f bmxDir = vec3f(((float*)h_bmxdir.ptr)[0], ((float*)h_bmxdir.ptr)[1], ((float*)h_bmxdir.ptr)[2]);
    vec3f bmyDir = vec3f(((float*)h_bmydir.ptr)[0], ((float*)h_bmydir.ptr)[1], ((float*)h_bmydir.ptr)[2]);
    vec3f sourcePos = vec3f(((float*)h_source.ptr)[0], ((float*)h_source.ptr)[1], ((float*)h_source.ptr)[2]);

    Grid ctGrid;
    vec3f iddDepth, profileDepth, rayweqSetting;

    memcpy(&(ctGrid.corner), ((vec3f*)h_corner.ptr), sizeof(vec3f));
    memcpy(&(ctGrid.resolution), ((vec3f*)h_resolution.ptr), sizeof(vec3f));
    memcpy(&iddDepth, (float*)h_iddsetting.ptr, sizeof(vec3f));
    memcpy(&profileDepth, (float*)h_profilesetting.ptr, sizeof(vec3f));
    memcpy(&(ctGrid.dims), ((vec3i*)h_dims.ptr), sizeof(vec3i));
    memcpy(&rayweqSetting, ((float*)h_weq.ptr), sizeof(vec3f));

    int nStep = ((float*)h_weq.ptr)[2];
    int ny = ((float*)h_weq.ptr)[5];
    int nx = ((float*)h_weq.ptr)[8];

    cudaArray_t iddArray;
    cudaTextureObject_t iddObj = 0;
    create2DTexture((float*)h_idddata.ptr, &iddArray, &iddObj, weq::LINEAR,
                    weq::R, h_idddata.shape[1], h_idddata.shape[0], gpuId);

    //suplementary 
#ifdef DEBUG
    printf("iddArray: %d %d %d\n", h_idddata.shape[1], h_idddata.shape[0], h_idddata.size);
#endif
    float* d_idd_per_depth;
    cudaMalloc((void **)&d_idd_per_depth, sizeof(float) * idd_per_depth.size());
    cudaMemset(d_idd_per_depth, 0, sizeof(float) * idd_per_depth.size());

    cudaArray_t profileArray;
    cudaTextureObject_t profileObj = 0;
    create3DTexture((float*)h_profiledata.ptr, &profileArray, &profileObj, weq::LINEAR,
                    weq::R, h_profiledata.shape[2], h_profiledata.shape[1], h_profiledata.shape[0], gpuId);

    cudaArray_t subSpotDataArray;
    cudaTextureObject_t subSpotDataObj = 0;
    create3DTexture((float*)h_subspotdata.ptr, &subSpotDataArray, &subSpotDataObj, weq::POINTS,
                    weq::R, h_subspotdata.shape[2], h_subspotdata.shape[1], h_subspotdata.shape[0], gpuId);

    cudaArray_t rayweqArray;
    cudaTextureObject_t rayweqObj = 0;
    create3DTexture((float*)h_weq.ptr + 9, &rayweqArray, &rayweqObj, weq::LINEAR,
                    weq::R, nStep, ny, nx, gpuId);

    int*   d_sumNNZ;
    int*   d_tmpRowIndex;
    int*   d_tmpColIndex;
    int*   d_tmpCscPtr;
    vec3i* d_roiInd;
    vec3f* d_bmzDir;
    float* d_tmpValues;
    float* d_idbeamxy;

    float sparseRatio = 1.0f;
    size_t nnzSizeThisLayer = size_t((float)maximumLayerSize * (float)nRoi * sparseRatio);

    if (nnzSizeThisLayer * 4 >= 4294967295)
        nnzSizeThisLayer = size_t(4294967295 / 4);

    checkCudaErrors(cudaMalloc((void **)&d_roiInd, sizeof(vec3i) * h_roiIndex.size / 3));
    checkCudaErrors(cudaMalloc((void **)&d_sumNNZ, sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_tmpCscPtr, sizeof(int) * (maximumLayerSize + 1)));
    checkCudaErrors(cudaMalloc((void **)&d_bmzDir, sizeof(vec3f) * maximumLayerSize));
    checkCudaErrors(cudaMalloc((void **)&d_idbeamxy, sizeof(float) * maximumLayerSize * 2));

    printf("%zu\n", nnzSizeThisLayer);

    checkCudaErrors(cudaMalloc((void **)&d_tmpValues, sizeof(float) * nnzSizeThisLayer));
    checkCudaErrors(cudaMalloc((void **)&d_tmpRowIndex, sizeof(int) * nnzSizeThisLayer));
    checkCudaErrors(cudaMalloc((void **)&d_tmpColIndex, sizeof(int) * nnzSizeThisLayer));

    checkCudaErrors(
        cudaMemcpy(d_roiInd, (vec3i *)h_roiIndex.ptr, sizeof(vec3i) * h_roiIndex.size / 3, cudaMemcpyHostToDevice));


    int returnFlag = 1;
    int spots_simulated = 0;

    for (int i = 0; i < layerInfo.size(); i++)
    {
        int layerSize = ((int*)h_layer_info.ptr)[i];
        int energyIdx = binarySearchEneIdx(((float*)h_layer_energy.ptr)[i], ((float*)h_enelist.ptr), nEne);

        float rtheta, theta2, longitudalCutoff, r2;
        r2     = ((float*)h_beamparadata.ptr)[energyIdx * 3];
        rtheta = ((float*)h_beamparadata.ptr)[energyIdx * 3 + 1];
        theta2 = ((float*)h_beamparadata.ptr)[energyIdx * 3 + 2];
        longitudalCutoff = *((float*)h_longitudalCutoff.ptr + beamOffset);

        int spots_left_this_layer = layerSize;
        while (spots_left_this_layer > 0)
        {
            int spot_size_this_batch = 0;
            if (spots_left_this_layer >= MAXIMUM_SPOTS_ONE_BATCH)
            {
                spot_size_this_batch = MAXIMUM_SPOTS_ONE_BATCH;
            }
            else
            {
                spot_size_this_batch = spots_left_this_layer;
            }
            checkCudaErrors(cudaMemset(d_tmpCscPtr, 0, sizeof(int) * (maximumLayerSize + 1)));
            checkCudaErrors(cudaMemset(d_bmzDir, 0, sizeof(vec3f) * maximumLayerSize));
            checkCudaErrors(cudaMemset(d_tmpColIndex, 0, sizeof(int) * nnzSizeThisLayer));
            checkCudaErrors(cudaMemset(d_tmpRowIndex, 0, sizeof(int) * nnzSizeThisLayer));
            checkCudaErrors(cudaMemset(d_tmpValues, 0, sizeof(float) * nnzSizeThisLayer));
            checkCudaErrors(cudaMemset(d_sumNNZ, 0, sizeof(int))); // reset sumNNZ value to 0

            checkCudaErrors(cudaMemcpy(d_bmzDir,
                (vec3f *)h_bmdir.ptr + beamOffset,
                sizeof(vec3f) * spot_size_this_batch,
                cudaMemcpyHostToDevice));

            checkCudaErrors(
                cudaMemcpy(d_idbeamxy, (float *)h_idbeamxy.ptr + beamOffset * 2, sizeof(float) * 2 *
                    spot_size_this_batch,
                    cudaMemcpyHostToDevice));

            int nnz = calOneLayerDoseOld_(d_tmpValues,
                                         d_tmpRowIndex,
                                         d_tmpColIndex,
                                         d_tmpCscPtr,
                                         d_idd_per_depth,//suplementary 
                                         nnzSizeThisLayer,
                                         d_sumNNZ,
                                         d_bmzDir,
                                         spot_size_this_batch,
                                         sourcePos,
                                         bmxDir,
                                         bmyDir,
                                         d_roiInd,
                                         nRoi,
                                         ctGrid,
                                         rayweqObj,
                                         iddObj,
                                         profileObj,
                                         subSpotDataObj,
                                         rayweqSetting,
                                         iddDepth,
                                         profileDepth,
                                         d_idbeamxy,
                                         nsubspot,
                                         nGauss,
                                         energyIdx,
                                         beamOffset,
                                         beamParaPos,
                                         longitudalCutoff,
                                         cutoff,
                                         rtheta,
                                         theta2,
                                         r2,
                                         sad,
                                         gpuId
            );
            spots_simulated += spot_size_this_batch;
            printf("spots simulated = %d ", spots_simulated);

            if (nnz > nnzSizeThisLayer)
            {
                printf("nnz larger than nnz size this batch....!\n");
                throw -1;
            }
            if (nnzOffset + nnz < pythonNNZSize)
            {
                checkCudaErrors(
                    cudaMemcpy((float *)h_out_cscdata1.ptr + nnzOffset, d_tmpValues, sizeof(float) * nnz,
                        cudaMemcpyDeviceToHost));
                checkCudaErrors(
                    cudaMemcpy((int *)h_out_cscrowind1.ptr + nnzOffset, d_tmpRowIndex, sizeof(int) * nnz,
                        cudaMemcpyDeviceToHost));
                checkCudaErrors(
                    cudaMemcpy((int *)h_out_cscptr.ptr + 1 + beamOffset, d_tmpCscPtr + 1, sizeof(int) * spot_size_this_batch,
                        cudaMemcpyDeviceToHost));
            }
            else
            {
                returnFlag = -1;
                break;
            }
            beamOffset += spot_size_this_batch;
            nnzOffset += nnz;
            spots_left_this_layer -= spot_size_this_batch;
            printf("layer:%d spots left this layer:%d ", i, spots_left_this_layer);
            printf("nnz offset:%lu nnz size:%u\n", nnzOffset, nnz);
        }
    }

    //suplementary
    auto h_idd_per_depth = pybind11::cast<pybind11::array_t<float>>(idd_per_depth).request();
    checkCudaErrors(cudaMemcpy((float*)h_idd_per_depth.ptr, d_idd_per_depth, sizeof(float) * idd_per_depth.size(),
        cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_idd_per_depth));

    checkCudaErrors(cudaFree(d_roiInd));
    checkCudaErrors(cudaFree(d_sumNNZ));
    checkCudaErrors(cudaFree(d_bmzDir));
    checkCudaErrors(cudaFree(d_tmpValues));
    checkCudaErrors(cudaFree(d_tmpRowIndex));
    checkCudaErrors(cudaFree(d_tmpColIndex));
    checkCudaErrors(cudaFree(d_tmpCscPtr));
    checkCudaErrors(cudaFree(d_idbeamxy));

    checkCudaErrors(cudaFreeArray(iddArray));
    checkCudaErrors(cudaFreeArray(profileArray));
    checkCudaErrors(cudaFreeArray(subSpotDataArray));
    checkCudaErrors(cudaFreeArray(rayweqArray));

    checkCudaErrors(cudaDestroyTextureObject(iddObj));
    checkCudaErrors(cudaDestroyTextureObject(profileObj));
    checkCudaErrors(cudaDestroyTextureObject(subSpotDataObj));
    checkCudaErrors(cudaDestroyTextureObject(rayweqObj));
    //  delete[] h_cscPtr;
    ((uint64_t*)(h_outNNZ.ptr))[0] = nnzOffset;
    return returnFlag;
}

EXPORT
void cudaFinalDose(pybind11::array out_finalDose,
                   pybind11::array weqData,
                   pybind11::array roiIndex,
                   pybind11::array sourceEne,           // shape N*1
                   pybind11::array source,              // shape N*3, start consider x/y difference
                   pybind11::array bmdir,               // shape N*3
                   pybind11::array bmxdir,
                   pybind11::array bmydir,
                   pybind11::array corner,
                   pybind11::array resolution,
                   pybind11::array dims,
                   pybind11::array longitudalCutoff,
                   pybind11::array enelist,
                   pybind11::array idddata,
                   pybind11::array iddsetting,
                   pybind11::array profiledata,
                   pybind11::array profilesetting,
                   pybind11::array beamparadata, // shape K*3
                   pybind11::array subSpotData,
                   pybind11::array layerInfo, // how many spots for each layer
                   pybind11::array layerEnergy, // energy for each layer
                   pybind11::array idbeamxy,
                   pybind11::array numParticlesPerBeam,
                   float sad,
                   float cutoff,
                   float beamParaPos,
                   int   gpuId
)
{
    printf("2024/09/02PM!!!!\n");

    checkCudaErrors(cudaSetDevice(gpuId));
    auto h_out_dose = pybind11::cast<pybind11::array_t<float>>(out_finalDose).request();
    auto h_sourceEne = pybind11::cast<pybind11::array_t<float>>(sourceEne).request();
    auto h_source = pybind11::cast<pybind11::array_t<float>>(source).request();
    auto h_bmdir = pybind11::cast<pybind11::array_t<float>>(bmdir).request();
    auto h_bmxdir = pybind11::cast<pybind11::array_t<float>>(bmxdir).request();
    auto h_bmydir = pybind11::cast<pybind11::array_t<float>>(bmydir).request();

    auto h_corner = pybind11::cast<pybind11::array_t<float>>(corner).request();
    auto h_resolution = pybind11::cast<pybind11::array_t<float>>(resolution).request();
    auto h_longitudalCutoff = pybind11::cast<pybind11::array_t<float>>(longitudalCutoff).request();
    auto h_dims = pybind11::cast<pybind11::array_t<int>>(dims).request();

    auto h_enelist = pybind11::cast<pybind11::array_t<float>>(enelist).request();
    auto h_idddata = pybind11::cast<pybind11::array_t<float>>(idddata).request();
    auto h_iddsetting = pybind11::cast<pybind11::array_t<float>>(iddsetting).request();
    auto h_profiledata = pybind11::cast<pybind11::array_t<float>>(profiledata).request();
    auto h_profilesetting = pybind11::cast<pybind11::array_t<float>>(profilesetting).request();
    auto h_beamparadata = pybind11::cast<pybind11::array_t<float>>(beamparadata).request();
    auto h_subspotdata = pybind11::cast<pybind11::array_t<float>>(subSpotData).request();
    auto h_layer_info = pybind11::cast<pybind11::array_t<int>>(layerInfo).request();
    auto h_layer_energy = pybind11::cast<pybind11::array_t<float>>(layerEnergy).request();
    auto h_weq = pybind11::cast<pybind11::array_t<float>>(weqData).request();
    auto h_roiIndex = pybind11::cast<pybind11::array_t<int>>(roiIndex).request();
    auto h_idbeamxy = pybind11::cast<pybind11::array_t<float>>(idbeamxy).request();
    auto h_numPar = pybind11::cast<pybind11::array_t<int>>(numParticlesPerBeam).request();

    int nEne = h_enelist.size;
    int nRoi = h_roiIndex.size / 3;

    int beamOffset = 0;
    int maximumLayerSize = thrust::reduce((int*)h_layer_info.ptr, (int*)h_layer_info.ptr + layerInfo.size(), -1,
                                          thrust::maximum<int>());

    int nsubspot = h_subspotdata.shape[1];
    int nProfilePara = h_profiledata.shape[2];
    int nGauss = (nProfilePara - 1) / 2;

    vec3f bmxDir = vec3f(((float*)h_bmxdir.ptr)[0], ((float*)h_bmxdir.ptr)[1], ((float*)h_bmxdir.ptr)[2]);
    vec3f bmyDir = vec3f(((float*)h_bmydir.ptr)[0], ((float*)h_bmydir.ptr)[1], ((float*)h_bmydir.ptr)[2]);
    vec3f sourcePos = vec3f(((float*)h_source.ptr)[0], ((float*)h_source.ptr)[1], ((float*)h_source.ptr)[2]);

    Grid ctGrid;
    vec3f iddDepth, profileDepth, rayweqSetting;

    memcpy(&(ctGrid.corner), ((vec3f*)h_corner.ptr), sizeof(vec3f));
    memcpy(&(ctGrid.resolution), ((vec3f*)h_resolution.ptr), sizeof(vec3f));
    memcpy(&iddDepth, (float*)h_iddsetting.ptr, sizeof(vec3f));
    memcpy(&profileDepth, (float*)h_profilesetting.ptr, sizeof(vec3f));
    memcpy(&(ctGrid.dims), ((vec3i*)h_dims.ptr), sizeof(vec3i));
    memcpy(&rayweqSetting, ((float*)h_weq.ptr), sizeof(vec3f));

    int nStep = ((float*)h_weq.ptr)[2];
    int ny = ((float*)h_weq.ptr)[5];
    int nx = ((float*)h_weq.ptr)[8];


    int gridSize = ctGrid.dims.x * ctGrid.dims.y * ctGrid.dims.z;

    cudaArray_t iddArray;
    cudaTextureObject_t iddObj = 0;
    create2DTexture((float*)h_idddata.ptr, &iddArray, &iddObj, weq::LINEAR,
                    weq::R, h_idddata.shape[1], h_idddata.shape[0], gpuId);

    cudaArray_t profileArray;
    cudaTextureObject_t profileObj = 0;
    create3DTexture((float*)h_profiledata.ptr, &profileArray, &profileObj, weq::LINEAR,
                    weq::R, h_profiledata.shape[2], h_profiledata.shape[1], h_profiledata.shape[0], gpuId);

    cudaArray_t subSpotDataArray;
    cudaTextureObject_t subSpotDataObj = 0;
    create3DTexture((float*)h_subspotdata.ptr, &subSpotDataArray, &subSpotDataObj, weq::POINTS,
                    weq::R, h_subspotdata.shape[2], h_subspotdata.shape[1], h_subspotdata.shape[0], gpuId);

    cudaArray_t rayweqArray;
    cudaTextureObject_t rayweqObj = 0;
    create3DTexture((float*)h_weq.ptr + 9, &rayweqArray, &rayweqObj, weq::LINEAR,
                    weq::R, nStep, ny, nx, gpuId);

    float* d_final;
    int*   d_numPar;
    vec3i* d_roiInd;
    vec3f* d_bmzDir;
    float* d_idbeamxy;

    checkCudaErrors(cudaMalloc((void **)&d_roiInd, sizeof(vec3i) * h_roiIndex.size / 3));
    checkCudaErrors(cudaMalloc((void **)&d_bmzDir, sizeof(vec3f) * maximumLayerSize));
    checkCudaErrors(cudaMalloc((void **)&d_numPar, sizeof(int) * maximumLayerSize));
    checkCudaErrors(cudaMalloc((void **)&d_idbeamxy, sizeof(float) * maximumLayerSize * 2));
    checkCudaErrors(cudaMalloc((void **)&d_final, sizeof(float) * gridSize));
    checkCudaErrors(cudaMemset(d_final, 0, sizeof(float) * gridSize));
    checkCudaErrors(
        cudaMemcpy(d_roiInd, (vec3i *)h_roiIndex.ptr, sizeof(vec3i) * h_roiIndex.size / 3, cudaMemcpyHostToDevice));

    for (int i = 0; i < layerInfo.size(); i++)
    {
        int layerSize = ((int*)h_layer_info.ptr)[i];
        int energyIdx = binarySearchEneIdx(((float*)h_layer_energy.ptr)[i], ((float*)h_enelist.ptr), nEne);

        float rtheta, theta2, longitudalCutoff, r2;
        r2     = ((float*)h_beamparadata.ptr)[energyIdx * 3];
        rtheta = ((float*)h_beamparadata.ptr)[energyIdx * 3 + 1];
        theta2 = ((float*)h_beamparadata.ptr)[energyIdx * 3 + 2];
        longitudalCutoff = *((float*)h_longitudalCutoff.ptr + beamOffset);

        checkCudaErrors(cudaMemset(d_bmzDir, 0, sizeof(vec3f) * maximumLayerSize));
        checkCudaErrors(cudaMemset(d_idbeamxy, 0, sizeof(float) * maximumLayerSize * 2));
        checkCudaErrors(cudaMemset(d_numPar, 0, sizeof(int) * maximumLayerSize));

        checkCudaErrors(
            cudaMemcpy(d_bmzDir, (vec3f *)h_bmdir.ptr + beamOffset, sizeof(vec3f) * layerSize, cudaMemcpyHostToDevice));
        checkCudaErrors(
            cudaMemcpy(d_numPar, (int *)h_numPar.ptr + beamOffset, sizeof(int) * layerSize, cudaMemcpyHostToDevice));
        checkCudaErrors(
            cudaMemcpy(d_idbeamxy, (float *)h_idbeamxy.ptr + beamOffset * 2, sizeof(float) * 2 * layerSize,
                cudaMemcpyHostToDevice));

        calFinalDose(d_final,
                     d_bmzDir,
                     layerSize,
                     sourcePos,
                     bmxDir,
                     bmyDir,
                     d_roiInd,
                     nRoi,
                     ctGrid,
                     rayweqObj,
                     iddObj,
                     profileObj,
                     subSpotDataObj,
                     rayweqSetting,
                     iddDepth,
                     profileDepth,
                     d_idbeamxy,
                     d_numPar,
                     nsubspot,
                     nGauss,
                     energyIdx,
                     beamOffset,
                     beamParaPos,
                     longitudalCutoff,
                     cutoff,
                     rtheta,
                     theta2,
                     r2,
                     sad,
                     gpuId
        );

        beamOffset += layerSize;
    }

    checkCudaErrors(cudaMemcpy((float *)h_out_dose.ptr, d_final, sizeof(float) * gridSize, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_roiInd));
    checkCudaErrors(cudaFree(d_bmzDir));
    checkCudaErrors(cudaFree(d_numPar));
    checkCudaErrors(cudaFree(d_final));
    checkCudaErrors(cudaFree(d_idbeamxy));

    checkCudaErrors(cudaFreeArray(iddArray));
    checkCudaErrors(cudaFreeArray(profileArray));
    checkCudaErrors(cudaFreeArray(subSpotDataArray));
    checkCudaErrors(cudaFreeArray(rayweqArray));

    checkCudaErrors(cudaDestroyTextureObject(iddObj));
    checkCudaErrors(cudaDestroyTextureObject(profileObj));
    checkCudaErrors(cudaDestroyTextureObject(subSpotDataObj));
    checkCudaErrors(cudaDestroyTextureObject(rayweqObj));
}

EXPORT
void cuFinalPhysDoseAndRBEDose(pybind11::array out_finalDose,
                               pybind11::array out_rbe,
                               pybind11::array out_rbe2,
                               pybind11::array letData,
                               pybind11::array lqData,
                               pybind11::array weqData,
                               pybind11::array Z1DData,
                               pybind11::array roiIndex,
                               pybind11::array sourceEne,            // shape N*1
                               pybind11::array source,               // shape N*3, start consider x/y difference
                               pybind11::array bmdir,                // shape N*3
                               pybind11::array bmxdir,
                               pybind11::array bmydir,
                               pybind11::array corner,
                               pybind11::array resolution,
                               pybind11::array dims,
                               pybind11::array longitudalCutoff,
                               pybind11::array enelist,
                               pybind11::array idddata,
                               pybind11::array iddsetting,
                               pybind11::array profiledata,
                               pybind11::array profilesetting,
                               pybind11::array beamparadata,         // shape K*3
                               pybind11::array subSpotData,
                               pybind11::array layerInfo,            // how many spots for each layer
                               pybind11::array layerEnergy,          // energy for each layer
                               pybind11::array idbeamxy,
                               pybind11::array numParticlesPerBeam,
                               pybind11::array fieldInfo,
                               pybind11::array sad,
                               pybind11::array rs,
                               pybind11::array cellData,
                               float           cutoff,
                               float           beamParaPos,
                               int             nField,
                               int             mode,
                               int             gpuId)
{
    printf("2025/3/26\n");
    checkCudaErrors(cudaSetDevice(gpuId));
    auto h_out_dose         = pybind11::cast<pybind11::array_t<float>>(out_finalDose).request();
    auto h_sourceEne        = pybind11::cast<pybind11::array_t<float>>(sourceEne).request();
    auto h_source           = pybind11::cast<pybind11::array_t<float>>(source).request();
    auto h_bmdir            = pybind11::cast<pybind11::array_t<float>>(bmdir).request();
    auto h_bmxdir           = pybind11::cast<pybind11::array_t<float>>(bmxdir).request();
    auto h_bmydir           = pybind11::cast<pybind11::array_t<float>>(bmydir).request();

    auto h_corner           = pybind11::cast<pybind11::array_t<float>>(corner).request();
    auto h_resolution       = pybind11::cast<pybind11::array_t<float>>(resolution).request();
    auto h_longitudalCutoff = pybind11::cast<pybind11::array_t<float>>(longitudalCutoff).request();
    auto h_dims             = pybind11::cast<pybind11::array_t<int>>(dims).request();

    auto h_enelist          = pybind11::cast<pybind11::array_t<float>>(enelist).request();
    auto h_idddata          = pybind11::cast<pybind11::array_t<float>>(idddata).request();
    auto h_iddsetting       = pybind11::cast<pybind11::array_t<float>>(iddsetting).request();
    auto h_profiledata      = pybind11::cast<pybind11::array_t<float>>(profiledata).request();
    auto h_profilesetting   = pybind11::cast<pybind11::array_t<float>>(profilesetting).request();
    auto h_beamparadata     = pybind11::cast<pybind11::array_t<float>>(beamparadata).request();
    auto h_subspotdata      = pybind11::cast<pybind11::array_t<float>>(subSpotData).request();
    auto h_layer_info                = pybind11::cast<pybind11::array_t<int>>(layerInfo).request();
    auto h_layer_energy     = pybind11::cast<pybind11::array_t<float>>(layerEnergy).request();
    auto h_weq              = pybind11::cast<pybind11::array_t<float>>(weqData).request();
    auto h_let              = pybind11::cast<pybind11::array_t<float>>(letData).request();
    auto h_lq               = pybind11::cast<pybind11::array_t<float>>(lqData).request();
    auto h_out_rbe          = pybind11::cast<pybind11::array_t<float>>(out_rbe).request();
    auto h_out_rbe2         = pybind11::cast<pybind11::array_t<float>>(out_rbe2).request();

    auto h_roiIndex         = pybind11::cast<pybind11::array_t<int>>(roiIndex).request();
    auto h_idbeamxy         = pybind11::cast<pybind11::array_t<float>>(idbeamxy).request();
    auto h_numPar           = pybind11::cast<pybind11::array_t<int>>(numParticlesPerBeam).request();
    auto h_fieldInfo                 = pybind11::cast<pybind11::array_t<int>>(fieldInfo).request();
    auto h_sad              = pybind11::cast<pybind11::array_t<float>>(sad).request();
    auto h_rs               = pybind11::cast<pybind11::array_t<float>>(rs).request();
    auto h_z1D              = pybind11::cast<pybind11::array_t<float>>(Z1DData).request();
    auto h_cellData         = pybind11::cast<pybind11::array_t<float>>(cellData).request();

    Grid ctGrid;
    vec2f alpha0AndBeta0;
    memcpy(&(ctGrid.corner),     ((vec3f*)h_corner.ptr),     sizeof(vec3f));
    memcpy(&(ctGrid.resolution), ((vec3f*)h_resolution.ptr), sizeof(vec3f));
    memcpy(&(ctGrid.dims),       ((vec3i*)h_dims.ptr),       sizeof(vec3i));
    memcpy(&alpha0AndBeta0,      ((vec3f*)h_cellData.ptr),   sizeof(vec2f));

    printf("alpha0:%f beta0:%f\n", alpha0AndBeta0.x, alpha0AndBeta0.y);

    int gridSize         = ctGrid.dims.x * ctGrid.dims.y * ctGrid.dims.z;
    int nEne             = h_enelist.size;
    int nRoi             = h_roiIndex.size / 3;
    int maximumLayerSize = thrust::reduce((int*)h_layer_info.ptr, (int*)h_layer_info.ptr + h_layer_info.size, -1,
                                          thrust::maximum<int>());

    int nsubspot         = h_subspotdata.shape[2];
    int nProfilePara     = h_profiledata.shape[3];
    int nGauss           = (nProfilePara - 1) / 2;

    int weqOffset        = 0;
    int layerOffset      = 0;
    int beamOffset       = 0;

    float* d_final, *d_rbemap, *d_rbemap2, *d_alpha, *d_beta, *d_Z1Dmix, *d_counter;
    vec3i* d_roiInd;

#ifdef DEBUG
    GPU_Clock timer;
    timer.start();
    float total_time1=0.f;
    float total_time2=0.f;
    float total_time3=0.f;
    float total_time4=0.f;
    float time=0.f;
#endif

    checkCudaErrors(cudaMalloc((void **)&d_final,  nField * sizeof(float) * gridSize));
    checkCudaErrors(cudaMalloc((void **)&d_rbemap, sizeof(float) * gridSize));
    checkCudaErrors(cudaMalloc((void **)&d_rbemap2, sizeof(float) * gridSize));
    checkCudaErrors(cudaMalloc((void **)&d_alpha,  sizeof(float) * gridSize));
    checkCudaErrors(cudaMalloc((void **)&d_beta,   sizeof(float) * gridSize));
    checkCudaErrors(cudaMalloc((void **)&d_Z1Dmix, sizeof(float) * gridSize));
    checkCudaErrors(cudaMalloc((void **)&d_counter, sizeof(float) * gridSize));
    checkCudaErrors(cudaMalloc((void **)&d_roiInd, sizeof(vec3i) * nRoi));

    checkCudaErrors(cudaMemset(d_final,  0, nField * sizeof(float) * gridSize));
    checkCudaErrors(cudaMemcpy(d_rbemap, (float *)h_out_rbe.ptr, sizeof(float) * gridSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_rbemap2, (float *)h_out_rbe2.ptr, sizeof(float) * gridSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_roiInd, (vec3i *)h_roiIndex.ptr, sizeof(vec3i) * nRoi, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemset(d_alpha,  0, sizeof(float) * gridSize));
    checkCudaErrors(cudaMemset(d_beta,   0, sizeof(float) * gridSize));
    checkCudaErrors(cudaMemset(d_Z1Dmix,   0, sizeof(float) * gridSize));
    checkCudaErrors(cudaMemset(d_counter,   0, sizeof(float) * gridSize));

    cudaArray_t lqDataArray;
    cudaTextureObject_t lqDataObj = 0;
    create3DTexture((float*)h_lq.ptr, &lqDataArray, &lqDataObj, weq::LINEAR,
                    weq::R, h_lq.shape[2], h_lq.shape[1], h_lq.shape[0], gpuId);

    cudaArray_t Z1DArray;
    cudaTextureObject_t Z1DObj = 0;
    create2DTexture((float*)h_z1D.ptr, &Z1DArray, &Z1DObj, weq::LINEAR, weq::R, h_z1D.shape[1], h_z1D.shape[0],
                         gpuId);

#ifdef DEBUG                         
    time=timer.seconds();
    printf("1.malloc and memcpy time:%f\n", time);
    total_time1 += time; 
#endif  

    for (int i = 0; i < nField; i++)
    {
        int nStep          = ((float*)h_weq.ptr)[weqOffset + 2];
        int ny             = ((float*)h_weq.ptr)[weqOffset + 5];
        int nx             = ((float*)h_weq.ptr)[weqOffset + 8];

        printf("calculating field:%d now....\n", i);

        float SAD          = ((float*)h_sad.ptr)[i];
        float RS           = ((float*)h_rs.ptr)[i];
        vec3f iddDepth, profileDepth, rayweqSetting;

        memcpy(&iddDepth,      (float*)h_iddsetting.ptr + i * 3,     sizeof(vec3f));
        memcpy(&profileDepth,  (float*)h_profilesetting.ptr + i * 3, sizeof(vec3f));
        memcpy(&rayweqSetting, (float*)h_weq.ptr + weqOffset,        sizeof(vec3f));

        vec3f bmxDir    = vec3f(((float*)h_bmxdir.ptr + i * 3)[0], ((float*)h_bmxdir.ptr + i * 3)[1], ((float*)h_bmxdir.ptr + i * 3)[2]);
        vec3f bmyDir    = vec3f(((float*)h_bmydir.ptr + i * 3)[0], ((float*)h_bmydir.ptr + i * 3)[1], ((float*)h_bmydir.ptr + i * 3)[2]);
        vec3f sourcePos = vec3f(((float*)h_source.ptr + i * 3)[0], ((float*)h_source.ptr + i * 3)[1], ((float*)h_source.ptr + i * 3)[2]);

        int   nLayers   = ((int*)h_fieldInfo.ptr)[i];

#ifdef DEBUG
        timer.start();
#endif

        cudaArray_t iddArray;
        cudaTextureObject_t iddObj = 0;
        create2DTexture((float*)h_idddata.ptr + i * h_idddata.shape[1] * h_idddata.shape[2], &iddArray, &iddObj, weq::LINEAR,
                        weq::R, h_idddata.shape[2], h_idddata.shape[1], gpuId);


        cudaArray_t profileArray;
        cudaTextureObject_t profileObj = 0;
        create3DTexture((float*)h_profiledata.ptr + i * h_profiledata.shape[1] * h_profiledata.shape[2] * h_profiledata.shape[3], &profileArray, &profileObj, weq::LINEAR,
                        weq::R, h_profiledata.shape[3], h_profiledata.shape[2], h_profiledata.shape[1], gpuId);


        cudaArray_t subSpotDataArray;
        cudaTextureObject_t subSpotDataObj = 0;
        create3DTexture((float*)h_subspotdata.ptr + i * h_subspotdata.shape[1] * h_subspotdata.shape[2] * h_subspotdata.shape[3], &subSpotDataArray, &subSpotDataObj, weq::POINTS,
                        weq::R, h_subspotdata.shape[3], h_subspotdata.shape[2], h_subspotdata.shape[1], gpuId);


        cudaArray_t letArray;
        cudaTextureObject_t letObj = 0;
        create2DTexture((float *)h_let.ptr + i * h_let.shape[1] * h_let.shape[2], &letArray, &letObj, weq::LINEAR, weq::R, h_let.shape[2], h_let.shape[1], gpuId);


        cudaArray_t rayweqArray;
        cudaTextureObject_t rayweqObj = 0;
        create3DTexture((float*)h_weq.ptr + weqOffset + 9, &rayweqArray, &rayweqObj, weq::LINEAR,
                        weq::R, nStep, ny, nx, gpuId);

        vec3f* d_bmzDir;
        float* d_idbeamxy;
        int*   d_numPar;

        checkCudaErrors(cudaMalloc((void **)&d_bmzDir, sizeof(vec3f) * maximumLayerSize));
        checkCudaErrors(cudaMalloc((void **)&d_numPar, sizeof(int) * maximumLayerSize));
        checkCudaErrors(cudaMalloc((void **)&d_idbeamxy, sizeof(float) * maximumLayerSize * 2));

#ifdef DEBUG
        time=timer.seconds();
        printf("2.malloc and memcpy time:%f\n", time);
        total_time2 += time;
#endif

        for (int j = 0; j < nLayers; j++)
        {
            // printf("calculating layer:%d\n", j);
            checkCudaErrors(cudaMemset(d_bmzDir, 0, sizeof(vec3f) * maximumLayerSize));
            checkCudaErrors(cudaMemset(d_idbeamxy, 0, sizeof(float) * maximumLayerSize * 2));
            checkCudaErrors(cudaMemset(d_numPar, 0, sizeof(int) * maximumLayerSize));

            float rtheta, theta2, longitudalCutoff, r2;
            int   layerSize = ((int*)h_layer_info.ptr)[j + layerOffset];
            int   energyIdx = binarySearchEneIdx(((float*)h_layer_energy.ptr)[j + layerOffset], ((float*)h_enelist.ptr), nEne);

            r2     = ((float*)h_beamparadata.ptr)[i * h_beamparadata.shape[1] * h_beamparadata.shape[2] + energyIdx * 3];
            rtheta = ((float*)h_beamparadata.ptr)[i * h_beamparadata.shape[1] * h_beamparadata.shape[2] + energyIdx * 3 + 1];
            theta2 = ((float*)h_beamparadata.ptr)[i * h_beamparadata.shape[1] * h_beamparadata.shape[2] + energyIdx * 3 + 2];
            longitudalCutoff = *((float*)h_longitudalCutoff.ptr + beamOffset);

#ifdef DEBUG
            timer.start();
#endif

            checkCudaErrors(cudaMemcpy(d_bmzDir, (vec3f *)h_bmdir.ptr + beamOffset, sizeof(vec3f) * layerSize, cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_numPar, (int *)h_numPar.ptr + beamOffset, sizeof(int) * layerSize, cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_idbeamxy, (float *)h_idbeamxy.ptr + beamOffset * 2, sizeof(float) * 2 * layerSize,cudaMemcpyHostToDevice));

#ifdef DEBUG
            time=timer.seconds();
            printf("3.malloc and memcpy time:%f\n", time);
            total_time3 += time;
#endif

            float estDimension = cbrtf(ctGrid.resolution.x * ctGrid.resolution.y * ctGrid.resolution.z);
            if (mode == 0)
            {
                // printf("calculating RBE using LQ mode.\n");

#ifdef DEBUG
                timer.start();
#endif

                calFinalDoseAndRBE(d_final,
                                   d_rbemap2,
                                   d_counter,
                                   d_alpha,
                                   d_beta,
                                   d_bmzDir,
                                   layerSize,
                                   sourcePos,
                                   bmxDir,
                                   bmyDir,
                                   d_roiInd,
                                   nRoi,
                                   ctGrid,
                                   rayweqObj,
                                   iddObj,
                                   profileObj,
                                   subSpotDataObj,
                                   letObj,
                                   lqDataObj,
                                   rayweqSetting,
                                   iddDepth,
                                   profileDepth,
                                   d_idbeamxy,
                                   d_numPar,
                                   nsubspot,
                                   nGauss,
                                   energyIdx,
                                   i,
                                   beamParaPos,
                                   longitudalCutoff,
                                   cutoff,
                                   rtheta,
                                   theta2,
                                   r2,
                                   estDimension,
                                   SAD,
                                   RS,
                                   gpuId
                );

#ifdef DEBUG
                time=timer.seconds();
                printf("4.kernel time:%f\n", time);
                total_time4 += time;
#endif

            }
            else
            {
                // printf("calculating RBE using mMKM mode.\n");
                calFinalDoseAndMKM(d_final,
                                   d_Z1Dmix,
                                   d_bmzDir,
                                   layerSize,
                                   sourcePos,
                                   bmxDir,
                                   bmyDir,
                                   d_roiInd,
                                   nRoi,
                                   ctGrid,
                                   rayweqObj,
                                   iddObj,
                                   profileObj,
                                   subSpotDataObj,
                                   Z1DObj,
                                   rayweqSetting,
                                   iddDepth,
                                   profileDepth,
                                   d_idbeamxy,
                                   d_numPar,
                                   nsubspot,
                                   nGauss,
                                   energyIdx,
                                   i,
                                   beamParaPos,
                                   longitudalCutoff,
                                   cutoff,
                                   rtheta,
                                   theta2,
                                   r2,
                                   estDimension,
                                   SAD,
                                   RS,
                                   ((float*)h_layer_energy.ptr)[j + layerOffset],
                                   gpuId);
            }

            beamOffset  += layerSize;
        }

        layerOffset += nLayers;
        weqOffset   += (9 + 10000 * ny * nx);

        checkCudaErrors(cudaFree(d_bmzDir));
        checkCudaErrors(cudaFree(d_idbeamxy));
        checkCudaErrors(cudaFree(d_numPar));

        checkCudaErrors(cudaFreeArray(iddArray));
        checkCudaErrors(cudaFreeArray(profileArray));
        checkCudaErrors(cudaFreeArray(subSpotDataArray));
        checkCudaErrors(cudaFreeArray(rayweqArray));
        checkCudaErrors(cudaFreeArray(letArray));

        checkCudaErrors(cudaDestroyTextureObject(iddObj));
        checkCudaErrors(cudaDestroyTextureObject(profileObj));
        checkCudaErrors(cudaDestroyTextureObject(subSpotDataObj));
        checkCudaErrors(cudaDestroyTextureObject(rayweqObj));
        checkCudaErrors(cudaDestroyTextureObject(letObj));
    }

#ifdef DEBUG
    printf("total time1:%f\n", total_time1);
    printf("total time2:%f\n", total_time2);
    printf("total time3:%f\n", total_time3);
    printf("total time4:%f\n", total_time4);
#endif

    if (mode == 0)
        RBEMap(nRoi, ctGrid, d_roiInd, d_alpha, d_beta, d_rbemap, d_rbemap2, d_counter, d_final, nField, gpuId);
    else
        mkmRBEMap(nRoi, ctGrid, d_roiInd, d_rbemap, d_rbemap2, d_Z1Dmix, d_final, alpha0AndBeta0, nField, gpuId);

    checkCudaErrors(cudaMemcpy((float*)h_out_rbe.ptr, d_rbemap, sizeof(float) * gridSize, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy((float*)h_out_rbe2.ptr, d_rbemap2, sizeof(float) * gridSize, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy((float*)h_out_dose.ptr, d_final, sizeof(float) * gridSize * nField, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_final));
    checkCudaErrors(cudaFree(d_rbemap));
    checkCudaErrors(cudaFree(d_roiInd));
    checkCudaErrors(cudaFree(d_alpha));
    checkCudaErrors(cudaFree(d_beta));
    checkCudaErrors(cudaFree(d_Z1Dmix));
    checkCudaErrors(cudaFree(d_rbemap2));
    checkCudaErrors(cudaFree(d_counter));

    checkCudaErrors(cudaFreeArray(lqDataArray));
    checkCudaErrors(cudaDestroyTextureObject(lqDataObj));

    checkCudaErrors(cudaFreeArray(Z1DArray));
    checkCudaErrors(cudaDestroyTextureObject(Z1DObj));
}

EXPORT
void cuRotate3DArray(pybind11::array old_corner, //np.array   shape((3,), dtype=np.float32)  corner vox center position
                     pybind11::array old_resolution, //np.array   shape((3,), dtype=np.float32)
                     pybind11::array old_dims, //np.array   shape((3,), dtype=np.int32)
                     pybind11::array new_corner, //input np.zeros((3,), dtype=np.float32) return new array corner
                     pybind11::array new_resolution,
                     //input np.zeros((3,), dtype=np.float32) return new array resolution
                     pybind11::array new_dims, //input np.zeros((3,), dtype=np.int32)   return new array dims
                     pybind11::array old_array, //np.array   shape((size1,), dtype=np.float32)
                     pybind11::array new_array,
                     //np.array   shape((size2,), dtype=np.float32)  给一个大一些的size， 然后使用 a = a[:new_dims.x * new_dims.y * new_dims.z] 取
                     pybind11::array rotMat,
                     //np.array   shape((16,), dtype=np.float32)  a 4*4 transform matrix forward mapping
                     int type, //1, set outside values to 0; 0, set outside values to boundary value
                     int gpuid //0
)
{
    auto h_old_corner = pybind11::cast<pybind11::array_t<float>>(old_corner).request();
    auto h_old_resolution = pybind11::cast<pybind11::array_t<float>>(old_resolution).request();
    auto h_old_dims = pybind11::cast<pybind11::array_t<int>>(old_dims).request();
    auto h_new_corner = pybind11::cast<pybind11::array_t<float>>(new_corner).request();
    auto h_new_resolution = pybind11::cast<pybind11::array_t<float>>(new_resolution).request();
    auto h_new_dims = pybind11::cast<pybind11::array_t<int>>(new_dims).request();
    auto h_old_array = pybind11::cast<pybind11::array_t<float>>(old_array).request();
    auto h_new_array = pybind11::cast<pybind11::array_t<float>>(new_array).request();
    auto h_rotMat = pybind11::cast<pybind11::array_t<float>>(rotMat).request();

    Grid old_g, new_g;
    old_g.corner = vec3f(((float*)h_old_corner.ptr)[0], ((float*)h_old_corner.ptr)[1], ((float*)h_old_corner.ptr)[2]);
    old_g.resolution = vec3f(((float*)h_old_resolution.ptr)[0],
                             ((float*)h_old_resolution.ptr)[1],
                             ((float*)h_old_resolution.ptr)[2]);
    old_g.dims = vec3i(((int*)h_old_dims.ptr)[0], ((int*)h_old_dims.ptr)[1], ((int*)h_old_dims.ptr)[2]);
    old_g.upperCorner = old_g.corner
        + vec3f(old_g.dims.x - 1.0f, old_g.dims.y - 1.0f, old_g.dims.z - 1.0f) * old_g.resolution;
    rotate3DArray(new_g, old_g, (float*)h_new_array.ptr, (float*)h_old_array.ptr, (float*)h_rotMat.ptr, type, gpuid);

    ((float*)h_new_corner.ptr)[0] = new_g.corner.x;
    ((float*)h_new_corner.ptr)[1] = new_g.corner.y;
    ((float*)h_new_corner.ptr)[2] = new_g.corner.z;

    ((float*)h_new_resolution.ptr)[0] = new_g.resolution.x;
    ((float*)h_new_resolution.ptr)[1] = new_g.resolution.y;
    ((float*)h_new_resolution.ptr)[2] = new_g.resolution.z;

    ((int*)h_new_dims.ptr)[0] = new_g.dims.x;
    ((int*)h_new_dims.ptr)[1] = new_g.dims.y;
    ((int*)h_new_dims.ptr)[2] = new_g.dims.z;
}

PYBIND11_MODULE(cudaCalDose1, m)
{
    m.def("cuCalDose3", cudaCalDose3);
    m.def("cuCalDose3_", cudaCalDose3_);//suplementary
    m.def("cuCalDoseNorm", cudaCalDoseNorm);
    m.def("cuCalFluenceMapAlphaBeta", cudaCalFluenceMapAlphaBeta);
    m.def("cuFinalDose", cudaFinalDose);
    // m.def("calWEQ", calWEQ);
    m.def("cuRotate3DArray", cuRotate3DArray);
    m.def("cuFinalDoseAndRBEMap", cuFinalPhysDoseAndRBEDose);
    m.def("cuCalFluenceMapRBE", cudaCalFluenceRBE);
}
