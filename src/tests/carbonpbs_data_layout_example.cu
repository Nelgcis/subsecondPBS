/**
 * \file
 * \brief CarbonPBS data layout example (for cuCalDose3 inputs)
 *
 * This file is NOT part of the RayTraceDicom port itself. It documents the
 * expected shapes/layouts of the main CarbonPBS lookup tables:
 *   - rayweqData (3D)
 *   - iddData (2D)
 *   - profileData (3D)
 *   - subSpotData (3D)
 *
 * It is based on the access patterns in carbonPBS/cudaCalDose.cpp and
 * carbonPBS/deviceCalDose.cu.
 */

#include <iostream>
#include <vector>
#include <cmath>

// Only for documenting the layout.
#include "core/common.cuh"

static inline size_t idx3(size_t i, size_t j, size_t k, size_t dimJ, size_t dimK) {
    return (i * dimJ + j) * dimK + k; // k is fastest
}

int main() {
    // -----------------------------
    // Example dimensions
    // -----------------------------
    const int nEne = 2;

    // rayweqData (3D): dims are (nx, ny, nStep), but is uploaded as a 3D texture with
    //   width  = nStep (depth samples)
    //   height = ny
    //   depth  = nx
    const int nStep = 128;
    const int ny = 64;
    const int nx = 64;

    // iddData (2D): uploaded as a 2D texture with
    //   width  = nIdd
    //   height = nEne
    const int nIdd = 256;

    // profileData (3D): uploaded as a 3D texture with
    //   width  = nProfilePara = 2*nGauss + 1
    //   height = nProfileDepth
    //   depth  = nEne
    const int nProfileDepth = 256;
    const int nGauss = 2;
    const int nProfilePara = 2 * nGauss + 1;

    // subSpotData (3D): uploaded as a 3D texture with
    //   width  = 5   (deltax, deltay, weight, sigmax, sigmay)
    //   height = nSubspot
    //   depth  = nEne
    const int nSubspot = 9;

    // -----------------------------
    // 1) rayweqData layout
    // -----------------------------
    // In carbonPBS/cudaCalDose.cpp:
    //   vec3f rayweqSetting = ((vec3f*)h_weq.ptr)[0];
    //   int nStep = ((float*)h_weq.ptr)[2];
    //   int ny    = ((float*)h_weq.ptr)[5];
    //   int nx    = ((float*)h_weq.ptr)[8];
    //   create3DTexture(weqData+9, nStep, ny, nx)
    //
    // So the first 9 floats form 3 vec3f blocks:
    //   [0..2] depth (start, step, nStep)
    //   [3..5] y     (start, step, ny)
    //   [6..8] x     (start, step, nx)
    std::vector<float> rayweqData(9 + static_cast<size_t>(nx) * ny * nStep, 0.0f);
    // header
    rayweqData[0] = 0.0f;           // depth start (mm)
    rayweqData[1] = 1.0f;           // depth step  (mm)
    rayweqData[2] = static_cast<float>(nStep);
    rayweqData[3] = -32.0f;         // y start (mm)
    rayweqData[4] = 1.0f;           // y step  (mm)
    rayweqData[5] = static_cast<float>(ny);
    rayweqData[6] = -32.0f;         // x start (mm)
    rayweqData[7] = 1.0f;           // x step  (mm)
    rayweqData[8] = static_cast<float>(nx);

    float* rayweqVol = rayweqData.data() + 9;
    for (int ix = 0; ix < nx; ++ix) {
        for (int iy = 0; iy < ny; ++iy) {
            for (int k = 0; k < nStep; ++k) {
                // Simple example: WEQ depth equals geometric depth sample (not physically meaningful).
                rayweqVol[idx3(ix, iy, k, static_cast<size_t>(ny), static_cast<size_t>(nStep))] =
                    rayweqData[0] + k * rayweqData[1];
            }
        }
    }

    // -----------------------------
    // 2) iddData layout
    // -----------------------------
    // In carbonPBS/cudaCalDose.cpp:
    //   create2DTexture(idddata, width=shape[1], height=shape[0])
    // Which matches a flat array laid out as: idd[e][k] with k fastest.
    std::vector<float> iddData(static_cast<size_t>(nEne) * nIdd, 0.0f);
    for (int e = 0; e < nEne; ++e) {
        for (int k = 0; k < nIdd; ++k) {
            // Toy depth-dose curve: decays with depth.
            float depth_mm = static_cast<float>(k);
            iddData[static_cast<size_t>(e) * nIdd + k] = std::exp(-depth_mm / 50.0f);
        }
    }

    // Settings arrays are separate inputs in CarbonPBS:
    //   vec3f iddDepth = ((vec3f*)h_iddsetting.ptr)[0];
    //   iddDepth = (start_mm, step_mm, nIdd)
    vec3f iddDepthSetting(0.0f, 1.0f, static_cast<float>(nIdd));

    // -----------------------------
    // 3) profileData layout
    // -----------------------------
    // In carbonPBS/deviceCalDose.cu, for a given energy and depth index:
    //   w_j     = tex3D(profileData, j + 0.5, depthIdx + 0.5, eneIdx + 0.5)
    //   sigma_j = tex3D(profileData, (j+nGauss)+0.5, depthIdx + 0.5, eneIdx + 0.5)
    //   overall = tex3D(profileData, (2*nGauss)+0.5, depthIdx + 0.5, eneIdx + 0.5)
    // So per (ene, depth) we store:
    //   [0..nGauss-1]           weights
    //   [nGauss..2*nGauss-1]    sigmas
    //   [2*nGauss]              overallWeight
    std::vector<float> profileData(static_cast<size_t>(nEne) * nProfileDepth * nProfilePara, 0.0f);
    for (int e = 0; e < nEne; ++e) {
        for (int d = 0; d < nProfileDepth; ++d) {
            const size_t base = (static_cast<size_t>(e) * nProfileDepth + d) * nProfilePara;
            // weights
            for (int g = 0; g < nGauss; ++g) {
                profileData[base + g] = 1.0f / static_cast<float>(nGauss); // simple equal weights
            }
            // sigmas (mm)
            for (int g = 0; g < nGauss; ++g) {
                profileData[base + nGauss + g] = 2.0f + 0.5f * g; // arbitrary
            }
            profileData[base + 2 * nGauss] = 1.0f; // overallWeight
        }
    }

    // profilesetting is also a separate vec3f input:
    //   vec3f profileDepth = ((vec3f*)h_profilesetting.ptr)[0];
    //   profileDepth = (start_mm, step_mm, nProfileDepth)
    vec3f profileDepthSetting(0.0f, 1.0f, static_cast<float>(nProfileDepth));

    // -----------------------------
    // 4) subSpotData layout
    // -----------------------------
    // In carbonPBS/deviceCalDose.cu, per (eneIdx, isubspot):
    //   deltax       = tex3D(subspotData, 0, isubspot, eneIdx)
    //   deltay       = tex3D(subspotData, 1, isubspot, eneIdx)
    //   subspotweight= tex3D(subspotData, 2, isubspot, eneIdx)
    //   sigmax       = tex3D(subspotData, 3, isubspot, eneIdx)
    //   sigmay       = tex3D(subspotData, 4, isubspot, eneIdx)
    std::vector<float> subSpotData(static_cast<size_t>(nEne) * nSubspot * 5, 0.0f);
    for (int e = 0; e < nEne; ++e) {
        for (int s = 0; s < nSubspot; ++s) {
            const size_t base = (static_cast<size_t>(e) * nSubspot + s) * 5;
            subSpotData[base + 0] = 0.0f;                  // deltax (mm)
            subSpotData[base + 1] = 0.0f;                  // deltay (mm)
            subSpotData[base + 2] = 1.0f / nSubspot;       // weight
            subSpotData[base + 3] = 2.0f;                  // sigmax (mm)
            subSpotData[base + 4] = 2.0f;                  // sigmay (mm)
        }
    }

    // -----------------------------
    // Summary
    // -----------------------------
    std::cout << "CarbonPBS lookup table example sizes:\n";
    std::cout << "  rayweqData: header 9 floats + volume nx*ny*nStep = "
              << rayweqData.size() << " floats (" << (rayweqData.size() * sizeof(float)) << " bytes)\n";
    std::cout << "    nx=" << nx << " ny=" << ny << " nStep=" << nStep << "\n";

    std::cout << "  iddData: nEne*nIdd = " << iddData.size() << " floats (" << (iddData.size() * sizeof(float))
              << " bytes), nEne=" << nEne << " nIdd=" << nIdd << "\n";

    std::cout << "  profileData: nEne*nProfileDepth*nProfilePara = "
              << profileData.size() << " floats (" << (profileData.size() * sizeof(float)) << " bytes)\n";
    std::cout << "    nEne=" << nEne << " nProfileDepth=" << nProfileDepth << " nProfilePara=" << nProfilePara
              << " (nGauss=" << nGauss << ")\n";

    std::cout << "  subSpotData: nEne*nSubspot*5 = " << subSpotData.size() << " floats ("
              << (subSpotData.size() * sizeof(float)) << " bytes), nSubspot=" << nSubspot << "\n";

    std::cout << "\nSettings vec3f examples (start, step, count):\n";
    std::cout << "  iddDepthSetting    = (" << iddDepthSetting.x << ", " << iddDepthSetting.y << ", " << iddDepthSetting.z
              << ")\n";
    std::cout << "  profileDepthSetting= (" << profileDepthSetting.x << ", " << profileDepthSetting.y << ", "
              << profileDepthSetting.z << ")\n";

    return 0;
}
