#include "../include/core/raytracedicom_integration.h"
#include "../include/core/ray_tracing.h"
#include "../include/core/bev_ray_tracing.h"
#include "../include/algorithms/idd_sigma.h"
#include "../include/algorithms/superposition.h"
#include "../include/algorithms/fill_idd_and_sigma_params.cuh"
#include "../include/core/forward_declarations.h"
#include "../include/utils/utils.h"
#include "../include/core/common.cuh"
#include "../include/core/Macro.cuh"
#include "../include/utils/debug_tools.h"
#include "../include/utils/nuclear_table_utils.h"
#include "../include/utils/vector_find.h"
#include "../include/utils/vector_interpolate.h"
#include "../include/algorithms/convolution.h"
#include "../include/algorithms/prim_transf_kernel.h"
#include "../include/algorithms/transfer_param_struct_div3.cuh"
#include "../include/algorithms/transfer_param_helper.h"
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <cctype>
#include <sstream>


// RayTraceDicom CT image use(HU + 1000)


namespace {

struct FloatSummaryStats {
    double sumFinite = 0.0;
    float maxFinite = -std::numeric_limits<float>::infinity();
    float minFinite =  std::numeric_limits<float>::infinity();
    float minPositive = std::numeric_limits<float>::infinity();
    int countFinite = 0;
    int countPositive = 0;
    int countGT = 0;
    int countNaN = 0;
    int countInf = 0;
};

struct BeamStageTiming {
    double setupMs = 0.0;
    double textureSetupMs = 0.0;
    double cpbConvolutionMs = 0.0;
    double cpbToRayMs = 0.0;
    double rayWeightMs = 0.0;
    double bevTraceMs = 0.0;
    double weplReduceMs = 0.0;
    double iddSigmaMs = 0.0;
    double superpositionMs = 0.0;
    double layerTextureMs = 0.0;
    double doseTransferMs = 0.0;
    double layerAllocMs = 0.0;
    double layerCleanupMs = 0.0;
    double outputCopyMs = 0.0;
    double totalMs = 0.0;
};

struct LayerPerfTiming {
    size_t layerIdx = 0;
    float energy = 0.0f;
    int spots = 0;
    int activeFirst = 0;
    int activeLast = 0;
    int activeCount = 0;
    int rayDimsX = 0;
    int rayDimsY = 0;
    int superpRayDimsX = 0;
    int superpRayDimsY = 0;
    int transferBoxVoxels = 0;
    int transferGridX = 0;
    int transferGridY = 0;
    double allocMs = 0.0;
    double iddSigmaMs = 0.0;
    double superpositionMs = 0.0;
    double textureMs = 0.0;
    double transferMs = 0.0;
    double cleanupMs = 0.0;
    double totalMs = 0.0;
};

struct CarbonSigmaDebugSample {
    int x = 0;
    int y = 0;
    int step = 0;
    float idd = 0.0f;
    float weqDepth = 0.0f;
    float meanWidth = 0.0f;
    float rSigmaEff = 0.0f;
    float approxSigma = 0.0f;
    float phyDepth = 0.0f;
    float profileDepthIdx = 0.0f;
    float sumW = 0.0f;
    float profileSigmaRad = 0.0f;
    float initR2 = 0.0f;
    float carbonSigmaAxis = 0.0f;
};

struct SigmaFieldStats {
    int positiveIddCount = 0;
    int finiteSigmaCount = 0;
    int overflowLikeCount = 0;
    float minApproxSigma = 0.0f;
    float p50ApproxSigma = 0.0f;
    float p90ApproxSigma = 0.0f;
    float p99ApproxSigma = 0.0f;
    float maxApproxSigma = 0.0f;
    float maxMeanWidth = 0.0f;
    float minRSigmaEff = 0.0f;
};

static inline bool hostIsNaN(float v) {
    return std::isnan(v);
}

static inline bool hostIsFinite(float v) {
    return std::isfinite(v);
}

static bool rtdSigmaDebugEnabled() {
    const char* v = std::getenv("RTD_SIGMA_DEBUG");
    if (!v) return false;
    const std::string s(v);
    return !(s == "0" || s == "false" || s == "FALSE" || s == "off" || s == "OFF");
}

static bool rtdInputAuditEnabled() {
    const char* v = std::getenv("RTD_INPUT_AUDIT");
    if (!v) return false;
    const std::string s(v);
    return !(s == "0" || s == "false" || s == "FALSE" || s == "off" || s == "OFF");
}

static bool rtdTransferAuditEnabled() {
    const char* v = std::getenv("RTD_TRANSFER_AUDIT");
    if (!v) return false;
    const std::string s(v);
    return !(s == "0" || s == "false" || s == "FALSE" || s == "off" || s == "OFF");
}

static bool rtdSuperpOverflowAuditEnabled() {
    const char* v = std::getenv("RTD_SUPERP_OVERFLOW_DEBUG");
    if (!v) return false;
    const std::string s(v);
    return !(s == "0" || s == "false" || s == "FALSE" || s == "off" || s == "OFF");
}

static bool rtdEnvFlag(const char* name) {
    const char* v = std::getenv(name);
    if (!v) return false;
    const std::string s(v);
    return !(s == "0" || s == "false" || s == "FALSE" || s == "off" || s == "OFF");
}

static bool rtdHaloAuditEnabled() {
    return rtdEnvFlag("RTD_HALO_AUDIT");
}

static bool rtdPerfProfileEnabled() {
    return rtdEnvFlag("RTD_PERF_PROFILE");
}

static bool rtdPerfProfileLayersEnabled() {
    return rtdEnvFlag("RTD_PERF_PROFILE_LAYERS");
}

using PerfClock = std::chrono::high_resolution_clock;

static inline PerfClock::time_point perfNow() {
    return PerfClock::now();
}

static inline double perfElapsedMs(PerfClock::time_point start) {
    const auto end = PerfClock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

static bool isRepresentativeLayer(size_t layerIdx, int numLayers) {
    if (numLayers <= 0) return false;
    return layerIdx == 0 || layerIdx == static_cast<size_t>(numLayers / 2) || layerIdx + 1 == static_cast<size_t>(numLayers);
}

static float normalizeLongitudinalCutoffToMm(float cutoff, float lenToMm) {
    if (!(cutoff > 0.0f)) return 0.0f;
    if (lenToMm > 1.0f && cutoff < 100.0f) return cutoff * 10.0f;
    return cutoff;
}

static SigmaFieldStats summarizeSigmaField(const std::vector<float>& rayIdd,
                                          const std::vector<float>& rayRSigmaEff,
                                          const FillIddAndSigmaParams& params,
                                          int rayDimsX,
                                          int rayDimsY,
                                          int steps) {
    SigmaFieldStats out;
    const int plane = rayDimsX * rayDimsY;
    if (plane <= 0 || steps <= 0) return out;

    std::vector<float> approxSigmas;
    approxSigmas.reserve(rayIdd.size() / 8);
    float minRSigma = std::numeric_limits<float>::infinity();

    for (int step = 0; step < steps; ++step) {
        const vec2f vw = params.voxelWidth(step);
        const float meanWidth = 0.5f * (vw.x + vw.y);
        const float sigmaAtRadiusLimit = (float(MAX_SUPERP_RADIUS) + 0.5f) * meanWidth / KS_SIGMA_CUTOFF - SIGMA_DELTA;
        if (meanWidth > out.maxMeanWidth) out.maxMeanWidth = meanWidth;

        const int base = step * plane;
        for (int i = 0; i < plane; ++i) {
            const float idd = rayIdd[static_cast<size_t>(base + i)];
            if (!(idd > 0.0f)) continue;
            out.positiveIddCount++;

            const float rSigmaEff = rayRSigmaEff[static_cast<size_t>(base + i)];
            if (!(std::isfinite(rSigmaEff) && rSigmaEff > 0.0f)) continue;
            out.finiteSigmaCount++;
            if (rSigmaEff < minRSigma) minRSigma = rSigmaEff;

            const float approxSigma = HALF * meanWidth / (SQRT2 * rSigmaEff) - SIGMA_DELTA;
            approxSigmas.push_back(approxSigma);
            if (approxSigma > sigmaAtRadiusLimit) {
                out.overflowLikeCount++;
            }
        }
    }

    if (out.finiteSigmaCount == 0 || approxSigmas.empty()) {
        out.minRSigmaEff = 0.0f;
        return out;
    }

    std::sort(approxSigmas.begin(), approxSigmas.end());
    auto percentile = [&](float q) {
        const size_t idx = std::min(
            approxSigmas.size() - 1,
            static_cast<size_t>(std::floor(q * float(approxSigmas.size() - 1)))
        );
        return approxSigmas[idx];
    };

    out.minApproxSigma = approxSigmas.front();
    out.p50ApproxSigma = percentile(0.50f);
    out.p90ApproxSigma = percentile(0.90f);
    out.p99ApproxSigma = percentile(0.99f);
    out.maxApproxSigma = approxSigmas.back();
    out.minRSigmaEff = std::isfinite(minRSigma) ? minRSigma : 0.0f;
    return out;
}

static FloatSummaryStats summarizeFloatVector(const std::vector<float>& v, float gtThr) {
    FloatSummaryStats s;
    for (float x : v) {
        if (hostIsNaN(x)) {
            s.countNaN++;
            continue;
        }
        if (!hostIsFinite(x)) {
            s.countInf++;
            continue;
        }
        s.countFinite++;
        s.sumFinite += static_cast<double>(x);
        if (x > s.maxFinite) s.maxFinite = x;
        if (x < s.minFinite) s.minFinite = x;
        if (x > 0.0f) {
            s.countPositive++;
            if (x < s.minPositive) s.minPositive = x;
        }
        if (x > gtThr) s.countGT++;
    }
    if (s.countFinite == 0) {
        s.maxFinite = 0.0f;
        s.minFinite = 0.0f;
    }
    if (s.countPositive == 0) {
        s.minPositive = 0.0f;
    }
    return s;
}

static int totalSpotCountHost(const std::vector<int>& layerSpotCounts) {
    int total = 0;
    for (int c : layerSpotCounts) {
        if (c > 0) total += c;
    }
    return total;
}

static bool isMonotonicEnergies(const std::vector<float>& e, bool ascending) {
    if (e.size() < 2) return true;
    for (size_t i = 1; i < e.size(); ++i) {
        if (ascending) {
            if (e[i] < e[i - 1]) return false;
        } else {
            if (e[i] > e[i - 1]) return false;
        }
    }
    return true;
}

static void printZSliceSummary(const char* name, const std::vector<float>& vol,
                               int nx, int ny, int nz, float thr) {
    if (nx <= 0 || ny <= 0 || nz <= 0) return;
    const size_t sliceStride = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    if (vol.size() < sliceStride * static_cast<size_t>(nz)) return;

    std::vector<double> sumZ(nz, 0.0);
    std::vector<float> maxZ(nz, 0.0f);
    std::vector<int> cntZ(nz, 0);
    std::vector<int> nanZ(nz, 0);
    std::vector<int> infZ(nz, 0);

    for (int z = 0; z < nz; ++z) {
        const size_t base = static_cast<size_t>(z) * sliceStride;
        for (size_t i = 0; i < sliceStride; ++i) {
            const float v = vol[base + i];
            if (hostIsNaN(v)) {
                nanZ[z]++;
                continue;
            }
            if (!hostIsFinite(v)) {
                infZ[z]++;
                continue;
            }
            sumZ[z] += static_cast<double>(v);
            if (v > maxZ[z]) maxZ[z] = v;
            if (v > thr) cntZ[z]++;
        }
    }

    int active = 0;
    int first = -1;
    int last = -1;
    int maxSumZ = 0;
    double maxSum = -1.0;
    for (int z = 0; z < nz; ++z) {
        if (sumZ[z] > 0.0 || cntZ[z] > 0) {
            active++;
            if (first < 0) first = z;
            last = z;
        }
        if (sumZ[z] > maxSum) {
            maxSum = sumZ[z];
            maxSumZ = z;
        }
    }

    std::cout << "  " << name << " z-slice summary (thr=" << thr << "):\n";
    std::cout << "    activeSlices=" << active << "/" << nz;
    if (active > 0) {
        std::cout << "  activeRange=[" << first << "," << last << "]";
    }
    std::cout << "  maxSumSlice=" << maxSumZ << " (sum=" << maxSum << ")\n";

    // Print up to first 6 slices and the max-sum slice (helps detect 'only z=0')
    const int preview = std::min(nz, 6);
    for (int z = 0; z < preview; ++z) {
        std::cout << "    z=" << z << " sum=" << sumZ[z] << " max=" << maxZ[z]
                  << " cnt>thr=" << cntZ[z] << " nan=" << nanZ[z] << " inf=" << infZ[z] << "\n";
    }
    if (maxSumZ >= preview && maxSumZ < nz) {
        std::cout << "    z=" << maxSumZ << " sum=" << sumZ[maxSumZ] << " max=" << maxZ[maxSumZ]
                  << " cnt>thr=" << cntZ[maxSumZ] << " nan=" << nanZ[maxSumZ] << " inf=" << infZ[maxSumZ] << "\n";
    }
}

static void printCenterlinePeakSummary(const std::string& name,
                                       const std::vector<float>& vol,
                                       int nx,
                                       int ny,
                                       int nz,
                                       float thr,
                                       const char* zAxisLabel) {
    if (nx <= 0 || ny <= 0 || nz <= 0) return;
    const size_t sliceStride = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    if (vol.size() < sliceStride * static_cast<size_t>(nz)) return;

    const char* axis = (zAxisLabel != nullptr && zAxisLabel[0] != '\0') ? zAxisLabel : "z";
    const int cx = nx / 2;
    const int cy = ny / 2;
    const size_t centerOffset = static_cast<size_t>(cy) * static_cast<size_t>(nx) + static_cast<size_t>(cx);

    int peakZ = -1;
    float peakValue = -std::numeric_limits<float>::infinity();
    int first = -1;
    int last = -1;
    int nonzero = 0;
    int nanCount = 0;
    int infCount = 0;

    for (int z = 0; z < nz; ++z) {
        const float v = vol[static_cast<size_t>(z) * sliceStride + centerOffset];
        if (hostIsNaN(v)) {
            nanCount++;
            continue;
        }
        if (!hostIsFinite(v)) {
            infCount++;
            continue;
        }
        if (peakZ < 0 || v > peakValue) {
            peakZ = z;
            peakValue = v;
        }
        if (v > thr) {
            if (first < 0) first = z;
            last = z;
            nonzero++;
        }
    }

    if (peakZ < 0) {
        std::cout << "  " << name << " centerline peak: no finite samples"
                  << "  xy=(" << cx << "," << cy << ")"
                  << "  " << axis << "-count=" << nz
                  << "  nan=" << nanCount
                  << "  inf=" << infCount
                  << std::endl;
        return;
    }

    std::cout << "  " << name << " centerline peak:"
              << " xy=(" << cx << "," << cy << ")"
              << " " << axis << "=" << peakZ
              << " value=" << peakValue
              << " nonzero(>" << thr << ")=" << nonzero << "/" << nz
              << " nan=" << nanCount
              << " inf=" << infCount;
    if (first >= 0) {
        std::cout << " activeRange=[" << first << "," << last << "]";
    }
    std::cout << std::endl;
}

static void printStageVolumeSummary(const std::string& stageName,
                                    const std::vector<float>& vol,
                                    int nx,
                                    int ny,
                                    int nz,
                                    float thr,
                                    const char* zAxisLabel) {
    const size_t expected = static_cast<size_t>(std::max(nx, 0)) *
                            static_cast<size_t>(std::max(ny, 0)) *
                            static_cast<size_t>(std::max(nz, 0));
    if (nx <= 0 || ny <= 0 || nz <= 0 || vol.size() < expected) {
        std::cout << "[INPUT_AUDIT][WRAPPER][" << stageName << "] skipped invalid volume"
                  << " dims=(" << nx << "," << ny << "," << nz << ")"
                  << " size=" << vol.size()
                  << " expected>=" << expected
                  << std::endl;
        return;
    }

    const char* axis = (zAxisLabel != nullptr && zAxisLabel[0] != '\0') ? zAxisLabel : "z";
    const FloatSummaryStats st = summarizeFloatVector(vol, thr);
    std::cout << "[INPUT_AUDIT][WRAPPER][" << stageName << "]"
              << " dims=(" << nx << "," << ny << "," << nz << ")"
              << " zAxis=" << axis
              << " sum=" << st.sumFinite
              << " max=" << st.maxFinite
              << " nonzero(>" << thr << ")=" << st.countGT << "/" << expected
              << " nan=" << st.countNaN
              << " inf=" << st.countInf
              << std::endl;
    printZSliceSummary(stageName.c_str(), vol, nx, ny, nz, thr);
    printCenterlinePeakSummary(stageName, vol, nx, ny, nz, thr, axis);
}

static void printDoseGridSupportSummary(const std::string& stageName,
                                        const std::vector<float>& vol,
                                        int nx,
                                        int ny,
                                        int nz,
                                        float thr) {
    const size_t expected = static_cast<size_t>(std::max(nx, 0)) *
                            static_cast<size_t>(std::max(ny, 0)) *
                            static_cast<size_t>(std::max(nz, 0));
    if (nx <= 0 || ny <= 0 || nz <= 0 || vol.size() < expected) {
        std::cout << "[INPUT_AUDIT][WRAPPER][" << stageName << "_DOSE_GRID_SUPPORT]"
                  << " skipped invalid volume dims=(" << nx << "," << ny << "," << nz << ")"
                  << " size=" << vol.size()
                  << " expected>=" << expected
                  << std::endl;
        return;
    }

    int nnz = 0;
    double sum = 0.0;
    float maxV = 0.0f;
    int minX = nx, minY = ny, minZ = nz;
    int maxX = -1, maxY = -1, maxZ = -1;
    std::vector<double> sumY(static_cast<size_t>(ny), 0.0);
    std::vector<int> cntY(static_cast<size_t>(ny), 0);

    for (int x = 0; x < nx; ++x) {
        for (int y = 0; y < ny; ++y) {
            for (int z = 0; z < nz; ++z) {
                const size_t idx =
                    (static_cast<size_t>(x) * static_cast<size_t>(ny) + static_cast<size_t>(y)) *
                    static_cast<size_t>(nz) + static_cast<size_t>(z);
                const float v = vol[idx];
                if (!hostIsFinite(v)) continue;
                sum += static_cast<double>(v);
                if (v > maxV) maxV = v;
                if (v > thr) {
                    ++nnz;
                    sumY[static_cast<size_t>(y)] += static_cast<double>(v);
                    ++cntY[static_cast<size_t>(y)];
                    minX = std::min(minX, x);
                    minY = std::min(minY, y);
                    minZ = std::min(minZ, z);
                    maxX = std::max(maxX, x);
                    maxY = std::max(maxY, y);
                    maxZ = std::max(maxZ, z);
                }
            }
        }
    }

    std::cout << "[INPUT_AUDIT][WRAPPER][" << stageName << "_DOSE_GRID_SUPPORT]"
              << " layout=c_order_xyz"
              << " sum=" << sum
              << " max=" << maxV
              << " nonzero(>" << thr << ")=" << nnz << "/" << expected;
    if (nnz > 0) {
        std::cout << " bbox=[(" << minX << "," << minY << "," << minZ << ")-("
                  << maxX << "," << maxY << "," << maxZ << ")]";
    }
    std::cout << std::endl;

    int activeY = 0;
    int firstY = -1;
    int lastY = -1;
    int maxSumY = 0;
    double maxYSum = -1.0;
    for (int y = 0; y < ny; ++y) {
        if (cntY[static_cast<size_t>(y)] > 0) {
            ++activeY;
            if (firstY < 0) firstY = y;
            lastY = y;
        }
        if (sumY[static_cast<size_t>(y)] > maxYSum) {
            maxYSum = sumY[static_cast<size_t>(y)];
            maxSumY = y;
        }
    }
    std::cout << "  " << stageName << " y-slice support (dose-grid y, thr=" << thr << "):"
              << " activeSlices=" << activeY << "/" << ny;
    if (activeY > 0) {
        std::cout << " activeRange=[" << firstY << "," << lastY << "]";
    }
    std::cout << " maxSumY=" << maxSumY << " (sum=" << maxYSum << ")" << std::endl;

    const int probes[] = {0, 1, 14, 15, 18, 93, 94, 120, 146};
    for (int y : probes) {
        if (y < 0 || y >= ny) continue;
        std::cout << "    y=" << y
                  << " sum=" << sumY[static_cast<size_t>(y)]
                  << " cnt>thr=" << cntY[static_cast<size_t>(y)]
                  << std::endl;
    }
}

struct PlaneShapeStats {
    bool valid = false;
    int z = -1;
    double sum = 0.0;
    float max = 0.0f;
    int nonzero = 0;
    int maxX = -1;
    int maxY = -1;
    int minX = -1;
    int maxXBox = -1;
    int minY = -1;
    int maxYBox = -1;
    double centroidX = 0.0;
    double centroidY = 0.0;
    double sigmaX = 0.0;
    double sigmaY = 0.0;
};

struct VolumePeakStats {
    bool valid = false;
    int x = -1;
    int y = -1;
    int z = -1;
    float value = 0.0f;
};

static PlaneShapeStats summarizeXYPlane(const std::vector<float>& vol,
                                        int nx,
                                        int ny,
                                        int nz,
                                        int z,
                                        float thr) {
    PlaneShapeStats out;
    out.z = z;
    if (nx <= 0 || ny <= 0 || nz <= 0 || z < 0 || z >= nz) return out;
    const size_t sliceStride = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    if (vol.size() < sliceStride * static_cast<size_t>(nz)) return out;

    out.minX = nx;
    out.minY = ny;
    const size_t base = static_cast<size_t>(z) * sliceStride;
    double momentX = 0.0;
    double momentY = 0.0;

    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            const float v = vol[base + static_cast<size_t>(y) * static_cast<size_t>(nx) + static_cast<size_t>(x)];
            if (!hostIsFinite(v) || !(v > 0.0f)) continue;
            out.valid = true;
            out.sum += static_cast<double>(v);
            if (v > out.max) {
                out.max = v;
                out.maxX = x;
                out.maxY = y;
            }
            momentX += static_cast<double>(v) * static_cast<double>(x);
            momentY += static_cast<double>(v) * static_cast<double>(y);
            if (v > thr) {
                out.nonzero++;
                out.minX = std::min(out.minX, x);
                out.maxXBox = std::max(out.maxXBox, x);
                out.minY = std::min(out.minY, y);
                out.maxYBox = std::max(out.maxYBox, y);
            }
        }
    }

    if (!out.valid || !(out.sum > 0.0)) return out;

    out.centroidX = momentX / out.sum;
    out.centroidY = momentY / out.sum;

    double varX = 0.0;
    double varY = 0.0;
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            const float v = vol[base + static_cast<size_t>(y) * static_cast<size_t>(nx) + static_cast<size_t>(x)];
            if (!hostIsFinite(v) || !(v > 0.0f)) continue;
            const double dx = static_cast<double>(x) - out.centroidX;
            const double dy = static_cast<double>(y) - out.centroidY;
            varX += static_cast<double>(v) * dx * dx;
            varY += static_cast<double>(v) * dy * dy;
        }
    }
    out.sigmaX = std::sqrt(varX / out.sum);
    out.sigmaY = std::sqrt(varY / out.sum);

    if (out.nonzero == 0) {
        out.minX = out.maxX;
        out.maxXBox = out.maxX;
        out.minY = out.maxY;
        out.maxYBox = out.maxY;
    }
    return out;
}

static VolumePeakStats summarizeVolumePeak(const std::vector<float>& vol,
                                           int nx,
                                           int ny,
                                           int nz) {
    VolumePeakStats out;
    if (nx <= 0 || ny <= 0 || nz <= 0) return out;
    const size_t expected = static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);
    if (vol.size() < expected) return out;

    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                const float v = vol[(static_cast<size_t>(z) * static_cast<size_t>(ny) + static_cast<size_t>(y)) *
                                        static_cast<size_t>(nx) +
                                    static_cast<size_t>(x)];
                if (!hostIsFinite(v)) continue;
                if (!out.valid || v > out.value) {
                    out.valid = true;
                    out.value = v;
                    out.x = x;
                    out.y = y;
                    out.z = z;
                }
            }
        }
    }
    return out;
}

static void printVolumePeakLocationSummary(const std::string& stageName,
                                           const std::vector<float>& vol,
                                           int nx,
                                           int ny,
                                           int nz,
                                           const char* zAxisLabel,
                                           int logicalXOffset,
                                           int logicalYOffset) {
    const VolumePeakStats peak = summarizeVolumePeak(vol, nx, ny, nz);
    const char* axis = (zAxisLabel != nullptr && zAxisLabel[0] != '\0') ? zAxisLabel : "z";
    if (!peak.valid) {
        std::cout << "[INPUT_AUDIT][WRAPPER][" << stageName << "] no finite peak location" << std::endl;
        return;
    }
    std::cout << "[INPUT_AUDIT][WRAPPER][" << stageName << "] peak="
              << " (" << (peak.x + logicalXOffset)
              << "," << (peak.y + logicalYOffset)
              << "," << peak.z << ")"
              << " axis=" << axis
              << " value=" << peak.value
              << std::endl;
}

static void printSelectedPlaneShapeSummary(const std::string& stageName,
                                           const std::vector<float>& vol,
                                           int nx,
                                           int ny,
                                           int nz,
                                           const std::vector<int>& zSlices,
                                           float thr,
                                           int logicalXOffset,
                                           int logicalYOffset,
                                           const char* zAxisLabel) {
    const char* axis = (zAxisLabel != nullptr && zAxisLabel[0] != '\0') ? zAxisLabel : "z";
    std::vector<int> uniqueSlices = zSlices;
    std::sort(uniqueSlices.begin(), uniqueSlices.end());
    uniqueSlices.erase(std::unique(uniqueSlices.begin(), uniqueSlices.end()), uniqueSlices.end());
    for (int z : uniqueSlices) {
        if (z < 0 || z >= nz) continue;
        const PlaneShapeStats s = summarizeXYPlane(vol, nx, ny, nz, z, thr);
        if (!s.valid) {
            std::cout << "[INPUT_AUDIT][WRAPPER][" << stageName << "] "
                      << axis << "=" << z << " no positive finite samples" << std::endl;
            continue;
        }
        std::cout << "[INPUT_AUDIT][WRAPPER][" << stageName << "] "
                  << axis << "=" << z
                  << " sum=" << s.sum
                  << " max=" << s.max
                  << " maxPos=(" << (s.maxX + logicalXOffset) << "," << (s.maxY + logicalYOffset) << ")"
                  << " nnz(>" << thr << ")=" << s.nonzero
                  << " bbox=[(" << (s.minX + logicalXOffset) << "," << (s.minY + logicalYOffset)
                  << ")-(" << (s.maxXBox + logicalXOffset) << "," << (s.maxYBox + logicalYOffset) << ")]"
                  << " centroid=(" << (s.centroidX + logicalXOffset) << "," << (s.centroidY + logicalYOffset) << ")"
                  << " sigma=(" << s.sigmaX << "," << s.sigmaY << ")"
                  << std::endl;
    }
}

struct DepthProjectionStats {
    bool valid = false;
    int peakZ = -1;
    double peakSum = 0.0;
    float peakMax = 0.0f;
    std::vector<double> sumZ;
    std::vector<float> maxZ;
};

static DepthProjectionStats summarizeDepthProjection(const std::vector<float>& vol,
                                                     int nx,
                                                     int ny,
                                                     int nz) {
    DepthProjectionStats out;
    if (nx <= 0 || ny <= 0 || nz <= 0) return out;
    const size_t expected = static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);
    if (vol.size() < expected) return out;

    out.sumZ.assign(nz, 0.0);
    out.maxZ.assign(nz, 0.0f);
    const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    for (int z = 0; z < nz; ++z) {
        const size_t base = static_cast<size_t>(z) * plane;
        for (size_t off = 0; off < plane; ++off) {
            const float v = vol[base + off];
            if (!hostIsFinite(v)) continue;
            out.valid = true;
            out.sumZ[static_cast<size_t>(z)] += static_cast<double>(v);
            out.maxZ[static_cast<size_t>(z)] = std::max(out.maxZ[static_cast<size_t>(z)], v);
        }
        if (out.sumZ[static_cast<size_t>(z)] > out.peakSum) {
            out.peakSum = out.sumZ[static_cast<size_t>(z)];
            out.peakMax = out.maxZ[static_cast<size_t>(z)];
            out.peakZ = z;
        }
    }
    return out;
}

static void printDepthProjectionSummary(const std::string& stageName,
                                        const std::vector<float>& vol,
                                        int nx,
                                        int ny,
                                        int nz,
                                        const std::vector<int>& zSlices,
                                        const char* zAxisLabel) {
    const char* axis = (zAxisLabel != nullptr && zAxisLabel[0] != '\0') ? zAxisLabel : "z";
    const DepthProjectionStats s = summarizeDepthProjection(vol, nx, ny, nz);
    if (!s.valid) {
        std::cout << "[INPUT_AUDIT][WRAPPER][" << stageName << "] no finite depth projection" << std::endl;
        return;
    }

    std::cout << "[INPUT_AUDIT][WRAPPER][" << stageName << "] "
              << "peak" << axis << "=" << s.peakZ
              << " peakSum=" << s.peakSum
              << " peakMax=" << s.peakMax
              << std::endl;

    std::vector<int> uniqueSlices = zSlices;
    uniqueSlices.push_back(0);
    uniqueSlices.push_back(std::max(0, nz / 2));
    uniqueSlices.push_back(std::max(0, nz - 1));
    if (s.peakZ >= 0) uniqueSlices.push_back(s.peakZ);
    std::sort(uniqueSlices.begin(), uniqueSlices.end());
    uniqueSlices.erase(std::unique(uniqueSlices.begin(), uniqueSlices.end()), uniqueSlices.end());
    for (int z : uniqueSlices) {
        if (z < 0 || z >= nz) continue;
        std::cout << "[INPUT_AUDIT][WRAPPER][" << stageName << "] "
                  << axis << "=" << z
                  << " sum=" << s.sumZ[static_cast<size_t>(z)]
                  << " max=" << s.maxZ[static_cast<size_t>(z)]
                  << std::endl;
    }
}

static void printSuperpositionSliceDeltaSummary(const std::string& stageName,
                                                const PlaneShapeStats& before,
                                                const PlaneShapeStats& after,
                                                int beforeXOffset,
                                                int beforeYOffset,
                                                int afterXOffset,
                                                int afterYOffset,
                                                const char* zAxisLabel) {
    const char* axis = (zAxisLabel != nullptr && zAxisLabel[0] != '\0') ? zAxisLabel : "z";
    if (!before.valid || !after.valid) {
        std::cout << "[INPUT_AUDIT][WRAPPER][" << stageName << "] "
                  << axis << "=" << before.z << " delta skipped (missing active plane)"
                  << std::endl;
        return;
    }

    const int beforeMaxX = before.maxX + beforeXOffset;
    const int beforeMaxY = before.maxY + beforeYOffset;
    const int afterMaxX = after.maxX + afterXOffset;
    const int afterMaxY = after.maxY + afterYOffset;
    const int growLeft = before.minX + beforeXOffset - (after.minX + afterXOffset);
    const int growRight = (after.maxXBox + afterXOffset) - (before.maxXBox + beforeXOffset);
    const int growBottom = before.minY + beforeYOffset - (after.minY + afterYOffset);
    const int growTop = (after.maxYBox + afterYOffset) - (before.maxYBox + beforeYOffset);
    const int asymX = std::abs(growLeft - growRight);
    const int asymY = std::abs(growBottom - growTop);

    const char* heuristic = "symmetric_broadening";
    if (std::abs(afterMaxX - beforeMaxX) > 1 || std::abs(afterMaxY - beforeMaxY) > 1 || asymX > 1 || asymY > 1) {
        heuristic = "asymmetric_shift";
    } else if (after.nonzero <= before.nonzero) {
        heuristic = "under_spread_or_clipped";
    }

    std::cout << "[INPUT_AUDIT][WRAPPER][" << stageName << "] "
              << axis << "=" << before.z
              << " maxShift=(" << (afterMaxX - beforeMaxX) << "," << (afterMaxY - beforeMaxY) << ")"
              << " bboxGrow(L,R,B,T)=(" << growLeft << "," << growRight << "," << growBottom << "," << growTop << ")"
              << " sigmaDelta=(" << (after.sigmaX - before.sigmaX) << "," << (after.sigmaY - before.sigmaY) << ")"
              << " sumRatio=" << (before.sum > 0.0 ? (after.sum / before.sum) : 0.0)
              << " heuristic=" << heuristic
              << std::endl;
}

static bool computeDoseRoiBounds(const std::vector<int>& roiLinearIndices,
                                 const int3& doseDims,
                                 int3& minIdx,
                                 int3& maxIdx) {
    if (roiLinearIndices.empty()) return false;

    // Python/numpy C-order for shape (Nx, Ny, Nz):
    //   linear = ix*(Ny*Nz) + iy*Nz + iz
    const int planeYZ = doseDims.y * doseDims.z;
    if (planeYZ <= 0) return false;

    minIdx = make_int3(doseDims.x, doseDims.y, doseDims.z);
    maxIdx = make_int3(-1, -1, -1);

    for (int linear : roiLinearIndices) {
        if (linear < 0) continue;
        const int x = linear / planeYZ;
        const int rem = linear - x * planeYZ;
        const int y = rem / doseDims.z;
        const int z = rem % doseDims.z;

        if (x < 0 || x >= doseDims.x || y < 0 || y >= doseDims.y || z < 0 || z >= doseDims.z) {
            continue;
        }

        minIdx.x = std::min(minIdx.x, x);
        minIdx.y = std::min(minIdx.y, y);
        minIdx.z = std::min(minIdx.z, z);
        maxIdx.x = std::max(maxIdx.x, x);
        maxIdx.y = std::max(maxIdx.y, y);
        maxIdx.z = std::max(maxIdx.z, z);
    }

    return maxIdx.x >= minIdx.x && maxIdx.y >= minIdx.y && maxIdx.z >= minIdx.z;
}

struct RawSpotLattice {
    bool valid = false;
    bool usedFallback = false;  // true when spotDelta came from weqHeader, not canonical PB inference
    uint3 spotGridDims = make_uint3(0, 0, 0);
    float3 spotDelta = make_float3(0.0f, 0.0f, 0.0f);
    float3 spotOffset = make_float3(0.0f, 0.0f, 0.0f);
    int skippedZeroWeightSpots = 0;
    std::string failureReason;
    std::vector<float> spotWeights;
};

struct HaloLatticePlan {
    bool valid = false;
    int layerIdx = -1;
    int2 nucRayDims = make_int2(0, 0);
    size_t nucPlaneN = 0;
    float3 spotDelta = make_float3(0.0f, 0.0f, 0.0f);
    float3 spotOffset = make_float3(0.0f, 0.0f, 0.0f);
    int activeSpotCount = 0;
    int mappedRayCenters = 0;
    std::string failureReason;
    std::vector<int> rayToNucSpotIdx;
    std::vector<float> paddedSpotWeights;
};

static const std::vector<float>& getActiveWeqHeader(const RTDBeamSettings& beam);
static inline bool decodeSpotPosition(const RTDBeamSettings& beam,
                                      const std::vector<float>& weqHeader,
                                      float rawX,
                                      float rawY,
                                      float& outX,
                                      float& outY);
static bool rasterizeContinuousSpotToDensePlane(std::vector<float>& dense,
                                                int nx,
                                                int ny,
                                                int layer,
                                                float latticeX,
                                                float latticeY,
                                                float weight,
                                                double* writtenWeight);

static bool approxEq(float a, float b, float tol = 1e-4f) {
    return std::fabs(a - b) <= tol;
}

static int roundUpToMultiple(int value, int multiple) {
    if (multiple <= 1) return value;
    return ((value + multiple - 1) / multiple) * multiple;
}

// UNIT CONTRACT (must match upstream beam.getSpotIdxToGantry()):
//   spotDelta   : mm per physical PB grid step (typically 2-10 mm)
//   spotOffset  : mm of physical PB cell (0,0) in gantry frame
// Downstream consumers (nucIdxToFan at ~5132, spotDistInRays at ~3911 and ~4036)
// require mm. Do NOT multiply by lenToMm at construction sites; downstream sites
// must NOT re-multiply by lenToMm. See halo-energy-conservation.md (RC2) for the
// regression this contract is intended to prevent.
static bool buildHaloLatticePlan(const RawSpotLattice& rawSpotLattice,
                                 int layerIdx,
                                 const vec3f& cpbCorner,
                                 const vec3f& cpbResolution,
                                 const int2& rayDims,
                                 HaloLatticePlan& out,
                                 bool verbose) {
    out = HaloLatticePlan{};
    out.layerIdx = layerIdx;
    auto fail = [&](const std::string& msg) {
        out.failureReason = msg;
        if (verbose) {
            std::cerr << "  [RTD_HALO] layer=" << layerIdx << " " << msg << std::endl;
        }
        return false;
    };

    // [9.42 Halo Contract]
    // When RawSpotLattice is built via rasterizeContinuousSpotToDensePlane (bilinear weight
    // redistribution), the weights are split across adjacent texels. This is acceptable for
    // primary CPB helper convolution but NOT for halo nuclear PB mapping:
    // - Upstream: each physical PB weight maps to one fixed HPB spot index (deterministic)
    // - Current: if rasterization splits weights, different runs might map differently
    //   depending on float precision, or the weights become spread across HPB cells.
    // 
    // For now, we proceed with the current weights and rely on RawSpotLattice validation
    // (9.43) to ensure the source was indeed from a true physical PB grid, not a sparse
    // rasterization artifact. If bilinear drift is detected here, fail explicitly.
    // TODO: Implement 9.43 separation to provide an explicit non-rasterized HPB view.

    if (!(cpbResolution.x > 0.0f) || !(cpbResolution.y > 0.0f)) {
        return fail("CPB/ray spacing must be positive");
    }
    if (rayDims.x <= 0 || rayDims.y <= 0) {
        return fail("rayDims must be positive");
    }
    if (!rawSpotLattice.valid) {
        return fail("rawSpotLattice must be valid before halo planning");
    }
    if (layerIdx < 0 || layerIdx >= static_cast<int>(rawSpotLattice.spotGridDims.z)) {
        return fail("layerIdx is outside rawSpotLattice.spotGridDims.z");
    }
    const int rawX = static_cast<int>(rawSpotLattice.spotGridDims.x);
    const int rawY = static_cast<int>(rawSpotLattice.spotGridDims.y);
    if (rawX <= 0 || rawY <= 0) {
        return fail("rawSpotLattice dimensions must be positive");
    }
    const float dxRef = rawSpotLattice.spotDelta.x;
    const float dyRef = rawSpotLattice.spotDelta.y;
    const float oxRef = rawSpotLattice.spotOffset.x;
    const float oyRef = rawSpotLattice.spotOffset.y;
    if (!(dxRef > 0.0f) || !(dyRef > 0.0f)) {
        return fail("rawSpotLattice spotDelta must be positive");
    }
    const size_t densePlaneN = static_cast<size_t>(rawX) * static_cast<size_t>(rawY);
    const size_t denseWeightsN = densePlaneN * static_cast<size_t>(rawSpotLattice.spotGridDims.z);
    if (rawSpotLattice.spotWeights.size() != denseWeightsN) {
        return fail("rawSpotLattice weight buffer size does not match dimensions");
    }

    const int nucX = roundUpToMultiple(rawX, SUPERP_TILE_X);
    const int nucY = roundUpToMultiple(rawY, SUPERP_TILE_Y);
    const size_t nucPlaneN = static_cast<size_t>(nucX) * static_cast<size_t>(nucY);

    out.nucRayDims = make_int2(nucX, nucY);
    out.nucPlaneN = nucPlaneN;
    out.spotDelta = make_float3(dxRef, dyRef, 0.0f);
    out.spotOffset = make_float3(oxRef, oyRef, 0.0f);
    out.rayToNucSpotIdx.assign(static_cast<size_t>(rayDims.x) * static_cast<size_t>(rayDims.y), -1);
    out.paddedSpotWeights.assign(nucPlaneN, 0.0f);

    const size_t layerPlaneOffset = static_cast<size_t>(layerIdx) * densePlaneN;
    for (int spotY = 0; spotY < rawY; ++spotY) {
        for (int spotX = 0; spotX < rawX; ++spotX) {
            const size_t denseIdx = layerPlaneOffset +
                                    static_cast<size_t>(spotY) * static_cast<size_t>(rawX) +
                                    static_cast<size_t>(spotX);
            const float weight = rawSpotLattice.spotWeights[denseIdx];
            out.paddedSpotWeights[static_cast<size_t>(spotY) * static_cast<size_t>(nucX) +
                                 static_cast<size_t>(spotX)] = weight;
            if (weight > 0.0f) {
                out.activeSpotCount++;
            }
        }
    }

    int mapped = 0;
    for (int spotY = 0; spotY < rawY; ++spotY) {
        const float gantryY = oyRef + static_cast<float>(spotY) * dyRef;
        const int rayY = static_cast<int>(std::lround((gantryY - cpbCorner.y) / cpbResolution.y));
        if (rayY < 0 || rayY >= rayDims.y) {
            return fail("dense halo lattice row mapped outside active ray grid");
        }

        for (int spotX = 0; spotX < rawX; ++spotX) {
            const float gantryX = oxRef + static_cast<float>(spotX) * dxRef;
            const int rayX = static_cast<int>(std::lround((gantryX - cpbCorner.x) / cpbResolution.x));
            if (rayX < 0 || rayX >= rayDims.x) {
                return fail("dense halo lattice column mapped outside active ray grid");
            }

            out.rayToNucSpotIdx[static_cast<size_t>(rayY) * static_cast<size_t>(rayDims.x) + static_cast<size_t>(rayX)] =
                spotY * nucX + spotX;
            mapped++;
        }
    }

    if (mapped <= 0) {
        return fail("no raw spot lattice positions mapped onto the active ray grid");
    }

    out.mappedRayCenters = mapped;
    out.valid = true;
    if (verbose) {
        std::cout << "  [RTD_HALO] layer=" << layerIdx
                  << " nucRayDims=(" << out.nucRayDims.x << "," << out.nucRayDims.y << ")"
                  << " rawDims=(" << rawX << "," << rawY << ",1)"
                  << " spotOffset=(" << out.spotOffset.x << "," << out.spotOffset.y << "," << out.spotOffset.z << ")"
                  << " spotDelta=(" << out.spotDelta.x << "," << out.spotDelta.y << "," << out.spotDelta.z << ")"
                  << " activeSpots=" << out.activeSpotCount
                  << " mappedRayCenters=" << out.mappedRayCenters
                  << " source=primary_dense_spot_lattice"
                  << std::endl;
    }
    return true;
}

static bool buildExplicitPhysicalPBHaloLatticePlan(const RTDBeamSettings& beam,
                                                   int layerIdx,
                                                   const vec3f& cpbCorner,
                                                   const vec3f& cpbResolution,
                                                   const int2& rayDims,
                                                   HaloLatticePlan& out,
                                                   bool verbose) {
    out = HaloLatticePlan{};
    out.layerIdx = layerIdx;
    auto fail = [&](const std::string& msg) {
        out.failureReason = msg;
        if (verbose) {
            std::cerr << "  [RTD_HALO] layer=" << layerIdx
                      << " source=explicit_physical_pb_rasterized_lattice " << msg << std::endl;
        }
        return false;
    };

    const int numLayers = static_cast<int>(beam.energies.size());
    if (layerIdx < 0 || layerIdx >= numLayers) {
        return fail("layerIdx is outside beam.energies");
    }
    if (beam.layerSpotDeltas.size() != static_cast<size_t>(numLayers)) {
        return fail("explicit layerSpotDeltas must contain one spacing per energy layer");
    }
    if (beam.layerSpotCounts.size() != static_cast<size_t>(numLayers)) {
        return fail("layerSpotCounts size must match beam.energies size");
    }
    if (beam.spotPositions.empty() || beam.spotWeights.empty()) {
        return fail("missing spotPositions/spotWeights input");
    }
    if ((beam.spotPositions.size() % 2u) != 0u) {
        return fail("spotPositions does not contain [N][2] rows");
    }
    if (!(cpbResolution.x > 0.0f) || !(cpbResolution.y > 0.0f)) {
        return fail("CPB/ray spacing must be positive");
    }
    if (rayDims.x <= 0 || rayDims.y <= 0) {
        return fail("rayDims must be positive");
    }

    int totalSpots = 0;
    int layerSpotOffset = 0;
    for (int l = 0; l < numLayers; ++l) {
        const int count = beam.layerSpotCounts[static_cast<size_t>(l)];
        if (count <= 0) {
            return fail("layerSpotCounts must be positive for every layer");
        }
        if (l < layerIdx) layerSpotOffset += count;
        totalSpots += count;
    }
    if (static_cast<int>(beam.spotPositions.size() / 2u) != totalSpots) {
        return fail("spotPositions row count does not match summed layerSpotCounts");
    }
    if (static_cast<int>(beam.spotWeights.size()) != totalSpots) {
        return fail("spotWeights size does not match summed layerSpotCounts");
    }

    const float2 layerDelta = beam.layerSpotDeltas[static_cast<size_t>(layerIdx)];
    const float dxRef = layerDelta.x;
    const float dyRef = layerDelta.y;
    if (!(dxRef > 0.0f) || !(dyRef > 0.0f) ||
        !hostIsFinite(dxRef) || !hostIsFinite(dyRef)) {
        return fail("explicit layerSpotDeltas must be finite and positive");
    }

    const std::vector<float>& weqHeader = getActiveWeqHeader(beam);
    if (beam.spotPositionsAreIndices && weqHeader.size() < 9u) {
        return fail("spotPositionsAreIndices requires a 9-value WEQ header");
    }

    struct DecodedSpot {
        float x = 0.0f;
        float y = 0.0f;
        float weight = 0.0f;
    };

    const int layerSpotCount = beam.layerSpotCounts[static_cast<size_t>(layerIdx)];
    std::vector<DecodedSpot> decoded;
    decoded.reserve(static_cast<size_t>(layerSpotCount));

    float minX = INF;
    float maxX = -INF;
    float minY = INF;
    float maxY = -INF;
    double inputWeightSum = 0.0;
    int positiveInputSpots = 0;

    for (int i = 0; i < layerSpotCount; ++i) {
        const int spotIdx = layerSpotOffset + i;
        const float rawX = beam.spotPositions[static_cast<size_t>(spotIdx) * 2u + 0u];
        const float rawY = beam.spotPositions[static_cast<size_t>(spotIdx) * 2u + 1u];
        const float weight = beam.spotWeights[static_cast<size_t>(spotIdx)];
        if (!hostIsFinite(weight)) {
            return fail("non-finite spot weight at localSpot=" + std::to_string(i));
        }
        if (weight < 0.0f) {
            return fail("negative spot weight at localSpot=" + std::to_string(i));
        }

        float x = 0.0f;
        float y = 0.0f;
        if (!decodeSpotPosition(beam, weqHeader, rawX, rawY, x, y)) {
            return fail("failed to decode spot position at localSpot=" + std::to_string(i));
        }
        if (!hostIsFinite(x) || !hostIsFinite(y)) {
            return fail("decoded spot position is non-finite at localSpot=" + std::to_string(i));
        }

        decoded.push_back(DecodedSpot{x, y, weight});
        minX = std::min(minX, x);
        maxX = std::max(maxX, x);
        minY = std::min(minY, y);
        maxY = std::max(maxY, y);
        inputWeightSum += static_cast<double>(weight);
        if (weight > 0.0f) {
            ++positiveInputSpots;
        }
    }

    if (decoded.empty()) {
        return fail("layer contains no decoded spots");
    }
    if (!(minX <= maxX) || !(minY <= maxY)) {
        return fail("decoded spot bounds are invalid");
    }

    const float originX = minX;
    const float originY = minY;
    const float extentX = (maxX - originX) / dxRef;
    const float extentY = (maxY - originY) / dyRef;
    if (!hostIsFinite(extentX) || !hostIsFinite(extentY) || extentX < -1.0e-4f || extentY < -1.0e-4f) {
        return fail("decoded spot extent is invalid for explicit physical PB rasterization");
    }

    // Add one high-side cell so bilinear rasterization of an off-lattice max
    // coordinate can write its upper neighbor without losing weight.
    const int rawX = std::max(1, static_cast<int>(std::floor(std::max(0.0f, extentX))) + 2);
    const int rawY = std::max(1, static_cast<int>(std::floor(std::max(0.0f, extentY))) + 2);
    if (rawX <= 0 || rawY <= 0) {
        return fail("explicit physical PB rasterized lattice dimensions are non-positive");
    }

    const int nucX = roundUpToMultiple(rawX, SUPERP_TILE_X);
    const int nucY = roundUpToMultiple(rawY, SUPERP_TILE_Y);
    const size_t nucPlaneN = static_cast<size_t>(nucX) * static_cast<size_t>(nucY);
    const size_t rawPlaneN = static_cast<size_t>(rawX) * static_cast<size_t>(rawY);
    const size_t rayPlaneN = static_cast<size_t>(rayDims.x) * static_cast<size_t>(rayDims.y);
    constexpr size_t kMaxExplicitPhysicalHaloCells = 4ull * 1024ull * 1024ull;
    if (rawX > rayDims.x || rawY > rayDims.y || rawPlaneN > rayPlaneN) {
        return fail("explicit physical PB rasterized lattice is larger than primary ray lattice: rawDims=(" +
                    std::to_string(rawX) + "," + std::to_string(rawY) + ") rayDims=(" +
                    std::to_string(rayDims.x) + "," + std::to_string(rayDims.y) + ")");
    }
    if (nucPlaneN > kMaxExplicitPhysicalHaloCells) {
        return fail("explicit physical PB rasterized lattice exceeds sanity cap: paddedCells=" +
                    std::to_string(nucPlaneN) + " cap=" +
                    std::to_string(kMaxExplicitPhysicalHaloCells));
    }

    out.nucRayDims = make_int2(nucX, nucY);
    out.nucPlaneN = nucPlaneN;
    out.spotDelta = make_float3(dxRef, dyRef, 0.0f);
    out.spotOffset = make_float3(originX, originY, 0.0f);
    out.rayToNucSpotIdx.assign(static_cast<size_t>(rayDims.x) * static_cast<size_t>(rayDims.y), -1);
    out.paddedSpotWeights.assign(nucPlaneN, 0.0f);

    double rasterizedWeightSum = 0.0;
    double fracAbsSumX = 0.0;
    double fracAbsSumY = 0.0;
    float maxNearestFracX = 0.0f;
    float maxNearestFracY = 0.0f;
    int offLatticeInputSpots = 0;
    for (const DecodedSpot& spot : decoded) {
        if (spot.weight == 0.0f) continue;
        const float latticeX = (spot.x - originX) / dxRef;
        const float latticeY = (spot.y - originY) / dyRef;
        if (!hostIsFinite(latticeX) || !hostIsFinite(latticeY)) {
            return fail("decoded spot produced non-finite explicit physical PB lattice coordinate");
        }
        const float nearestFracX = std::fabs(latticeX - std::round(latticeX));
        const float nearestFracY = std::fabs(latticeY - std::round(latticeY));
        fracAbsSumX += static_cast<double>(nearestFracX);
        fracAbsSumY += static_cast<double>(nearestFracY);
        maxNearestFracX = std::max(maxNearestFracX, nearestFracX);
        maxNearestFracY = std::max(maxNearestFracY, nearestFracY);
        if (nearestFracX > 1.0e-3f || nearestFracY > 1.0e-3f) {
            ++offLatticeInputSpots;
        }
        if (!rasterizeContinuousSpotToDensePlane(out.paddedSpotWeights,
                                                 nucX,
                                                 nucY,
                                                 0,
                                                 latticeX,
                                                 latticeY,
                                                 spot.weight,
                                                 &rasterizedWeightSum)) {
            return fail("positive-weight decoded spot could not be rasterized onto explicit physical PB lattice");
        }
    }
    const double conservationTol = std::max(1.0e-3, std::fabs(inputWeightSum) * 1.0e-5);
    if (std::fabs(rasterizedWeightSum - inputWeightSum) > conservationTol) {
        return fail("explicit physical PB rasterization did not conserve weight: input=" +
                    std::to_string(inputWeightSum) + " rasterized=" +
                    std::to_string(rasterizedWeightSum) + " tol=" +
                    std::to_string(conservationTol));
    }

    for (float w : out.paddedSpotWeights) {
        if (w > 0.0f) out.activeSpotCount++;
    }

    int mapped = 0;
    for (int spotY = 0; spotY < rawY; ++spotY) {
        const float gantryY = originY + static_cast<float>(spotY) * dyRef;
        const int rayY = static_cast<int>(std::lround((gantryY - cpbCorner.y) / cpbResolution.y));
        if (rayY < 0 || rayY >= rayDims.y) {
            return fail("explicit physical PB rasterized lattice row mapped outside active ray grid");
        }
        for (int spotX = 0; spotX < rawX; ++spotX) {
            const float gantryX = originX + static_cast<float>(spotX) * dxRef;
            const int rayX = static_cast<int>(std::lround((gantryX - cpbCorner.x) / cpbResolution.x));
            if (rayX < 0 || rayX >= rayDims.x) {
                return fail("explicit physical PB rasterized lattice column mapped outside active ray grid");
            }
            int& mappedIdx = out.rayToNucSpotIdx[static_cast<size_t>(rayY) *
                                                 static_cast<size_t>(rayDims.x) +
                                                 static_cast<size_t>(rayX)];
            if (mappedIdx >= 0) {
                return fail("multiple explicit physical PB rasterized cells mapped to the same primary ray center");
            }
            mappedIdx = spotY * nucX + spotX;
            mapped++;
        }
    }
    if (mapped <= 0) {
        return fail("no explicit physical PB rasterized lattice positions mapped onto the active ray grid");
    }

    out.mappedRayCenters = mapped;
    out.valid = true;
    if (verbose) {
        std::cout << "  [RTD_HALO] layer=" << layerIdx
                  << " nucRayDims=(" << out.nucRayDims.x << "," << out.nucRayDims.y << ")"
                  << " rawDims=(" << rawX << "," << rawY << ",1)"
                  << " spotOffset=(" << out.spotOffset.x << "," << out.spotOffset.y << "," << out.spotOffset.z << ")"
                  << " spotDelta=(" << out.spotDelta.x << "," << out.spotDelta.y << "," << out.spotDelta.z << ")"
                  << " inputSpots=" << layerSpotCount
                  << " positiveInputSpots=" << positiveInputSpots
                  << " inputWeightSum=" << inputWeightSum
                  << " rasterizedWeightSum=" << rasterizedWeightSum
                  << " activeSpots=" << out.activeSpotCount
                  << " mappedRayCenters=" << out.mappedRayCenters
                  << " decodedBounds=(" << minX << "," << maxX << "," << minY << "," << maxY << ")"
                  << " fractionalOffsetMax=(" << maxNearestFracX << "," << maxNearestFracY << ")"
                  << " fractionalOffsetMean=("
                  << (positiveInputSpots > 0 ? fracAbsSumX / static_cast<double>(positiveInputSpots) : 0.0)
                  << ","
                  << (positiveInputSpots > 0 ? fracAbsSumY / static_cast<double>(positiveInputSpots) : 0.0)
                  << ")"
                  << " offLatticeInputSpots=" << offLatticeInputSpots
                  << " source=explicit_physical_pb_rasterized_lattice"
                  << std::endl;
    }
    return true;
}

static bool rasterizeContinuousSpotToDensePlane(std::vector<float>& dense,
                                                int nx,
                                                int ny,
                                                int layer,
                                                float latticeX,
                                                float latticeY,
                                                float weight,
                                                double* writtenWeight = nullptr) {
    if (nx <= 0 || ny <= 0 || layer < 0 || dense.empty()) return false;

    const int ix0 = static_cast<int>(std::floor(latticeX));
    const int iy0 = static_cast<int>(std::floor(latticeY));
    const int ix1 = ix0 + 1;
    const int iy1 = iy0 + 1;
    const float tx = latticeX - static_cast<float>(ix0);
    const float ty = latticeY - static_cast<float>(iy0);

    double sumWritten = 0.0;
    auto accum = [&](int ix, int iy, float frac) {
        if (!(frac > 0.0f)) return;
        if (ix < 0 || ix >= nx || iy < 0 || iy >= ny) return;
        const size_t idx =
            (static_cast<size_t>(layer) * static_cast<size_t>(ny) + static_cast<size_t>(iy)) * static_cast<size_t>(nx) +
            static_cast<size_t>(ix);
        dense[idx] += weight * frac;
        sumWritten += static_cast<double>(weight) * static_cast<double>(frac);
    };

    accum(ix0, iy0, (1.0f - tx) * (1.0f - ty));
    accum(ix1, iy0, tx * (1.0f - ty));
    accum(ix0, iy1, (1.0f - tx) * ty);
    accum(ix1, iy1, tx * ty);

    if (writtenWeight != nullptr) {
        *writtenWeight += sumWritten;
    }
    return sumWritten > 0.0;
}

static bool alignGridToReferencePhase(float minCoord,
                                      float maxCoord,
                                      float refOrigin,
                                      float step,
                                      int alignedMultiple,
                                      float& alignedOrigin,
                                      int& alignedCount,
                                      int* firstStepOut = nullptr,
                                      int* lastStepOut = nullptr) {
    if (!(step > 0.0f) || !hostIsFinite(minCoord) || !hostIsFinite(maxCoord) || !hostIsFinite(refOrigin)) {
        return false;
    }
    if (maxCoord < minCoord) std::swap(minCoord, maxCoord);

    const float eps = 1.0e-4f;
    const int firstStep = static_cast<int>(std::ceil((minCoord - refOrigin) / step - eps));
    const int lastStep = static_cast<int>(std::floor((maxCoord - refOrigin) / step + eps));
    if (lastStep < firstStep) return false;

    alignedOrigin = refOrigin + static_cast<float>(firstStep) * step;
    alignedCount = std::max(1, lastStep - firstStep + 1);
    if (alignedMultiple > 1) {
        alignedCount = ((alignedCount + alignedMultiple - 1) / alignedMultiple) * alignedMultiple;
    }
    if (firstStepOut != nullptr) *firstStepOut = firstStep;
    if (lastStepOut != nullptr) *lastStepOut = lastStep;
    return true;
}

static const std::vector<float>& getActiveWeqHeader(const RTDBeamSettings& beam) {
    if (beam.waterEquivalence.size() >= 9) return beam.waterEquivalence;
    return beam.rayWeqHeader;
}

static inline bool decodeSpotPosition(const RTDBeamSettings& beam,
                                      const std::vector<float>& weqHeader,
                                      float rawX,
                                      float rawY,
                                      float& outX,
                                      float& outY) {
    if (!beam.spotPositionsAreIndices) {
        outX = rawX;
        outY = rawY;
        return true;
    }
    if (weqHeader.size() < 9) return false;

    // CarbonPBS idbeamxy is exported as rayweq texture coordinates centered on
    // texels via +0.5. Convert back to the physical reference-plane coordinate
    // of the corresponding texel center.
    outX = weqHeader[6] + (rawX - 0.5f) * weqHeader[7];
    outY = weqHeader[3] + (rawY - 0.5f) * weqHeader[4];
    return true;
}

static bool validateWrapperGlobalInputs(const float* ctData,
                                        const int3& ctDims,
                                        const float3& ctResolution,
                                        float* doseData,
                                        const int3& doseDims,
                                        const float3& doseResolution,
                                        const RTDBeamSettings* beamSettings,
                                        size_t numBeams,
                                        const RTDEnergyStruct* energyData) {
    auto fail = [](const char* msg) {
        std::cerr << "[RTD_ENTRY_ASSERT] " << msg << std::endl;
        return false;
    };

    if (ctData == nullptr) return fail("ctData must not be null");
    if (doseData == nullptr) return fail("doseData must not be null");
    if (beamSettings == nullptr) return fail("beamSettings must not be null");
    if (energyData == nullptr) return fail("energyData must not be null");
    if (numBeams == 0) return fail("numBeams must be positive");

    if (ctDims.x <= 0 || ctDims.y <= 0 || ctDims.z <= 0) return fail("ctDims must be positive");
    if (doseDims.x <= 0 || doseDims.y <= 0 || doseDims.z <= 0) return fail("doseDims must be positive");
    if (!(ctResolution.x > 0.0f && ctResolution.y > 0.0f && ctResolution.z > 0.0f)) {
        return fail("ctResolution must be positive");
    }
    if (!(doseResolution.x > 0.0f && doseResolution.y > 0.0f && doseResolution.z > 0.0f)) {
        return fail("doseResolution must be positive");
    }

    if (!(energyData->nEnergies > 0 && energyData->nEnergySamples > 0)) {
        return fail("energyData dimensions must be positive");
    }
    if (energyData->energiesPerU.size() != static_cast<size_t>(energyData->nEnergies)) {
        return fail("energiesPerU size must match energyData->nEnergies");
    }
    if (energyData->peakDepths.size() != static_cast<size_t>(energyData->nEnergies)) {
        return fail("peakDepths size must match energyData->nEnergies");
    }
    if (energyData->scaleFacts.size() != static_cast<size_t>(energyData->nEnergies)) {
        return fail("scaleFacts size must match energyData->nEnergies");
    }
    if (energyData->ciddMatrix.size() !=
        static_cast<size_t>(energyData->nEnergies) * static_cast<size_t>(energyData->nEnergySamples)) {
        return fail("ciddMatrix size must equal nEnergies*nEnergySamples");
    }
    if (energyData->densityVector.empty() || energyData->spVector.empty() || energyData->rRlVector.empty()) {
        return fail("density, stopping-power, and radiation-length LUTs must not be empty");
    }

    return true;
}

static void enforceNuclearCorrectionContract(const RTDEnergyStruct* energyData,
                                             bool nuclearCorrection) {
    if (!nuclearCorrection) return;
    if (energyData == nullptr) {
        throw std::runtime_error("nuclear_correction=true requires a non-null RTDEnergyStruct");
    }
#ifndef NUCLEAR_CORR
    throw std::runtime_error(
        "nuclear_correction=true requested, but this binary was built with NUCLEAR_CORR=OFF");
#else
    if (!rtd::nuclear::hasNuclearTables(*energyData)) {
        throw std::runtime_error(
            "nuclear_correction=true requested, but RTDEnergyStruct does not carry nuclear_weight/sigma tables");
    }
    if (!energyData->nuclearTablesAlignedToPrimaryAxis ||
        !rtd::nuclear::nuclearPayloadMatchesPrimaryAxis(*energyData)) {
        throw std::runtime_error(
            "nuclear_correction=true requested, but nuclear LUTs are not aligned to the active primary energy-depth axis");
    }
#endif
}

static bool validateWrapperEntryBeam(const RTDBeamSettings& beam, size_t beamIdx) {
    auto fail = [beamIdx](const std::string& msg) {
        std::cerr << "[RTD_ENTRY_ASSERT] beamIdx=" << beamIdx << " " << msg << std::endl;
        return false;
    };

    if (beam.energies.empty()) return fail("beam.energies must not be empty");
    if (beam.maxSubspotsPerLayer <= 0) return fail("beam.maxSubspotsPerLayer must be positive");

    const size_t expectedSubspotN =
        beam.energies.size() * static_cast<size_t>(beam.maxSubspotsPerLayer) * 5ull;
    if (beam.subspotData.size() != expectedSubspotN) {
        return fail("beam.subspotData size must equal numLayers*maxSubspotsPerLayer*5");
    }

    if (!beam.layerSpotCounts.empty()) {
        if (beam.layerSpotCounts.size() != beam.energies.size()) {
            return fail("layerSpotCounts size must match beam.energies size");
        }
        const int totalSpots = totalSpotCountHost(beam.layerSpotCounts);
        if (!beam.spotWeights.empty() && totalSpots != static_cast<int>(beam.spotWeights.size())) {
            return fail("sum(layerSpotCounts) must match spotWeights size");
        }
        if (!beam.spotPositions.empty()) {
            if ((beam.spotPositions.size() % 2u) != 0u) {
                return fail("spotPositions must be shaped as [N][2]");
            }
            if (totalSpots != static_cast<int>(beam.spotPositions.size() / 2u)) {
                return fail("sum(layerSpotCounts) must match spotPositions row count");
            }
        }
    }

    if (!beam.spotBeamDirections.empty()) {
        if ((beam.spotBeamDirections.size() % 3u) != 0u) {
            return fail("spotBeamDirections must be shaped as [N][3]");
        }
        if (!beam.spotPositions.empty() &&
            beam.spotBeamDirections.size() / 3u != beam.spotPositions.size() / 2u) {
            return fail("spotBeamDirections row count must match spotPositions row count");
        }
    }

    if (!beam.layerLongitudinalCutoffs.empty() &&
        beam.layerLongitudinalCutoffs.size() != beam.energies.size()) {
        return fail("layerLongitudinalCutoffs size must match beam.energies size");
    }
    if (!beam.layerSpotDeltas.empty() && beam.layerSpotDeltas.size() != beam.energies.size()) {
        return fail("layerSpotDeltas size must match beam.energies size");
    }
    if (!beam.profileData.empty() && beam.profileSetting.size() < 3u) {
        return fail("profileData requires profileSetting[depth0,step,n]");
    }
    if (!beam.beamParaData.empty() && (beam.beamParaData.size() % 3u) != 0u) {
        return fail("beamParaData must be shaped as [rows][3]");
    }
    if (beam.spotPositionsAreIndices && getActiveWeqHeader(beam).size() < 9u) {
        return fail("spotPositionsAreIndices=true requires a 9-value WEQ header");
    }

    return true;
}

static void printWrapperEntryAuditSummary(const RTDBeamSettings& beam,
                                          const RTDEnergyStruct* energyData,
                                          const int3& ctDims,
                                          const float3& ctResolution,
                                          const float3& ctCorner,
                                          const int3& doseDims,
                                          const float3& doseResolution,
                                          const float3& doseCorner,
                                          size_t beamIdx) {
    if (!rtdInputAuditEnabled() || energyData == nullptr) return;

    const int totalSpots = totalSpotCountHost(beam.layerSpotCounts);
    const int spotPosRows = static_cast<int>(beam.spotPositions.size() / 2u);
    const int spotDirRows = static_cast<int>(beam.spotBeamDirections.size() / 3u);
    const int beamParaRows = static_cast<int>(beam.beamParaData.size() / 3u);
    const int profileDepthN =
        (beam.profileSetting.size() >= 3u) ? std::max(0, static_cast<int>(std::lround(beam.profileSetting[2]))) : 0;
    const size_t expectedSubspotN =
        beam.energies.size() * static_cast<size_t>(std::max(beam.maxSubspotsPerLayer, 0)) * 5ull;
    const std::vector<float>& weqHeader = getActiveWeqHeader(beam);

    std::cout << "[INPUT_AUDIT][WRAPPER_ENTRY] beamIdx=" << beamIdx
              << " GRID"
              << " ctDims=(" << ctDims.x << "," << ctDims.y << "," << ctDims.z << ")"
              << " ctRes=(" << ctResolution.x << "," << ctResolution.y << "," << ctResolution.z << ")"
              << " ctCorner=(" << ctCorner.x << "," << ctCorner.y << "," << ctCorner.z << ")"
              << " doseDims=(" << doseDims.x << "," << doseDims.y << "," << doseDims.z << ")"
              << " doseRes=(" << doseResolution.x << "," << doseResolution.y << "," << doseResolution.z << ")"
              << " doseCorner=(" << doseCorner.x << "," << doseCorner.y << "," << doseCorner.z << ")"
              << std::endl;

    std::cout << "[INPUT_AUDIT][WRAPPER_ENTRY] beamIdx=" << beamIdx
              << " GEOM"
              << " source=(" << beam.sourcePosition.x << "," << beam.sourcePosition.y << "," << beam.sourcePosition.z << ")"
              << " sad=" << beam.sad
              << " sourceDist=(" << beam.sourceDist.x << "," << beam.sourceDist.y << ")"
              << " beamDir=(" << beam.beamDirection.x << "," << beam.beamDirection.y << "," << beam.beamDirection.z << ")"
              << " bmX=(" << beam.bmXDirection.x << "," << beam.bmXDirection.y << "," << beam.bmXDirection.z << ")"
              << " bmY=(" << beam.bmYDirection.x << "," << beam.bmYDirection.y << "," << beam.bmYDirection.z << ")"
              << " refPlaneZ=" << beam.refPlaneZ
              << std::endl;

    std::cout << "[INPUT_AUDIT][WRAPPER_ENTRY] beamIdx=" << beamIdx
              << " SPOTS"
              << " numLayers=" << beam.energies.size()
              << " layerSpotCounts=" << beam.layerSpotCounts.size()
              << " totalSpots=" << totalSpots
              << " spotPosRows=" << spotPosRows
              << " spotWeights=" << beam.spotWeights.size()
              << " spotDirRows=" << spotDirRows
              << " raySpacing=(" << beam.raySpacing.x << "," << beam.raySpacing.y << ")"
              << " spotDelta=(" << beam.spotDelta.x << "," << beam.spotDelta.y << "," << beam.spotDelta.z << ")"
              << " layerSpotDeltas=" << beam.layerSpotDeltas.size()
              << " maxSubspots=" << beam.maxSubspotsPerLayer
              << " layerCutoffs=" << beam.layerLongitudinalCutoffs.size()
              << " roiLinear=" << beam.roiLinearIndices.size()
              << " spotPositionsAreIndices=" << (beam.spotPositionsAreIndices ? 1 : 0)
              << std::endl;

    std::cout << "[INPUT_AUDIT][WRAPPER_ENTRY] beamIdx=" << beamIdx
              << " WEQ"
              << " activeHeaderLen=" << std::min<size_t>(weqHeader.size(), 9u)
              << " weqVectorLen=" << beam.waterEquivalence.size()
              << " bodyLen=" << (beam.waterEquivalence.size() >= 9 ? (beam.waterEquivalence.size() - 9u) : 0u);
    if (weqHeader.size() >= 9u) {
        std::cout << " header9=("
                  << weqHeader[0] << "," << weqHeader[1] << "," << weqHeader[2] << ","
                  << weqHeader[3] << "," << weqHeader[4] << "," << weqHeader[5] << ","
                  << weqHeader[6] << "," << weqHeader[7] << "," << weqHeader[8] << ")";
        if (beam.spotPositions.size() >= 2u) {
            const float rawX = beam.spotPositions[0];
            const float rawY = beam.spotPositions[1];
            const int ix = static_cast<int>(std::floor(rawX));
            const int iy = static_cast<int>(std::floor(rawY));
            float originX = rawX;
            float originY = rawY;
            decodeSpotPosition(beam, weqHeader, rawX, rawY, originX, originY);
            const float centeredX = weqHeader[6] + (rawX - 0.5f) * weqHeader[7];
            const float centeredY = weqHeader[3] + (rawY - 0.5f) * weqHeader[4];
            std::cout << " spot0Raw=(" << rawX << "," << rawY << ")"
                      << " spot0Floor=(" << ix << "," << iy << ")"
                      << " spot0Origin=(" << originX << "," << originY << ")"
                      << " spot0Centered=(" << centeredX << "," << centeredY << ")";
        }
    }
    std::cout << std::endl;

    std::cout << "[INPUT_AUDIT][WRAPPER_ENTRY] beamIdx=" << beamIdx
              << " LUT"
              << " energyRows=" << energyData->nEnergies
              << " energySamples=" << energyData->nEnergySamples
              << " layerEnergies=" << beam.energies.size()
              << " profileEnergyRows=" << beam.profileEnergies.size()
              << " profileDepthN=" << profileDepthN
              << " profileRaw=" << beam.profileData.size()
              << " beamParaRows=" << beamParaRows
              << " subspotExpected=" << expectedSubspotN
              << " subspotActual=" << beam.subspotData.size()
              << std::endl;
}

static bool buildRawSpotLattice(const RTDBeamSettings& beam, RawSpotLattice& out, bool verbose) {
    out = RawSpotLattice{};
    auto fail = [&](const std::string& msg) {
        out.failureReason = msg;
        if (verbose) {
            std::cerr << "  [RTD_SPOT_GRID] " << msg << std::endl;
        }
        return false;
    };

    const int numLayers = static_cast<int>(beam.energies.size());
    if (beam.layerSpotCounts.empty() || beam.spotPositions.empty() || beam.spotWeights.empty()) {
        return fail("missing layerSpotCounts/spotPositions/spotWeights input");
    }
    if (static_cast<int>(beam.layerSpotCounts.size()) != numLayers) {
        return fail("layerSpotCounts size does not match beam.energies size");
    }
    if ((beam.spotPositions.size() % 2) != 0) {
        return fail("spotPositions does not contain [N][2] rows");
    }

    int totalSpots = 0;
    for (int c : beam.layerSpotCounts) totalSpots += c;
    if (totalSpots <= 0) return fail("beam contains no spots");
    if (static_cast<int>(beam.spotPositions.size() / 2) != totalSpots) {
        return fail("spotPositions row count does not match summed layerSpotCounts");
    }
    if (static_cast<int>(beam.spotWeights.size()) != totalSpots) {
        return fail("spotWeights size does not match summed layerSpotCounts");
    }

    const std::vector<float>& weqHeader = getActiveWeqHeader(beam);
    if (beam.spotPositionsAreIndices && weqHeader.size() < 9) {
        return fail("spotPositionsAreIndices requires a 9-value WEQ header");
    }

    if (beam.spotPositionsAreIndices) {
        const int nxRef = std::max(1, static_cast<int>(std::lround(weqHeader[8])));
        const int nyRef = std::max(1, static_cast<int>(std::lround(weqHeader[5])));
        const float dxRef = weqHeader[7];
        const float dyRef = weqHeader[4];
        // Input lattice index 0 corresponds to the first rayweq texel center at
        // physical coordinate x0/y0 after removing the exported +0.5 offset.
        const float oxRef = weqHeader[6];
        const float oyRef = weqHeader[3];

        const size_t denseN = static_cast<size_t>(nxRef) * static_cast<size_t>(nyRef) * static_cast<size_t>(numLayers);
        std::vector<float> dense(denseN, 0.0f);
        double totalInputWeight = 0.0;
        double totalRasterizedWeight = 0.0;

        int offset = 0;
        for (int layer = 0; layer < numLayers; ++layer) {
            const int count = beam.layerSpotCounts[layer];
            if (count <= 0) return false;
            for (int i = 0; i < count; ++i) {
                const int spotIdx = offset + i;
                const float rawX = beam.spotPositions[spotIdx * 2 + 0];
                const float rawY = beam.spotPositions[spotIdx * 2 + 1];
                const float weight = beam.spotWeights[spotIdx];
                if (!hostIsFinite(weight)) {
                    return fail("non-finite spot weight at layer=" + std::to_string(layer) +
                                " localSpot=" + std::to_string(i));
                }
                if (weight < 0.0f) {
                    return fail("negative spot weight at layer=" + std::to_string(layer) +
                                " localSpot=" + std::to_string(i));
                }
                if (weight == 0.0f) {
                    out.skippedZeroWeightSpots++;
                    continue;
                }
                totalInputWeight += static_cast<double>(weight);
                // CarbonPBS uses idbeamxy directly as continuous rayweq texture coordinates.
                // Preserve that sub-texel position when rasterizing onto the dense departure
                // plane instead of hard-flooring to the lower texel.
                const float latticeX = rawX - 0.5f;
                const float latticeY = rawY - 0.5f;
                if (!rasterizeContinuousSpotToDensePlane(dense, nxRef, nyRef, layer, latticeX, latticeY, weight, &totalRasterizedWeight)) {
                    return fail("positive-weight spot could not be rasterized onto rayweq lattice at layer=" +
                                std::to_string(layer) + " localSpot=" + std::to_string(i) +
                                " raw=(" + std::to_string(rawX) + "," + std::to_string(rawY) + ")" +
                                " lattice=(" + std::to_string(latticeX) + "," + std::to_string(latticeY) + ")" +
                                " weight=" + std::to_string(weight));
                }
            }
            offset += count;
        }

        out.valid = true;
        out.spotGridDims = make_uint3(nxRef, nyRef, numLayers);
        out.spotDelta = make_float3(dxRef, dyRef, 0.0f);
        out.spotOffset = make_float3(oxRef, oyRef, 0.0f);
        out.spotWeights = std::move(dense);
        if (verbose) {
            std::cout << "  [RTD_SPOT_GRID] rasterized onto full rayweq departure plane grid:"
                      << " dims=(" << out.spotGridDims.x << "," << out.spotGridDims.y << "," << out.spotGridDims.z << ")"
                      << " offset=(" << out.spotOffset.x << "," << out.spotOffset.y << "," << out.spotOffset.z << ")"
                      << " delta=(" << out.spotDelta.x << "," << out.spotDelta.y << "," << out.spotDelta.z << ")"
                      << " weightIn=" << totalInputWeight
                      << " weightRasterized=" << totalRasterizedWeight
                      << " skippedZeroWeightSpots=" << out.skippedZeroWeightSpots
                      << "\n";
        }
        return true;
    }

    std::vector<float> allX;
    std::vector<float> allY;
    allX.reserve(totalSpots);
    allY.reserve(totalSpots);
    for (int i = 0; i < totalSpots; ++i) {
        float x = 0.0f, y = 0.0f;
        const float rawX = beam.spotPositions[i * 2 + 0];
        const float rawY = beam.spotPositions[i * 2 + 1];
        if (!decodeSpotPosition(beam, weqHeader, rawX, rawY, x, y)) {
            return fail("failed to decode spot position while inferring lattice bounds");
        }
        allX.push_back(x);
        allY.push_back(y);
    }

    auto mkUnique = [](std::vector<float> v) {
        std::sort(v.begin(), v.end());
        std::vector<float> u;
        for (float x : v) {
            if (u.empty() || !approxEq(u.back(), x)) u.push_back(x);
        }
        return u;
    };

    const std::vector<float> ux = mkUnique(allX);
    const std::vector<float> uy = mkUnique(allY);
    if (ux.empty() || uy.empty()) return fail("failed to infer non-empty raw spot lattice axes");

    auto inferGridStep = [](const std::vector<float>& u) {
        float step = INF;
        for (size_t i = 1; i < u.size(); ++i) {
            const float d = u[i] - u[i - 1];
            if (d > 1.0e-4f && d < step) step = d;
        }
        if (step == INF) step = 1.0f;
        return step;
    };

    const float dxRef = inferGridStep(ux);
    const float dyRef = inferGridStep(uy);
    const float oxRef = ux.front();
    const float oyRef = uy.front();

    const int nxRef = std::max(1, static_cast<int>(std::lround((ux.back() - oxRef) / dxRef)) + 1);
    const int nyRef = std::max(1, static_cast<int>(std::lround((uy.back() - oyRef) / dyRef)) + 1);
    if (nxRef <= 0 || nyRef <= 0) return fail("inferred raw spot lattice dimensions are non-positive");

    const size_t denseN = static_cast<size_t>(nxRef) * static_cast<size_t>(nyRef) * static_cast<size_t>(numLayers);
    std::vector<float> dense(denseN, 0.0f);

    int offset = 0;
    for (int layer = 0; layer < numLayers; ++layer) {
        const int count = beam.layerSpotCounts[layer];
        if (count <= 0) return false;
        for (int i = 0; i < count; ++i) {
            const int spotIdx = offset + i;
            const float rawX = beam.spotPositions[spotIdx * 2 + 0];
            const float rawY = beam.spotPositions[spotIdx * 2 + 1];
            const float weight = beam.spotWeights[spotIdx];
            if (!hostIsFinite(weight)) {
                return fail("non-finite spot weight at layer=" + std::to_string(layer) +
                            " localSpot=" + std::to_string(i));
            }
            if (weight < 0.0f) {
                return fail("negative spot weight at layer=" + std::to_string(layer) +
                            " localSpot=" + std::to_string(i));
            }
            if (weight == 0.0f) {
                out.skippedZeroWeightSpots++;
                continue;
            }
            float x = 0.0f;
            float y = 0.0f;
            if (!decodeSpotPosition(beam, weqHeader, rawX, rawY, x, y)) {
                return fail("failed to decode positive-weight spot position at layer=" +
                            std::to_string(layer) + " localSpot=" + std::to_string(i));
            }
            const float fx = (dxRef != 0.0f) ? ((x - oxRef) / dxRef) : 0.0f;
            const float fy = (dyRef != 0.0f) ? ((y - oyRef) / dyRef) : 0.0f;
            const int ix = static_cast<int>(std::lround(fx));
            const int iy = static_cast<int>(std::lround(fy));
            if (ix < 0 || ix >= nxRef || iy < 0 || iy >= nyRef) {
                return fail("positive-weight decoded spot is outside inferred lattice bounds at layer=" +
                            std::to_string(layer) + " localSpot=" + std::to_string(i));
            }
            if (!approxEq(oxRef + ix * dxRef, x, 2e-2f) || !approxEq(oyRef + iy * dyRef, y, 2e-2f)) {
                return fail("positive-weight decoded spot does not align with inferred lattice phase at layer=" +
                            std::to_string(layer) + " localSpot=" + std::to_string(i));
            }
            dense[(static_cast<size_t>(layer) * nyRef + static_cast<size_t>(iy)) * nxRef + static_cast<size_t>(ix)] +=
                weight;
        }
        offset += count;
    }

    out.valid = true;
    out.spotGridDims = make_uint3(nxRef, nyRef, numLayers);
    out.spotDelta = make_float3(dxRef, dyRef, 0.0f);
    out.spotOffset = make_float3(oxRef, oyRef, 0.0f);
    out.spotWeights = std::move(dense);
    if (verbose) {
        std::cout << "  [RTD_SPOT_GRID] dims=(" << out.spotGridDims.x << "," << out.spotGridDims.y << "," << out.spotGridDims.z
                  << ") offset=(" << out.spotOffset.x << "," << out.spotOffset.y << "," << out.spotOffset.z
                  << ") delta=(" << out.spotDelta.x << "," << out.spotDelta.y << "," << out.spotDelta.z << ")"
                  << " skippedZeroWeightSpots=" << out.skippedZeroWeightSpots << "\n";
    }
    return true;
}

// Build a physical PB-like spot lattice view that avoids bilinear/barycentric
// weight splitting onto a dense WEQ departure plane.
//
// This is needed for halo/nuclear mapping when spotPositionsAreIndices=true:
// in that mode, the regular buildRawSpotLattice(...) intentionally rasterizes
// onto the full ray/WEQ texel lattice (dense), which is NOT the physical PB
// grid contract expected by the upstream halo normalization.
// UNIT CONTRACT (must match upstream beam.getSpotIdxToGantry()):
//   spotDelta   : mm per physical PB grid step (typically 2-10 mm)
//   spotOffset  : mm of physical PB cell (0,0) in gantry frame
// The lattice axes returned in spotDelta/spotOffset are consumed downstream as
// mm; do NOT multiply by lenToMm here, and downstream sites must not multiply
// by lenToMm again. Decoded (x,y) below come from decodeSpotPosition which
// already returns mm.
static bool buildPhysicalPBLatticeView(const RTDBeamSettings& beam, RawSpotLattice& out, bool verbose) {
    out = RawSpotLattice{};
    auto fail = [&](const std::string& msg) {
        out.failureReason = msg;
        if (verbose) {
            std::cerr << "  [RTD_PB_GRID] " << msg << std::endl;
        }
        return false;
    };

    const int numLayers = static_cast<int>(beam.energies.size());
    if (beam.layerSpotCounts.empty() || beam.spotPositions.empty() || beam.spotWeights.empty()) {
        return fail("missing layerSpotCounts/spotPositions/spotWeights input");
    }
    if (static_cast<int>(beam.layerSpotCounts.size()) != numLayers) {
        return fail("layerSpotCounts size does not match beam.energies size");
    }
    if ((beam.spotPositions.size() % 2) != 0) {
        return fail("spotPositions does not contain [N][2] rows");
    }

    int totalSpots = 0;
    for (int c : beam.layerSpotCounts) totalSpots += c;
    if (totalSpots <= 0) return fail("beam contains no spots");
    if (static_cast<int>(beam.spotPositions.size() / 2) != totalSpots) {
        return fail("spotPositions row count does not match summed layerSpotCounts");
    }
    if (static_cast<int>(beam.spotWeights.size()) != totalSpots) {
        return fail("spotWeights size does not match summed layerSpotCounts");
    }

    const std::vector<float>& weqHeader = getActiveWeqHeader(beam);
    if (beam.spotPositionsAreIndices && weqHeader.size() < 9) {
        return fail("spotPositionsAreIndices requires a 9-value WEQ header");
    }

    // Infer physical PB lattice axes from the decoded spot center coordinates.
    // Important: do NOT infer axes from only positive-weight spots; allow zero
    // weights to keep geometry complete (the caller can still leave weight=0 cells empty).
    std::vector<float> allX;
    std::vector<float> allY;
    allX.reserve(static_cast<size_t>(totalSpots));
    allY.reserve(static_cast<size_t>(totalSpots));

    for (int i = 0; i < totalSpots; ++i) {
        const float rawX = beam.spotPositions[i * 2 + 0];
        const float rawY = beam.spotPositions[i * 2 + 1];
        float x = 0.0f, y = 0.0f;
        if (!decodeSpotPosition(beam, weqHeader, rawX, rawY, x, y)) {
            return fail("failed to decode spot position while inferring PB lattice bounds");
        }
        allX.push_back(x);
        allY.push_back(y);
    }

    // [9.23] Try to infer the canonical physical PB lattice from the decoded spot
    // center coordinates themselves. Upstream RTD-main exposes this as
    // beam.getSpotIdxToGantry() (kernel_wrapper.cu:903-908). Local equivalent:
    // project the decoded (x,y) cloud onto a uniform 2D lattice and compute
    // (nx, ny, dx, dy, ox, oy) from the projection.
    //
    // BEST-EFFORT: when canonical inference fails (non-uniform plan, sub-texel
    // jitter, very dense plans where every WEQ texel hosts a spot), the caller
    // falls back to weqHeader[8/5/7/4/6/3] so halo can still run with the RC2 BEV
    // fix active. The fallback is NOT upstream-equivalent for lateral profile;
    // see halo-energy-conservation.md (RC1/RC3/RC4) and the fallback warning.
    auto inferAxis = [](const std::vector<float>& vals,
                        float& outDelta,
                        float& outOffset,
                        int& outCount,
                        std::string& whyFailed) -> bool {
        whyFailed.clear();
        if (vals.empty()) {
            whyFailed = "no decoded values";
            return false;
        }
        std::vector<float> uniq;
        uniq.reserve(vals.size());
        for (float v : vals) {
            bool found = false;
            for (float u : uniq) {
                if (std::fabs(v - u) <= 0.05f) { found = true; break; }
            }
            if (!found) uniq.push_back(v);
        }
        std::sort(uniq.begin(), uniq.end());

        outOffset = uniq.front();
        if (uniq.size() == 1) {
            // Single PB on this axis: spacing is undefined. Set a nominal positive
            // value so downstream divides do not fault; cardinality is 1 so delta
            // does not actually drive any indexing.
            outDelta = 1.0f;
            outCount = 1;
            return true;
        }

        // Compute consecutive diffs. Use the median as the canonical step.
        std::vector<float> diffs;
        diffs.reserve(uniq.size() - 1);
        for (size_t i = 1; i < uniq.size(); ++i) {
            diffs.push_back(uniq[i] - uniq[i - 1]);
        }
        std::vector<float> sortedDiffs = diffs;
        std::sort(sortedDiffs.begin(), sortedDiffs.end());
        const float median = sortedDiffs[sortedDiffs.size() / 2];
        if (!(median > 0.0f)) {
            whyFailed = "median lattice step must be positive";
            return false;
        }

        // Validate uniformity. Tolerances relaxed to 25% per-diff and k <= 32 to
        // accept realistic plans with small placement jitter without rejecting
        // the canonical lattice entirely.
        for (float d : diffs) {
            const float ratio = d / median;
            const float roundedRatio = std::round(ratio);
            if (roundedRatio < 1.0f || roundedRatio > 32.0f) {
                whyFailed = "lattice step deviates from canonical multiples (ratio=" +
                            std::to_string(ratio) + ")";
                return false;
            }
            if (std::fabs(ratio - roundedRatio) > 0.25f) {
                whyFailed = "non-integer lattice spacing detected (ratio=" +
                            std::to_string(ratio) + ")";
                return false;
            }
        }
        outDelta = median;
        const float extent = uniq.back() - uniq.front();
        outCount = static_cast<int>(std::lround(extent / median)) + 1;
        if (outCount < 1) outCount = 1;

        // Final sanity: every observed value must hit a lattice point within 25%
        // of a cell. This is the binning tolerance for nearest-cell weight
        // assignment; sub-cell jitter is acceptable.
        for (float v : vals) {
            const float frac = (v - outOffset) / outDelta;
            const float rounded = std::round(frac);
            if (rounded < 0.0f || rounded > static_cast<float>(outCount - 1)) {
                whyFailed = "decoded value outside inferred lattice bounds";
                return false;
            }
            if (std::fabs(frac - rounded) > 0.25f) {
                whyFailed = "decoded value not aligned to inferred lattice (frac=" +
                            std::to_string(frac) + ")";
                return false;
            }
        }
        return true;
    };

    int nxRef = 0;
    int nyRef = 0;
    float dxRef = 0.0f, dyRef = 0.0f, oxRef = 0.0f, oyRef = 0.0f;
    bool usingFallback = false;
    std::string whyXFailed, whyYFailed;
    const bool xOk = inferAxis(allX, dxRef, oxRef, nxRef, whyXFailed);
    const bool yOk = inferAxis(allY, dyRef, oyRef, nyRef, whyYFailed);
    if (!xOk || !yOk) {
        // Global inference failed. Some legacy callers provide layer-dependent
        // placement phases that make the cross-layer cloud appear denser than a
        // single layer. Try the densest single layer before falling back to the
        // WEQ header. This branch is not used for explicit CarbonPBS pybind spacing.
        int bestLayerIdx = -1;
        int bestLayerCount = 0;
        int spotOffset2 = 0;
        for (int layer = 0; layer < numLayers; ++layer) {
            if (beam.layerSpotCounts[layer] > bestLayerCount) {
                bestLayerCount = beam.layerSpotCounts[layer];
                bestLayerIdx = layer;
            }
            spotOffset2 += beam.layerSpotCounts[layer];
        }
        bool perLayerOk = false;
        if (bestLayerIdx >= 0 && bestLayerCount >= 3) {
            std::vector<float> layerX, layerY;
            layerX.reserve(static_cast<size_t>(bestLayerCount));
            layerY.reserve(static_cast<size_t>(bestLayerCount));
            int startSpot = 0;
            for (int layer = 0; layer < bestLayerIdx; ++layer) startSpot += beam.layerSpotCounts[layer];
            for (int i = 0; i < bestLayerCount; ++i) {
                const float rawX2 = beam.spotPositions[(startSpot + i) * 2 + 0];
                const float rawY2 = beam.spotPositions[(startSpot + i) * 2 + 1];
                float lx = 0.0f, ly = 0.0f;
                if (decodeSpotPosition(beam, weqHeader, rawX2, rawY2, lx, ly)) {
                    layerX.push_back(lx);
                    layerY.push_back(ly);
                }
            }
            std::string whyXLayer, whyYLayer;
            float dxLayer = 0.0f, dyLayer = 0.0f, oxLayer = 0.0f, oyLayer = 0.0f;
            int nxLayer = 0, nyLayer = 0;
            const bool xLayerOk = inferAxis(layerX, dxLayer, oxLayer, nxLayer, whyXLayer);
            const bool yLayerOk = inferAxis(layerY, dyLayer, oyLayer, nyLayer, whyYLayer);
            if (xLayerOk && yLayerOk) {
                dxRef = dxLayer; dyRef = dyLayer;
                oxRef = oxLayer; oyRef = oyLayer;
                // Cardinality comes from the full spot range across all layers, not just this layer.
                // Re-use global allX/allY for extent; delta is now known from per-layer inference.
                float minX2 = *std::min_element(allX.begin(), allX.end());
                float maxX2 = *std::max_element(allX.begin(), allX.end());
                float minY2 = *std::min_element(allY.begin(), allY.end());
                float maxY2 = *std::max_element(allY.begin(), allY.end());
                nxRef = (dxRef > 0.0f) ? (static_cast<int>(std::lround((maxX2 - minX2) / dxRef)) + 1) : 1;
                nyRef = (dyRef > 0.0f) ? (static_cast<int>(std::lround((maxY2 - minY2) / dyRef)) + 1) : 1;
                oxRef = minX2;
                oyRef = minY2;
                if (nxRef < 1) nxRef = 1;
                if (nyRef < 1) nyRef = 1;
                perLayerOk = true;
                if (verbose) {
                    std::cout << "  [RTD_PB_GRID] Per-layer PB inference succeeded"
                              << " using layer=" << bestLayerIdx << " (n=" << bestLayerCount << " spots)"
                              << " delta=(" << dxRef << "," << dyRef << ")"
                              << " dims=(" << nxRef << "," << nyRef << ")"
                              << std::endl;
                }
            }
        }

        if (!perLayerOk) {
        // Fall back to the WEQ texel grid. For CarbonPBS exports with
        // spotPositionsAreIndices=true this IS the actual spot-bearing grid; the
        // canonical "physical PB lattice" concept may not apply. RC2 (BEV
        // positioning) still gets fixed because nucIdxToFan no longer doubles
        // lenToMm; the lateral profile is not upstream-equivalent in fallback
        // mode -- expect halo width = primary width and missing broad umbrella.
        if (weqHeader.size() < 9) {
            return fail("canonical PB lattice inference failed and WEQ header is too short for fallback");
        }
        nxRef = std::max(1, static_cast<int>(std::lround(weqHeader[8])));
        nyRef = std::max(1, static_cast<int>(std::lround(weqHeader[5])));
        dxRef = weqHeader[7];
        dyRef = weqHeader[4];
        oxRef = weqHeader[6];
        oyRef = weqHeader[3];
        usingFallback = true;
        if (verbose || true) {
            std::cerr << "  [RTD_PB_GRID] WARN: canonical PB lattice inference failed";
            if (!xOk) std::cerr << " (axis x: " << whyXFailed << ")";
            if (!yOk) std::cerr << " (axis y: " << whyYFailed << ")";
            std::cerr << "; falling back to WEQ-header lattice ("
                      << "dims=(" << nxRef << "," << nyRef << ") "
                      << "delta=(" << dxRef << "," << dyRef << ") mm). "
                      << "RC2 BEV fix is active but lateral profile will NOT be "
                      << "upstream-equivalent for halo (RC3/RC4). "
                      << "See halo-energy-conservation.md."
                      << std::endl;
        }
        } // end if (!perLayerOk)
    }
    if (!(dxRef > 0.0f) || !(dyRef > 0.0f) || nxRef <= 0 || nyRef <= 0) {
        return fail("inferred lattice has invalid cardinality or step");
    }

    const size_t denseN = static_cast<size_t>(nxRef) * static_cast<size_t>(nyRef) * static_cast<size_t>(numLayers);
    std::vector<float> dense(denseN, 0.0f);

    // Place each spot's weight onto exactly one physical PB cell via nearest lattice assignment.
    int offset = 0;
    for (int layer = 0; layer < numLayers; ++layer) {
        const int count = beam.layerSpotCounts[layer];
        if (count <= 0) return false;

        for (int i = 0; i < count; ++i) {
            const int spotIdx = offset + i;
            const float rawX = beam.spotPositions[spotIdx * 2 + 0];
            const float rawY = beam.spotPositions[spotIdx * 2 + 1];
            const float weight = beam.spotWeights[spotIdx];

            if (!hostIsFinite(weight)) {
                return fail("non-finite spot weight at layer=" + std::to_string(layer) +
                            " localSpot=" + std::to_string(i));
            }
            if (weight < 0.0f) {
                return fail("negative spot weight at layer=" + std::to_string(layer) +
                            " localSpot=" + std::to_string(i));
            }

            // Decode the physical PB lattice coordinate used by upstream.
            float x = 0.0f, y = 0.0f;
            if (!decodeSpotPosition(beam, weqHeader, rawX, rawY, x, y)) {
                return fail("failed to decode spot position while assigning PB weights at layer=" +
                            std::to_string(layer) + " localSpot=" + std::to_string(i));
            }

            if (weight == 0.0f) {
                // Keep geometry complete, but zero-weight spots do not change the weight field.
                continue;
            }

            const float fx = (dxRef != 0.0f) ? ((x - oxRef) / dxRef) : 0.0f;
            const float fy = (dyRef != 0.0f) ? ((y - oyRef) / dyRef) : 0.0f;
            int ix = static_cast<int>(std::lround(fx));
            int iy = static_cast<int>(std::lround(fy));
            if (ix < 0 || ix >= nxRef || iy < 0 || iy >= nyRef) {
                if (usingFallback) {
                    // Fallback path: soft-bin out-of-range spots to nearest cell
                    // rather than aborting. Better lossy halo than no halo.
                    ix = std::max(0, std::min(nxRef - 1, ix));
                    iy = std::max(0, std::min(nyRef - 1, iy));
                } else {
                    return fail("decoded spot is outside inferred PB lattice bounds at layer=" +
                                std::to_string(layer) + " localSpot=" + std::to_string(i));
                }
            }

            dense[(static_cast<size_t>(layer) * static_cast<size_t>(nyRef) + static_cast<size_t>(iy)) * nxRef +
                  static_cast<size_t>(ix)] += weight;
        }

        offset += count;
    }

    out.valid = true;
    out.usedFallback = usingFallback;
    out.spotGridDims = make_uint3(nxRef, nyRef, numLayers);
    out.spotDelta = make_float3(dxRef, dyRef, 0.0f);
    out.spotOffset = make_float3(oxRef, oyRef, 0.0f);
    out.spotWeights = std::move(dense);

    if (verbose) {
        std::cout << "  [RTD_PB_GRID] " << (usingFallback ? "WEQ-fallback" : "canonical") << " PB lattice:"
                  << " dims=(" << out.spotGridDims.x << "," << out.spotGridDims.y << "," << out.spotGridDims.z << ")"
                  << " offset=(" << out.spotOffset.x << "," << out.spotOffset.y << "," << out.spotOffset.z << ")"
                  << " delta=(" << out.spotDelta.x << "," << out.spotDelta.y << "," << out.spotDelta.z << ")"
                  << "\n";
    }

    return true;
}

static bool computeDecodedSpotBounds(const RTDBeamSettings& beam,
                                     float& minX,
                                     float& maxX,
                                     float& minY,
                                     float& maxY) {
    if (beam.layerSpotCounts.empty() || beam.spotPositions.empty()) return false;
    const std::vector<float>& weqHeader = getActiveWeqHeader(beam);
    if (beam.spotPositionsAreIndices && weqHeader.size() < 9) return false;

    minX = INF; maxX = -INF;
    minY = INF; maxY = -INF;
    const int totalSpots = static_cast<int>(beam.spotPositions.size() / 2);
    for (int i = 0; i < totalSpots; ++i) {
        float x = 0.0f, y = 0.0f;
        const float rawX = beam.spotPositions[i * 2 + 0];
        const float rawY = beam.spotPositions[i * 2 + 1];
        if (!decodeSpotPosition(beam, weqHeader, rawX, rawY, x, y)) return false;
        minX = std::min(minX, x);
        maxX = std::max(maxX, x);
        minY = std::min(minY, y);
        maxY = std::max(maxY, y);
    }
    return minX <= maxX && minY <= maxY;
}

static bool estimateVirtualSourceDistancesFromSpots(const RTDBeamSettings& beam,
                                                    const vec3f& bmX,
                                                    const vec3f& bmY,
                                                    const vec3f& bmZ,
                                                    float sad,
                                                    float& sadX,
                                                    float& sadY) {
    if (!(sad > 1.0e-6f)) return false;
    if (beam.spotBeamDirections.empty() || beam.spotPositions.empty()) return false;
    if ((beam.spotBeamDirections.size() % 3) != 0 || (beam.spotPositions.size() % 2) != 0) return false;

    const size_t nDirs = beam.spotBeamDirections.size() / 3;
    const size_t nPos = beam.spotPositions.size() / 2;
    const size_t n = std::min(nDirs, nPos);
    if (n == 0) return false;

    const std::vector<float>& weqHeader = getActiveWeqHeader(beam);
    const bool haveHeader = weqHeader.size() >= 9;
    const float x0 = haveHeader ? weqHeader[6] : 0.0f;
    const float y0 = haveHeader ? weqHeader[3] : 0.0f;
    const float dx = haveHeader ? weqHeader[7] : 1.0f;
    const float dy = haveHeader ? weqHeader[4] : 1.0f;

    std::vector<float> sadXs;
    std::vector<float> sadYs;
    sadXs.reserve(n);
    sadYs.reserve(n);

    for (size_t i = 0; i < n; ++i) {
        const float rawX = beam.spotPositions[i * 2 + 0];
        const float rawY = beam.spotPositions[i * 2 + 1];

        float spotX = rawX;
        float spotY = rawY;
        if (beam.spotPositionsAreIndices) {
            if (!haveHeader) return false;
            // dosecal.py exports idbeamxy = physical_index + offset + 0.5
            // Recover the physical reference-plane coordinate used to build
            // spot-specific beam directions: coord = start + (raw - 0.5) * step.
            spotX = x0 + (rawX - 0.5f) * dx;
            spotY = y0 + (rawY - 0.5f) * dy;
        }

        vec3f dir = make_vec3f(beam.spotBeamDirections[i * 3 + 0],
                               beam.spotBeamDirections[i * 3 + 1],
                               beam.spotBeamDirections[i * 3 + 2]);
        const float dirLen = sqrtf(dot(dir, dir));
        if (!(dirLen > 1.0e-6f)) continue;
        dir /= dirLen;

        const float a = dot(dir, bmZ);
        if (!(fabsf(a) > 1.0e-6f)) continue;
        const float bx = dot(dir, bmX);
        const float by = dot(dir, bmY);

        if (fabsf(spotX) > 1.0e-3f && fabsf(bx) > 1.0e-6f) {
            const float candidate = fabsf(spotX * a / bx);
            if (candidate > 1.0f && hostIsFinite(candidate)) sadXs.push_back(candidate);
        }
        if (fabsf(spotY) > 1.0e-3f && fabsf(by) > 1.0e-6f) {
            const float candidate = fabsf(spotY * a / by);
            if (candidate > 1.0f && hostIsFinite(candidate)) sadYs.push_back(candidate);
        }
    }

    auto robustMedian = [](std::vector<float>& v) -> float {
        if (v.empty()) return 0.0f;
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };

    sadX = robustMedian(sadXs);
    sadY = robustMedian(sadYs);
    return sadX > 0.0f || sadY > 0.0f;
}

__global__ void fillBevFromWeqVolumeKernel(
    float* __restrict__ bevDensity,
    float* __restrict__ bevCumulSp,
    int* __restrict__ beamFirstInside,
    int* __restrict__ firstStepOutside,
    int rayDimsX,
    int rayDimsY,
    int steps,
    const float* __restrict__ weqVolume,
    int weqNx,
    int weqNy,
    int weqStoredSteps,
    int weqActiveSteps,
    float weqStepMm,
    float rayOriginX,
    float rayOriginY,
    float rayStepX,
    float rayStepY,
    float weqX0,
    float weqY0,
    float weqDx,
    float weqDy
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= rayDimsX || y >= rayDimsY) return;

    const int idx2d = y * rayDimsX + x;
    const int plane = rayDimsX * rayDimsY;
    const float safeWeqStepMm = (weqStepMm > 0.0f) ? weqStepMm : 1.0f;
    const float densityThreshold = 1.0e-4f;

    int firstMaterial = -1;
    int lastMaterial = -1;
    float prevWeq = 0.0f;
    float lastKnownWeq = 0.0f;
    const float rayX = rayOriginX + static_cast<float>(x) * rayStepX;
    const float rayY = rayOriginY + static_cast<float>(y) * rayStepY;
    const int weqX = (fabsf(weqDx) > 0.0f) ? static_cast<int>(lroundf((rayX - weqX0) / weqDx)) : -1;
    const int weqY = (fabsf(weqDy) > 0.0f) ? static_cast<int>(lroundf((rayY - weqY0) / weqDy)) : -1;
    const bool rayInsideWeq = (weqX >= 0 && weqX < weqNx && weqY >= 0 && weqY < weqNy);

    for (int k = 0; k < steps; ++k) {
        float weq = lastKnownWeq;
        if (rayInsideWeq && k < weqActiveSteps) {
            const size_t weqIdx =
                (static_cast<size_t>(weqX) * static_cast<size_t>(weqNy) + static_cast<size_t>(weqY)) *
                    static_cast<size_t>(weqStoredSteps) +
                static_cast<size_t>(k);
            weq = weqVolume[weqIdx];
            lastKnownWeq = weq;
        }
        const float density = fmaxf(0.0f, weq - prevWeq) / safeWeqStepMm;
        const int idx = k * plane + idx2d;
        bevCumulSp[idx] = weq;
        bevDensity[idx] = density;
        if (density > densityThreshold) {
            if (firstMaterial < 0) firstMaterial = k;
            lastMaterial = k;
        }
        prevWeq = weq;
    }

    beamFirstInside[idx2d] = (firstMaterial >= 0) ? (firstMaterial + 1) : steps;
    firstStepOutside[idx2d] = (lastMaterial >= 0) ? (lastMaterial + 1) : 0;
}

template<int blockSize>
__global__ void sliceMinVarIgnoreNeverEnter(const float* __restrict__ devIn,
                                            const int* __restrict__ firstStepOutside,
                                            float* __restrict__ devOut,
                                            const int n) {
    __shared__ float sdata[blockSize];

    const int tid = threadIdx.x;
    const int base = n * blockIdx.z;
    int rayIdx = tid;
    float myMin = INF;

    while (rayIdx < n) {
        if (firstStepOutside[rayIdx] > 0) {
            const float v = devIn[base + rayIdx];
            if (v < myMin) myMin = v;
        }
        rayIdx += blockSize;
    }

    sdata[tid] = myMin;
    __syncthreads();

    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] < sdata[tid]) {
            sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        devOut[blockIdx.z] = sdata[0];
    }
}

// Pad ray IDD / sigma arrays to the superposition tile pitch (RayTraceDicom expects
// rayDimsX % SUPERP_TILE_X == 0 and rayDimsY % SUPERP_TILE_Y == 0 for the tile kernels).
__global__ void padRayIddSigmaKernel(const float* __restrict__ srcIdd,
                                    const float* __restrict__ srcSigma,
                                    float* __restrict__ dstIdd,
                                    float* __restrict__ dstSigma,
                                    int srcX, int srcY,
                                    int dstX, int dstY,
                                    int steps) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z;

    if (x >= dstX || y >= dstY || z >= steps) return;

    const int dstPlane = dstX * dstY;
    const int srcPlane = srcX * srcY;

    const int dstIdx = z * dstPlane + y * dstX + x;

    if (x < srcX && y < srcY) {
        const int srcIdx = z * srcPlane + y * srcX + x;
        dstIdd[dstIdx] = srcIdd[srcIdx];
        dstSigma[dstIdx] = srcSigma[srcIdx];
    } else {
        dstIdd[dstIdx] = 0.0f;
        dstSigma[dstIdx] = __int_as_float(0x7f800000); // +inf
    }
}

__global__ void scalePlaneKernel(const float* __restrict__ src,
                                 float* __restrict__ dst,
                                 int n,
                                 float scale) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx] = src[idx] * scale;
}

__global__ void addPlaneKernel(float* __restrict__ dst,
                               const float* __restrict__ src,
                               int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx] += src[idx];
}

__global__ void fillFloatKernel(float* __restrict__ dst,
                                int n,
                                float value) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx] = value;
}

__global__ void overrideRSigmaFromCarbonProfileKernel(
    const float* __restrict__ bevCumulSp,
    const float* __restrict__ bevIdd,
    float* __restrict__ bevRSigmaEff,
    FillIddAndSigmaParams params,
    int rayDimsX,
    int rayDimsY,
    int steps,
    cudaTextureObject_t profileTex,
    float profileDepth0,
    float profileDepthStep,
    int profileDepthN,
    int profileChannels,
    float profileRowIdx,
    float beamPara0,
    float beamPara1,
    float beamPara2,
    float beamParaPos
) {
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= (unsigned)rayDimsX || y >= (unsigned)rayDimsY) return;

    const unsigned int plane = (unsigned int)(rayDimsX * rayDimsY);
    unsigned int idx = y * (unsigned int)rayDimsX + x;
    const int nGauss = (profileChannels - 1) / 2;
    if (nGauss <= 0 || profileDepthN <= 0 || !(profileDepthStep > 0.0f)) return;

    const float sad = 0.5f * (params.dist.x + params.dist.y);

    for (int stepNo = 0; stepNo < steps; ++stepNo, idx += plane) {
        const float idd = bevIdd[idx];
        if (!(idd > 0.0f)) {
            bevRSigmaEff[idx] = __int_as_float(0x7f800000);
            continue;
        }

        const float weqDepth = bevCumulSp[idx];
        const float profileDepthIdx = (weqDepth - profileDepth0) / profileDepthStep;
        const float clampedDepthIdx = fminf(fmaxf(profileDepthIdx, 0.0f), float(profileDepthN - 1));

        float sumW = 0.0f;
        float sumSigma2Rad = 0.0f;
        for (int g = 0; g < nGauss; ++g) {
            const float w = tex3D<float>(profileTex, float(g) + 0.5f, clampedDepthIdx + 0.5f, profileRowIdx + 0.5f);
            const float sigmaRad = tex3D<float>(profileTex, float(g + nGauss) + 0.5f, clampedDepthIdx + 0.5f, profileRowIdx + 0.5f);
            sumW += w;
            sumSigma2Rad += w * sigmaRad * sigmaRad;
        }
        if (!(sumW > 1.0e-8f)) {
            bevRSigmaEff[idx] = __int_as_float(0x7f800000);
            continue;
        }

        // Reconstruct CarbonPBS geometric depth:
        //   projectedLength - (sad - beamParaPos)
        // where projectedLength is the distance from the virtual source on this
        // divergent ray to the evaluation point.
        const float fanX = params.corner.x + float(x) * params.delta.x;
        const float fanY = params.corner.y + float(y) * params.delta.y;
        const float fanZ = params.corner.z + float(stepNo) * params.delta.z;
        const float axialFromSource = sad - fanZ;
        const float projectedLength =
            axialFromSource * sqrtf(1.0f + (fanX * fanX) / (params.dist.x * params.dist.x) +
                                             (fanY * fanY) / (params.dist.y * params.dist.y));
        const float phyDepth = projectedLength - (sad - beamParaPos);
        const float initR2 = beamPara0 + 2.0f * beamPara1 * phyDepth + beamPara2 * phyDepth * phyDepth;
        const float sigmaAxisSq = 0.5f * fmaxf(0.0f, initR2 + sumSigma2Rad / sumW);
        const float sigmaAxis = sqrtf(fmaxf(0.0f, sigmaAxisSq));
        const float meanWidth = 0.5f * (params.voxelWidth(stepNo).x + params.voxelWidth(stepNo).y);
        bevRSigmaEff[idx] = HALF * meanWidth / (SQRT2 * (sigmaAxis + SIGMA_DELTA));
    }
}

__global__ void applyCarbonProfileOverallWeightKernel(
    const float* __restrict__ bevCumulSp,
    float* __restrict__ bevIdd,
    int rayDimsX,
    int rayDimsY,
    int steps,
    cudaTextureObject_t profileTex,
    float profileDepth0,
    float profileDepthStep,
    int profileDepthN,
    int profileChannels,
    float profileRowIdx
) {
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= (unsigned)rayDimsX || y >= (unsigned)rayDimsY) return;
    if (profileTex == 0 || profileDepthN <= 0 || !(profileDepthStep > 0.0f) || profileChannels <= 0) return;

    const unsigned int plane = (unsigned int)(rayDimsX * rayDimsY);
    unsigned int idx = y * (unsigned int)rayDimsX + x;
    const float overallChannel = float(profileChannels - 1) + 0.5f;

    for (int stepNo = 0; stepNo < steps; ++stepNo, idx += plane) {
        const float idd = bevIdd[idx];
        if (!(idd > 0.0f)) {
            continue;
        }

        const float weqDepth = bevCumulSp[idx];
        const float profileDepthIdx = (weqDepth - profileDepth0) / profileDepthStep;
        const float clampedDepthIdx = fminf(fmaxf(profileDepthIdx, 0.0f), float(profileDepthN - 1));
        const float overallWeight = fmaxf(0.0f, tex3D<float>(profileTex, overallChannel, clampedDepthIdx + 0.5f, profileRowIdx + 0.5f));
        bevIdd[idx] = idd * overallWeight;
    }
}

__global__ void gatherCarbonSigmaDebugKernel(
    const float* __restrict__ bevCumulSp,
    const float* __restrict__ bevIdd,
    const float* __restrict__ bevRSigmaEff,
    FillIddAndSigmaParams params,
    int rayDimsX,
    int rayDimsY,
    int steps,
    cudaTextureObject_t profileTex,
    float profileDepth0,
    float profileDepthStep,
    int profileDepthN,
    int profileChannels,
    float profileRowIdx,
    float beamPara0,
    float beamPara1,
    float beamPara2,
    float beamParaPos,
    const int* __restrict__ sampleXs,
    const int* __restrict__ sampleYs,
    const int* __restrict__ sampleSteps,
    int sampleCount,
    CarbonSigmaDebugSample* __restrict__ outSamples
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= sampleCount) return;

    CarbonSigmaDebugSample sample;
    sample.x = sampleXs[tid];
    sample.y = sampleYs[tid];
    sample.step = sampleSteps[tid];

    if (sample.x < 0 || sample.x >= rayDimsX ||
        sample.y < 0 || sample.y >= rayDimsY ||
        sample.step < 0 || sample.step >= steps) {
        outSamples[tid] = sample;
        return;
    }

    const int plane = rayDimsX * rayDimsY;
    const int idx = sample.step * plane + sample.y * rayDimsX + sample.x;

    sample.idd = bevIdd[idx];
    sample.weqDepth = bevCumulSp[idx];
    const vec2f vw = params.voxelWidth(sample.step);
    sample.meanWidth = 0.5f * (vw.x + vw.y);
    sample.rSigmaEff = bevRSigmaEff[idx];
    if (isfinite(sample.rSigmaEff) && sample.rSigmaEff > 0.0f) {
        sample.approxSigma = HALF * sample.meanWidth / (SQRT2 * sample.rSigmaEff) - SIGMA_DELTA;
    } else {
        sample.approxSigma = __int_as_float(0x7f800000);
    }

    const int nGauss = (profileChannels - 1) / 2;
    if (profileTex == 0 || nGauss <= 0 || profileDepthN <= 0 || !(profileDepthStep > 0.0f)) {
        outSamples[tid] = sample;
        return;
    }

    sample.profileDepthIdx = (sample.weqDepth - profileDepth0) / profileDepthStep;
    const float clampedDepthIdx = fminf(fmaxf(sample.profileDepthIdx, 0.0f), float(profileDepthN - 1));

    float sumW = 0.0f;
    float sumSigma2Rad = 0.0f;
    for (int g = 0; g < nGauss; ++g) {
        const float w = tex3D<float>(profileTex, float(g) + 0.5f, clampedDepthIdx + 0.5f, profileRowIdx + 0.5f);
        const float sigmaRad = tex3D<float>(profileTex, float(g + nGauss) + 0.5f, clampedDepthIdx + 0.5f, profileRowIdx + 0.5f);
        sumW += w;
        sumSigma2Rad += w * sigmaRad * sigmaRad;
    }
    sample.sumW = sumW;
    sample.profileSigmaRad = (sumW > 1.0e-8f) ? sqrtf(fmaxf(0.0f, sumSigma2Rad / sumW)) : 0.0f;

    const float sad = 0.5f * (params.dist.x + params.dist.y);
    const float fanX = params.corner.x + float(sample.x) * params.delta.x;
    const float fanY = params.corner.y + float(sample.y) * params.delta.y;
    const float fanZ = params.corner.z + float(sample.step) * params.delta.z;
    const float axialFromSource = sad - fanZ;
    const float projectedLength =
        axialFromSource * sqrtf(1.0f + (fanX * fanX) / (params.dist.x * params.dist.x) +
                                         (fanY * fanY) / (params.dist.y * params.dist.y));
    sample.phyDepth = projectedLength - (sad - beamParaPos);
    sample.initR2 = beamPara0 + 2.0f * beamPara1 * sample.phyDepth + beamPara2 * sample.phyDepth * sample.phyDepth;
    if (sumW > 1.0e-8f) {
        const float sigmaAxisSq = 0.5f * fmaxf(0.0f, sample.initR2 + sumSigma2Rad / sumW);
        sample.carbonSigmaAxis = sqrtf(fmaxf(0.0f, sigmaAxisSq));
    }

    outSamples[tid] = sample;
}


enum class CtInputType {
    Auto,
    HU,           // [-1000, 3000]
    HUPlus1000,   // [0, 4000]
    Density,      // ~[0.5, 2.0]
    SPR           // ~[0.5, 2.0]
};

static CtInputType parseCtInputTypeEnv() {
    const char* v = std::getenv("RTD_CT_INPUT_TYPE");
    if (!v) return CtInputType::Auto;
    std::string s(v);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return (char)std::tolower(c); });
    if (s == "hu") return CtInputType::HU;
    if (s == "hu+1000" || s == "huplus1000" || s == "hup1000") return CtInputType::HUPlus1000;
    if (s == "density" || s == "rho") return CtInputType::Density;
    if (s == "spr" || s == "stoppingpower" || s == "stopping_power") return CtInputType::SPR;
    return CtInputType::Auto;
}

struct CtStats {
    float minV = 0.0f;
    float maxV = 0.0f;
};

static CtStats sampleCtStats(const float* data, size_t n) {
    CtStats s;
    if (!data || n == 0) return s;

    // Sample at most ~100k points to keep overhead low for large volumes.
    const size_t maxSamples = 100000;
    const size_t stride = (n > maxSamples) ? (n / maxSamples) : 1;

    float mn = data[0];
    float mx = data[0];
    for (size_t i = 0; i < n; i += stride) {
        float v = data[i];
        mn = std::min(mn, v);
        mx = std::max(mx, v);
    }
    s.minV = mn;
    s.maxV = mx;
    return s;
}

static CtInputType autoDetectCtType(const CtStats& st) {
    // Heuristics:
    //  - HU contains negatives.
    //  - HU+1000 is usually non-negative, up to a few thousand.
    //  - Density / SPR are around 1.0.
    if (st.minV < -200.0f) return CtInputType::HU;
    if (st.maxV <= 20.0f && st.minV >= 0.0f) return CtInputType::Density;
    return CtInputType::HUPlus1000;
}

// Find fractional index for a monotonic LUT (vec[idx] ≈ value).
// Returns index in [0, n-1].
static float findFractionalIndexMonotonic(const std::vector<float>& vec, float value) {
    const int n = static_cast<int>(vec.size());
    if (n <= 1) return 0.0f;

    const bool increasing = (vec.back() >= vec.front());
    // Clamp
    if (increasing) {
        if (value <= vec.front()) return 0.0f;
        if (value >= vec.back())  return float(n - 1);
    } else {
        if (value >= vec.front()) return 0.0f;
        if (value <= vec.back())  return float(n - 1);
    }

    int lo = 0;
    int hi = n - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) >> 1;
        const float vm = vec[mid];
        if (increasing) {
            if (vm <= value) lo = mid; else hi = mid;
        } else {
            if (vm >= value) lo = mid; else hi = mid;
        }
    }

    const float v0 = vec[lo];
    const float v1 = vec[lo + 1];
    const float denom = (v1 - v0);
    float t = 0.0f;
    if (denom != 0.0f) {
        t = (value - v0) / denom;
    }
    // For decreasing LUT, denom is negative and t still interpolates correctly.
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    return float(lo) + t;
}

static bool convertCtToHUPlus1000(
    const float* ctIn,
    size_t n,
    const RTDEnergyStruct* energyData,
    std::vector<float>& ctOutHUPlus1000,
    CtInputType forcedType,
    bool verbose
) {
    if (!ctIn || n == 0) return false;

    CtStats st = sampleCtStats(ctIn, n);
    CtInputType type = forcedType;
    if (type == CtInputType::Auto) {
        type = autoDetectCtType(st);
    }

    if (verbose) {
        std::cout << "[CT_ADAPTER] Input stats: min=" << st.minV << ", max=" << st.maxV << std::endl;
        std::cout << "[CT_ADAPTER] Input type: "
                  << (type == CtInputType::HU ? "HU" :
                      type == CtInputType::HUPlus1000 ? "HU+1000" :
                      type == CtInputType::Density ? "Density" :
                      type == CtInputType::SPR ? "SPR" : "Auto")
                  << std::endl;
    }

    // If already HU+1000, just reference input (no conversion needed).
    if (type == CtInputType::HUPlus1000) {
        return false; // indicates no conversion (caller can use ctIn directly)
    }

    ctOutHUPlus1000.resize(n);

    if (type == CtInputType::HU) {
        for (size_t i = 0; i < n; ++i) ctOutHUPlus1000[i] = ctIn[i] + 1000.0f;
        return true;
    }

    // Density / SPR inversion through LUT
    if (!energyData) {
        std::cerr << "[CT_ADAPTER] Error: energyData is null, cannot invert LUT for Density/SPR." << std::endl;
        return false;
    }

    const bool useDensity = (type == CtInputType::Density);
    const std::vector<float>& lut = useDensity ? energyData->densityVector : energyData->spVector;
    const float scale = useDensity ? energyData->densityScaleFact : energyData->spScaleFact;

    if (lut.empty() || scale == 0.0f) {
        std::cerr << "[CT_ADAPTER] Error: LUT is empty or scale factor is 0; cannot convert." << std::endl;
        ctOutHUPlus1000.clear();
        return false;
    }

    for (size_t i = 0; i < n; ++i) {
        const float v = ctIn[i];
        const float idx = findFractionalIndexMonotonic(lut, v);
        // Texture coordinate = HUplus1000 * scale + 0.5 -> index+0.5
        // => HUplus1000 * scale = index
        ctOutHUPlus1000[i] = idx / scale;
    }

    return true;
}

} // namespace

// Main wrapper function implementation
void subsecondWrapper(
    const float* ctData,          // ctdata (corner/resolution/dims naming matches carbonPBS caldose)
    const int3& ctDims,           // ct grid dims
    const float3& ctResolution,   // ct grid resolution
    const float3& ctCorner,       // ct grid corner
    float* doseData,              // output dose grid
    const int3& doseDims,
    const float3& doseResolution,
    const float3& doseCorner,
    const RTDBeamSettings* beamSettings, size_t numBeams,
    const RTDEnergyStruct* energyData,
    int gpuId, bool nuclearCorrection, int verbose
) {
    CPU_TIMER_START_SUMMARY();
    CPU_TIMER_START();
    setRTDVerbose(verbose);
    const bool fineTiming = rtdVerboseFineTiming();  // verbose==1 only; verbose>=2 is summary-only
    const bool haloAudit = rtdHaloAuditEnabled();
    const bool perfProfile = rtdPerfProfileEnabled();
    const bool perfProfileLayers = rtdPerfProfileLayersEnabled();
    if (fineTiming) {
        std::cout << "Starting RTD wrapper" << std::endl;
    }

    if (!validateWrapperGlobalInputs(
            ctData, ctDims, ctResolution,
            doseData, doseDims, doseResolution,
            beamSettings, numBeams, energyData)) {
        return;
    }
    enforceNuclearCorrectionContract(energyData, nuclearCorrection);

    for (size_t beamIdx = 0; beamIdx < numBeams; ++beamIdx) {
        const RTDBeamSettings& beam = beamSettings[beamIdx];
        if (!validateWrapperEntryBeam(beam, beamIdx)) {
            return;
        }
        printWrapperEntryAuditSummary(
            beam, energyData,
            ctDims, ctResolution, ctCorner,
            doseDims, doseResolution, doseCorner,
            beamIdx
        );
    }
    
    // Initialize CUDA
    cudaSetDevice(gpuId);
    cudaFree(0);
    
    if (fineTiming) {
        printDeviceInfo();
        printMemoryInfo();
    }

    // ------------------------------------------------------------------------
    // CT input conversion (CarbonPBS -> RayTraceDicom)
    // ------------------------------------------------------------------------
    const size_t ctElemN = static_cast<size_t>(ctDims.x) * static_cast<size_t>(ctDims.y) * static_cast<size_t>(ctDims.z);
    std::vector<float> ctHUPlus1000;
    const float* ctForTexture = ctData;

    CtInputType forcedType = parseCtInputTypeEnv();
    CtInputType chosenType = forcedType;
    if (chosenType == CtInputType::Auto) {
        chosenType = autoDetectCtType(sampleCtStats(ctData, ctElemN));
    }

    const bool needConversion = (chosenType != CtInputType::HUPlus1000);
    bool converted = false;
    if (needConversion) {
        // If conversion returns true -> use ctHUPlus1000 buffer.
        // If conversion returns false here -> conversion failed.
        converted = convertCtToHUPlus1000(ctData, ctElemN, energyData, ctHUPlus1000, chosenType, fineTiming);
        if (!converted || ctHUPlus1000.empty()) {
            std::cerr << "[CT_ADAPTER] Fatal: CT conversion to HU+1000 failed. Set RTD_CT_INPUT_TYPE=HU|HU+1000|DENSITY|SPR to override." << std::endl;
            return;
        }
        ctForTexture = ctHUPlus1000.data();
    }

    cudaTextureObject_t imVolTex = create3DTexture(ctForTexture, ctDims, cudaFilterModeLinear, cudaAddressModeBorder);
    float* devCtLinear = nullptr;
    if (imVolTex == 0) {
        if (fineTiming) {
            std::cerr << "[RTD] CT 3D texture creation failed; falling back to linear-memory trilinear sampling." << std::endl;
        }
        devCtLinear = (float*)allocateDeviceMemory(ctElemN * sizeof(float));
        copyToDevice(devCtLinear, ctForTexture, ctElemN * sizeof(float));
    }
    
    cudaTextureObject_t cumulIddTex = create2DTexture(&energyData->ciddMatrix[0], 
                                                     make_int2(energyData->nEnergySamples, energyData->nEnergies),
                                                     cudaFilterModeLinear, cudaAddressModeClamp);
    if (cumulIddTex == 0) {
        std::cerr << "[RTD] Fatal: failed to create cumulative IDD texture. Dose computation aborted." << std::endl;
        destroyTextureObjectAndArray(imVolTex);
        return;
    }
    
    cudaTextureObject_t densityTex = create1DTexture(&energyData->densityVector[0], 
                                                    energyData->nDensitySamples,
                                                    cudaFilterModeLinear, cudaAddressModeClamp);
    if (densityTex == 0) {
        std::cerr << "[RTD] Fatal: failed to create density texture. Dose computation aborted." << std::endl;
        destroyTextureObjectAndArray(imVolTex);
        destroyTextureObjectAndArray(cumulIddTex);
        return;
    }
    
    cudaTextureObject_t stoppingPowerTex = create1DTexture(&energyData->spVector[0], 
                                                          energyData->nSpSamples,
                                                          cudaFilterModeLinear, cudaAddressModeClamp);
    if (stoppingPowerTex == 0) {
        std::cerr << "[RTD] Fatal: failed to create stopping-power texture. Dose computation aborted." << std::endl;
        destroyTextureObjectAndArray(imVolTex);
        destroyTextureObjectAndArray(cumulIddTex);
        destroyTextureObjectAndArray(densityTex);
        return;
    }
    
    cudaTextureObject_t rRadiationLengthTex = create1DTexture(&energyData->rRlVector[0], 
                                                             energyData->nRRlSamples,
                                                             cudaFilterModeLinear, cudaAddressModeClamp);
    if (rRadiationLengthTex == 0) {
        std::cerr << "[RTD] Fatal: failed to create radiation-length texture. Dose computation aborted." << std::endl;
        destroyTextureObjectAndArray(imVolTex);
        destroyTextureObjectAndArray(cumulIddTex);
        destroyTextureObjectAndArray(densityTex);
        destroyTextureObjectAndArray(stoppingPowerTex);
        return;
    }

#ifdef NUCLEAR_CORR
    cudaTextureObject_t nucWeightTex = 0;
    cudaTextureObject_t nucSqSigmaTex = 0;
    if (nuclearCorrection) {
        nucWeightTex = create2DTexture(energyData->nucWeightMatrix.data(),
                                       make_int2(energyData->nEnergySamples, energyData->nEnergies),
                                       cudaFilterModeLinear, cudaAddressModeClamp);
        nucSqSigmaTex = create2DTexture(energyData->nucSqSigmaMatrix.data(),
                                        make_int2(energyData->nEnergySamples, energyData->nEnergies),
                                        cudaFilterModeLinear, cudaAddressModeClamp);
        if (nucWeightTex == 0 || nucSqSigmaTex == 0) {
            std::cerr << "[RTD] Fatal: failed to create nuclear LUT textures. Dose computation aborted." << std::endl;
            destroyTextureObjectAndArray(nucWeightTex);
            destroyTextureObjectAndArray(nucSqSigmaTex);
            destroyTextureObjectAndArray(imVolTex);
            destroyTextureObjectAndArray(cumulIddTex);
            destroyTextureObjectAndArray(densityTex);
            destroyTextureObjectAndArray(stoppingPowerTex);
            destroyTextureObjectAndArray(rRadiationLengthTex);
            return;
        }
    }
#endif

    // ------------------------------------------------------------------------
    // Energy table diagnostics (helps detect clamping / unsorted energy vectors)
    // ------------------------------------------------------------------------
    if (fineTiming && energyData && !energyData->energiesPerU.empty()) {
        const bool ascending = energyData->energiesPerU.front() < energyData->energiesPerU.back();
        const bool monotonic = isMonotonicEnergies(energyData->energiesPerU, ascending);
        std::cout << "\n[ENERGY_TABLE] nEnergies=" << energyData->nEnergies
                  << " nSamples=" << energyData->nEnergySamples
                  << " energiesPerU=[" << energyData->energiesPerU.front() << ", " << energyData->energiesPerU.back() << "]"
                  << " order=" << (ascending ? "ascending" : "descending")
                  << " monotonic=" << (monotonic ? "true" : "false")
                  << std::endl;
        if (!monotonic) {
            std::cout << "[ENERGY_TABLE] Warning: energiesPerU is NOT monotonic. Energy indexing/interpolation assumes sorted energies.\n";
            const int preview = std::min(10, energyData->nEnergies);
            std::cout << "[ENERGY_TABLE] energiesPerU first " << preview << ": ";
            for (int i = 0; i < preview; ++i) std::cout << energyData->energiesPerU[i] << (i + 1 < preview ? ", " : "");
            std::cout << std::endl;
        }
        if (!energyData->peakDepths.empty()) {
            std::cout << "[ENERGY_TABLE] peakDepths=[" << energyData->peakDepths.front() << ", " << energyData->peakDepths.back() << "] (table units)" << std::endl;
        }
        if (!energyData->scaleFacts.empty()) {
            std::cout << "[ENERGY_TABLE] scaleFacts=[" << energyData->scaleFacts.front() << ", " << energyData->scaleFacts.back() << "]" << std::endl;
        }
        // Quick sanity check: are CIDD curves actually different across energy?
        // If rows are identical, changing energy will not change the dose (by design/bug).
        if (!energyData->ciddMatrix.empty() && energyData->nEnergies > 1 && energyData->nEnergySamples > 0) {
            const int nS = energyData->nEnergySamples;
            const int e0 = 0;
            const int eN = energyData->nEnergies - 1;
            const float* row0 = &energyData->ciddMatrix[e0 * nS];
            const float* rowN = &energyData->ciddMatrix[eN * nS];

            double meanAbsDiff = 0.0;
            float maxAbsDiff = 0.0f;
            for (int s = 0; s < nS; ++s) {
                const float d = std::fabs(row0[s] - rowN[s]);
                meanAbsDiff += (double)d;
                if (d > maxAbsDiff) maxAbsDiff = d;
            }
            meanAbsDiff /= (double)nS;

            std::cout << "[ENERGY_TABLE] CIDD row0 vs rowLast: meanAbsDiff=" << meanAbsDiff
                      << " maxAbsDiff=" << maxAbsDiff << std::endl;

            const int sMid = nS / 2;
            std::cout << "[ENERGY_TABLE] CIDD samples (row0,rowLast): s0=(" << row0[0] << "," << rowN[0]
                      << ") sMid=(" << row0[sMid] << "," << rowN[sMid]
                      << ") sEnd=(" << row0[nS - 1] << "," << rowN[nS - 1] << ")" << std::endl;
        }

    }
    
    size_t doseSize = doseDims.x * doseDims.y * doseDims.z;
    float* devDoseVol = (float*)allocateDeviceMemory(doseSize * sizeof(float));
    
    // Debug: Check if devDoseVol allocation was successful
    if (devDoseVol == nullptr) {
        std::cout << "Error: Failed to allocate devDoseVol memory!" << std::endl;
        return;
    }
    
    copyToDevice(devDoseVol, doseData, doseSize * sizeof(float));
    
    for (size_t beamIdx = 0; beamIdx < numBeams; ++beamIdx) {
        const RTDBeamSettings& beam = beamSettings[beamIdx];
        BeamStageTiming beamTiming;
        const auto beamPerfStart = perfNow();
        int totalBeamSpots = 0;
        for (int c : beam.layerSpotCounts) {
            totalBeamSpots += c;
        }
        std::vector<LayerPerfTiming> layerPerfRows;
        if (fineTiming) {
            std::cout << "Processing beam " << beamIdx << " with " << beam.energies.size() << " energy layers" << std::endl;
        }
        if (fineTiming) {
            std::cout << "  [BEAM] energies (first up to 8): ";
            const size_t nE = beam.energies.size();
            const size_t nPrint = std::min<size_t>(nE, 8);
            for (size_t i = 0; i < nPrint; ++i) {
                std::cout << beam.energies[i];
                if (i + 1 < nPrint) std::cout << ", ";
            }
            if (nE > nPrint) {
                std::cout << " ... last=" << beam.energies.back();
            }
            std::cout << std::endl;
        }

        
        // ============================================================================
        // Pre-compute CPB convolution for all layers (energy layer  is executed inside)
        // ============================================================================
        
        const int numLayers = static_cast<int>(beam.energies.size());
        const int maxSubspotsPerLayer = beam.maxSubspotsPerLayer;

        // IMPORTANT:
        //  - Per requirements: wrapper must NOT synthesize / hard-code subspot data.
        //  - subspotData must be supplied by the caller (CarbonPBS plan parsing pipeline).
        if (maxSubspotsPerLayer <= 0 || beam.subspotData.empty()) {
            std::cerr << "[RTD] ERROR: beam.maxSubspotsPerLayer/subspotData not provided (beamIdx="
                      << beamIdx << ", numLayers=" << numLayers
                      << "). This wrapper no longer generates test subspot data." << std::endl;
            continue;
        }
        const size_t expectedSubspotN = static_cast<size_t>(numLayers) * static_cast<size_t>(maxSubspotsPerLayer) * 5ull;
        if (beam.subspotData.size() != expectedSubspotN) {
            std::cerr << "[RTD] ERROR: beam.subspotData size mismatch. Expected " << expectedSubspotN
                      << " floats (= numLayers*maxSubspotsPerLayer*5), got " << beam.subspotData.size()
                      << " (beamIdx=" << beamIdx << ")" << std::endl;
            continue;
        }
        const float* subspotData = beam.subspotData.data();
        
        // Create subspot texture
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
        cudaTextureObject_t subspotTexture = 0;
        cudaArray* subspotArray = nullptr;

        // ------------------------------------------------------------------------
        // Geometry + coordinate system (CarbonPBS -> RayTraceDicom)
        // ------------------------------------------------------------------------
        // RayTraceDicom's BEV/fan geometry is defined in a beam-centric ("gantry") coordinate
        // system where:
        //   - z points *away* from the beam direction
        //   - the source is at (0,0,dist)
        //   - the reference plane (isocenter) is at z=0
        // Here we construct a world<->gantry rotation from CarbonPBS inputs:
        //   bmdir (beam direction), bmxdir, bmydir, source position, SAD.
        //
        // Unit handling:
        //   CarbonPBS commonly uses cm (e.g. 0.1 == 1 mm). RayTraceDicom energy LUTs use mm.
        //   We infer a length scale from ctResolution.z (heuristic retained from step2).
        const float stepLength_input = ctResolution.z;
        const float lenToMm = (stepLength_input > 0.0f && stepLength_input < 0.3f) ? 10.0f : 1.0f;

        // ------------------------------------------------------------------------
        // Energy depth unit normalization (EnergyStruct peakDepth/scaleFacts -> mm)
        //
        // RayTraceDicom internally treats:
        //   - cumulSp (WEPL) in *length units* (typically mm)
        //   - peakDepth in the same units as cumulSp
        //   - energyScaleFact converts cumulSp to the depth-index of cumulIddTex
        //
        // CarbonPBS tables may be stored in cm. We detect/override and convert
        // peakDepth and scaleFacts so that WEPL(mm) maps to the correct IDD depth.
        // Override options:
        //   RTD_ENERGY_DEPTH_UNIT=mm|cm
        //   RTD_ENERGY_DEPTH_SCALE=<float>   (e.g. 10 for cm->mm)
        // ------------------------------------------------------------------------
                float energyDepthToMm = lenToMm;
        if (energyData && !energyData->peakDepths.empty()) {
            const char* envScale = std::getenv("RTD_ENERGY_DEPTH_SCALE");
            const char* envUnit  = std::getenv("RTD_ENERGY_DEPTH_UNIT");

            // Peak depth magnitude is a strong indicator of table units:
            //   - Proton therapy peak depth is ~30-320 mm for ~60-230 MeV.
            //   - The reference RayTraceDicom proton LUT (proton_cumul_ddd_data.txt) uses mm.
            float peakMax = 0.0f;
            for (float v : energyData->peakDepths) peakMax = std::max(peakMax, v);

            // --------------------------------------------------------------------
            // User override wins (for full determinism)
            // --------------------------------------------------------------------
            if (envScale && envScale[0] != '\0') {
                energyDepthToMm = std::max(0.0f, static_cast<float>(std::atof(envScale)));
                if (energyDepthToMm == 0.0f) energyDepthToMm = lenToMm;
            } else if (envUnit && envUnit[0] != '\0') {
                std::string u(envUnit);
                for (char& c : u) c = (char)std::tolower((unsigned char)c);
                if (u == "cm") energyDepthToMm = 10.0f;
                else if (u == "mm") energyDepthToMm = 1.0f;
            } else {
                // ----------------------------------------------------------------
                // Auto-detect when the CT geometry uses cm (lenToMm==10) but the LUT
                // peakDepth values are clearly in mm (hundreds).
                // This avoids a common pitfall where peakDepth(mm) is mistakenly
                // treated as cm and multiplied by 10, which stretches the depth axis
                // and makes IDD values near-zero for typical WEPL(mm) values.
                // ----------------------------------------------------------------
                if (lenToMm == 10.0f && peakMax > 200.0f) {
                    energyDepthToMm = 1.0f;
                    if (fineTiming) {
                        std::cout << "  [ENERGY_UNITS] Auto-detected peakDepth table units as mm (peakMax=" << peakMax
                                  << ", CT lenToMm=10). Using energyDepthToMm=1." << std::endl;
                    }
                }

                // Retain diagnostic warnings to help catch rare/ambiguous cases.
                if (fineTiming) {
                    if (lenToMm == 1.0f && peakMax > 0.0f && peakMax < 80.0f) {
                        std::cout << "  [ENERGY_UNITS] Warning: peakDepth max=" << peakMax
                                  << " could be cm, but CT unit looks like mm (lenToMm=1). "
                                  << "Set RTD_ENERGY_DEPTH_UNIT=cm if needed." << std::endl;
                    }
                    if (lenToMm == 10.0f && peakMax > 200.0f) {
                        // This warning is kept, but auto-correction above should already handle it.
                        std::cout << "  [ENERGY_UNITS] Note: peakDepth max=" << peakMax
                                  << " looks like mm while CT uses cm; auto-correct applied (energyDepthToMm=1)." << std::endl;
                    }
                }
            }
        }

if (fineTiming) {
            std::cout << "  [ENERGY_UNITS] energyDepthToMm=" << energyDepthToMm
                      << " (peakDepth table units -> mm)" << std::endl;
        }

        auto layerLongitudinalCutoffMm = [&](size_t layerIdx) {
            if (layerIdx >= beam.layerLongitudinalCutoffs.size()) return 0.0f;
            return normalizeLongitudinalCutoffToMm(beam.layerLongitudinalCutoffs[layerIdx], lenToMm);
        };

        // Strict RTD-baseline mode for the current wrapper convergence pass:
        // keep CarbonPBS-only profile/subspot/cutoff inputs available for audit,
        // but do not let them alter the RTD physical chain.
        const bool useCarbonSubspotConvolution = false;
        const bool useCarbonProfileSigmaOverride = false;
        const bool useCarbonProfileOverallWeight = false;

        cudaTextureObject_t profileTex = 0;
        int profileRows = 0;
        int profileDepthN = 0;
        int profileChannels = 0;
        const bool hasBeamParaData = beam.beamParaData.size() >= static_cast<size_t>(numLayers) * 3ull;
        const bool hasProfileModel =
            beam.profileSetting.size() >= 3 &&
            static_cast<int>(std::lround(beam.profileSetting[2])) > 0 &&
            !beam.profileData.empty();
        if (hasProfileModel) {
            profileDepthN = std::max(1, static_cast<int>(std::lround(beam.profileSetting[2])));
            const int beamParaRows = static_cast<int>(beam.beamParaData.size() / 3ull);
            const size_t profilePlane = static_cast<size_t>(profileDepthN);
            if (profileDepthN > 0) {
                if (beamParaRows > 0 &&
                    beam.beamParaData.size() == static_cast<size_t>(beamParaRows) * 3ull &&
                    beam.profileData.size() % (static_cast<size_t>(beamParaRows) * profilePlane) == 0) {
                    profileRows = beamParaRows;
                } else if (!beam.profileEnergies.empty() &&
                           beam.profileData.size() % (beam.profileEnergies.size() * profilePlane) == 0) {
                    profileRows = static_cast<int>(beam.profileEnergies.size());
                } else if (numLayers > 0 &&
                           beam.profileData.size() % (static_cast<size_t>(numLayers) * profilePlane) == 0) {
                    profileRows = numLayers;
                } else if (profilePlane > 0 && beam.profileData.size() % profilePlane == 0) {
                    // Last-resort shape inference for compact slice exports:
                    // profileData is laid out as [row][depth][channel].
                    // If the row count differs from both numLayers and beamParaRows,
                    // derive it from the exported tensor size and channel count.
                    const size_t perPlaneCount = beam.profileData.size() / profilePlane;
                    if (perPlaneCount % 11ull == 0ull) {
                        profileRows = static_cast<int>(perPlaneCount / 11ull);
                    }
                }
            }
            const size_t perRowProfile =
                (profileRows > 0) ? (beam.profileData.size() / static_cast<size_t>(profileRows)) : 0ull;
            if (profileRows > 0 && profileDepthN > 0 && perRowProfile >= static_cast<size_t>(profileDepthN)) {
                profileChannels = static_cast<int>(perRowProfile / static_cast<size_t>(profileDepthN));
                if (profileChannels > 0 &&
                    static_cast<size_t>(profileRows) * static_cast<size_t>(profileDepthN) * static_cast<size_t>(profileChannels) == beam.profileData.size() &&
                    (useCarbonProfileSigmaOverride || useCarbonProfileOverallWeight)) {
                    profileTex = create3DTexture(beam.profileData.data(),
                                                 make_int3(profileChannels, profileDepthN, profileRows),
                                                 cudaFilterModeLinear, cudaAddressModeClamp);
                }
            }
            if (fineTiming || profileTex == 0) {
                const size_t expectedAfterChannels =
                    static_cast<size_t>(std::max(profileRows, 0)) *
                    static_cast<size_t>(profileDepthN) *
                    static_cast<size_t>(std::max(profileChannels, 0));
                std::cout << "  [CARBON_PROFILE] enabled=" << (profileTex != 0 ? "true" : "false")
                          << " rawSize=" << beam.profileData.size()
                          << " numLayers=" << numLayers
                          << " profileRows=" << profileRows
                          << " depthN=" << profileDepthN
                          << " perRow=" << perRowProfile
                          << " channels=" << profileChannels
                          << " expectedSize=" << expectedAfterChannels
                          << " beamParaRows=" << beamParaRows
                          << " energyTableRows=" << (energyData ? energyData->nEnergies : 0)
                          << " beamPara=" << (hasBeamParaData ? "true" : "false")
                          << std::endl;
            }
        }

        auto toVec3f = [](const float3& v) {
            return make_vec3f(v.x, v.y, v.z);
        };
        auto normalizeSafe = [](vec3f v) {
            const float len = sqrtf(dot(v, v));
            if (len > 1e-8f) v /= len;
            return v;
        };

        // Beam geometry parameters (from CarbonPBS / caller)
        vec3f beamDirectionW = normalizeSafe(toVec3f(beam.beamDirection));
        vec3f bmXDirectionW  = normalizeSafe(toVec3f(beam.bmXDirection));
        vec3f bmYDirectionW  = normalizeSafe(toVec3f(beam.bmYDirection));
        vec3f sourcePositionW_cm = toVec3f(beam.sourcePosition);

        // SAD fallback: if not provided explicitly, fall back to sourceDist.x (legacy)
        const float sad_cm = (beam.sad > 0.0f) ? beam.sad : beam.sourceDist.x;

        // CarbonPBS exports beam_xdir / beam_ydir as an orthonormal gantry basis.
        // Use that basis as the primary source of beam-axis geometry and only use
        // mean beam_dir to disambiguate the sign. This preserves the exported
        // isocenter relation source + bmZ * SAD much better than averaging spot
        // directions, which can introduce a small tilt.
        const vec3f basisCross = normalizeSafe(cross(bmXDirectionW, bmYDirectionW));
        const bool haveBasisCross = dot(basisCross, basisCross) > 1.0e-8f;
        const bool haveBeamDir = dot(beamDirectionW, beamDirectionW) > 1.0e-8f;
        vec3f bmZ = haveBasisCross ? basisCross : beamDirectionW;
        if (haveBasisCross && haveBeamDir && dot(bmZ, beamDirectionW) < 0.0f) {
            bmZ *= -1.0f;
        }
        if (!(dot(bmZ, bmZ) > 1.0e-8f)) {
            bmZ = make_vec3f(0.0f, 0.0f, 1.0f);
        }

        vec3f bmX = normalizeSafe(bmXDirectionW - bmZ * dot(bmXDirectionW, bmZ));
        if (!(dot(bmX, bmX) > 1.0e-8f)) {
            bmX = normalizeSafe(cross(bmYDirectionW, bmZ));
        }
        if (!(dot(bmX, bmX) > 1.0e-8f)) {
            bmX = make_vec3f(1.0f, 0.0f, 0.0f);
        }

        vec3f bmY = normalizeSafe(cross(bmZ, bmX));
        if (dot(bmY, bmYDirectionW) < 0.0f) bmY *= -1.0f;
        bmX = normalizeSafe(cross(bmY, bmZ));

        // Build a right-handed gantry basis that matches RayTraceDicom convention:
        //   beam direction is along -z in gantry.
        const vec3f gX = bmX;
        const vec3f gY = bmY;
        const vec3f gZ = bmZ * -1.0f;

        // Gantry origin at isocenter (source + beamDir * SAD)
        const vec3f sourcePositionW_mm = sourcePositionW_cm * lenToMm;
        const float sad_mm = sad_cm * lenToMm;
        const vec3f isoW_mm = sourcePositionW_mm + bmZ * sad_mm;

        if (fineTiming) {
            std::cout << "  [BEAM_BASIS] beamDirW=(" << beamDirectionW.x << "," << beamDirectionW.y << "," << beamDirectionW.z << ")"
                      << " crossXY=(" << basisCross.x << "," << basisCross.y << "," << basisCross.z << ")"
                      << " bmZ=(" << bmZ.x << "," << bmZ.y << "," << bmZ.z << ")"
                      << std::endl;
            std::cout << "  [BEAM_BASIS] source_mm=(" << sourcePositionW_mm.x << "," << sourcePositionW_mm.y << "," << sourcePositionW_mm.z << ")"
                      << " sad_mm=" << sad_mm
                      << " iso_mm=(" << isoW_mm.x << "," << isoW_mm.y << "," << isoW_mm.z << ")"
                      << std::endl;
        }

        RawSpotLattice rawSpotLattice;
        const bool hasRawSpotLattice = buildRawSpotLattice(beam, rawSpotLattice, fineTiming || haloAudit);
        if ((fineTiming || haloAudit) && !hasRawSpotLattice) {
            std::cout << "  [RTD_SPOT_GRID] raw spot lattice unavailable; wrapper still uses legacy CPB projection path for this beam";
            if (!rawSpotLattice.failureReason.empty()) {
                std::cout << " reason=" << rawSpotLattice.failureReason;
            }
            std::cout << "\n";
        }

        // Gantry->World affine transform (mm)
        const Matrix3x3 gantryToWorldMat(
            make_vec3f(gX.x, gY.x, gZ.x),
            make_vec3f(gX.y, gY.y, gZ.y),
            make_vec3f(gX.z, gY.z, gZ.z)
        );
        const Float3AffineTransform gantryToWorld(gantryToWorldMat, isoW_mm);
        const Float3AffineTransform worldToGantry = gantryToWorld.inverse();
        
        // ------------------------------------------------------------------------
        // CPB / ray grid definition (in gantry coordinates, on z=0 reference plane)
        // ------------------------------------------------------------------------
        // We define the CPB/ray grid to cover the *dose volume projection* in gantry X/Y.
        // This avoids the "world==gantry" simplification and makes the pipeline work for
        // rotated beams.
        const vec3f doseRes_mm = make_vec3f(doseResolution.x * lenToMm, doseResolution.y * lenToMm, doseResolution.z * lenToMm);
        const vec3f doseCorner_mm = make_vec3f(doseCorner.x * lenToMm, doseCorner.y * lenToMm, doseCorner.z * lenToMm);

        float minGX = 1e30f, maxGX = -1e30f;
        float minGY = 1e30f, maxGY = -1e30f;
        int3 roiMinIdx = make_int3(0, 0, 0);
        int3 roiMaxIdx = make_int3(doseDims.x - 1, doseDims.y - 1, doseDims.z - 1);
        const bool hasDoseRoi = computeDoseRoiBounds(beam.roiLinearIndices, doseDims, roiMinIdx, roiMaxIdx);

        // CPB grid must cover the FULL dose volume projection in gantry XY.
        // (ROI bounds are used elsewhere for dose masking, but cannot restrict the
        //  CPB grid or the beam center / high-energy rays will fall outside.)
        const vec3f projCorner_mm = doseCorner_mm;  // always full volume
        // CarbonPBS CSV corner/origin is voxel-center based. Use (dims-1)*spacing
        // for geometric extents; using dims*spacing shifts the traced fan one voxel
        // beyond the CT/dose box and causes border-grazing rays.
        const vec3f projSize_mm = make_vec3f(
            float(std::max(doseDims.x - 1, 0)) * doseRes_mm.x,
            float(std::max(doseDims.y - 1, 0)) * doseRes_mm.y,
            float(std::max(doseDims.z - 1, 0)) * doseRes_mm.z
        );

        for (int cx = 0; cx <= 1; ++cx) {
            for (int cy = 0; cy <= 1; ++cy) {
                for (int cz = 0; cz <= 1; ++cz) {
                    const vec3f w = projCorner_mm + make_vec3f(
                        float(cx) * projSize_mm.x,
                        float(cy) * projSize_mm.y,
                        float(cz) * projSize_mm.z
                    );
                    const vec3f g = worldToGantry.transformPoint(w);
                    minGX = fminf(minGX, g.x);
                    maxGX = fmaxf(maxGX, g.x);
                    minGY = fminf(minGY, g.y);
                    maxGY = fmaxf(maxGY, g.y);
                }
            }
        }
        // Also ensure all subspot gantry positions are inside the grid.
        const int nLayers = static_cast<int>(beam.energies.size());
        const int mspl = beam.maxSubspotsPerLayer;
        if (mspl > 0 && !beam.subspotData.empty()) {
            for (int li = 0; li < nLayers; ++li) {
                for (int si = 0; si < mspl; ++si) {
                    const int base = (li * mspl + si) * 5;
                    if (static_cast<int>(beam.subspotData.size()) <= base + 4) break;
                    if (beam.subspotData[base + 2] > 0.0f) {  // weight > 0
                        const float sx = beam.subspotData[base + 0] / lenToMm;
                        const float sy = beam.subspotData[base + 1] / lenToMm;
                        minGX = fminf(minGX, sx); maxGX = fmaxf(maxGX, sx);
                        minGY = fminf(minGY, sy); maxGY = fmaxf(maxGY, sy);
                    }
                }
            }
        }

        auto roundUpTo = [](int v, int pitch) {
            return ((v + pitch - 1) / pitch) * pitch;
        };

        const std::vector<float>& weqHeader = getActiveWeqHeader(beam);
        const bool hasWeqHeader = weqHeader.size() >= 9;

        vec3f cpbCorner;
        vec3f cpbResolution;
        vec3i cpbDims;
        int2 rayDims;

        if (hasRawSpotLattice && beam.raySpacing.x > 0.0f && beam.raySpacing.y > 0.0f) {
            float maxSigmaX = 0.0f;
            float maxSigmaY = 0.0f;
            for (const float2& s : beam.spotSigmas) {
                maxSigmaX = std::max(maxSigmaX, s.x);
                maxSigmaY = std::max(maxSigmaY, s.y);
            }
            const float marginX = 3.0f * maxSigmaX + HALF * beam.raySpacing.x;
            const float marginY = 3.0f * maxSigmaY + HALF * beam.raySpacing.y;
            float cpbOriginX = 0.0f;
            float cpbOriginY = 0.0f;
            int cpbNX = 0;
            int cpbNY = 0;
            int xFirstStep = 0;
            int xLastStep = 0;
            int yFirstStep = 0;
            int yLastStep = 0;
            const bool alignedX = alignGridToReferencePhase(
                rawSpotLattice.spotOffset.x - marginX,
                rawSpotLattice.spotOffset.x + float(int(rawSpotLattice.spotGridDims.x) - 1) * rawSpotLattice.spotDelta.x + marginX,
                rawSpotLattice.spotOffset.x,
                beam.raySpacing.x,
                SUPERP_TILE_X,
                cpbOriginX,
                cpbNX,
                &xFirstStep,
                &xLastStep
            );
            const bool alignedY = alignGridToReferencePhase(
                rawSpotLattice.spotOffset.y - marginY,
                rawSpotLattice.spotOffset.y + float(int(rawSpotLattice.spotGridDims.y) - 1) * rawSpotLattice.spotDelta.y + marginY,
                rawSpotLattice.spotOffset.y,
                beam.raySpacing.y,
                SUPERP_TILE_Y,
                cpbOriginY,
                cpbNY,
                &yFirstStep,
                &yLastStep
            );
            if (!alignedX || !alignedY) {
                throw std::runtime_error("Failed to align CPB grid to raw spot lattice phase");
            }

            cpbCorner = make_vec3f(cpbOriginX, cpbOriginY, 0.0f);
            cpbResolution = make_vec3f(beam.raySpacing.x, beam.raySpacing.y, doseResolution.z);
            cpbDims = make_vec3i(
                cpbNX,
                cpbNY,
                numLayers
            );
            rayDims = make_int2(cpbDims.x, cpbDims.y);
            if (fineTiming) {
                std::cout << "  [CPB_ALIGN] rawSpotPhase ref=(" << rawSpotLattice.spotOffset.x << "," << rawSpotLattice.spotOffset.y << ")"
                          << " step=(" << beam.raySpacing.x << "," << beam.raySpacing.y << ")"
                          << " stepRangeX=[" << xFirstStep << "," << xLastStep << "]"
                          << " stepRangeY=[" << yFirstStep << "," << yLastStep << "]"
                          << " corner=(" << cpbCorner.x << "," << cpbCorner.y << ")"
                          << std::endl;
            }
        } else if (hasWeqHeader) {
            bool croppedBySpotBounds = false;
            float spotMinX = 0.0f, spotMaxX = 0.0f, spotMinY = 0.0f, spotMaxY = 0.0f;
            if (beam.raySpacing.x > 0.0f && beam.raySpacing.y > 0.0f &&
                computeDecodedSpotBounds(beam, spotMinX, spotMaxX, spotMinY, spotMaxY)) {
                float maxSigmaX = 0.0f;
                float maxSigmaY = 0.0f;
                for (const float2& s : beam.spotSigmas) {
                    maxSigmaX = std::max(maxSigmaX, s.x);
                    maxSigmaY = std::max(maxSigmaY, s.y);
                }
                const float marginX = 3.0f * maxSigmaX + HALF * beam.raySpacing.x;
                const float marginY = 3.0f * maxSigmaY + HALF * beam.raySpacing.y;
                const float refX = weqHeader[6] / lenToMm;
                const float refY = weqHeader[3] / lenToMm;
                float cpbOriginX = 0.0f;
                float cpbOriginY = 0.0f;
                int cpbNX = 0;
                int cpbNY = 0;
                int xFirstStep = 0;
                int xLastStep = 0;
                int yFirstStep = 0;
                int yLastStep = 0;
                const bool alignedX = alignGridToReferencePhase(
                    spotMinX - marginX,
                    spotMaxX + marginX,
                    refX,
                    beam.raySpacing.x,
                    SUPERP_TILE_X,
                    cpbOriginX,
                    cpbNX,
                    &xFirstStep,
                    &xLastStep
                );
                const bool alignedY = alignGridToReferencePhase(
                    spotMinY - marginY,
                    spotMaxY + marginY,
                    refY,
                    beam.raySpacing.y,
                    SUPERP_TILE_Y,
                    cpbOriginY,
                    cpbNY,
                    &yFirstStep,
                    &yLastStep
                );
                if (alignedX && alignedY) {
                    cpbCorner = make_vec3f(cpbOriginX, cpbOriginY, 0.0f);
                    cpbResolution = make_vec3f(beam.raySpacing.x, beam.raySpacing.y, doseResolution.z);
                    cpbDims = make_vec3i(cpbNX, cpbNY, numLayers);
                    rayDims = make_int2(cpbDims.x, cpbDims.y);
                    croppedBySpotBounds = true;
                    if (fineTiming) {
                        std::cout << "  [RTD_SPOT_GRID] using decoded spot bounds for CPB/ray crop:"
                                  << " x=[" << spotMinX << "," << spotMaxX << "]"
                                  << " y=[" << spotMinY << "," << spotMaxY << "]"
                                  << " ref=(" << refX << "," << refY << ")"
                                  << " stepRangeX=[" << xFirstStep << "," << xLastStep << "]"
                                  << " stepRangeY=[" << yFirstStep << "," << yLastStep << "]"
                                  << std::endl;
                    }
                }
            }
            if (!croppedBySpotBounds) {
                cpbCorner = make_vec3f(weqHeader[6] / lenToMm, weqHeader[3] / lenToMm, 0.0f);
                cpbResolution = make_vec3f(weqHeader[7] / lenToMm, weqHeader[4] / lenToMm, doseResolution.z);
                cpbDims = make_vec3i(
                    roundUpTo(std::max(1, static_cast<int>(std::lround(weqHeader[8]))), SUPERP_TILE_X),
                    roundUpTo(std::max(1, static_cast<int>(std::lround(weqHeader[5]))), SUPERP_TILE_Y),
                    numLayers
                );
                rayDims = make_int2(cpbDims.x, cpbDims.y);
            }
        } else {
            if (!(beam.raySpacing.x > 0.0f) || !(beam.raySpacing.y > 0.0f)) {
                throw std::runtime_error(
                    "Legacy CPB path requires explicit beam.raySpacing; "
                    "doseResolution*0.5 fallback is not RTD-main-equivalent");
            }
            const float cpbResX = beam.raySpacing.x;
            const float cpbResY = beam.raySpacing.y;
            const float cpbResZ = doseResolution.z;

            float covX_cm = (maxGX - minGX) / lenToMm;
            float covY_cm = (maxGY - minGY) / lenToMm;
            if (!(covX_cm > 0.0f) || !(covY_cm > 0.0f)) {
                throw std::runtime_error(
                    "Legacy CPB path produced non-positive projected coverage; "
                    "full dose-box coverage fallback is disabled");
            }

            if (const char* env = std::getenv("RTD_CPB_COVERAGE_CM")) {
                const float v = std::atof(env);
                if (v > 0.0f && std::isfinite(v)) {
                    covX_cm = v;
                    covY_cm = v;
                    if (fineTiming) {
                        std::cout << "  [CPB] Using RTD_CPB_COVERAGE_CM override: " << v << " cm" << std::endl;
                    }
                }
            }

            cpbCorner = make_vec3f(minGX / lenToMm, minGY / lenToMm, 0.0f);
            cpbResolution = make_vec3f(cpbResX, cpbResY, cpbResZ);
            cpbDims = make_vec3i(
                std::max(1, (int)std::ceil(covX_cm / cpbResolution.x)),
                std::max(1, (int)std::ceil(covY_cm / cpbResolution.y)),
                numLayers
            );
            rayDims = make_int2(cpbDims.x, cpbDims.y);
        }

        if (fineTiming) {
            const float covXDebug = cpbDims.x * cpbResolution.x;
            const float covYDebug = cpbDims.y * cpbResolution.y;
            std::cout << "  [CPB] gantry XY bounds (mm): x=[" << minGX << "," << maxGX << "] y=[" << minGY << "," << maxGY << "]\n";
            std::cout << "  [CPB] corner_gantry_cm=(" << cpbCorner.x << ", " << cpbCorner.y << ", " << cpbCorner.z << ")"
                      << " res_cm=(" << cpbResolution.x << ", " << cpbResolution.y << ", " << cpbResolution.z << ")"
                      << " dims=(" << cpbDims.x << ", " << cpbDims.y << ", " << cpbDims.z << ")"
                      << " cov_cm=(" << covXDebug << ", " << covYDebug << ")"
                      << std::endl;
            if (hasDoseRoi) {
                std::cout << "  [ROI] projected dose ROI bounds idx min=("
                          << roiMinIdx.x << "," << roiMinIdx.y << "," << roiMinIdx.z << ") max=("
                          << roiMaxIdx.x << "," << roiMaxIdx.y << "," << roiMaxIdx.z << ")" << std::endl;
            }
        }

        // ------------------------------------------------------------------------
        // Gantry -> image/dose index transforms (mm)
        // ------------------------------------------------------------------------
        const vec3f ctRes_mm = make_vec3f(ctResolution.x * lenToMm, ctResolution.y * lenToMm, ctResolution.z * lenToMm);
        const vec3f ctCorner_mm = make_vec3f(ctCorner.x * lenToMm, ctCorner.y * lenToMm, ctCorner.z * lenToMm);

        const Matrix3x3 worldToImMat(1.0f / ctRes_mm.x, 1.0f / ctRes_mm.y, 1.0f / ctRes_mm.z);
        const vec3f worldToImOff = make_vec3f(-ctCorner_mm.x / ctRes_mm.x,
                                              -ctCorner_mm.y / ctRes_mm.y,
                                              -ctCorner_mm.z / ctRes_mm.z);
        const Float3AffineTransform worldToImIdx(worldToImMat, worldToImOff);
        const Float3AffineTransform gantryToImIdx = concat(gantryToWorld, worldToImIdx);

        const Matrix3x3 worldToDoseMat(1.0f / doseRes_mm.x, 1.0f / doseRes_mm.y, 1.0f / doseRes_mm.z);
        const vec3f worldToDoseOff = make_vec3f(-doseCorner_mm.x / doseRes_mm.x,
                                                -doseCorner_mm.y / doseRes_mm.y,
                                                -doseCorner_mm.z / doseRes_mm.z);
        const Float3AffineTransform worldToDoseIdx(worldToDoseMat, worldToDoseOff);
        const Float3AffineTransform gantryToDoseIdx = concat(gantryToWorld, worldToDoseIdx);

        float inferredSadX = 0.0f;
        float inferredSadY = 0.0f;
        const bool inferredSourceDist =
            estimateVirtualSourceDistancesFromSpots(beam, bmX, bmY, bmZ, sad_cm, inferredSadX, inferredSadY);

        // Source distance in mm (per-axis). Prefer explicitly provided sourceDist.
        // Otherwise recover CarbonPBS's virtual source distances from the exported
        // spot-specific beam directions, which encode sadx/sady implicitly.
        float distX_mm = beam.sourceDist.x * lenToMm;
        float distY_mm = beam.sourceDist.y * lenToMm;
        const bool sourceDistLooksLegacySad =
            fabsf(beam.sourceDist.x - beam.sad) < 1.0e-3f &&
            fabsf(beam.sourceDist.y - beam.sad) < 1.0e-3f;
        if (inferredSourceDist && (!(distX_mm > 1e-6f) || sourceDistLooksLegacySad)) {
            distX_mm = inferredSadX * lenToMm;
        }
        if (inferredSourceDist && (!(distY_mm > 1e-6f) || sourceDistLooksLegacySad)) {
            distY_mm = inferredSadY * lenToMm;
        }
        if (!(distX_mm > 1e-6f)) distX_mm = sad_mm;
        if (!(distY_mm > 1e-6f)) distY_mm = sad_mm;
        const vec2f sourceDistVec = make_vec2f(distX_mm, distY_mm);

        if (fineTiming) {
            std::cout << "  [SOURCE_DIST] explicit=(" << beam.sourceDist.x << "," << beam.sourceDist.y << ")"
                      << " inferred=(" << inferredSadX << "," << inferredSadY << ")"
                      << " used_mm=(" << sourceDistVec.x << "," << sourceDistVec.y << ")"
                      << std::endl;
        }
        if (rtdSigmaDebugEnabled()) {
            std::cout << "  [RTD_DEBUG_GEOM] source_mm=(" << sourcePositionW_mm.x << "," << sourcePositionW_mm.y << "," << sourcePositionW_mm.z << ")"
                      << " iso_mm=(" << isoW_mm.x << "," << isoW_mm.y << "," << isoW_mm.z << ")"
                      << " sourceDist_mm=(" << sourceDistVec.x << "," << sourceDistVec.y << ")"
                      << std::endl;
        }

        float minGZ = 1e30f, maxGZ = -1e30f;
        {
            const vec3f ctSize_mm = make_vec3f(float(std::max(ctDims.x - 1, 0)) * ctRes_mm.x,
                                               float(std::max(ctDims.y - 1, 0)) * ctRes_mm.y,
                                               float(std::max(ctDims.z - 1, 0)) * ctRes_mm.z);
            for (int cx = 0; cx <= 1; ++cx) {
                for (int cy = 0; cy <= 1; ++cy) {
                    for (int cz = 0; cz <= 1; ++cz) {
                        const vec3f w = ctCorner_mm + make_vec3f(float(cx) * ctSize_mm.x,
                                                                float(cy) * ctSize_mm.y,
                                                                float(cz) * ctSize_mm.z);
                        const vec3f g = worldToGantry.transformPoint(w);
                        minGZ = fminf(minGZ, g.z);
                        maxGZ = fmaxf(maxGZ, g.z);
                    }
                }
            }
        }

        const bool hasWeqVolume = hasWeqHeader && beam.waterEquivalence.size() > 9;
        float stepLength_mm = hasWeqVolume ? (weqHeader[1] * lenToMm) : (ctResolution.z * lenToMm);
        if (!(stepLength_mm > 0.0f)) {
            stepLength_mm = ctResolution.z * lenToMm;
        }
        float startZ_mm = maxGZ;
        float startZ_ct_mm = maxGZ;
        float startZ_weq_mm = maxGZ;
        bool startZAlignedToWeq = false;
        bool startZAlignSkippedBogusHeader = false;
        float startZAlignDeltaMm = 0.0f;
        if (hasWeqVolume && weqHeader.size() >= 9) {
            // Task 0.21: align BEV start plane to the WEQ projected origin.
            // RayTraceDicom-main marches a ray tracer from the fan start plane
            // (derived from the CT extent) and accumulates WEPL through CT.
            // With CT replaced by WEQ, fillBevFromWeqVolumeKernel samples
            // WEQ[k] at BEV step k, which is only physically correct when
            // BEV step 0 and WEQ step 0 share the same projected depth from
            // source. The CT-derived maxGZ and the exporter-chosen weqHeader[0]
            // are computed independently, so any difference between them
            // rotates the depth axis and can shift the Bragg peak, collapse
            // BEV support, or make some plans produce NaN/Inf downstream.
            // The fix is to derive the BEV start plane from weqHeader[0] when
            // a WEQ volume is present, which keeps the k->weqVolume[k] contract
            // in the tracer kernel physically consistent and lets every
            // downstream stage (entry plane, pxSpMult, iddParams.corner,
            // primTransfDiv) inherit one aligned depth axis.
            const float meanSourceDistMm = 0.5f * (distX_mm + distY_mm);
            const float weqProjectedStartMm = weqHeader[0] * lenToMm;
            const float alignedStartZMm = meanSourceDistMm - weqProjectedStartMm;
            // Sanity-bound: only accept the alignment when both the header and
            // the resulting start plane look physical. A bogus exporter value
            // (e.g. weqHeader[0]==0 or NaN) would otherwise move BEV start by
            // thousands of millimetres and regress plans that happened to
            // match maxGZ by coincidence. The allowed delta is bounded by the
            // WEQ volume's own depth span so we never realign beyond the
            // material the exporter can actually back.
            const float weqStepAbsMm = fabsf(weqHeader[1] * lenToMm);
            const int weqStepsHdr = std::max(0, static_cast<int>(std::lround(weqHeader[2])));
            const float weqDepthSpanMm = weqStepAbsMm * static_cast<float>(weqStepsHdr);
            const float maxAllowedDeltaMm = std::max(weqDepthSpanMm, 50.0f);
            if (std::isfinite(alignedStartZMm) && std::isfinite(weqProjectedStartMm) &&
                weqProjectedStartMm > 0.0f) {
                const float deltaMm = alignedStartZMm - startZ_ct_mm;
                if (fabsf(deltaMm) <= maxAllowedDeltaMm) {
                    startZ_weq_mm = alignedStartZMm;
                    startZ_mm = alignedStartZMm;
                    startZAlignedToWeq = true;
                    startZAlignDeltaMm = deltaMm;
                } else {
                    startZAlignSkippedBogusHeader = true;
                    startZAlignDeltaMm = deltaMm;
                }
            } else {
                startZAlignSkippedBogusHeader = true;
            }
        }
        int tracerSteps = hasWeqVolume
            ? std::max(1, static_cast<int>(std::lround(weqHeader[2])))
            : std::max(1, (int)std::ceil((maxGZ - minGZ) / fabsf(stepLength_mm)) + 1);

        if (fineTiming) {
            std::cout << "  [GEOM] gantry Z bounds (mm): z=[" << minGZ << "," << maxGZ << "]  startZ=" << startZ_mm
                      << " stepLen=" << stepLength_mm << " steps=" << tracerSteps
                      << (hasWeqVolume ? "  [WEQ depth sampling + CT entry plane]" : "  [from CT extent]")
                      << std::endl;
            if (startZAlignedToWeq) {
                std::cout << "  [WEQ_ALIGN] startZ ct_maxGZ=" << startZ_ct_mm
                          << " -> weq_aligned=" << startZ_weq_mm
                          << " delta=" << startZAlignDeltaMm
                          << " (weqHeader[0]_mm=" << (weqHeader[0] * lenToMm)
                          << " meanSourceDist_mm=" << (0.5f * (distX_mm + distY_mm)) << ")"
                          << std::endl;
            } else if (startZAlignSkippedBogusHeader) {
                std::cout << "  [WEQ_ALIGN] skipped: weqHeader[0] outside sanity bound"
                          << " ct_maxGZ=" << startZ_ct_mm
                          << " proposedAligned=" << (0.5f * (distX_mm + distY_mm) - weqHeader[0] * lenToMm)
                          << " delta=" << startZAlignDeltaMm
                          << " (weqHeader[0]_mm=" << (weqHeader[0] * lenToMm) << ")"
                          << std::endl;
            }
        }

        // Fan grid definition (in gantry coords, mm).
        const vec3f fanCorner_mm = make_vec3f(cpbCorner.x * lenToMm, cpbCorner.y * lenToMm, startZ_mm);
        const vec3f fanDelta_mm  = make_vec3f(cpbResolution.x * lenToMm,
                                              cpbResolution.y * lenToMm,
                                              -fabsf(stepLength_mm));
        if (rtdSigmaDebugEnabled()) {
            std::cout << "  [RTD_DEBUG_FAN] cpbCorner=(" << cpbCorner.x << "," << cpbCorner.y << "," << cpbCorner.z << ")"
                      << " cpbRes=(" << cpbResolution.x << "," << cpbResolution.y << "," << cpbResolution.z << ")"
                      << " rayDims=(" << rayDims.x << "," << rayDims.y << ")"
                      << " fanCorner_mm=(" << fanCorner_mm.x << "," << fanCorner_mm.y << "," << fanCorner_mm.z << ")"
                      << " fanDelta_mm=(" << fanDelta_mm.x << "," << fanDelta_mm.y << "," << fanDelta_mm.z << ")"
                      << std::endl;
        }
        const Float3IdxTransform fanIdxToFan(fanDelta_mm, fanCorner_mm);
        const Float3FromFanTransform fanIdxToImIdx(fanIdxToFan, sourceDistVec, gantryToImIdx);
        const DensityAndSpTracerParams densityTracerParams(energyData ? energyData->densityScaleFact : 1.0f,
                                                           energyData ? energyData->spScaleFact : 1.0f,
                                                           tracerSteps,
                                                           fanIdxToImIdx);
        
        beamTiming.setupMs = perfElapsedMs(beamPerfStart);
        const auto rayWeightPerfStart = perfNow();
        size_t rayWeightsSize = (size_t)numLayers * rayDims.x * rayDims.y * sizeof(float);
        float* devRayWeightsAllLayers = (float*)allocateDeviceMemory(rayWeightsSize);
        cudaMemset(devRayWeightsAllLayers, 0, rayWeightsSize);

        bool runtimeNuclearEnabled = false;
        std::vector<HaloLatticePlan> haloPlans;
#ifdef NUCLEAR_CORR
        runtimeNuclearEnabled = nuclearCorrection;
        if (runtimeNuclearEnabled) {
            if (!hasRawSpotLattice) {
                throw std::runtime_error(
                    "nuclear_correction=true requires raw spot lattice inputs; legacy CPB fallback cannot drive halo mode");
            }
            haloPlans.resize(static_cast<size_t>(numLayers));
            const bool hasExplicitLayerSpotDeltas =
                beam.layerSpotDeltas.size() == static_cast<size_t>(numLayers);
            const bool useExplicitPhysicalHaloLattice =
                hasExplicitLayerSpotDeltas && beam.spotPositionsAreIndices;
            RawSpotLattice haloSpotLattice = rawSpotLattice;
            const char* haloLatticeSource =
                useExplicitPhysicalHaloLattice ? "explicit_physical_pb_rasterized_lattice" : "raw_spot_lattice";
            if (!useExplicitPhysicalHaloLattice && !hasExplicitLayerSpotDeltas && beam.spotPositionsAreIndices) {
                // Legacy non-pybind callers may still require inference. Pybind
                // CarbonPBS imports must supply explicit layerSpotDeltas so halo
                // never falls back to the WEQ/depth-step lattice.
                RawSpotLattice physicalPbLattice;
                if (!buildPhysicalPBLatticeView(beam, physicalPbLattice, fineTiming || haloAudit)) {
                    throw std::runtime_error(
                        "nuclear_correction=true requires a physical PB lattice view when spotPositionsAreIndices=true. "
                        "Gate 9.41 upstream contract failed: " + physicalPbLattice.failureReason);
                }
                haloSpotLattice = std::move(physicalPbLattice);
                haloLatticeSource = "inferred_physical_pb_lattice";
            }
            if (haloAudit) {
                std::cout << "  [RTD_HALO_SELECT] beam=" << beamIdx
                          << " source=" << haloLatticeSource
                          << " selectedMode="
                          << (useExplicitPhysicalHaloLattice ? "per_layer_explicit_physical_pb" : "single_lattice")
                          << " hasExplicitLayerSpotDeltas=" << (hasExplicitLayerSpotDeltas ? 1 : 0)
                          << " spotPositionsAreIndices=" << (beam.spotPositionsAreIndices ? 1 : 0)
                          << " rawDims=(" << rawSpotLattice.spotGridDims.x << "," << rawSpotLattice.spotGridDims.y << "," << rawSpotLattice.spotGridDims.z << ")"
                          << " rawDelta=(" << rawSpotLattice.spotDelta.x << "," << rawSpotLattice.spotDelta.y << "," << rawSpotLattice.spotDelta.z << ")"
                          << " rawOffset=(" << rawSpotLattice.spotOffset.x << "," << rawSpotLattice.spotOffset.y << "," << rawSpotLattice.spotOffset.z << ")"
                          << " rawUsedFallback=" << (rawSpotLattice.usedFallback ? 1 : 0)
                          << " beamSpotDelta=(" << beam.spotDelta.x << "," << beam.spotDelta.y << "," << beam.spotDelta.z << ")"
                          << " raySpacing=(" << beam.raySpacing.x << "," << beam.raySpacing.y << ")";
                if (!useExplicitPhysicalHaloLattice) {
                    std::cout << " selectedDims=(" << haloSpotLattice.spotGridDims.x << "," << haloSpotLattice.spotGridDims.y << "," << haloSpotLattice.spotGridDims.z << ")"
                              << " selectedDelta=(" << haloSpotLattice.spotDelta.x << "," << haloSpotLattice.spotDelta.y << "," << haloSpotLattice.spotDelta.z << ")"
                              << " selectedOffset=(" << haloSpotLattice.spotOffset.x << "," << haloSpotLattice.spotOffset.y << "," << haloSpotLattice.spotOffset.z << ")";
                }
                if (hasExplicitLayerSpotDeltas && !beam.layerSpotDeltas.empty()) {
                    const float2 firstDelta = beam.layerSpotDeltas.front();
                    const float2 lastDelta = beam.layerSpotDeltas.back();
                    std::cout << " firstLayerSpotDelta=(" << firstDelta.x << "," << firstDelta.y << ")"
                              << " lastLayerSpotDelta=(" << lastDelta.x << "," << lastDelta.y << ")";
                }
                std::cout << std::endl;
            }
            for (int layerNo = 0; layerNo < numLayers; ++layerNo) {
                const bool planOk =
                    useExplicitPhysicalHaloLattice
                        ? buildExplicitPhysicalPBHaloLatticePlan(
                              beam,
                              layerNo,
                              cpbCorner,
                              cpbResolution,
                              rayDims,
                              haloPlans[static_cast<size_t>(layerNo)],
                              fineTiming || haloAudit)
                        : buildHaloLatticePlan(haloSpotLattice,
                                               layerNo,
                                               cpbCorner,
                                               cpbResolution,
                                               rayDims,
                                               haloPlans[static_cast<size_t>(layerNo)],
                                               fineTiming || haloAudit);
                if (!planOk) {
                    throw std::runtime_error("failed to build halo lattice plan for layer " +
                                             std::to_string(layerNo) + ": " +
                                             haloPlans[static_cast<size_t>(layerNo)].failureReason);
                }
            }
        }
#endif

        if (!hasRawSpotLattice) {
            // ------------------------------------------------------------------------
            // Legacy CPB fallback path
            // ------------------------------------------------------------------------
            cudaExtent subspotExtent = make_cudaExtent(5, maxSubspotsPerLayer, numLayers);
            cudaMalloc3DArray(&subspotArray, &channelDesc, subspotExtent);

            cudaMemcpy3DParms copyParams = {};
            copyParams.srcPtr = make_cudaPitchedPtr((void*)subspotData, 5 * sizeof(float), 5, maxSubspotsPerLayer);
            copyParams.dstArray = subspotArray;
            copyParams.extent = subspotExtent;
            copyParams.kind = cudaMemcpyHostToDevice;
            cudaMemcpy3D(&copyParams);

            cudaResourceDesc resDesc = {};
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = subspotArray;

            cudaTextureDesc texDesc = {};
            texDesc.filterMode = cudaFilterModePoint;
            texDesc.addressMode[0] = cudaAddressModeClamp;
            texDesc.addressMode[1] = cudaAddressModeClamp;
            texDesc.addressMode[2] = cudaAddressModeClamp;
            texDesc.readMode = cudaReadModeElementType;
            texDesc.normalizedCoords = 0;
            cudaCreateTextureObject(&subspotTexture, &resDesc, &texDesc, nullptr);

            float* d_cpbWeights;
            size_t cpbWeightsSize = cpbDims.x * cpbDims.y * cpbDims.z * sizeof(float);
            cudaMalloc(&d_cpbWeights, cpbWeightsSize);
            cudaMemset(d_cpbWeights, 0, cpbWeightsSize);
            const vec3f beamDirectionG = make_vec3f(0.0f, 0.0f, -1.0f);
            const vec3f bmXDirectionG  = make_vec3f(1.0f, 0.0f, 0.0f);
            const vec3f bmYDirectionG  = make_vec3f(0.0f, -1.0f, 0.0f);
            const vec3f sourcePositionG_cm = make_vec3f(0.0f, 0.0f, sad_cm);
            const float refPlaneZ = 0.0f;
            performSubspotToCPBConvolution(subspotTexture, numLayers, maxSubspotsPerLayer,
                                          cpbCorner, cpbResolution, cpbDims, d_cpbWeights,
                                          beamDirectionG, bmXDirectionG, bmYDirectionG,
                                          sourcePositionG_cm, sad_cm, refPlaneZ);
            vec3i rayDimsVec = make_vec3i(rayDims.x, rayDims.y, 1);
            vec3f rayCorner = cpbCorner;
            vec3f rayResolution = cpbResolution;
            for (int l = 0; l < numLayers; ++l) {
                float* devRayWeightsLayer = devRayWeightsAllLayers + l * rayDims.x * rayDims.y;
                performCPBToRayWeightMapping(d_cpbWeights, cpbDims, cpbCorner, cpbResolution,
                                            devRayWeightsLayer, rayDimsVec,
                                            rayCorner, rayResolution,
                                            l,
                                            beamDirectionG, bmXDirectionG, bmYDirectionG,
                                            sourcePositionG_cm, sad_cm, refPlaneZ);
            }
            cudaFree(d_cpbWeights);
        }

        // Cleanup subspot texture (keep rayWeightsAllLayers)
        if (subspotTexture != 0) cudaDestroyTextureObject(subspotTexture);
        if (subspotArray != nullptr) cudaFreeArray(subspotArray);

        const size_t raySize = static_cast<size_t>(rayDims.x) * static_cast<size_t>(rayDims.y) * static_cast<size_t>(tracerSteps);
        dim3 tracerBlock(16, 16);
        dim3 tracerGrid((rayDims.x + tracerBlock.x - 1) / tracerBlock.x,
                        (rayDims.y + tracerBlock.y - 1) / tracerBlock.y);

        size_t bevArraySize = raySize;
        float* devBevDensity = (float*)allocateDeviceMemory(bevArraySize * sizeof(float));
        float* devBevCumulSp = (float*)allocateDeviceMemory(bevArraySize * sizeof(float));
        int* devBeamFirstInside = (int*)allocateDeviceMemory(rayDims.x * rayDims.y * sizeof(int));
        int* devFirstStepOutside = (int*)allocateDeviceMemory(rayDims.x * rayDims.y * sizeof(int));
        cudaMemset(devBevDensity, 0, bevArraySize * sizeof(float));
        cudaMemset(devBevCumulSp, 0, bevArraySize * sizeof(float));
        cudaMemset(devBeamFirstInside, 0, rayDims.x * rayDims.y * sizeof(int));
        cudaMemset(devFirstStepOutside, 0, rayDims.x * rayDims.y * sizeof(int));

        FillIddAndSigmaParams tracingParams;
        tracingParams.rayDimsX = rayDims.x;
        tracingParams.rayDimsY = rayDims.y;
        tracingParams.first = 0;
        tracingParams.afterLast = tracerSteps;

        if (fineTiming) {
            const vec3f p00 = densityTracerParams.getStart(0, 0);
            const vec3f p10 = densityTracerParams.getStart(rayDims.x - 1, 0);
            const vec3f p01 = densityTracerParams.getStart(0, rayDims.y - 1);
            const vec3f p11 = densityTracerParams.getStart(rayDims.x - 1, rayDims.y - 1);
            const float xMin = fminf(fminf(p00.x, p10.x), fminf(p01.x, p11.x));
            const float xMax = fmaxf(fmaxf(p00.x, p10.x), fmaxf(p01.x, p11.x));
            const float yMin = fminf(fminf(p00.y, p10.y), fminf(p01.y, p11.y));
            const float yMax = fmaxf(fmaxf(p00.y, p10.y), fmaxf(p01.y, p11.y));
            const bool intersectsX = !(xMax < 0.0f || xMin > float(ctDims.x - 1));
            const bool intersectsY = !(yMax < 0.0f || yMin > float(ctDims.y - 1));
            if (!intersectsX || !intersectsY) {
                std::cerr << "[GEOM] Warning: Ray grid does not overlap CT volume (XY). "
                          << "This usually indicates a coordinate/unit mismatch and will produce near-zero dose.\n"
                          << "       RayX idx range: [" << xMin << ", " << xMax << "] vs CT [0, " << (ctDims.x - 1) << "]\n"
                          << "       RayY idx range: [" << yMin << ", " << yMax << "] vs CT [0, " << (ctDims.y - 1) << "]\n";
            }
            const vec3f s00 = densityTracerParams.getStart(0, 0);
            const vec3f di00 = densityTracerParams.getInc(0, 0);
            const float sl00 = densityTracerParams.stepLen(0, 0);
            std::cout << "  DensityAndSpTracerParams:" << std::endl;
            std::cout << "    steps: " << densityTracerParams.getSteps() << std::endl;
            std::cout << "    densityScale: " << densityTracerParams.getDensityScale() << std::endl;
            std::cout << "    spScale: " << densityTracerParams.getSpScale() << std::endl;
            std::cout << "    fanCorner_mm: (" << fanCorner_mm.x << ", " << fanCorner_mm.y << ", " << fanCorner_mm.z << ")" << std::endl;
            std::cout << "    fanDelta_mm : (" << fanDelta_mm.x  << ", " << fanDelta_mm.y  << ", " << fanDelta_mm.z  << ")" << std::endl;
            std::cout << "    sourceDist_mm: (" << sourceDistVec.x << ", " << sourceDistVec.y << ")" << std::endl;
            std::cout << "    startIdx(0,0): (" << s00.x << ", " << s00.y << ", " << s00.z << ")" << std::endl;
            std::cout << "    incIdx  (0,0): (" << di00.x << ", " << di00.y << ", " << di00.z << ")" << std::endl;
            std::cout << "    stepLen_mm(0,0): " << sl00 << std::endl;
        }

        const auto bevTracePerfStart = perfNow();
        if (hasWeqVolume) {
            const size_t bodyCount = beam.waterEquivalence.size() - 9;
            const float* body = beam.waterEquivalence.data() + 9;
            const int weqSteps = std::max(1, static_cast<int>(std::lround(weqHeader[2])));
            const int weqNy = std::max(1, static_cast<int>(std::lround(weqHeader[5])));
            const int weqNx = std::max(1, static_cast<int>(std::lround(weqHeader[8])));
            const size_t weqRayCount = static_cast<size_t>(weqNx) * static_cast<size_t>(weqNy);
            if (weqRayCount == 0) {
                throw std::runtime_error("WEQ header produced an empty ray lattice");
            }
            const size_t headerPackedElems = weqRayCount * static_cast<size_t>(weqSteps);
            if (bodyCount < headerPackedElems) {
                throw std::runtime_error(
                    "water_equivalence payload is smaller than header-declared nx*ny*nStep");
            }
            // dosecal.py preallocates rayweq with nMaxStep=10000, but legacy
            // CarbonPBS only uploads the first nx*ny*header.nStep samples to the
            // 3D texture. The remaining tail is reserved capacity, not valid per-ray
            // depth samples, so the header step count must stay authoritative here.
            const int weqStoredSteps = weqSteps;
            const size_t usedElems = headerPackedElems;
            const size_t ignoredTailElems = bodyCount - usedElems;
            const bool hasExactTiling = (ignoredTailElems == 0);
            const int weqActiveSteps = std::max(0, std::min(tracerSteps, weqStoredSteps));
            const float weqStepMm = fabsf(weqHeader[1] * lenToMm);
            const float weqX0 = weqHeader[6] / lenToMm;
            const float weqY0 = weqHeader[3] / lenToMm;
            const float weqDx = weqHeader[7] / lenToMm;
            const float weqDy = weqHeader[4] / lenToMm;

            if (fineTiming) {
                int firstPositive = -1;
                int lastPositive = -1;
                float maxWeq = 0.0f;
                if (weqNx > 0 && weqNy > 0 && weqActiveSteps > 0) {
                    for (int k = 0; k < weqActiveSteps; ++k) {
                        const float v = body[static_cast<size_t>(k)];
                        if (v > 0.0f) {
                            if (firstPositive < 0) firstPositive = k;
                            lastPositive = k;
                            maxWeq = std::max(maxWeq, v);
                        }
                    }
                }
                std::cout << "  [WEQ] Imported full WEQ volume: bodyCount=" << bodyCount
                          << " usedElems=" << usedElems
                          << " weqDims=(" << weqNx << "," << weqNy << "," << weqSteps << ")"
                          << " storedSteps=" << weqStoredSteps
                          << " activeSteps=" << weqActiveSteps
                          << " tracerSteps=" << tracerSteps
                          << " exactTiling=" << (hasExactTiling ? "true" : "false")
                          << " ignoredTail=" << ignoredTailElems
                          << " firstPositive=" << firstPositive
                          << " lastPositive=" << lastPositive
                          << " maxWeq=" << maxWeq
                          << std::endl;
            }

            float* devWeqVolume = nullptr;
            if (usedElems > 0) {
                devWeqVolume = (float*)allocateDeviceMemory(usedElems * sizeof(float));
                copyToDevice(devWeqVolume, body, usedElems * sizeof(float));
            }
            fillBevFromWeqVolumeKernel<<<tracerGrid, tracerBlock>>>(
                devBevDensity, devBevCumulSp, devBeamFirstInside, devFirstStepOutside,
                rayDims.x, rayDims.y, tracerSteps, devWeqVolume,
                weqNx, weqNy, weqStoredSteps, weqActiveSteps, weqStepMm,
                cpbCorner.x, cpbCorner.y, cpbResolution.x, cpbResolution.y,
                weqX0, weqY0, weqDx, weqDy
            );
            checkCudaErrors(cudaDeviceSynchronize());
            if (devWeqVolume != nullptr) {
                freeDeviceMemory(devWeqVolume);
            }
        } else {
            rayTracingBEVKernel<<<tracerGrid, tracerBlock>>>(
                devBevDensity, devBevCumulSp, nullptr, nullptr,
                devRayWeightsAllLayers, devBeamFirstInside, devFirstStepOutside, nullptr,
                densityTracerParams, tracingParams, nullptr,
                imVolTex, devCtLinear, densityTex, stoppingPowerTex, cumulIddTex, rRadiationLengthTex,
                ctDims
            );
            checkCudaErrors(cudaDeviceSynchronize());
        }
        beamTiming.bevTraceMs += perfElapsedMs(bevTracePerfStart);

        int beamFirstInsideRT = 0;
        int beamFirstOutsideRT = tracerSteps;
        {
            const auto weplReducePerfStart = perfNow();
            int* devBeamFirstInsideMinTmp = (int*)allocateDeviceMemory(sizeof(int));
            int* devBeamFirstOutsideMaxTmp = (int*)allocateDeviceMemory(sizeof(int));
            dim3 oneSliceGrid(1, 1, 1);
            sliceMinVar<int, 1024><<<oneSliceGrid, 1024, 1024 * sizeof(int)>>>(
                devBeamFirstInside, devBeamFirstInsideMinTmp, rayDims.x * rayDims.y);
            sliceMaxVar<int, 1024><<<oneSliceGrid, 1024, 1024 * sizeof(int)>>>(
                devFirstStepOutside, devBeamFirstOutsideMaxTmp, rayDims.x * rayDims.y);
            checkCudaErrors(cudaDeviceSynchronize());
            copyToHost(&beamFirstInsideRT, devBeamFirstInsideMinTmp, sizeof(int));
            copyToHost(&beamFirstOutsideRT, devBeamFirstOutsideMaxTmp, sizeof(int));
            freeDeviceMemory(devBeamFirstInsideMinTmp);
            freeDeviceMemory(devBeamFirstOutsideMaxTmp);
            beamTiming.weplReduceMs += perfElapsedMs(weplReducePerfStart);
        }

        std::vector<float> weplMinHost(tracerSteps, INF);
        {
            const auto weplReducePerfStart = perfNow();
            float* devWeplMin = (float*)allocateDeviceMemory(tracerSteps * sizeof(float));
            const int tracerThreadN = rayDims.x * rayDims.y;
            const int weplMinBlockSize = 128;
            dim3 weplMinGridDim(1, 1, tracerSteps);
            sliceMinVarIgnoreNeverEnter<weplMinBlockSize><<<weplMinGridDim, weplMinBlockSize>>>(
                devBevCumulSp, devFirstStepOutside, devWeplMin, tracerThreadN);
            checkCudaErrors(cudaDeviceSynchronize());
            copyToHost(weplMinHost.data(), devWeplMin, tracerSteps * sizeof(float));
            freeDeviceMemory(devWeplMin);
            beamTiming.weplReduceMs += perfElapsedMs(weplReducePerfStart);
        }

        if (rtdInputAuditEnabled()) {
            std::vector<float> hBevCumulSp(raySize);
            copyToHost(hBevCumulSp.data(), devBevCumulSp, raySize * sizeof(float));
            printStageVolumeSummary("WEPL_BEV_CUMUL_SP", hBevCumulSp, rayDims.x, rayDims.y, tracerSteps, 1.0e-6f, "step");
        }

        const float maxEnergy = findMax(beam.energies);
        const float maxEnergyIdx = findDecimalOrdered(energyData->energiesPerU, maxEnergy);
        const float maxPeakDepth_table = vectorInterpolate(energyData->peakDepths, maxEnergyIdx);
        float maxPeakDepth_mm = maxPeakDepth_table * energyDepthToMm;
        float maxLongitudinalCutoffMm = 0.0f;
        for (size_t layerIdx = 0; layerIdx < beam.energies.size(); ++layerIdx) {
            const float layerCutoffMm = layerLongitudinalCutoffMm(layerIdx);
            if (layerCutoffMm > maxLongitudinalCutoffMm) {
                maxLongitudinalCutoffMm = layerCutoffMm;
            }
        }
        const float longitudinalLimitAllMm = std::max(BP_DEPTH_CUTOFF * maxPeakDepth_mm,
                                                      maxLongitudinalCutoffMm);
        const int firstPastCutoffAll = findFirstLargerOrdered(weplMinHost, longitudinalLimitAllMm);
        const int beamFirstGuaranteedPassiveRT = std::min(firstPastCutoffAll, beamFirstOutsideRT);

        if (rtdInputAuditEnabled() && hasWeqVolume) {
            const float meanSourceDistMm = 0.5f * (sourceDistVec.x + sourceDistVec.y);
            const float weqProjectedStartMm = weqHeader[0] * lenToMm;
            const float geomProjectedStartMm = meanSourceDistMm - fanCorner_mm.z;
            const float entryZMm = float(beamFirstInsideRT) * fanDelta_mm.z + fanCorner_mm.z;
            const float geomProjectedEntryMm = meanSourceDistMm - entryZMm;
            const float weqProjectedEntryMm = weqProjectedStartMm + float(beamFirstInsideRT) * fabsf(weqHeader[1] * lenToMm);
            auto sampleWepl = [&](int step) -> float {
                if (step < 0 || step >= tracerSteps) return -1.0f;
                const float v = weplMinHost[static_cast<size_t>(step)];
                return hostIsFinite(v) ? v : -1.0f;
            };
            const int midStep = std::min(tracerSteps - 1,
                                         std::max(0, beamFirstInsideRT +
                                                     (beamFirstGuaranteedPassiveRT - beamFirstInsideRT) / 2));
            std::cout << "[INPUT_AUDIT][WRAPPER][WEQ_ALIGN]"
                      << " projectedStart(geom/weq)=(" << geomProjectedStartMm << "," << weqProjectedStartMm << ")"
                      << " delta=" << (geomProjectedStartMm - weqProjectedStartMm)
                      << " entryProjected(geom/weq)=(" << geomProjectedEntryMm << "," << weqProjectedEntryMm << ")"
                      << " delta=" << (geomProjectedEntryMm - weqProjectedEntryMm)
                      << " beamFirstInside=" << beamFirstInsideRT
                      << " beamFirstOutside=" << beamFirstOutsideRT
                      << " firstPastCutoffAll=" << firstPastCutoffAll
                      << " beamFirstGuaranteedPassive=" << beamFirstGuaranteedPassiveRT
                      << " weplMin(sample)=(" << sampleWepl(std::max(0, beamFirstInsideRT - 1))
                      << "," << sampleWepl(beamFirstInsideRT)
                      << "," << sampleWepl(midStep)
                      << "," << sampleWepl(std::max(0, beamFirstGuaranteedPassiveRT - 1)) << ")"
                      << std::endl;
        }

        if (hasRawSpotLattice) {
            const float entryZ_mm = float(beamFirstInsideRT) * fanDelta_mm.z + fanCorner_mm.z;
            const float2 pxSpMult = make_float2(1.0f - entryZ_mm / sourceDistVec.x,
                                                1.0f - entryZ_mm / sourceDistVec.y);
            const size_t spotPlaneN = (size_t)rawSpotLattice.spotGridDims.x * rawSpotLattice.spotGridDims.y;
            const size_t convIntermPlaneN = (size_t)rayDims.x * rawSpotLattice.spotGridDims.y;
            const size_t rayPlaneN = (size_t)rayDims.x * rayDims.y;
            const bool hasSubspotLutAvailable =
                beam.maxSubspotsPerLayer > 0 &&
                beam.subspotData.size() >= static_cast<size_t>(numLayers) * static_cast<size_t>(beam.maxSubspotsPerLayer) * 5ull;

            float* devBaseSpotPlane = (float*)allocateDeviceMemory(spotPlaneN * sizeof(float));
            float* devWorkSpotPlane = (float*)allocateDeviceMemory(spotPlaneN * sizeof(float));
            float* devConvIntermPlane = (float*)allocateDeviceMemory(convIntermPlaneN * sizeof(float));
            float* devTempRayPlane = (float*)allocateDeviceMemory(rayPlaneN * sizeof(float));
            float2* devEntrySigma = (float2*)allocateDeviceMemory(sizeof(float2));

            const int scaleThreads = 256;
            const int spotBlocks = static_cast<int>((spotPlaneN + scaleThreads - 1) / scaleThreads);
            const int rayBlocks = static_cast<int>((rayPlaneN + scaleThreads - 1) / scaleThreads);

            auto sigmaToMm = [lenToMm](float sigma) {
                if (lenToMm > 1.0f && sigma > 0.0f && sigma < 1.0f) return sigma * 10.0f;
                return sigma;
            };

            for (int layerNo = 0; layerNo < numLayers; ++layerNo) {
                const float energyPerU = beam.energies[layerNo];
                const float energyIdx = findDecimalOrdered(energyData->energiesPerU, energyPerU);
                float peakDepth = vectorInterpolate(energyData->peakDepths, energyIdx) * energyDepthToMm;
                const vec2f sigmaSqCoefs = FillIddAndSigmaParams().sigmaSqAirCoefs(peakDepth);
                const float* hostLayerSpotWeights = rawSpotLattice.spotWeights.data() + static_cast<size_t>(layerNo) * spotPlaneN;
                float* devRayWeightsLayer = devRayWeightsAllLayers + static_cast<size_t>(layerNo) * rayPlaneN;
                copyToDevice(devBaseSpotPlane, hostLayerSpotWeights, spotPlaneN * sizeof(float));
                cudaMemset(devRayWeightsLayer, 0, rayPlaneN * sizeof(float));

                bool accumulatedSubspot = false;
                if (useCarbonSubspotConvolution && hasSubspotLutAvailable) {
                    for (int subspotNo = 0; subspotNo < beam.maxSubspotsPerLayer; ++subspotNo) {
                        const size_t base = (static_cast<size_t>(layerNo) * static_cast<size_t>(beam.maxSubspotsPerLayer) +
                                             static_cast<size_t>(subspotNo)) * 5ull;
                        const float subspotWeight = beam.subspotData[base + 2];
                        if (!(subspotWeight > 1.0e-8f)) continue;

                        const float sigmaXmm = sigmaToMm(beam.subspotData[base + 3]);
                        const float sigmaYmm = sigmaToMm(beam.subspotData[base + 4]);
                        if (!(sigmaXmm > 1.0e-8f) || !(sigmaYmm > 1.0e-8f)) continue;

                        scalePlaneKernel<<<spotBlocks, scaleThreads>>>(devBaseSpotPlane, devWorkSpotPlane, static_cast<int>(spotPlaneN), subspotWeight);
                        checkCudaErrors(cudaGetLastError());

                        float2 entrySigma = make_float2(
                            sqrtf(std::max(0.0f, sigmaSqCoefs.x * entryZ_mm * entryZ_mm + sigmaSqCoefs.y * entryZ_mm + sigmaXmm * sigmaXmm)),
                            sqrtf(std::max(0.0f, sigmaSqCoefs.x * entryZ_mm * entryZ_mm + sigmaSqCoefs.y * entryZ_mm + sigmaYmm * sigmaYmm))
                        );
#ifdef NUCLEAR_CORR
#if NUCLEAR_CORR == GAUSS_FIT
                        if (runtimeNuclearEnabled) {
                            entrySigma.x *= 0.97f;
                            entrySigma.y *= 0.97f;
                        }
#endif
#endif
                        copyToDevice(devEntrySigma, &entrySigma, sizeof(float2));

                        const float3 subspotOffset = make_float3(
                            rawSpotLattice.spotOffset.x + beam.subspotData[base + 0] / lenToMm,
                            rawSpotLattice.spotOffset.y + beam.subspotData[base + 1] / lenToMm,
                            rawSpotLattice.spotOffset.z
                        );

                        performExactRTDConvolution2D(
                            devWorkSpotPlane,
                            devConvIntermPlane,
                            devTempRayPlane,
                            devEntrySigma,
                            make_uint3(rawSpotLattice.spotGridDims.x, rawSpotLattice.spotGridDims.y, 1),
                            make_uint3(rayDims.x, rayDims.y, 1),
                            rawSpotLattice.spotDelta,
                            subspotOffset,
                            make_float3(cpbResolution.x, cpbResolution.y, 0.0f),
                            make_float3(cpbCorner.x, cpbCorner.y, 0.0f),
                            pxSpMult
                        );
                        checkCudaErrors(cudaDeviceSynchronize());
                        addPlaneKernel<<<rayBlocks, scaleThreads>>>(devRayWeightsLayer, devTempRayPlane, static_cast<int>(rayPlaneN));
                        checkCudaErrors(cudaGetLastError());
                        accumulatedSubspot = true;
                    }
                }

                if (!accumulatedSubspot) {
                    const float2 spotSigma = (layerNo < static_cast<int>(beam.spotSigmas.size()))
                        ? beam.spotSigmas[layerNo]
                        : make_float2(1.0f, 1.0f);
                    const float sigmaXmm = sigmaToMm(spotSigma.x);
                    const float sigmaYmm = sigmaToMm(spotSigma.y);
                    float2 entrySigma = make_float2(
                        sqrtf(std::max(0.0f, sigmaSqCoefs.x * entryZ_mm * entryZ_mm + sigmaSqCoefs.y * entryZ_mm + sigmaXmm * sigmaXmm)),
                        sqrtf(std::max(0.0f, sigmaSqCoefs.x * entryZ_mm * entryZ_mm + sigmaSqCoefs.y * entryZ_mm + sigmaYmm * sigmaYmm))
                    );
#ifdef NUCLEAR_CORR
#if NUCLEAR_CORR == GAUSS_FIT
                    if (runtimeNuclearEnabled) {
                        entrySigma.x *= 0.97f;
                        entrySigma.y *= 0.97f;
                    }
#endif
#endif
                    checkCudaErrors(cudaMemcpy(devWorkSpotPlane, devBaseSpotPlane, spotPlaneN * sizeof(float), cudaMemcpyDeviceToDevice));
                    copyToDevice(devEntrySigma, &entrySigma, sizeof(float2));
                    performExactRTDConvolution2D(
                        devWorkSpotPlane,
                        devConvIntermPlane,
                        devRayWeightsLayer,
                        devEntrySigma,
                        make_uint3(rawSpotLattice.spotGridDims.x, rawSpotLattice.spotGridDims.y, 1),
                        make_uint3(rayDims.x, rayDims.y, 1),
                        rawSpotLattice.spotDelta,
                        rawSpotLattice.spotOffset,
                        make_float3(cpbResolution.x, cpbResolution.y, 0.0f),
                        make_float3(cpbCorner.x, cpbCorner.y, 0.0f),
                        pxSpMult
                    );
                    checkCudaErrors(cudaDeviceSynchronize());
                }
            }

            freeDeviceMemory(devBaseSpotPlane);
            freeDeviceMemory(devWorkSpotPlane);
            freeDeviceMemory(devConvIntermPlane);
            freeDeviceMemory(devTempRayPlane);
            freeDeviceMemory(devEntrySigma);

            if (fineTiming) {
                std::cout << "  [RTD_CONV] entryZ_mm=" << entryZ_mm
                          << " pxSpMult=(" << pxSpMult.x << "," << pxSpMult.y << ")"
                          << " mode=" << ((useCarbonSubspotConvolution && hasSubspotLutAvailable) ? "per-layer-subspot" : "per-layer-sigma")
                          << " subspotAvailable=" << (hasSubspotLutAvailable ? 1 : 0)
                          << " using beamFirstInside=" << beamFirstInsideRT
                          << std::endl;
            }
        }

        beamTiming.rayWeightMs += perfElapsedMs(rayWeightPerfStart);

        if (fineTiming) {
            const int nRays = rayDims.x * rayDims.y;
            double sumAllLayers = 0.0;
            float maxAllLayers = 0.0f;
            int nnzAllLayers = 0;
            std::vector<float> hRayWeights(nRays);
            for (int l = 0; l < numLayers; ++l) {
                float* devRayWeightsLayer = devRayWeightsAllLayers + l * nRays;
                copyToHost(hRayWeights.data(), devRayWeightsLayer, nRays * sizeof(float));
                const FloatSummaryStats st = summarizeFloatVector(hRayWeights, 1e-6f);
                std::cout << "  [RAY_W] Layer " << l
                          << " sum=" << st.sumFinite
                          << " max=" << st.maxFinite
                          << " nnz(>0)=" << st.countPositive << "/" << nRays
                          << " nnz(>1e-6)=" << st.countGT << "/" << nRays
                          << " nan=" << st.countNaN
                          << " inf=" << st.countInf
                          << std::endl;
                sumAllLayers += st.sumFinite;
                if (st.maxFinite > maxAllLayers) maxAllLayers = st.maxFinite;
                nnzAllLayers += st.countPositive;
            }
            std::cout << "  [RAY_W] All layers total sum=" << sumAllLayers
                      << " global max=" << maxAllLayers
                      << " total nnz(>0)=" << nnzAllLayers
                      << " (note: nnz counts are per-layer, not unique rays)" << std::endl;
        }

        if (rtdInputAuditEnabled()) {
            const size_t rayWeightsElems =
                static_cast<size_t>(numLayers) *
                static_cast<size_t>(rayDims.x) *
                static_cast<size_t>(rayDims.y);
            std::vector<float> hRayWeightsAll(rayWeightsElems);
            copyToHost(hRayWeightsAll.data(), devRayWeightsAllLayers, rayWeightsElems * sizeof(float));
            printStageVolumeSummary("SPOT_SUBSPOT_TO_RAY", hRayWeightsAll, rayDims.x, rayDims.y, numLayers, 1.0e-6f, "layer");
        }

        const int maxSuperpR = MAX_SUPERP_RADIUS;
        const int superpRayDimsXBeam = ((rayDims.x + SUPERP_TILE_X - 1) / SUPERP_TILE_X) * SUPERP_TILE_X;
        const int superpRayDimsYBeam = ((rayDims.y + SUPERP_TILE_Y - 1) / SUPERP_TILE_Y) * SUPERP_TILE_Y;
        const size_t rayPlaneElems = static_cast<size_t>(rayDims.x) * static_cast<size_t>(rayDims.y);
        const size_t paddedRayElems = static_cast<size_t>(superpRayDimsXBeam) *
                                      static_cast<size_t>(superpRayDimsYBeam) *
                                      static_cast<size_t>(tracerSteps);
        const int bevDoseXBeam = superpRayDimsXBeam + 2 * maxSuperpR;
        const int bevDoseYBeam = superpRayDimsYBeam + 2 * maxSuperpR;
        const int bevDoseZBeam = beamFirstGuaranteedPassiveRT;
        const size_t bevDoseSizeBeam = static_cast<size_t>(bevDoseXBeam) *
                                       static_cast<size_t>(bevDoseYBeam) *
                                       static_cast<size_t>(bevDoseZBeam) * sizeof(float);
        float* devRayIddScratch = (float*)allocateDeviceMemory(raySize * sizeof(float));
        float* devRayRSigmaEffScratch = (float*)allocateDeviceMemory(raySize * sizeof(float));
        int* devFirstPassiveScratch = (int*)allocateDeviceMemory(rayPlaneElems * sizeof(int));
        int* devBeamFirstPassiveMaxScratch = (int*)allocateDeviceMemory(sizeof(int));
        float* devRayIddPaddedScratch = nullptr;
        float* devRayRSigmaEffPaddedScratch = nullptr;
        if (superpRayDimsXBeam != rayDims.x || superpRayDimsYBeam != rayDims.y) {
            devRayIddPaddedScratch = (float*)allocateDeviceMemory(paddedRayElems * sizeof(float));
            devRayRSigmaEffPaddedScratch = (float*)allocateDeviceMemory(paddedRayElems * sizeof(float));
        }
        float* devBevPrimDoseScratch = (float*)allocateDeviceMemory(bevDoseSizeBeam);
#ifdef NUCLEAR_CORR
        size_t maxNucPlaneN = 0;
        int maxNucRayDimsX = 0;
        int maxNucRayDimsY = 0;
        if (runtimeNuclearEnabled) {
            for (const HaloLatticePlan& plan : haloPlans) {
                maxNucPlaneN = std::max(maxNucPlaneN, plan.nucPlaneN);
                maxNucRayDimsX = std::max(maxNucRayDimsX, plan.nucRayDims.x);
                maxNucRayDimsY = std::max(maxNucRayDimsY, plan.nucRayDims.y);
            }
        }
        const size_t nucSpotIdxSizeBeam =
            runtimeNuclearEnabled ? (rayPlaneElems * sizeof(int)) : 0u;
        const size_t nucRaySizeBeam =
            runtimeNuclearEnabled ? (maxNucPlaneN * static_cast<size_t>(tracerSteps)) : 0u;
        const size_t bevNucDoseSizeBeam =
            runtimeNuclearEnabled
                ? (static_cast<size_t>(maxNucRayDimsX + 2 * maxSuperpR) *
                   static_cast<size_t>(maxNucRayDimsY + 2 * maxSuperpR) *
                   static_cast<size_t>(bevDoseZBeam) * sizeof(float))
                : 0u;
        float* devNucRayWeightsScratch =
            runtimeNuclearEnabled ? (float*)allocateDeviceMemory(maxNucPlaneN * sizeof(float)) : nullptr;
        int* devNucSpotIdxScratch =
            runtimeNuclearEnabled ? (int*)allocateDeviceMemory(nucSpotIdxSizeBeam) : nullptr;
        float* devNucIddScratch =
            runtimeNuclearEnabled ? (float*)allocateDeviceMemory(nucRaySizeBeam * sizeof(float)) : nullptr;
        float* devNucRSigmaEffScratch =
            runtimeNuclearEnabled ? (float*)allocateDeviceMemory(nucRaySizeBeam * sizeof(float)) : nullptr;
        float* devBevNucDoseScratch =
            runtimeNuclearEnabled ? (float*)allocateDeviceMemory(bevNucDoseSizeBeam) : nullptr;
#endif
        
        // process for each energy layer
        
        for (size_t layerIdx = 0; layerIdx < beam.energies.size(); ++layerIdx) {
            float energy = beam.energies[layerIdx];
            LayerPerfTiming layerPerf;
            layerPerf.layerIdx = layerIdx;
            layerPerf.energy = energy;
            layerPerf.spots = (layerIdx < beam.layerSpotCounts.size()) ? beam.layerSpotCounts[layerIdx] : 0;
            layerPerf.rayDimsX = rayDims.x;
            layerPerf.rayDimsY = rayDims.y;
            const auto layerPerfStart = perfNow();
            
            // Get rayWeights for current layer
            float* devRayWeights = devRayWeightsAllLayers + layerIdx * rayDims.x * rayDims.y;
            
            // Find energy index (float index for interpolation).
// IMPORTANT: energiesPerU can be ascending or descending. The old patch8 logic clamped almost
// everything when the LUT table was descending, making "changing energy has no effect".
float energyIdx = 0.0f;
bool clampedLow = false;
bool clampedHigh = false;

const int nEnergies = energyData->nEnergies;
const bool ascendingE = (nEnergies > 1) ? (energyData->energiesPerU.front() < energyData->energiesPerU.back()) : true;

// Define physical low/high energy ends depending on ordering
const int idxLow  = ascendingE ? 0 : (nEnergies - 1);   // lowest energy index
const int idxHigh = ascendingE ? (nEnergies - 1) : 0;   // highest energy index
const float E_low  = energyData->energiesPerU[idxLow];
const float E_high = energyData->energiesPerU[idxHigh];

int floorIdxUsed = 0;
float corrUsed = 0.0f;
float e0 = energyData->energiesPerU.front();
float e1 = energyData->energiesPerU.front();

if (nEnergies <= 1) {
    energyIdx = 0.0f;
    clampedLow = clampedHigh = false;
    floorIdxUsed = 0;
    corrUsed = 0.0f;
    e0 = energyData->energiesPerU.front();
    e1 = energyData->energiesPerU.front();
} else if (energy <= E_low) {
    // Clamp to low-energy end (below LUT range)
    clampedLow = true;
    energyIdx = (float)idxLow;

    floorIdxUsed = ascendingE ? 0 : (nEnergies - 2);
    corrUsed = ascendingE ? 0.0f : 1.0f;

    e0 = energyData->energiesPerU[floorIdxUsed];
    e1 = energyData->energiesPerU[floorIdxUsed + 1];
} else if (energy >= E_high) {
    // Clamp to high-energy end (above LUT range)
    clampedHigh = true;
    energyIdx = (float)idxHigh;

    floorIdxUsed = ascendingE ? (nEnergies - 2) : 0;
    corrUsed = ascendingE ? 1.0f : 0.0f;

    e0 = energyData->energiesPerU[floorIdxUsed];
    e1 = energyData->energiesPerU[floorIdxUsed + 1];
} else {
    // Find segment containing energy (inclusive)
    for (int i = 0; i < nEnergies - 1; ++i) {
        const float a = energyData->energiesPerU[i];
        const float b = energyData->energiesPerU[i + 1];
        const float lo = (a < b) ? a : b;
        const float hi = (a < b) ? b : a;
        if (energy >= lo && energy <= hi) {
            floorIdxUsed = i;
            e0 = a;
            e1 = b;
            break;
        }
    }

    const float denom = (e1 - e0);
    corrUsed = (denom != 0.0f) ? ((energy - e0) / denom) : 0.0f;
    if (!std::isfinite(corrUsed)) corrUsed = 0.0f;
    corrUsed = std::min(1.0f, std::max(0.0f, corrUsed));

    energyIdx = float(floorIdxUsed) + corrUsed;
}

// Warn when energy is truly outside LUT range (silent failure otherwise)
const bool outOfRange = (energy < E_low) || (energy > E_high);
        if (outOfRange && fineTiming) {
            std::cout << "[ENERGY] Warning: layer " << layerIdx
                      << " E=" << energy
                      << " outside LUT range [" << E_low << ", " << E_high << "]"
              << " (order=" << (ascendingE ? "asc" : "desc") << ")"
              << " -> clamped energyIdx=" << energyIdx
              << std::endl;
}

if (fineTiming) {
    std::cout << "\n[ENERGY] Layer " << layerIdx
              << " E=" << energy
              << " -> energyIdx=" << energyIdx
              << " (floor=" << floorIdxUsed << ", corr=" << corrUsed
              << ", bracket=[" << e0 << ", " << e1 << "])"
              << (clampedLow ? " CLAMP_LOW" : "")
              << (clampedHigh ? " CLAMP_HIGH" : "")
              << std::endl;
}

// Calculate energyScaleFact using vectorInterpolate
            float energyScaleFact = 1.0f;
            if (energyIdx <= 0.0f) {
                energyScaleFact = energyData->scaleFacts[0];
            } else if (energyIdx >= float(energyData->nEnergies - 1)) {
                energyScaleFact = energyData->scaleFacts[energyData->nEnergies - 1];
            } else {
                float intPart;
                float decimals = std::modf(energyIdx, &intPart);
                int floorIdx = static_cast<int>(intPart);
                energyScaleFact = energyData->scaleFacts[floorIdx] + 
                                 (energyData->scaleFacts[floorIdx + 1] - energyData->scaleFacts[floorIdx]) * decimals;
            }
            
            // Calculate peakDepth using vectorInterpolate
            float peakDepth = 10.0f;
            if (energyIdx <= 0.0f) {
                peakDepth = energyData->peakDepths[0];
            } else if (energyIdx >= float(energyData->nEnergies - 1)) {
                peakDepth = energyData->peakDepths[energyData->nEnergies - 1];
            } else {
                float intPart;
                float decimals = std::modf(energyIdx, &intPart);
                int floorIdx = static_cast<int>(intPart);
                peakDepth = energyData->peakDepths[floorIdx] + 
                           (energyData->peakDepths[floorIdx + 1] - energyData->peakDepths[floorIdx]) * decimals;
            }

            // ------------------------------------------------------------------------
            // Normalize energy depth units to mm so that:
            //   depthIdx = WEPL_mm * energyScaleFact + 0.5
            // matches the depth axis of cumulIddTex.
            // ------------------------------------------------------------------------
            const float energyScaleFact_table = energyScaleFact;
            const float peakDepth_table = peakDepth;
            energyScaleFact = energyScaleFact_table / energyDepthToMm;
            peakDepth = peakDepth_table * energyDepthToMm;
            const float layerCutoffMm = layerLongitudinalCutoffMm(layerIdx);

            if (fineTiming) {
                std::cout << "[ENERGY]   scaleFact(table)=" << energyScaleFact_table
                          << "  peakDepth(table)=" << peakDepth_table
                          << "  -> scaleFact(mm)=" << energyScaleFact
                          << "  peakDepth(mm)=" << peakDepth
                          << "  rangeStopDepth(mm)=" << (BP_DEPTH_CUTOFF * peakDepth)
                          << "  externalLongitudalCutoff(mm)=" << layerCutoffMm
                          << "  (energyDepthToMm=" << energyDepthToMm << ")"
                          << std::endl;
            }
            
            const HaloLatticePlan* layerHaloPlan =
                runtimeNuclearEnabled ? &haloPlans[static_cast<size_t>(layerIdx)] : nullptr;
            // [9.33] Expected dimensionless spotDistInRays = halo physical-PB spacing / CPB
            // ray spacing. Post-9.23 the halo lattice is the canonical PB grid (e.g. 5x5
            // at 5mm) while the primary helper lattice (rawSpotLattice) stays at the WEQ
            // texel grid (e.g. 80x60 at 1mm); therefore expected MUST come from
            // layerHaloPlan, not rawSpotLattice.
            const float expectedHaloSpotDistInRays =
                (runtimeNuclearEnabled && layerHaloPlan != nullptr && beam.raySpacing.x > 0.0f)
                    ? (layerHaloPlan->spotDelta.x / beam.raySpacing.x)
                    : 0.0f;
            const bool hasExplicitLayerSpotDeltasForLayer =
                beam.layerSpotDeltas.size() == static_cast<size_t>(numLayers);
            const float3 physicalSpotDeltaForLayer =
                (hasExplicitLayerSpotDeltasForLayer && layerIdx < beam.layerSpotDeltas.size())
                    ? make_float3(beam.layerSpotDeltas[static_cast<size_t>(layerIdx)].x,
                                  beam.layerSpotDeltas[static_cast<size_t>(layerIdx)].y,
                                  0.0f)
                    : make_float3(
                          (layerHaloPlan != nullptr) ? layerHaloPlan->spotDelta.x : beam.spotDelta.x,
                          (layerHaloPlan != nullptr) ? layerHaloPlan->spotDelta.y : beam.spotDelta.y,
                          0.0f);
            const float expectedPhysicalSpotDistInRays =
                (runtimeNuclearEnabled && beam.raySpacing.x > 0.0f && physicalSpotDeltaForLayer.x > 0.0f)
                    ? (physicalSpotDeltaForLayer.x / beam.raySpacing.x)
                    : 0.0f;

            const auto layerAllocStart = perfNow();
            float* devRayIdd = devRayIddScratch;
            float* devRayRSigmaEff = devRayRSigmaEffScratch;
            int* devFirstPassive = devFirstPassiveScratch;
            checkCudaErrors(cudaMemset(devRayIdd, 0, raySize * sizeof(float)));
            const int initSigmaThreads = 256;
            const int initSigmaBlocks = static_cast<int>((raySize + static_cast<size_t>(initSigmaThreads) - 1) / static_cast<size_t>(initSigmaThreads));
            fillFloatKernel<<<initSigmaBlocks, initSigmaThreads>>>(
                devRayRSigmaEff,
                static_cast<int>(raySize),
                std::numeric_limits<float>::infinity());
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaMemset(devFirstPassive, 0, rayPlaneElems * sizeof(int)));
            // [9.59] Pre-allocation nuclear spotDist gate.
            // Compute spotDistInRays here (same priority as the full computation below)
            // so we can fail-fast before allocating devNucIdd / devNucRSigmaEff.
            // If spotDist < 1.5 nuclear σ_eff will not be compressed and tileRadCalc
            // will almost certainly exceed kMaxSuperpR; fail now with a diagnostic.
            if (runtimeNuclearEnabled && layerHaloPlan != nullptr && beam.raySpacing.x > 0.0f) {
                float preCheckDeltaX;
                if (hasExplicitLayerSpotDeltasForLayer && physicalSpotDeltaForLayer.x > 0.0f) {
                    preCheckDeltaX = physicalSpotDeltaForLayer.x;
                } else {
                    preCheckDeltaX = layerHaloPlan->spotDelta.x;
                }
                const float preCheckSpotDist = preCheckDeltaX / beam.raySpacing.x;
                if (preCheckSpotDist < 1.5f) {
                    throw std::runtime_error(
                        "[9.59] Nuclear superposition gate: spotDistInRays=" +
                        std::to_string(preCheckSpotDist) +
                        " < 1.5 for layer=" + std::to_string(layerIdx) +
                        ". Nuclear sigma will not be compressed; tileRadCalc would exceed kMaxSuperpR=32. "
                        "Cause: canonical PB lattice inference failed or beam.spotDelta not set. "
                        "beam.spotDelta=(" + std::to_string(beam.spotDelta.x) + "," +
                        std::to_string(beam.spotDelta.y) + ") "
                        "layerHaloPlan->spotDelta=(" + std::to_string(layerHaloPlan->spotDelta.x) + "," +
                        std::to_string(layerHaloPlan->spotDelta.y) + ") "
                        "beam.raySpacing.x=" + std::to_string(beam.raySpacing.x));
                }
            }

#ifdef NUCLEAR_CORR
            float* devNucRayWeights = nullptr;
            int* devNucSpotIdx = nullptr;
            const size_t nucSpotIdxSize =
                runtimeNuclearEnabled ? (static_cast<size_t>(rayDims.x) * static_cast<size_t>(rayDims.y) * sizeof(int)) : 0u;
            if (runtimeNuclearEnabled) {
                const size_t haloWeightsSize = layerHaloPlan->nucPlaneN * sizeof(float);
                devNucRayWeights = devNucRayWeightsScratch;
                copyToDevice(devNucRayWeights, layerHaloPlan->paddedSpotWeights.data(), haloWeightsSize);
                devNucSpotIdx = devNucSpotIdxScratch;
                copyToDevice(devNucSpotIdx, layerHaloPlan->rayToNucSpotIdx.data(), nucSpotIdxSize);
            }

            const size_t nucRaySize = runtimeNuclearEnabled ? (layerHaloPlan->nucPlaneN * static_cast<size_t>(tracerSteps)) : 0;
            float* devNucIdd = nullptr;
            float* devNucRSigmaEff = nullptr;
            if (runtimeNuclearEnabled) {
                devNucIdd = devNucIddScratch;
                devNucRSigmaEff = devNucRSigmaEffScratch;
                cudaMemset(devNucIdd, 0, nucRaySize * sizeof(float));
                const int haloFillThreads = 256;
                const int haloFillBlocks = static_cast<int>((nucRaySize + static_cast<size_t>(haloFillThreads) - 1) / static_cast<size_t>(haloFillThreads));
                fillFloatKernel<<<haloFillBlocks, haloFillThreads>>>(devNucRSigmaEff,
                                                                    static_cast<int>(nucRaySize),
                                                                    std::numeric_limits<float>::infinity());
                checkCudaErrors(cudaGetLastError());
            }
#endif
            layerPerf.allocMs += perfElapsedMs(layerAllocStart);
            
            // Create IDD parameters
            FillIddAndSigmaParams iddParams;
            iddParams.energyIdx = energyIdx;  // Float index for texture interpolation
            iddParams.energyScaleFact = energyScaleFact;  // Interpolated scale factor
            iddParams.peakDepth = peakDepth;  // Interpolated peak depth
            iddParams.rangeStopDepth = BP_DEPTH_CUTOFF * peakDepth;
            iddParams.rRlScale = energyData->rRlScaleFact;

            // spotDist is **dimensionless**: spot spacing expressed in number of rays.
            // RayTraceDicom reference: spotDistInRays = beam.spotDelta.x / beam.raySpacing.x
            // (kernel_wrapper.cu:941 — direct lookup, no inference needed when beam.spotDelta is
            // explicitly supplied by the caller with the canonical physical PB spacing).
            //
            // Priority:
            //   1. explicit per-layer CarbonPBS/dosecal spacing -> physical PB spacing.
            //   2. runtimeNuclearEnabled legacy path -> layerHaloPlan->spotDelta.
            //   3. fallback: beam.spotDelta for non-halo paths.
            float spotDistInRays = 1.0f;
            if (beam.raySpacing.x > 0.0f) {
                float chosenDeltaX;
                if (hasExplicitLayerSpotDeltasForLayer && physicalSpotDeltaForLayer.x > 0.0f) {
                    chosenDeltaX = physicalSpotDeltaForLayer.x;
                } else if (runtimeNuclearEnabled && layerHaloPlan != nullptr) {
                    chosenDeltaX = layerHaloPlan->spotDelta.x;
                } else {
                    chosenDeltaX = beam.spotDelta.x;
                }
                spotDistInRays = chosenDeltaX / beam.raySpacing.x;
                if (!(spotDistInRays > 0.0f)) spotDistInRays = 1.0f;
            }
            iddParams.spotDist = spotDistInRays;
            if (runtimeNuclearEnabled && layerHaloPlan != nullptr) {
                // [9.24] Legacy diagnostic warning (NOT a hard fail). In the
                // explicit CarbonPBS path, placement grid spacing and physical PB
                // spacing are intentionally separate: idbeamxy drives rayweight
                // placement, layerSpotDeltas drives spotDist/nuclear normalization.
                if (beam.layerSpotDeltas.size() != static_cast<size_t>(numLayers) &&
                    rawSpotLattice.spotDelta.x > 0.0f && rawSpotLattice.spotDelta.y > 0.0f) {
                    const float refDeltaX = rawSpotLattice.spotDelta.x;
                    const float refDeltaY = rawSpotLattice.spotDelta.y;
                    const float ratioX = layerHaloPlan->spotDelta.x / refDeltaX;
                    const float ratioY = layerHaloPlan->spotDelta.y / refDeltaY;
                    const float roundedX = std::round(ratioX);
                    const float roundedY = std::round(ratioY);
                    const bool ratioBad = !(ratioX >= 0.999f && ratioY >= 0.999f) ||
                                          std::fabs(ratioX - roundedX) > 0.25f ||
                                          std::fabs(ratioY - roundedY) > 0.25f ||
                                          roundedX > 32.0f || roundedY > 32.0f;
                    if (ratioBad) {
                        static thread_local int s_ratioWarn = 0;
                        if (s_ratioWarn++ < 3) {
                            std::cerr << "  [9.24] WARN: halo/primary lattice ratio=("
                                      << ratioX << "," << ratioY
                                      << ") not a clean integer multiple; halo lateral profile may be off"
                                      << std::endl;
                        }
                    }
                }
                if (!(expectedHaloSpotDistInRays > 0.0f) ||
                    !approxEq(iddParams.spotDist, expectedHaloSpotDistInRays, 1.0e-3f)) {
                    static thread_local int s_distWarn = 0;
                    if (s_distWarn++ < 3) {
                        if (fineTiming) {
                            std::cerr << "  [HALO_AUDIT] WARN: site-1 spotDist=" << iddParams.spotDist
                                      << " differs from expected=" << expectedHaloSpotDistInRays
                                      << " (delta=" << (iddParams.spotDist - expectedHaloSpotDistInRays) << ")"
                                      << std::endl;
                        }
                    }
                }
                if (fineTiming || haloAudit) {
                    // [9.50] Lateral-profile audit: nucRayDims and paddedSpotWeights
                    // occupancy directly indicate lattice identity.
                    //   Pre-9.23 (broken): nucRayDims = padded(CPB grid), e.g. (96, 64);
                    //                      occupancy ~ 25/6144 = 0.4% (sparse-on-dense).
                    //   Post-9.23 (fixed):  nucRayDims = padded(PB grid), e.g. (32, 32);
                    //                      occupancy ~ 25/1024 = 2.4% (sparse-on-coarse).
                    int paddedNonzero = 0;
                    double paddedWeightSum = 0.0;
                    for (float v : layerHaloPlan->paddedSpotWeights) {
                        if (v > 0.0f) ++paddedNonzero;
                        if (hostIsFinite(v)) paddedWeightSum += static_cast<double>(v);
                    }
                    const size_t totalCells = layerHaloPlan->paddedSpotWeights.size();
                    const double occupancy = (totalCells > 0)
                        ? (static_cast<double>(paddedNonzero) / static_cast<double>(totalCells))
                        : 0.0;
                    int validMapped = 0;
                    int invalidMapped = 0;
                    int mappedWeightNonzero = 0;
                    double mappedWeightSum = 0.0;
                    int mappedPrimaryRayWeightNonzero = 0;
                    double mappedPrimaryRayWeightSum = 0.0;
                    double primaryRayWeightSum = 0.0;
                    std::vector<float> hMappedRayWeights(static_cast<size_t>(rayDims.x) * static_cast<size_t>(rayDims.y), 0.0f);
                    copyToHost(hMappedRayWeights.data(), devRayWeights, hMappedRayWeights.size() * sizeof(float));
                    for (size_t mapIdx = 0; mapIdx < layerHaloPlan->rayToNucSpotIdx.size(); ++mapIdx) {
                        const float rw = (mapIdx < hMappedRayWeights.size()) ? hMappedRayWeights[mapIdx] : 0.0f;
                        if (hostIsFinite(rw)) primaryRayWeightSum += static_cast<double>(rw);
                        const int nucIdx = layerHaloPlan->rayToNucSpotIdx[mapIdx];
                        if (nucIdx < 0 || static_cast<size_t>(nucIdx) >= layerHaloPlan->paddedSpotWeights.size()) {
                            ++invalidMapped;
                            continue;
                        }
                        ++validMapped;
                        const float nw = layerHaloPlan->paddedSpotWeights[static_cast<size_t>(nucIdx)];
                        if (nw > 0.0f) {
                            ++mappedWeightNonzero;
                            mappedWeightSum += static_cast<double>(nw);
                        }
                        if (rw > 0.0f) {
                            ++mappedPrimaryRayWeightNonzero;
                            mappedPrimaryRayWeightSum += static_cast<double>(rw);
                        }
                    }
                    const double haloToPhysicalRatioX =
                        (physicalSpotDeltaForLayer.x > 0.0f)
                            ? static_cast<double>(layerHaloPlan->spotDelta.x) / static_cast<double>(physicalSpotDeltaForLayer.x)
                            : 0.0;
                    const double haloToPhysicalRatioY =
                        (physicalSpotDeltaForLayer.y > 0.0f)
                            ? static_cast<double>(layerHaloPlan->spotDelta.y) / static_cast<double>(physicalSpotDeltaForLayer.y)
                            : 0.0;
                    const bool physicalPbLike =
                        std::fabs(haloToPhysicalRatioX - 1.0) <= 0.05 &&
                        std::fabs(haloToPhysicalRatioY - 1.0) <= 0.05;
                    const char* gridIdentity =
                        physicalPbLike ? "physical_pb_like" :
                        (hasExplicitLayerSpotDeltasForLayer ? "dense_or_nonphysical_vs_explicit_pb" : "dense_or_nonphysical");
                    std::cout << "  [RTD_HALO_AUDIT] layer=" << layerIdx
                              << " primarySpotDelta=(" << rawSpotLattice.spotDelta.x << "," << rawSpotLattice.spotDelta.y << ")"
                              << " cpbSpacing=(" << beam.raySpacing.x << "," << beam.raySpacing.y << ")"
                              << " haloGridDelta=(" << layerHaloPlan->spotDelta.x << "," << layerHaloPlan->spotDelta.y << ")"
                              << " physicalPBDelta=(" << physicalSpotDeltaForLayer.x << "," << physicalSpotDeltaForLayer.y << ")"
                              << " haloToPhysicalRatio=(" << haloToPhysicalRatioX << "," << haloToPhysicalRatioY << ")"
                              << " gridIdentity=" << gridIdentity
                              << " spotDistInRays=" << iddParams.spotDist
                              << " expectedSpotDistInRays=" << expectedHaloSpotDistInRays
                              << " expectedPhysicalSpotDistInRays=" << expectedPhysicalSpotDistInRays
                              << " nucRayDims=(" << layerHaloPlan->nucRayDims.x << "," << layerHaloPlan->nucRayDims.y << ")"
                              << " validMapped=" << validMapped
                              << " invalidMapped=" << invalidMapped
                              << " mappedWeightNonzero=" << mappedWeightNonzero
                              << " mappedWeightSum=" << mappedWeightSum
                              << " mappedPrimaryRayWeightNonzero=" << mappedPrimaryRayWeightNonzero
                              << " mappedPrimaryRayWeightSum=" << mappedPrimaryRayWeightSum
                              << " primaryRayWeightSum=" << primaryRayWeightSum
                              << " paddedNonzero=" << paddedNonzero
                              << " paddedTotal=" << totalCells
                              << " paddedWeightSum=" << paddedWeightSum
                              << " occupancy=" << occupancy
                              << std::endl;
                }
            }
            iddParams.nucMemStep = 0;
            iddParams.rayDimsX = rayDims.x;
            iddParams.rayDimsY = rayDims.y;
            iddParams.first = 0;
            iddParams.afterLast = tracerSteps;

	            // ------------------------------------------------------------
	            // IMPORTANT (unit normalization):
	            //   RayTraceDicom LUT peakDepths/scaleFacts are in **mm**.
	            //   All "fan" geometry (corner/delta/dist) provided to the
	            //   IDD/Sigma stage must match the geometry used by the ray
	            //   tracer (DensityAndSpTracerParams).
	            // ------------------------------------------------------------

	            // Spot sigma is expected in mm for RayTraceDicom. If the input
	            // looks cm-like and sigma is very small (<1), treat it as cm.
	            // entrySigmaSq is the **variance** at iso in air.
	            // RTDBeamSettings stores spotSigmas as (sigmax, sigmay) per layer.
	            float2 spotSigmaIso = make_float2(0.0f, 0.0f);
	            if (layerIdx < beam.spotSigmas.size()) {
	                spotSigmaIso = beam.spotSigmas[layerIdx];
	            }
	            float sigmaX_mm = spotSigmaIso.x;
	            float sigmaY_mm = spotSigmaIso.y;
	            if (lenToMm > 1.0f) {
	                if (sigmaX_mm > 0.0f && sigmaX_mm < 1.0f) sigmaX_mm *= 10.0f;
	                if (sigmaY_mm > 0.0f && sigmaY_mm < 1.0f) sigmaY_mm *= 10.0f;
	            }
	            iddParams.entrySigmaSq = 0.5f * (sigmaX_mm * sigmaX_mm + sigmaY_mm * sigmaY_mm);
	            iddParams.stepLength = fabsf(fanDelta_mm.z);
	            // These are recomputed in initStepAndAirDiv(); initialize to 0.
	            iddParams.sigmaSqAirLin = 0.0f;
	            iddParams.sigmaSqAirQuad = 0.0f;
	            // Fan geometry (mm). Must match the tracer's fanIdxToFan settings.
	            iddParams.dist = make_vec3f(sourceDistVec.x, sourceDistVec.y, 0.0f);
	            iddParams.corner = fanCorner_mm;
	            iddParams.delta = fanDelta_mm;
            
            // Calculate volConst, volLin, volSq for stepVol calculation
            // Based on RayTracedicom formula: volPerDist(k) = volConst + k*volLin + k*k*volSq
            float deltaX = iddParams.delta.x;
            float deltaY = iddParams.delta.y;
            float deltaZ = iddParams.delta.z;
            float cornerZ = iddParams.corner.z;
            float distX = iddParams.dist.x;
            float distY = iddParams.dist.y;
            float deltaXYZ = fabsf(deltaX * deltaY * deltaZ);
            
            iddParams.volConst = deltaXYZ * (1.0f - cornerZ/distX - cornerZ/distY + 
                                             (cornerZ*cornerZ + deltaZ*deltaZ/12.0f)/(distX*distY));
            iddParams.volLin = deltaXYZ * deltaZ * (-1.0f/distX - 1.0f/distY + 2.0f*cornerZ/(distX*distY));
            iddParams.volSq = deltaXYZ * deltaZ * deltaZ / (distX * distY);
            
            // Initialize step and air division parameters
            iddParams.initStepAndAirDiv();
        // FillIddAndSigmaParams debug (helps diagnose rSigmaEff NaNs)
        if (rtdSigmaDebugEnabled()) {
            const float incDiv0 = iddParams.sigmaSqAirLin + (2.0f * static_cast<float>(iddParams.first) - 1.0f) * iddParams.sigmaSqAirQuad;
            std::cout << "  [SIGMA_AIR] peakDepth=" << iddParams.peakDepth
                      << " rangeStopDepth=" << iddParams.rangeStopDepth
                      << " deltaZ=" << iddParams.delta.z
                      << " cornerZ=" << iddParams.corner.z
                      << " sigmaSqAirLin=" << iddParams.sigmaSqAirLin
                      << " sigmaSqAirQuad=" << iddParams.sigmaSqAirQuad
                      << " incDiv(step0)=" << incDiv0
                      << " stepVol0=" << iddParams.stepVol(0)
                      << " stepVol1=" << iddParams.stepVol(1)
                      << "\n";
        }

            const float longitudinalLimitMm = std::max(BP_DEPTH_CUTOFF * peakDepth, layerCutoffMm);
            const int localAfterLastStep = findFirstLargerOrdered(weplMinHost, longitudinalLimitMm);
            const int afterLastStep = std::min(localAfterLastStep, beamFirstGuaranteedPassiveRT);

            if (fineTiming) {
                std::cout << "  [CUTOFF] maxPeakDepth(table)=" << maxPeakDepth_table
                          << " -> maxPeakDepth(mm)=" << maxPeakDepth_mm
                          << "  firstPastCutoffAll=" << firstPastCutoffAll
                          << "  beamFirstGuaranteedPassive=" << beamFirstGuaranteedPassiveRT
                          << "  localAfterLast=" << localAfterLastStep
                          << "  longitudalLimitMm=" << longitudinalLimitMm
                          << "  externalLayerCutoffMm=" << layerCutoffMm
                          << "  afterLast=" << afterLastStep
                          << std::endl;
            }

            // Entry sigma at entryZ (reference uses sigmaSqAirCoefs + spotSigma^2)
            const float entryZ = float(beamFirstInsideRT) * fanDelta_mm.z + fanCorner_mm.z;
            const vec2f sigmaSqCoefs = iddParams.sigmaSqAirCoefs(peakDepth);
            // spot sigma at the isocenter plane (already converted to mm above)
            const float spotSigmaXmm = sigmaX_mm;
            iddParams.entrySigmaSq = sigmaSqCoefs.x * entryZ * entryZ + sigmaSqCoefs.y * entryZ + spotSigmaXmm * spotSigmaXmm;

            // Spot distance in rays (used to cap effective sigma)
            if (fabsf(fanDelta_mm.x) > 1e-6f) {
                // [9.48] RC2 fix companion site: layerHaloPlan->spotDelta is already in mm.
                // Priority mirrors site-1 above: explicit CarbonPBS spacing is
                // physical PB spacing for spotDist; halo plan delta is only the
                // BEV placement grid when explicit layerSpotDeltas exist.
                float spotSpacingMm;
                if (hasExplicitLayerSpotDeltasForLayer && physicalSpotDeltaForLayer.x > 0.0f) {
                    spotSpacingMm = physicalSpotDeltaForLayer.x;
                } else if (runtimeNuclearEnabled && layerHaloPlan != nullptr) {
                    spotSpacingMm = layerHaloPlan->spotDelta.x;  // mm, from inference or fallback
                } else {
                    spotSpacingMm = (hasRawSpotLattice ? rawSpotLattice.spotDelta.x : beam.spotDelta.x) * lenToMm;
                }
                iddParams.spotDist = fabsf(spotSpacingMm / fanDelta_mm.x);
                // [9.55] Diagnostic warning (NOT a hard fail): dimensionless
                // spotDist >= 1.5 is the upstream HPB regime. If WEQ-header
                // fallback is active (canonical PB inference failed), spotDist
                // collapses to ~1 because the lattice equals the CPB grid.
                // Halo lateral profile then degenerates to primary-width
                // Gaussians (RC3/RC4 not addressed); RC2 BEV fix still applies,
                // so integrated halo dose returns to ~correct.
                if (runtimeNuclearEnabled && !(iddParams.spotDist >= 1.5f)) {
                    static thread_local int s_warnCount = 0;
                    if (s_warnCount++ < 3) {
                        std::cerr << "  [9.55] WARN: halo spotDist=" << iddParams.spotDist
                                  << " is < 1.5 (typical HPB regime is 2-16). Lateral profile "
                                  << "will not be upstream-equivalent. RC2 BEV fix still active."
                                  << std::endl;
                    }
                }
                if (runtimeNuclearEnabled && layerHaloPlan != nullptr) {
                    if (!(expectedHaloSpotDistInRays > 0.0f) ||
                        !approxEq(iddParams.spotDist, expectedHaloSpotDistInRays, 1.0e-3f)) {
                        static thread_local int s_distWarn2 = 0;
                        if (s_distWarn2++ < 3) {
                            if (fineTiming) {
                                std::cerr << "  [HALO_AUDIT] WARN: site-2 spotDist=" << iddParams.spotDist
                                          << " differs from expected=" << expectedHaloSpotDistInRays
                                          << " after fan-geometry normalization"
                                          << std::endl;
                            }
                        }
                    }
                }
            }

            // Apply per-energy step range for fillIddAndSigmaKernel, then recompute step-dependent coefficients
            iddParams.first = beamFirstInsideRT;
            iddParams.afterLast = afterLastStep;
            iddParams.nucMemStep = runtimeNuclearEnabled ? static_cast<unsigned int>(layerHaloPlan->nucPlaneN) : 0u;
            iddParams.initStepAndAirDiv();
            layerPerf.activeFirst = static_cast<int>(iddParams.first);
            layerPerf.activeLast = std::max(layerPerf.activeFirst, static_cast<int>(iddParams.afterLast) - 1);
            layerPerf.activeCount = std::max(0, static_cast<int>(iddParams.afterLast) - layerPerf.activeFirst);

            if (fineTiming) {
                const int activeFirst = iddParams.first;
                const int activeLast = std::max(activeFirst, static_cast<int>(iddParams.afterLast) - 1);
                const int activeMid = activeFirst + (activeLast - activeFirst) / 2;
                const vec2f vwFirst = iddParams.voxelWidth(activeFirst);
                const vec2f vwMid = iddParams.voxelWidth(activeMid);
                const vec2f vwLast = iddParams.voxelWidth(activeLast);
                const float meanWidthFirst = 0.5f * (vwFirst.x + vwFirst.y);
                const float meanWidthMid = 0.5f * (vwMid.x + vwMid.y);
                const float meanWidthLast = 0.5f * (vwLast.x + vwLast.y);
                auto sigmaAtRadiusLimit = [](float meanWidth) {
                    return (float(MAX_SUPERP_RADIUS) + 0.5f) * meanWidth / KS_SIGMA_CUTOFF - 0.21f;
                };
                std::cout << "  [SIGMA_GRID] activeSteps=[" << activeFirst << "," << activeLast << "]"
                          << " voxelWidthMean(mm): first=" << meanWidthFirst
                          << " mid=" << meanWidthMid
                          << " last=" << meanWidthLast
                          << " sigmaAtRad32(mm): first=" << sigmaAtRadiusLimit(meanWidthFirst)
                          << " mid=" << sigmaAtRadiusLimit(meanWidthMid)
                          << " last=" << sigmaAtRadiusLimit(meanWidthLast)
                          << std::endl;
            }
        if (rtdSigmaDebugEnabled() && (layerIdx == 0 || layerIdx == (numLayers / 2) || layerIdx + 1 == numLayers)) {
            const int activeFirst = iddParams.first;
            const int activeLast = std::max(activeFirst, static_cast<int>(iddParams.afterLast) - 1);
            const int activeMid = activeFirst + (activeLast - activeFirst) / 2;
            const vec2f vwFirst = iddParams.voxelWidth(activeFirst);
                const vec2f vwMid = iddParams.voxelWidth(activeMid);
                const vec2f vwLast = iddParams.voxelWidth(activeLast);
                const float meanWidthFirst = 0.5f * (vwFirst.x + vwFirst.y);
                const float meanWidthMid = 0.5f * (vwMid.x + vwMid.y);
                const float meanWidthLast = 0.5f * (vwLast.x + vwLast.y);
                auto sigmaAtRadiusLimit = [](float meanWidth) {
                    return (float(MAX_SUPERP_RADIUS) + 0.5f) * meanWidth / KS_SIGMA_CUTOFF - 0.21f;
                };
                std::cout << "  [RTD_DEBUG_VOXEL] layer=" << layerIdx
                          << " activeSteps=[" << activeFirst << "," << activeLast << "]"
                          << " voxelWidthMean_mm=(" << meanWidthFirst << "," << meanWidthMid << "," << meanWidthLast << ")"
                          << " sigmaAtRad32_mm=(" << sigmaAtRadiusLimit(meanWidthFirst)
                          << "," << sigmaAtRadiusLimit(meanWidthMid)
                          << "," << sigmaAtRadiusLimit(meanWidthLast) << ")"
                          << " peakDepth=" << peakDepth
                          << std::endl;
            }

            // ============================================================================
            // Step 3b: IDD and Sigma Calculation using proper RayTracedicom algorithm
            // ============================================================================
            
            // Launch fillIddAndSigmaKernel - uses energyIdx to query IDD lookup table
            const auto iddSigmaPerfStart = perfNow();
#ifdef NUCLEAR_CORR
            fillIddAndSigmaKernel<<<tracerGrid, tracerBlock>>>(
                devBevDensity, devBevCumulSp, devRayIdd, devRayRSigmaEff,
                devNucIdd, devNucRSigmaEff,
                runtimeNuclearEnabled ? devNucRayWeights : nullptr,
                runtimeNuclearEnabled ? devNucSpotIdx : nullptr,
                devRayWeights, devBeamFirstInside, devFirstStepOutside, devFirstPassive,
                iddParams,
                rayDims.x, rayDims.y, tracerSteps,
                cumulIddTex, rRadiationLengthTex,
                runtimeNuclearEnabled ? nucWeightTex : 0,
                runtimeNuclearEnabled ? nucSqSigmaTex : 0
            );
#else
            fillIddAndSigmaKernel<<<tracerGrid, tracerBlock>>>(
                devBevDensity, devBevCumulSp, devRayIdd, devRayRSigmaEff,
                devRayWeights, devBeamFirstInside, devFirstStepOutside, devFirstPassive,
                iddParams,
                rayDims.x, rayDims.y, tracerSteps,
                cumulIddTex, rRadiationLengthTex
            );
#endif
            checkCudaErrors(cudaDeviceSynchronize());

#ifdef NUCLEAR_CORR
            // [9.49] Per-layer energy-conservation audit print.
            // Necessary-but-not-sufficient gate. Compares pre-deposition source totals:
            //   sum_rayWeight  : sum of primary spot weights for this layer (rasterized)
            //   sum_nucRayWeight: sum of halo paddedSpotWeights (same physical source)
            //   spotDistSq     : iddParams.spotDist^2; appears in nucRes denominator
            // The ratio diagnoses RC3/RC4 lattice/scaling regressions but is INSENSITIVE
            // to RC2 (BEV positioning) since both quantities exist before BEV-to-dose
            // transfer. Use 9.50 occupancy + runtime fixture comparisons for RC2/lateral
            // verdicts. See halo-energy-conservation.md section 6.
            if (runtimeNuclearEnabled && layerHaloPlan != nullptr && (fineTiming || haloAudit)) {
                const size_t spotPlaneN =
                    static_cast<size_t>(rawSpotLattice.spotGridDims.x) *
                    static_cast<size_t>(rawSpotLattice.spotGridDims.y);
                const size_t layerOff = static_cast<size_t>(layerIdx) * spotPlaneN;
                double sumRayWeight = 0.0;
                if (layerOff + spotPlaneN <= rawSpotLattice.spotWeights.size()) {
                    for (size_t i = 0; i < spotPlaneN; ++i) {
                        sumRayWeight += rawSpotLattice.spotWeights[layerOff + i];
                    }
                }
                double sumNucRayWeight = 0.0;
                for (float v : layerHaloPlan->paddedSpotWeights) sumNucRayWeight += v;
                const double spotDistSq =
                    static_cast<double>(iddParams.spotDist) * static_cast<double>(iddParams.spotDist);
                const double ratioNucToRay =
                    (sumRayWeight > 0.0) ? (sumNucRayWeight / sumRayWeight) : 0.0;
                float nucWeightMin = std::numeric_limits<float>::infinity();
                float nucWeightMax = -std::numeric_limits<float>::infinity();
                if (energyData != nullptr &&
                    energyData->nEnergySamples > 0 &&
                    floorIdxUsed >= 0 &&
                    floorIdxUsed < energyData->nEnergies &&
                    energyData->nucWeightMatrix.size() >=
                        static_cast<size_t>(energyData->nEnergySamples) * static_cast<size_t>(energyData->nEnergies)) {
                    const size_t rowOff =
                        static_cast<size_t>(floorIdxUsed) * static_cast<size_t>(energyData->nEnergySamples);
                    for (int s = 0; s < energyData->nEnergySamples; ++s) {
                        const float w = energyData->nucWeightMatrix[rowOff + static_cast<size_t>(s)];
                        if (!hostIsFinite(w)) continue;
                        nucWeightMin = std::min(nucWeightMin, w);
                        nucWeightMax = std::max(nucWeightMax, w);
                    }
                }
                std::cout << "  [RTD_HALO_ENERGY] layer=" << layerIdx
                          << " sum_rayWeight=" << sumRayWeight
                          << " sum_nucRayWeight=" << sumNucRayWeight
                          << " spotDistSq=" << spotDistSq
                          << " nuc/ray_should_be_1=" << ratioNucToRay
                          << " expectedNucSourceScale="
                          << ((spotDistSq > 0.0) ? (sumNucRayWeight / spotDistSq) : 0.0)
                          << " expectedNucVsRayScale="
                          << ((sumRayWeight > 0.0 && spotDistSq > 0.0)
                                  ? (sumNucRayWeight / (sumRayWeight * spotDistSq))
                                  : 0.0)
                          << " nucWeightRow=" << floorIdxUsed;
                if (hostIsFinite(nucWeightMin) && hostIsFinite(nucWeightMax)) {
                    std::cout << " nucWeightRange=(" << nucWeightMin << "," << nucWeightMax << ")";
                } else {
                    std::cout << " nucWeightRange=(unavailable)";
                }
                std::cout
                          << " placementSource=decoded_idbeamxy_rasterized"
                          << " physicalPBDelta=(" << physicalSpotDeltaForLayer.x << "," << physicalSpotDeltaForLayer.y << ")"
                          << " haloGridDelta=(" << layerHaloPlan->spotDelta.x << "," << layerHaloPlan->spotDelta.y << ")"
                          << std::endl;
            }
#endif

#ifdef NUCLEAR_CORR
            if ((rtdInputAuditEnabled() || haloAudit) && runtimeNuclearEnabled && layerHaloPlan != nullptr &&
                isRepresentativeLayer(static_cast<int>(layerIdx), numLayers)) {
                std::vector<float> hNucIdd(nucRaySize);
                std::vector<float> hNucRSigma(nucRaySize);
                std::vector<float> hRayIddForCompare(raySize);
                copyToHost(hNucIdd.data(), devNucIdd, hNucIdd.size() * sizeof(float));
                copyToHost(hNucRSigma.data(), devNucRSigmaEff, hNucRSigma.size() * sizeof(float));
                copyToHost(hRayIddForCompare.data(), devRayIdd, hRayIddForCompare.size() * sizeof(float));
                const std::string nucIddStage = "NUC_IDD layer=" + std::to_string(layerIdx);
                const std::string nucSigmaStage = "NUC_RSIGMA layer=" + std::to_string(layerIdx);
                const FloatSummaryStats rayIddCompareStats = summarizeFloatVector(hRayIddForCompare, 1.0e-12f);
                const FloatSummaryStats nucIddStats = summarizeFloatVector(hNucIdd, 1.0e-12f);
                int nucPositiveSigmaFinite = 0;
                int nucPositiveSigmaBad = 0;
                const size_t nucCompareN = std::min(hNucIdd.size(), hNucRSigma.size());
                for (size_t i = 0; i < nucCompareN; ++i) {
                    if (!(hNucIdd[i] > 0.0f)) continue;
                    if (hostIsFinite(hNucRSigma[i]) && hNucRSigma[i] > 0.0f) {
                        ++nucPositiveSigmaFinite;
                    } else {
                        ++nucPositiveSigmaBad;
                    }
                }
                std::cout << "  [RTD_HALO_IDD_COMPARE] layer=" << layerIdx
                          << " primaryIddSum=" << rayIddCompareStats.sumFinite
                          << " nucIddSum=" << nucIddStats.sumFinite
                          << " nucToPrimaryIddSumRatio="
                          << ((rayIddCompareStats.sumFinite > 0.0)
                                  ? (nucIddStats.sumFinite / rayIddCompareStats.sumFinite)
                                  : 0.0)
                          << " nucIddNonzero=" << nucIddStats.countGT
                          << " primaryIddNonzero=" << rayIddCompareStats.countGT
                          << " nucRSigmaFiniteWhereIddPositive=" << nucPositiveSigmaFinite
                          << " nucRSigmaBadWhereIddPositive=" << nucPositiveSigmaBad
                          << std::endl;
                printStageVolumeSummary(nucIddStage,
                                        hNucIdd,
                                        layerHaloPlan->nucRayDims.x,
                                        layerHaloPlan->nucRayDims.y,
                                        tracerSteps,
                                        1.0e-12f,
                                        "step");
                printStageVolumeSummary(nucSigmaStage,
                                        hNucRSigma,
                                        layerHaloPlan->nucRayDims.x,
                                        layerHaloPlan->nucRayDims.y,
                                        tracerSteps,
                                        1.0e-12f,
                                        "step");
            }
#endif

            if (rtdInputAuditEnabled()) {
                std::vector<float> hRayIdd(raySize);
                std::vector<float> hRayRSigmaEff(raySize);
                copyToHost(hRayIdd.data(), devRayIdd, hRayIdd.size() * sizeof(float));
                copyToHost(hRayRSigmaEff.data(), devRayRSigmaEff, hRayRSigmaEff.size() * sizeof(float));
                const std::string iddStage = "IDD layer=" + std::to_string(layerIdx);
                const std::string sigmaStage = "RSIGMA layer=" + std::to_string(layerIdx);
                printStageVolumeSummary(iddStage, hRayIdd, rayDims.x, rayDims.y, tracerSteps, 1.0e-12f, "step");
                printStageVolumeSummary(sigmaStage, hRayRSigmaEff, rayDims.x, rayDims.y, tracerSteps, 1.0e-12f, "step");
            }

            if (rtdInputAuditEnabled() && isRepresentativeLayer(layerIdx, numLayers)) {
                std::vector<float> hRayWeights(rayDims.x * rayDims.y);
                std::vector<float> hRayIdd(raySize);
                copyToHost(hRayWeights.data(), devRayWeights, hRayWeights.size() * sizeof(float));
                copyToHost(hRayIdd.data(), devRayIdd, hRayIdd.size() * sizeof(float));
                const FloatSummaryStats rayWStats = summarizeFloatVector(hRayWeights, 1.0e-6f);
                const FloatSummaryStats bevIddStats = summarizeFloatVector(hRayIdd, 1.0e-12f);
                const int iddFirst = std::max(0, static_cast<int>(iddParams.first));
                const int iddAfterLast = std::max(0, static_cast<int>(iddParams.afterLast));
                const std::vector<int> iddSlices = {
                    iddFirst,
                    std::max(0, iddFirst + std::max(0, (iddAfterLast - iddFirst - 1) / 2)),
                    std::max(0, iddAfterLast - 1)
                };
                std::cout << "[INPUT_AUDIT][WRAPPER] layer=" << layerIdx
                          << " rayWeights(sum,max)=(" << rayWStats.sumFinite << "," << rayWStats.maxFinite << ")"
                          << " bevIdd(sum,max)=(" << bevIddStats.sumFinite << "," << bevIddStats.maxFinite << ")"
                          << std::endl;
                printDepthProjectionSummary("IDD_DEPTH_PROJECTION",
                                            hRayIdd,
                                            rayDims.x,
                                            rayDims.y,
                                            tracerSteps,
                                            iddSlices,
                                            "step");
            }

            std::vector<float> sigmaAuditRayIdd;
            std::vector<float> sigmaAuditBefore;
            std::vector<float> sigmaAuditAfter;
            const bool sigmaFieldAudit = (rtdInputAuditEnabled() || rtdSuperpOverflowAuditEnabled()) &&
                                         isRepresentativeLayer(layerIdx, numLayers);
            if (sigmaFieldAudit) {
                sigmaAuditRayIdd.resize(raySize);
                sigmaAuditBefore.resize(raySize);
                copyToHost(sigmaAuditRayIdd.data(), devRayIdd, raySize * sizeof(float));
                copyToHost(sigmaAuditBefore.data(), devRayRSigmaEff, raySize * sizeof(float));
                const SigmaFieldStats s = summarizeSigmaField(
                    sigmaAuditRayIdd, sigmaAuditBefore, iddParams, rayDims.x, rayDims.y, tracerSteps);
                std::cout << "[SIGMA_FIELD][before] layer=" << layerIdx
                          << " positiveIdd=" << s.positiveIddCount
                          << " finiteSigma=" << s.finiteSigmaCount
                          << " overflowLike=" << s.overflowLikeCount
                          << " approxSigma(min,p50,p90,p99,max)=(" << s.minApproxSigma
                          << "," << s.p50ApproxSigma
                          << "," << s.p90ApproxSigma
                          << "," << s.p99ApproxSigma
                          << "," << s.maxApproxSigma << ")"
                          << " minRSigmaEff=" << s.minRSigmaEff
                          << " maxMeanWidth=" << s.maxMeanWidth
                          << std::endl;
            }

            const bool sigmaDebugEnabled = rtdSigmaDebugEnabled();
            const bool debugSigmaLayer =
                sigmaDebugEnabled &&
                (layerIdx == 0 || layerIdx == (numLayers / 2) || layerIdx + 1 == numLayers);
            const bool debugSigmaCompare =
                useCarbonProfileSigmaOverride &&
                profileTex != 0 &&
                hasBeamParaData &&
                profileRows > 0 &&
                profileDepthN > 0 &&
                profileChannels >= 3 &&
                debugSigmaLayer;
            std::vector<CarbonSigmaDebugSample> sigmaDebugBefore;
            std::vector<CarbonSigmaDebugSample> sigmaDebugAfter;

            float profileRowIdx = static_cast<float>(layerIdx);
            if (profileRows == numLayers) {
                profileRowIdx = static_cast<float>(layerIdx);
            } else if (!beam.profileEnergies.empty() &&
                       profileRows == static_cast<int>(beam.profileEnergies.size())) {
                profileRowIdx = findFractionalIndexMonotonic(beam.profileEnergies, energy);
            } else if (energyData != nullptr &&
                       profileRows == energyData->nEnergies &&
                       !energyData->energiesPerU.empty()) {
                profileRowIdx = fminf(fmaxf(energyIdx, 0.0f), float(profileRows - 1));
            } else {
                profileRowIdx = fminf(fmaxf(static_cast<float>(layerIdx), 0.0f), float(std::max(0, profileRows - 1)));
            }
            profileRowIdx = fminf(fmaxf(profileRowIdx, 0.0f), float(std::max(0, profileRows - 1)));

            auto interpBeamParaCoeff = [&](int coeffIdx) {
                if (profileRows <= 0 || beam.beamParaData.size() < static_cast<size_t>(profileRows) * 3ull) {
                    return 0.0f;
                }
                const float clampedIdx = fminf(fmaxf(profileRowIdx, 0.0f), float(profileRows - 1));
                const int idx0 = std::max(0, std::min(profileRows - 1, static_cast<int>(floorf(clampedIdx))));
                const int idx1 = std::max(0, std::min(profileRows - 1, idx0 + 1));
                const float t = clampedIdx - float(idx0);
                const float v0 = beam.beamParaData[static_cast<size_t>(idx0) * 3ull + static_cast<size_t>(coeffIdx)];
                const float v1 = beam.beamParaData[static_cast<size_t>(idx1) * 3ull + static_cast<size_t>(coeffIdx)];
                return v0 + (v1 - v0) * t;
            };
            const float beamPara0Interp = interpBeamParaCoeff(0);
            const float beamPara1Interp = interpBeamParaCoeff(1);
            const float beamPara2Interp = interpBeamParaCoeff(2);

            if (rtdInputAuditEnabled() && isRepresentativeLayer(layerIdx, numLayers)) {
                std::cout << "[INPUT_AUDIT][WRAPPER] layer=" << layerIdx
                          << " energy=" << energy
                          << " energyIdx=" << energyIdx
                          << " profileRowIdx=" << profileRowIdx
                          << " peakDepth=" << peakDepth
                          << " longitudalCutoff=" << layerCutoffMm
                          << " beamPara=(" << beamPara0Interp << "," << beamPara1Interp << "," << beamPara2Interp << ")"
                          << std::endl;
                if (profileRows > 0 && profileDepthN > 0 && profileChannels > 0 && !beam.profileData.empty()) {
                    const int nGauss = std::max(0, (profileChannels - 1) / 2);
                    const int row = std::max(0, std::min(profileRows - 1, static_cast<int>(std::lround(profileRowIdx))));
                    const std::vector<int> depthSel = {0, profileDepthN / 2, profileDepthN - 1};
                    for (int depth : depthSel) {
                        const int d = std::max(0, std::min(profileDepthN - 1, depth));
                        const size_t base = (static_cast<size_t>(row) * static_cast<size_t>(profileDepthN) + static_cast<size_t>(d)) * static_cast<size_t>(profileChannels);
                        std::cout << "  [INPUT_AUDIT][WRAPPER] profile row=" << row << " depth=" << d << " weights=[";
                        for (int g = 0; g < std::min(5, nGauss); ++g) {
                            std::cout << beam.profileData[base + static_cast<size_t>(g)];
                            if (g + 1 != std::min(5, nGauss)) std::cout << ", ";
                        }
                        std::cout << "] sigmas=[";
                        for (int g = 0; g < std::min(5, nGauss); ++g) {
                            std::cout << beam.profileData[base + static_cast<size_t>(nGauss + g)];
                            if (g + 1 != std::min(5, nGauss)) std::cout << ", ";
                        }
                        std::cout << "]";
                        if (profileChannels > 2 * nGauss) {
                            std::cout << " overall=" << beam.profileData[base + static_cast<size_t>(profileChannels - 1)];
                        }
                        std::cout << std::endl;
                    }
                }
            }

            if (debugSigmaLayer) {
                std::cout << "  [RTD_DEBUG_SIGMA_HOOK] layer=" << layerIdx
                          << " profileTex=" << (profileTex != 0 ? 1 : 0)
                          << " hasBeamParaData=" << (hasBeamParaData ? 1 : 0)
                          << " profileRows=" << profileRows
                          << " profileDepthN=" << profileDepthN
                          << " profileChannels=" << profileChannels
                          << " profileRowIdx=" << profileRowIdx
                          << " energy=" << energy
                          << " enabled=" << (debugSigmaCompare ? 1 : 0)
                          << "\n";
            }

            if (useCarbonProfileSigmaOverride &&
                profileTex != 0 &&
                hasBeamParaData &&
                profileDepthN > 0 &&
                profileChannels >= 3) {
                if (debugSigmaCompare) {
                    const int activeFirst = iddParams.first;
                    const int activeLast = std::max(activeFirst, static_cast<int>(iddParams.afterLast) - 1);
                    const int activeMid = activeFirst + (activeLast - activeFirst) / 2;
                    const std::vector<int> sampleXs = {
                        std::max(0, rayDims.x / 8),
                        std::max(0, rayDims.x / 8),
                        std::max(0, rayDims.x / 2),
                        std::max(0, rayDims.x / 8)
                    };
                    const std::vector<int> sampleYs = {
                        std::max(0, rayDims.y / 10),
                        std::max(0, rayDims.y / 2),
                        std::max(0, rayDims.y / 2),
                        std::max(0, (9 * rayDims.y) / 10)
                    };
                    const std::vector<int> sampleSteps = {
                        activeFirst,
                        activeMid,
                        activeMid,
                        activeLast
                    };
                    const int sampleCount = static_cast<int>(sampleXs.size());

                    int* devSampleXs = nullptr;
                    int* devSampleYs = nullptr;
                    int* devSampleSteps = nullptr;
                    CarbonSigmaDebugSample* devSigmaDebug = nullptr;
                    checkCudaErrors(cudaMalloc(&devSampleXs, sampleCount * sizeof(int)));
                    checkCudaErrors(cudaMalloc(&devSampleYs, sampleCount * sizeof(int)));
                    checkCudaErrors(cudaMalloc(&devSampleSteps, sampleCount * sizeof(int)));
                    checkCudaErrors(cudaMalloc(&devSigmaDebug, sampleCount * sizeof(CarbonSigmaDebugSample)));
                    checkCudaErrors(cudaMemcpy(devSampleXs, sampleXs.data(), sampleCount * sizeof(int), cudaMemcpyHostToDevice));
                    checkCudaErrors(cudaMemcpy(devSampleYs, sampleYs.data(), sampleCount * sizeof(int), cudaMemcpyHostToDevice));
                    checkCudaErrors(cudaMemcpy(devSampleSteps, sampleSteps.data(), sampleCount * sizeof(int), cudaMemcpyHostToDevice));

                    const int dbgBlock = 64;
                    const int dbgGrid = (sampleCount + dbgBlock - 1) / dbgBlock;
                    gatherCarbonSigmaDebugKernel<<<dbgGrid, dbgBlock>>>(
                        devBevCumulSp, devRayIdd, devRayRSigmaEff,
                        iddParams,
                        rayDims.x, rayDims.y, tracerSteps,
                        profileTex,
                        beam.profileSetting[0],
                        beam.profileSetting[1],
                        profileDepthN,
                        profileChannels,
                        profileRowIdx,
                        beamPara0Interp,
                        beamPara1Interp,
                        beamPara2Interp,
                        beam.beamParaPos,
                        devSampleXs, devSampleYs, devSampleSteps,
                        sampleCount,
                        devSigmaDebug
                    );
                    checkCudaErrors(cudaDeviceSynchronize());
                    sigmaDebugBefore.resize(sampleCount);
                    copyToHost(sigmaDebugBefore.data(), devSigmaDebug, sampleCount * sizeof(CarbonSigmaDebugSample));

                    overrideRSigmaFromCarbonProfileKernel<<<tracerGrid, tracerBlock>>>(
                        devBevCumulSp, devRayIdd, devRayRSigmaEff,
                        iddParams,
                        rayDims.x, rayDims.y, tracerSteps,
                        profileTex,
                        beam.profileSetting[0],
                        beam.profileSetting[1],
                        profileDepthN,
                        profileChannels,
                        profileRowIdx,
                        beamPara0Interp,
                        beamPara1Interp,
                        beamPara2Interp,
                        beam.beamParaPos
                    );
                    checkCudaErrors(cudaDeviceSynchronize());

                    gatherCarbonSigmaDebugKernel<<<dbgGrid, dbgBlock>>>(
                        devBevCumulSp, devRayIdd, devRayRSigmaEff,
                        iddParams,
                        rayDims.x, rayDims.y, tracerSteps,
                        profileTex,
                        beam.profileSetting[0],
                        beam.profileSetting[1],
                        profileDepthN,
                        profileChannels,
                        profileRowIdx,
                        beamPara0Interp,
                        beamPara1Interp,
                        beamPara2Interp,
                        beam.beamParaPos,
                        devSampleXs, devSampleYs, devSampleSteps,
                        sampleCount,
                        devSigmaDebug
                    );
                    checkCudaErrors(cudaDeviceSynchronize());
                    sigmaDebugAfter.resize(sampleCount);
                    copyToHost(sigmaDebugAfter.data(), devSigmaDebug, sampleCount * sizeof(CarbonSigmaDebugSample));

                    checkCudaErrors(cudaFree(devSigmaDebug));
                    checkCudaErrors(cudaFree(devSampleSteps));
                    checkCudaErrors(cudaFree(devSampleYs));
                    checkCudaErrors(cudaFree(devSampleXs));
                } else {
                    overrideRSigmaFromCarbonProfileKernel<<<tracerGrid, tracerBlock>>>(
                        devBevCumulSp, devRayIdd, devRayRSigmaEff,
                        iddParams,
                        rayDims.x, rayDims.y, tracerSteps,
                        profileTex,
                        beam.profileSetting[0],
                        beam.profileSetting[1],
                        profileDepthN,
                        profileChannels,
                        profileRowIdx,
                        beamPara0Interp,
                        beamPara1Interp,
                        beamPara2Interp,
                        beam.beamParaPos
                    );
                    checkCudaErrors(cudaDeviceSynchronize());
                }
            }

            if (debugSigmaLayer) {
                std::cout << "  [RTD_DEBUG_SIGMA_HOOK_RESULT] layer=" << layerIdx
                          << " beforeCount=" << sigmaDebugBefore.size()
                          << " afterCount=" << sigmaDebugAfter.size()
                          << "\n";
            }

            if (!sigmaDebugBefore.empty() && sigmaDebugBefore.size() == sigmaDebugAfter.size()) {
                std::cout << "  [RTD_DEBUG_SIGMA_COMPARE] layer=" << layerIdx
                          << " samples=" << sigmaDebugBefore.size() << "\n";
                for (size_t si = 0; si < sigmaDebugBefore.size(); ++si) {
                    const auto& before = sigmaDebugBefore[si];
                    const auto& after = sigmaDebugAfter[si];
                    std::cout << "    sample[" << si << "] xy=(" << before.x << "," << before.y << ")"
                              << " step=" << before.step
                              << " idd=" << before.idd
                              << " weq=" << before.weqDepth
                              << " phy=" << after.phyDepth
                              << " profIdx=" << after.profileDepthIdx
                              << " meanWidth=" << before.meanWidth
                              << " rSigma(before,after)=(" << before.rSigmaEff << "," << after.rSigmaEff << ")"
                              << " approxSigma(before,after)=(" << before.approxSigma << "," << after.approxSigma << ")"
                              << " carbonSigmaAxis=" << after.carbonSigmaAxis
                              << " profileSigmaRad=" << after.profileSigmaRad
                              << " initR2=" << after.initR2
                              << " sumW=" << after.sumW
                              << "\n";
                }
            }

            const bool applyCarbonOverallWeight =
                useCarbonProfileOverallWeight &&
                profileTex != 0 &&
                profileDepthN > 0 &&
                profileChannels > 0;
            if (applyCarbonOverallWeight) {
                applyCarbonProfileOverallWeightKernel<<<tracerGrid, tracerBlock>>>(
                    devBevCumulSp,
                    devRayIdd,
                    rayDims.x, rayDims.y, tracerSteps,
                    profileTex,
                    beam.profileSetting[0],
                    beam.profileSetting[1],
                    profileDepthN,
                    profileChannels,
                    profileRowIdx
                );
                checkCudaErrors(cudaDeviceSynchronize());
            }

            if (sigmaFieldAudit) {
                sigmaAuditAfter.resize(raySize);
                copyToHost(sigmaAuditAfter.data(), devRayRSigmaEff, raySize * sizeof(float));
                const SigmaFieldStats s = summarizeSigmaField(
                    sigmaAuditRayIdd, sigmaAuditAfter, iddParams, rayDims.x, rayDims.y, tracerSteps);
                std::cout << "[SIGMA_FIELD][after] layer=" << layerIdx
                          << " positiveIdd=" << s.positiveIddCount
                          << " finiteSigma=" << s.finiteSigmaCount
                          << " overflowLike=" << s.overflowLikeCount
                          << " approxSigma(min,p50,p90,p99,max)=(" << s.minApproxSigma
                          << "," << s.p50ApproxSigma
                          << "," << s.p90ApproxSigma
                          << "," << s.p99ApproxSigma
                          << "," << s.maxApproxSigma << ")"
                          << " minRSigmaEff=" << s.minRSigmaEff
                          << " maxMeanWidth=" << s.maxMeanWidth
                          << std::endl;
            }
            layerPerf.iddSigmaMs += perfElapsedMs(iddSigmaPerfStart);
            beamTiming.iddSigmaMs += layerPerf.iddSigmaMs;
            
            if (fineTiming) {
                std::vector<float> rayIddData(raySize);
                std::vector<float> raySigmaData(raySize);
                copyToHost(rayIddData.data(), devRayIdd, raySize * sizeof(float));
                copyToHost(raySigmaData.data(), devRayRSigmaEff, raySize * sizeof(float));

                double totalRayIddFinite = 0.0;
                float maxRayIddFinite = 0.0f;
                float minPosRayIdd = std::numeric_limits<float>::infinity();
                int iddNaN = 0;
                int iddInf = 0;
                int iddPos = 0;
                int iddGT1e15 = 0;
                int iddGT1e6 = 0;

                float maxSigmaFinite = 0.0f;
                float minSigmaFinite = std::numeric_limits<float>::infinity();
                int sigmaNaN = 0;
                int sigmaInf = 0;
                int sigmaFinite = 0;
                int sigmaGT1e6 = 0;
                int sigmaBadOnIddPos = 0;
                int sigmaFiniteOnIddPos = 0;
                float maxSigmaOnIddPos = 0.0f;
                float minSigmaOnIddPos = std::numeric_limits<float>::infinity();

                for (int i = 0; i < raySize; i++) {
                    const float idd = rayIddData[i];
                    const float sig = raySigmaData[i];

                    if (hostIsNaN(idd)) {
                        iddNaN++;
                    } else if (!hostIsFinite(idd)) {
                        iddInf++;
                    } else {
                        totalRayIddFinite += static_cast<double>(idd);
                        if (idd > maxRayIddFinite) maxRayIddFinite = idd;
                        if (idd > 0.0f) {
                            iddPos++;
                            if (idd < minPosRayIdd) minPosRayIdd = idd;
                            if (idd > 1e-15f) iddGT1e15++;
                            if (idd > 1e-6f) iddGT1e6++;
                        }
                    }

                    if (hostIsNaN(sig)) {
                        sigmaNaN++;
                        if (idd > 0.0f) sigmaBadOnIddPos++;
                    } else if (!hostIsFinite(sig)) {
                        sigmaInf++;
                        if (idd > 0.0f) sigmaBadOnIddPos++;
                    } else {
                        sigmaFinite++;
                        if (sig > maxSigmaFinite) maxSigmaFinite = sig;
                        if (sig < minSigmaFinite) minSigmaFinite = sig;
                        if (sig > 1e-6f) sigmaGT1e6++;

                        if (idd > 0.0f) {
                            sigmaFiniteOnIddPos++;
                            if (sig > maxSigmaOnIddPos) maxSigmaOnIddPos = sig;
                            if (sig < minSigmaOnIddPos) minSigmaOnIddPos = sig;
                        }
                    }
                }

                if (minPosRayIdd == std::numeric_limits<float>::infinity()) minPosRayIdd = 0.0f;
                if (minSigmaFinite == std::numeric_limits<float>::infinity()) minSigmaFinite = 0.0f;
                if (minSigmaOnIddPos == std::numeric_limits<float>::infinity()) minSigmaOnIddPos = 0.0f;

                std::cout << "  Ray Tracing Results Analysis:" << std::endl;
                std::cout << "    Ray IDD  - sumFinite=" << totalRayIddFinite
                          << "  max=" << maxRayIddFinite
                          << "  minPos=" << minPosRayIdd
                          << "  nnz(>0)=" << iddPos << "/" << raySize
                          << "  nnz(>1e-15)=" << iddGT1e15 << "/" << raySize
                          << "  nnz(>1e-6)=" << iddGT1e6 << "/" << raySize
                          << "  nan=" << iddNaN
                          << "  inf=" << iddInf
                          << std::endl;
                std::cout << "    Ray Sigma- finiteCount=" << sigmaFinite << "/" << raySize
                          << "  minFinite=" << minSigmaFinite
                          << "  maxFinite=" << maxSigmaFinite
                          << "  nnz(>1e-6)=" << sigmaGT1e6 << "/" << raySize
                          << "  nan=" << sigmaNaN
                          << "  inf=" << sigmaInf
                          << std::endl;
                std::cout << "    Sigma on active IDD: finite=" << sigmaFiniteOnIddPos
                          << "  bad(nan/inf)=" << sigmaBadOnIddPos
                          << "  min=" << minSigmaOnIddPos
                          << "  max=" << maxSigmaOnIddPos
                          << std::endl;
            }
            
            // ============================================================================
            // Step 3c: Calculate beamFirstInside and beamFirstCalculatedPassive
            // ============================================================================
            
            // beamFirstInside/beamFirstOutside are BEV-geometry properties already
            // reduced once at beam scope; only firstPassive is layer-dependent.
            int beamFirstInside = beamFirstInsideRT;
            int beamFirstOutside = beamFirstOutsideRT;
            
            // Calculate beamFirstCalculatedPassive (maximum passive step)
            sliceMaxVar<int, 1024><<<1, 1024, 1024*sizeof(int)>>>(
                devFirstPassive, devBeamFirstPassiveMaxScratch, rayDims.x * rayDims.y);
            checkCudaErrors(cudaDeviceSynchronize());
            
            int beamFirstPassive;
            checkCudaErrors(cudaMemcpy(&beamFirstPassive, devBeamFirstPassiveMaxScratch, sizeof(int), cudaMemcpyDeviceToHost));

            // Reference RTD semantics: layerFirstPassive is taken from the maximum
            // firstPassive over rays after fillIddAndSigma. Do not shrink it again
            // with firstOutside here; afterLastStep already encodes the layer cut-off.
            int beamFirstCalculatedPassive = beamFirstPassive;
            if (beamFirstCalculatedPassive < beamFirstInside + 1) {
                beamFirstCalculatedPassive = beamFirstInside + 1;
            }
            if (beamFirstCalculatedPassive < 0) {
                beamFirstCalculatedPassive = beamFirstInside + 1;
            }
            if (beamFirstCalculatedPassive > afterLastStep) {
                beamFirstCalculatedPassive = afterLastStep;
            }
            if (beamFirstCalculatedPassive > tracerSteps) {
                beamFirstCalculatedPassive = tracerSteps;
            }
            
            if (fineTiming) {
                std::cout << "  Beam Entry/Exit Analysis:" << std::endl;
                std::cout << "    beamFirstInside: " << beamFirstInside << std::endl;
                std::cout << "    beamFirstOutside: " << beamFirstOutside << std::endl;
                std::cout << "    beamFirstPassive: " << beamFirstPassive << std::endl;
                std::cout << "    beamFirstCalculatedPassive: " << beamFirstCalculatedPassive << std::endl;
            }

            // Extra geometry diagnostics: how many rays actually enter the patient/phantom?
            if (fineTiming) {
                const int nRays = rayDims.x * rayDims.y;
                std::vector<int> hFirstInside(nRays);
                std::vector<int> hFirstOutside(nRays);
                std::vector<int> hFirstPassive(nRays);
                copyToHost(hFirstInside.data(), devBeamFirstInside, nRays * sizeof(int));
                copyToHost(hFirstOutside.data(), devFirstStepOutside, nRays * sizeof(int));
                copyToHost(hFirstPassive.data(), devFirstPassive, nRays * sizeof(int));

                int insideRays = 0;
                int noInsideRays = 0;
                int maxFirstInside = -1;
                for (int v : hFirstInside) {
                    if (v >= 0 && v < tracerSteps) {
                        insideRays++;
                        if (v > maxFirstInside) maxFirstInside = v;
                    } else {
                        noInsideRays++;
                    }
                }

                int outsideFound = 0;
                for (int v : hFirstOutside) if (v > 0 && v <= tracerSteps) outsideFound++;
                int passiveFound = 0;
                for (int v : hFirstPassive) if (v >= 0 && v <= tracerSteps) passiveFound++;

                std::cout << "    [GEOM] Rays entering medium: " << insideRays << "/" << nRays
                          << " (" << (100.0 * double(insideRays) / double(nRays)) << "%)"
                          << ", never-enter: " << noInsideRays << "/" << nRays
                          << ", maxFirstInside=" << maxFirstInside << std::endl;
                std::cout << "    [GEOM] Rays with firstOutside found: " << outsideFound << "/" << nRays
                          << ", rays with firstPassive found: " << passiveFound << "/" << nRays << std::endl;
            }
            
            // ============================================================================
                        // Step 4: Complete Tile-Based Superposition (RayTracedicom algorithm)

            // RayTraceDicom superposition assumes rayDimsX/Y are exact multiples of SUPERP_TILE_X/Y.
            // If not, we pad the per-ray (IDD, rSigmaEff) arrays to the next tile multiple.
            const int superpRayDimsX = ((rayDims.x + SUPERP_TILE_X - 1) / SUPERP_TILE_X) * SUPERP_TILE_X;
            const int superpRayDimsY = ((rayDims.y + SUPERP_TILE_Y - 1) / SUPERP_TILE_Y) * SUPERP_TILE_Y;
            layerPerf.superpRayDimsX = superpRayDimsX;
            layerPerf.superpRayDimsY = superpRayDimsY;

            float* devRayIddForSuperp = devRayIdd;
            float* devRayRSigmaEffForSuperp = devRayRSigmaEff;

            if (superpRayDimsX != rayDims.x || superpRayDimsY != rayDims.y) {
                std::cout << "  Padding ray IDD/rSigmaEff for superposition:" << std::endl;
                std::cout << "    Original ray dims: (" << rayDims.x << ", " << rayDims.y << ")" << std::endl;
                std::cout << "    Padded   ray dims: (" << superpRayDimsX << ", " << superpRayDimsY << ")" << std::endl;

                dim3 padBlock(16, 16, 1);
                dim3 padGrid((superpRayDimsX + padBlock.x - 1) / padBlock.x,
                             (superpRayDimsY + padBlock.y - 1) / padBlock.y,
                             tracerSteps);

                padRayIddSigmaKernel<<<padGrid, padBlock>>>(
                    devRayIdd, devRayRSigmaEff,
                    devRayIddPaddedScratch, devRayRSigmaEffPaddedScratch,
                    rayDims.x, rayDims.y,
                    superpRayDimsX, superpRayDimsY,
                    tracerSteps);
                checkCudaErrors(cudaDeviceSynchronize());

                devRayIddForSuperp = devRayIddPaddedScratch;
                devRayRSigmaEffForSuperp = devRayRSigmaEffPaddedScratch;
            }

            // Allocate BEV dose array with proper padding (based on padded ray dims)
            const int bevDoseX = superpRayDimsX + 2 * maxSuperpR;
            const int bevDoseY = superpRayDimsY + 2 * maxSuperpR;
            const int bevDoseZ = beamFirstGuaranteedPassiveRT;

            if (fineTiming) {
                std::cout << "  Allocating BEV dose array with padding:" << std::endl;
                std::cout << "    BEV dimensions: (" << bevDoseX << ", " << bevDoseY << ", " << bevDoseZ << ")" << std::endl;
                std::cout << "    Padding: " << maxSuperpR << " voxels on each side" << std::endl;
            }

            size_t bevDoseSize = (size_t)bevDoseX * bevDoseY * bevDoseZ * sizeof(float);
            float* devBevPrimDose = devBevPrimDoseScratch;
            checkCudaErrors(cudaMemset(devBevPrimDose, 0, bevDoseSize));

#ifdef NUCLEAR_CORR
            const int bevNucDoseX = runtimeNuclearEnabled ? (layerHaloPlan->nucRayDims.x + 2 * maxSuperpR) : 0;
            const int bevNucDoseY = runtimeNuclearEnabled ? (layerHaloPlan->nucRayDims.y + 2 * maxSuperpR) : 0;
            const int bevNucDoseZ = runtimeNuclearEnabled ? bevDoseZ : 0;
            size_t bevNucDoseSize = 0;
            float* devBevNucDose = nullptr;
            if (runtimeNuclearEnabled) {
                bevNucDoseSize =
                    static_cast<size_t>(bevNucDoseX) * static_cast<size_t>(bevNucDoseY) * static_cast<size_t>(bevNucDoseZ) * sizeof(float);
                devBevNucDose = devBevNucDoseScratch;
                checkCudaErrors(cudaMemset(devBevNucDose, 0, bevNucDoseSize));
            }
#endif

            const bool auditRepresentativeLayer =
                (rtdInputAuditEnabled() || haloAudit) && isRepresentativeLayer(layerIdx, numLayers);
            std::vector<float> hSuperpInputIdd;
            std::vector<float> hSuperpInputRSigma;
            std::vector<int> superpSlices;
            const float superpShapeThr = 1.0e-12f;
            if (auditRepresentativeLayer) {
                const size_t paddedElems = static_cast<size_t>(superpRayDimsX) *
                                           static_cast<size_t>(superpRayDimsY) *
                                           static_cast<size_t>(tracerSteps);
                hSuperpInputIdd.resize(paddedElems);
                hSuperpInputRSigma.resize(paddedElems);
                copyToHost(hSuperpInputIdd.data(), devRayIddForSuperp, paddedElems * sizeof(float));
                copyToHost(hSuperpInputRSigma.data(), devRayRSigmaEffForSuperp, paddedElems * sizeof(float));

                superpSlices.push_back(beamFirstInside);
                superpSlices.push_back(beamFirstCalculatedPassive - 1);
                superpSlices.push_back(beamFirstInside + std::max(0, (beamFirstCalculatedPassive - beamFirstInside - 1) / 2));
                std::sort(superpSlices.begin(), superpSlices.end());
                superpSlices.erase(std::unique(superpSlices.begin(), superpSlices.end()), superpSlices.end());

                printStageVolumeSummary("SUPERP_INPUT_IDD",
                                        hSuperpInputIdd,
                                        superpRayDimsX,
                                        superpRayDimsY,
                                        tracerSteps,
                                        1.0e-12f,
                                        "step");
                printVolumePeakLocationSummary("SUPERP_INPUT_IDD",
                                               hSuperpInputIdd,
                                               superpRayDimsX,
                                               superpRayDimsY,
                                               tracerSteps,
                                               "step",
                                               0,
                                               0);
                printSelectedPlaneShapeSummary("SUPERP_INPUT_IDD",
                                               hSuperpInputIdd,
                                               superpRayDimsX,
                                               superpRayDimsY,
                                               tracerSteps,
                                               superpSlices,
                                               superpShapeThr,
                                               0,
                                               0,
                                               "step");

                printStageVolumeSummary("SUPERP_INPUT_RSIGMA",
                                        hSuperpInputRSigma,
                                        superpRayDimsX,
                                        superpRayDimsY,
                                        tracerSteps,
                                        1.0e-12f,
                                        "step");
                printVolumePeakLocationSummary("SUPERP_INPUT_RSIGMA",
                                               hSuperpInputRSigma,
                                               superpRayDimsX,
                                               superpRayDimsY,
                                               tracerSteps,
                                               "step",
                                               0,
                                               0);
                printSelectedPlaneShapeSummary("SUPERP_INPUT_RSIGMA",
                                               hSuperpInputRSigma,
                                               superpRayDimsX,
                                               superpRayDimsY,
                                               tracerSteps,
                                               superpSlices,
                                               superpShapeThr,
                                               0,
                                               0,
                                               "step");
            }

            // Tile-based superposition: separate primary / nuclear functions, mirroring
            // RayTraceDicom-main kernel_wrapper.cu (lines 985-1108).
            //
            // Plan B layout: each branch owns its own tile counters and inOut idx scratch
            // and frees them before returning, so the two scratch sets are never live at
            // the same time on the default stream. Concurrent-stream execution is enabled
            // by passing distinct non-blocking streams to each call below; the trailing
            // cudaStreamSynchronize on each stream inside the helper guarantees BEV dose
            // is fully written before the wrapper proceeds to nucTransfDiv / primTransfDiv.
            SuperpositionDebugContext primarySuperpDebug;
            primarySuperpDebug.layerIdx = static_cast<int>(layerIdx);
            primarySuperpDebug.numLayers = numLayers;
            primarySuperpDebug.energy = energy;
            primarySuperpDebug.energyIdx = energyIdx;
            primarySuperpDebug.peakDepthMm = peakDepth;
            primarySuperpDebug.layerCutoffMm = layerCutoffMm;
            primarySuperpDebug.originalRayDimsX = rayDims.x;
            primarySuperpDebug.originalRayDimsY = rayDims.y;
            primarySuperpDebug.superpRayDimsX = superpRayDimsX;
            primarySuperpDebug.superpRayDimsY = superpRayDimsY;
            primarySuperpDebug.spotSigmaXmm = sigmaX_mm;
            primarySuperpDebug.spotSigmaYmm = sigmaY_mm;
            primarySuperpDebug.profileRowIdx = profileRowIdx;
            primarySuperpDebug.devBevCumulSp = devBevCumulSp;
            primarySuperpDebug.devBevDensity = devBevDensity;
            primarySuperpDebug.rangeStopDepthMm = iddParams.rangeStopDepth;
            primarySuperpDebug.stepVolConst = iddParams.volConst;
            primarySuperpDebug.stepVolLin = iddParams.volLin;
            primarySuperpDebug.stepVolSq = iddParams.volSq;
            {
                const int activeFirst = std::max(0, beamFirstInside);
                const int activeLast = std::max(activeFirst, beamFirstCalculatedPassive - 1);
                const int activeMid = activeFirst + (activeLast - activeFirst) / 2;
                const vec2f vwFirst = iddParams.voxelWidth(activeFirst);
                const vec2f vwMid = iddParams.voxelWidth(activeMid);
                const vec2f vwLast = iddParams.voxelWidth(activeLast);
                primarySuperpDebug.meanVoxelWidthFirstMm = 0.5f * (vwFirst.x + vwFirst.y);
                primarySuperpDebug.meanVoxelWidthMidMm = 0.5f * (vwMid.x + vwMid.y);
                primarySuperpDebug.meanVoxelWidthLastMm = 0.5f * (vwLast.x + vwLast.y);
                auto sigmaAtRadiusLimit = [](float meanWidth) {
                    return (float(MAX_SUPERP_RADIUS) + 0.5f) * meanWidth / KS_SIGMA_CUTOFF - 0.21f;
                };
                primarySuperpDebug.sigmaAtRad32FirstMm =
                    sigmaAtRadiusLimit(primarySuperpDebug.meanVoxelWidthFirstMm);
                primarySuperpDebug.sigmaAtRad32MidMm =
                    sigmaAtRadiusLimit(primarySuperpDebug.meanVoxelWidthMidMm);
                primarySuperpDebug.sigmaAtRad32LastMm =
                    sigmaAtRadiusLimit(primarySuperpDebug.meanVoxelWidthLastMm);
            }
            const auto superpositionPerfStart = perfNow();
            performPrimaryTileBasedSuperposition(devRayIddForSuperp, devRayRSigmaEffForSuperp, devBevPrimDose,
                                                 superpRayDimsX, superpRayDimsY, tracerSteps,
                                                 beamFirstInside, beamFirstCalculatedPassive,
                                                 0, &primarySuperpDebug);

#ifdef NUCLEAR_CORR
            if (runtimeNuclearEnabled) {
                performNuclearTileBasedSuperposition(devNucIdd, devNucRSigmaEff, devBevNucDose,
                                                     layerHaloPlan->nucRayDims.x, layerHaloPlan->nucRayDims.y, tracerSteps,
                                                     beamFirstInside, beamFirstCalculatedPassive);
            }
#endif
            layerPerf.superpositionMs += perfElapsedMs(superpositionPerfStart);
            beamTiming.superpositionMs += layerPerf.superpositionMs;

            if (auditRepresentativeLayer) {
                const size_t bevDoseElems = static_cast<size_t>(bevDoseX) * static_cast<size_t>(bevDoseY) * static_cast<size_t>(bevDoseZ);
                std::vector<float> hBevPrimDose(bevDoseElems);
                copyToHost(hBevPrimDose.data(), devBevPrimDose, bevDoseElems * sizeof(float));
                printStageVolumeSummary("SUPERP_OUTPUT_BEV",
                                        hBevPrimDose,
                                        bevDoseX,
                                        bevDoseY,
                                        bevDoseZ,
                                        1.0e-12f,
                                        "step");
                printVolumePeakLocationSummary("SUPERP_OUTPUT_BEV",
                                               hBevPrimDose,
                                               bevDoseX,
                                               bevDoseY,
                                               bevDoseZ,
                                               "step",
                                               -maxSuperpR,
                                               -maxSuperpR);
                printSelectedPlaneShapeSummary("SUPERP_OUTPUT_BEV",
                                               hBevPrimDose,
                                               bevDoseX,
                                               bevDoseY,
                                               bevDoseZ,
                                               superpSlices,
                                               superpShapeThr,
                                               -maxSuperpR,
                                               -maxSuperpR,
                                               "step");
                printDepthProjectionSummary("SUPERP_OUTPUT_BEV_DEPTH",
                                            hBevPrimDose,
                                            bevDoseX,
                                            bevDoseY,
                                            bevDoseZ,
                                            superpSlices,
                                            "step");
#ifdef NUCLEAR_CORR
                if (runtimeNuclearEnabled && layerHaloPlan != nullptr) {
                    const size_t bevNucDoseElems = static_cast<size_t>(bevNucDoseX) *
                                                   static_cast<size_t>(bevNucDoseY) *
                                                   static_cast<size_t>(bevNucDoseZ);
                    std::vector<float> hBevNucDose(bevNucDoseElems);
                    copyToHost(hBevNucDose.data(), devBevNucDose, bevNucDoseElems * sizeof(float));
                    printStageVolumeSummary("SUPERP_OUTPUT_NUC_BEV",
                                            hBevNucDose,
                                            bevNucDoseX,
                                            bevNucDoseY,
                                            bevNucDoseZ,
                                            1.0e-12f,
                                            "step");
                    printVolumePeakLocationSummary("SUPERP_OUTPUT_NUC_BEV",
                                                   hBevNucDose,
                                                   bevNucDoseX,
                                                   bevNucDoseY,
                                                   bevNucDoseZ,
                                                   "step",
                                                   -maxSuperpR,
                                                   -maxSuperpR);
                }
#endif

                const VolumePeakStats inputPeak = summarizeVolumePeak(hSuperpInputIdd, superpRayDimsX, superpRayDimsY, tracerSteps);
                const VolumePeakStats outputPeak = summarizeVolumePeak(hBevPrimDose, bevDoseX, bevDoseY, bevDoseZ);
                if (inputPeak.valid && outputPeak.valid) {
                    std::cout << "[INPUT_AUDIT][WRAPPER][SUPERP_PEAK_SHIFT]"
                              << " delta=(" << (outputPeak.x - maxSuperpR - inputPeak.x)
                              << "," << (outputPeak.y - maxSuperpR - inputPeak.y)
                              << "," << (outputPeak.z - inputPeak.z) << ")"
                              << " inputValue=" << inputPeak.value
                              << " outputValue=" << outputPeak.value
                              << std::endl;
                }

                for (int z : superpSlices) {
                    if (z < 0 || z >= tracerSteps || z >= bevDoseZ) continue;
                    const PlaneShapeStats before = summarizeXYPlane(hSuperpInputIdd,
                                                                    superpRayDimsX,
                                                                    superpRayDimsY,
                                                                    tracerSteps,
                                                                    z,
                                                                    superpShapeThr);
                    const PlaneShapeStats after = summarizeXYPlane(hBevPrimDose,
                                                                   bevDoseX,
                                                                   bevDoseY,
                                                                   bevDoseZ,
                                                                   z,
                                                                   superpShapeThr);
                    printSuperpositionSliceDeltaSummary("SUPERP_SLICE_DELTA",
                                                        before,
                                                        after,
                                                        0,
                                                        0,
                                                        -maxSuperpR,
                                                        -maxSuperpR,
                                                        "step");
                }
            }

            if (fineTiming) {
                std::vector<float> hostBevDose(bevDoseX * bevDoseY * bevDoseZ);
                checkCudaErrors(cudaMemcpy(hostBevDose.data(), devBevPrimDose, bevDoseSize, cudaMemcpyDeviceToHost));
                const FloatSummaryStats bevStats = summarizeFloatVector(hostBevDose, 1e-6f);
                std::cout << "  BEV Dose Analysis after Superposition:" << std::endl;
                std::cout << "    sumFinite=" << bevStats.sumFinite
                          << "  maxFinite=" << bevStats.maxFinite
                          << "  minPos=" << bevStats.minPositive
                          << "  nnz(>0)=" << bevStats.countPositive << "/" << hostBevDose.size()
                          << "  nnz(>1e-6)=" << bevStats.countGT << "/" << hostBevDose.size()
                          << "  nan=" << bevStats.countNaN
                          << "  inf=" << bevStats.countInf
                          << std::endl;
                printZSliceSummary("BEV dose", hostBevDose, bevDoseX, bevDoseY, bevDoseZ, 1e-15f);
            }

            // Step 5: Dose Transformation and Accumulation (3D BEV to Dose Grid)
            // ============================================================================
            
            // Create BEV dose 3D texture
            const auto layerTexturePerfStart = perfNow();
            cudaTextureObject_t bevPrimDoseTex = create3DTexture(
                devBevPrimDose,
                make_int3(bevDoseX, bevDoseY, bevDoseZ),
                cudaFilterModeLinear,
                cudaAddressModeBorder
            );
            if (bevPrimDoseTex == 0) {
                throw std::runtime_error("[RTD] Fatal: BEV primary-dose texture creation failed for layer " +
                                         std::to_string(layerIdx) +
                                         "; transfer cannot fall back to linear-memory sampling.");
            }

#ifdef NUCLEAR_CORR
            cudaTextureObject_t bevNucDoseTex = 0;
            if (runtimeNuclearEnabled) {
                bevNucDoseTex = create3DTexture(
                    devBevNucDose,
                    make_int3(bevNucDoseX, bevNucDoseY, bevNucDoseZ),
                    cudaFilterModeLinear,
                    cudaAddressModeBorder
                );
                if (bevNucDoseTex == 0) {
                    destroyTextureObjectAndArray(bevPrimDoseTex);
                    throw std::runtime_error("[RTD] Fatal: BEV nuclear-dose texture creation failed for layer " +
                                             std::to_string(layerIdx) +
                                             "; transfer cannot fall back to linear-memory sampling.");
                }
            }
#endif
            layerPerf.textureMs += perfElapsedMs(layerTexturePerfStart);
            beamTiming.layerTextureMs += layerPerf.textureMs;
            
            // --------------------------------------------------------------------
            // BEV -> global dose grid transform (reference RayTraceDicom logic)
            // --------------------------------------------------------------------
            // Build rayIdx -> doseIdx fan transform and then invert it to get
            // doseIdx -> rayIdx (fan index) used by primTransfDiv.
            const Float3FromFanTransform primRayIdxToDoseIdx =
                Float3FromFanTransform(fanIdxToFan, sourceDistVec, gantryToDoseIdx);

            vec3f primMaxPoint = make_vec3f(-1.0f, -1.0f, -1.0f);
            vec3f primMinPoint = make_vec3f(100000.0f, 100000.0f, 100000.0f);
            const float xVals[2] = {
                -float(maxSuperpR),
                float(superpRayDimsX + maxSuperpR - 1)
            };
            const float yVals[2] = {
                -float(maxSuperpR),
                float(superpRayDimsY + maxSuperpR - 1)
            };
            // Transfer launch support must be conservative. `beamFirstInside`
            // is a tracer/material diagnostic and can be one step deeper than
            // the first shallow voxels that still sample nonzero BEV dose after
            // fan projection. Keep the BEV sampling shift unchanged; only widen
            // the dose-box launch lower bound.
            const int transferFirstStep = 0;
            const float zVals[2] = {
                float(transferFirstStep),
                float(beamFirstCalculatedPassive - 1)
            };

            for (int zVal = 0; zVal < 2; ++zVal) {
                for (int yVal = 0; yVal < 2; ++yVal) {
                    for (int xVal = 0; xVal < 2; ++xVal) {
                        const vec3f primTransfPoint = primRayIdxToDoseIdx.transformPoint(
                            make_vec3f(xVals[xVal], yVals[yVal], zVals[zVal])
                        );
                        primMaxPoint.x = fmaxf(primMaxPoint.x, primTransfPoint.x);
                        primMaxPoint.y = fmaxf(primMaxPoint.y, primTransfPoint.y);
                        primMaxPoint.z = fmaxf(primMaxPoint.z, primTransfPoint.z);
                        primMinPoint.x = fminf(primMinPoint.x, primTransfPoint.x);
                        primMinPoint.y = fminf(primMinPoint.y, primTransfPoint.y);
                        primMinPoint.z = fminf(primMinPoint.z, primTransfPoint.z);
                    }
                }
            }

            auto roundUpTo = [](int value, int multiple) {
                return ((value + multiple - 1) / multiple) * multiple;
            };

            vec3i startIdx = make_vec3i(
                std::max((static_cast<int>(floorf(primMinPoint.x)) / 32) * 32, 0),
                std::max(static_cast<int>(floorf(primMinPoint.y)), 0),
                std::max(static_cast<int>(floorf(primMinPoint.z)), 0)
            );
            vec3i maxIdx = make_vec3i(
                std::min(static_cast<int>(ceilf(primMaxPoint.x)), doseDims.x - 1),
                std::min(static_cast<int>(ceilf(primMaxPoint.y)), doseDims.y - 1),
                std::min(static_cast<int>(ceilf(primMaxPoint.z)), doseDims.z - 1)
            );

            const Float3ToFanTransform doseIdxToPrimRayIdx = primRayIdxToDoseIdx.invertAndShift(
                make_vec3f(float(maxSuperpR), float(maxSuperpR), -float(beamFirstInside))
            );
            const TransferParamStructDiv3 transferParams(doseIdxToPrimRayIdx);
            const bool transferAudit = rtdTransferAuditEnabled();

            dim3 transfBlockDim(32, 8);
            dim3 transfGridDim(
                roundUpTo(std::max(maxIdx.x - startIdx.x + 1, 0), transfBlockDim.x) / transfBlockDim.x,
                roundUpTo(std::max(maxIdx.y - startIdx.y + 1, 0), transfBlockDim.y) / transfBlockDim.y
            );
            int maxZIdx = maxIdx.z;
            layerPerf.transferGridX = static_cast<int>(transfGridDim.x);
            layerPerf.transferGridY = static_cast<int>(transfGridDim.y);
            const int transferNx = std::max(maxIdx.x - startIdx.x + 1, 0);
            const int transferNy = std::max(maxIdx.y - startIdx.y + 1, 0);
            const int transferNz = std::max(maxIdx.z - startIdx.z + 1, 0);
            layerPerf.transferBoxVoxels = transferNx * transferNy * transferNz;
            
            if (fineTiming) {
                std::cout << "  Launching primTransfDiv kernel:" << std::endl;
                std::cout << "    doseBox minIdx: (" << startIdx.x << ", " << startIdx.y << ", " << startIdx.z << ")" << std::endl;
                std::cout << "    doseBox maxIdx: (" << maxIdx.x << ", " << maxIdx.y << ", " << maxIdx.z << ")" << std::endl;
                std::cout << "    Grid: (" << transfGridDim.x << ", " << transfGridDim.y << ")" << std::endl;
                std::cout << "    Block: (" << transfBlockDim.x << ", " << transfBlockDim.y << ")" << std::endl;
                std::cout << "    maxZ: " << maxZIdx << std::endl;
            }

            if (transferAudit && isRepresentativeLayer(layerIdx, numLayers)) {
                std::cout << "[TRANSFER_AUDIT] layer=" << layerIdx
                          << " energy=" << energy
                          << " beamFirstInside=" << beamFirstInside
                          << " transferFirstStep=" << transferFirstStep
                          << " beamFirstCalculatedPassive=" << beamFirstCalculatedPassive
                          << " maxSuperpR=" << maxSuperpR
                          << std::endl;
                std::cout << "  primMinPoint=(" << primMinPoint.x << "," << primMinPoint.y << "," << primMinPoint.z << ")"
                          << " primMaxPoint=(" << primMaxPoint.x << "," << primMaxPoint.y << "," << primMaxPoint.z << ")"
                          << std::endl;
                std::cout << "  startIdx=(" << startIdx.x << "," << startIdx.y << "," << startIdx.z << ")"
                          << " maxIdx=(" << maxIdx.x << "," << maxIdx.y << "," << maxIdx.z << ")"
                          << " doseDims=(" << doseDims.x << "," << doseDims.y << "," << doseDims.z << ")"
                          << std::endl;
                std::cout << "  normDist=(" << transferParams.normDist.x << "," << transferParams.normDist.y << ")"
                          << " globalOffset=(" << transferParams.globalOffset.x << "," << transferParams.globalOffset.y << "," << transferParams.globalOffset.z << ")"
                          << std::endl;

                const vec3f basisDoseOrigin = make_vec3f(float((startIdx.x + maxIdx.x) / 2),
                                                         float((startIdx.y + maxIdx.y) / 2),
                                                         float((startIdx.z + maxIdx.z) / 2));
                const vec3f fanAtOrigin = doseIdxToPrimRayIdx.transformPoint(basisDoseOrigin);
                const vec3f fanDx = doseIdxToPrimRayIdx.transformPoint(basisDoseOrigin + make_vec3f(1.0f, 0.0f, 0.0f)) - fanAtOrigin;
                const vec3f fanDy = doseIdxToPrimRayIdx.transformPoint(basisDoseOrigin + make_vec3f(0.0f, 1.0f, 0.0f)) - fanAtOrigin;
                const vec3f fanDz = doseIdxToPrimRayIdx.transformPoint(basisDoseOrigin + make_vec3f(0.0f, 0.0f, 1.0f)) - fanAtOrigin;
                std::cout << "  basisProbe doseOrigin=(" << basisDoseOrigin.x << "," << basisDoseOrigin.y << "," << basisDoseOrigin.z << ")"
                          << " -> bev=(" << fanAtOrigin.x << "," << fanAtOrigin.y << "," << fanAtOrigin.z << ")"
                          << std::endl;
                std::cout << "  doseAxisResponse dX=(" << fanDx.x << "," << fanDx.y << "," << fanDx.z << ")"
                          << " dY=(" << fanDy.x << "," << fanDy.y << "," << fanDy.z << ")"
                          << " dZ=(" << fanDz.x << "," << fanDz.y << "," << fanDz.z << ")"
                          << std::endl;
            }

            // ------------------------------------------------------------
            // Debug: sanity-check BEV sampling coordinates before launching primTransfDiv.
            // If these are OUT of [0..bevDim-1], the texture returns 0 and final dose can be all-zero.
            // ------------------------------------------------------------
            if (fineTiming || (transferAudit && isRepresentativeLayer(layerIdx, numLayers))) {
                if (fineTiming) {
                    std::cout << "  [TRANSF] normDist=(" << transferParams.normDist.x << ", " << transferParams.normDist.y << ")"
                              << "  globalOffset=(" << transferParams.globalOffset.x << ", " << transferParams.globalOffset.y << ", " << transferParams.globalOffset.z << ")" << std::endl;
                }

                auto probe = [&](int i, int j, int k) {
                    TransferParamStructDiv3 p = transferParams;
                    p.init(i, j);
                    vec3f fan = p.getFanIdx(k);
                    const bool in = (fan.x >= 0.0f && fan.x < float(bevDoseX) &&
                                     fan.y >= 0.0f && fan.y < float(bevDoseY) &&
                                     fan.z >= 0.0f && fan.z < float(bevDoseZ));
                    std::cout << "    [PROBE] doseIdx=(" << i << "," << j << "," << k << ") -> bevIdx=("
                              << fan.x << "," << fan.y << "," << fan.z << ") "
                              << (in ? "IN" : "OUT") << std::endl;
                };

                const int i0 = startIdx.x;
                const int j0 = startIdx.y;
                const int ic = (startIdx.x + maxIdx.x) / 2;
                const int jc = (startIdx.y + maxIdx.y) / 2;
                const int i1 = maxIdx.x;
                const int j1 = maxIdx.y;
                const int kA = std::max(0, startIdx.z);
                const int kB = std::max(0, maxZIdx);
                const int kMid = std::max(0, (startIdx.z + maxZIdx) / 2);

                probe(i0, j0, kA);
                probe(ic, jc, kA);
                probe(i1, j1, kA);
                if (kMid != kA && kMid != kB) {
                    probe(i0, j0, kMid);
                    probe(ic, jc, kMid);
                    probe(i1, j1, kMid);
                }
                if (kB != kA) {
                    probe(i0, j0, kB);
                    probe(ic, jc, kB);
                    probe(i1, j1, kB);
                }
            }
            
            const auto transferPerfStart = perfNow();
            if (startIdx.x <= maxIdx.x && startIdx.y <= maxIdx.y && startIdx.z <= maxIdx.z &&
                transfGridDim.x > 0 && transfGridDim.y > 0) {
                primTransfDiv<<<transfGridDim, transfBlockDim>>>(
                    devDoseVol,
                    transferParams,
                    startIdx,
                    maxZIdx,
                    make_vec3i(doseDims.x, doseDims.y, doseDims.z),
                    bevPrimDoseTex
                );
                checkCudaErrors(cudaDeviceSynchronize());
            } else if (fineTiming) {
                std::cout << "  [TRANSF] Skipping primTransfDiv because projected dose box is empty" << std::endl;
            }

#ifdef NUCLEAR_CORR
            if (runtimeNuclearEnabled) {
                // [9.53] Halo lattice deltas/offsets must be in mm to match primary
                // fan transform (line ~3192) and upstream beam.getSpotIdxToGantry()
                // contract. Catch unit-regressions early before constructing nucIdxToFan.
                if (!(layerHaloPlan->spotDelta.x > 0.0f && layerHaloPlan->spotDelta.x < 100.0f) ||
                    !(layerHaloPlan->spotDelta.y > 0.0f && layerHaloPlan->spotDelta.y < 100.0f)) {
                    throw std::runtime_error(
                        "[9.53] halo lattice spotDelta out of plausible mm range; "
                        "suspected unit regression (cm-vs-mm or stray *lenToMm)");
                }

                // [9.48] RC2 fix: layerHaloPlan->spotDelta/spotOffset are already in mm
                // (sourced from weqHeader[7]/[4]/[6]/[3] which are mm). The primary path
                // at line ~3192 uses Float3IdxTransform fanIdxToFan(fanDelta_mm, fanCorner_mm)
                // with no extra unit conversion. Upstream kernel_wrapper.cu:1240 uses
                // beam.getSpotIdxToGantry() (mm) directly. Multiplying by lenToMm again
                // here placed the halo BEV ~10x outside the dose volume and broke
                // nucTransfDiv sampling for every voxel. See halo-energy-conservation.md.
                const Float3IdxTransform nucIdxToFan(
                    make_vec3f(layerHaloPlan->spotDelta.x,
                               layerHaloPlan->spotDelta.y,
                               fanDelta_mm.z),
                    make_vec3f(layerHaloPlan->spotOffset.x,
                               layerHaloPlan->spotOffset.y,
                               fanCorner_mm.z)
                );
                const Float3FromFanTransform nucRayIdxToDoseIdx(
                    nucIdxToFan, sourceDistVec, gantryToDoseIdx);

                vec3f nucMaxPoint = make_vec3f(-1.0f, -1.0f, -1.0f);
                vec3f nucMinPoint = make_vec3f(100000.0f, 100000.0f, 100000.0f);
                const float nucXVals[2] = {
                    -float(maxSuperpR),
                    float(layerHaloPlan->nucRayDims.x + maxSuperpR - 1)
                };
                const float nucYVals[2] = {
                    -float(maxSuperpR),
                    float(layerHaloPlan->nucRayDims.y + maxSuperpR - 1)
                };
                for (int zVal = 0; zVal < 2; ++zVal) {
                    for (int yVal = 0; yVal < 2; ++yVal) {
                        for (int xVal = 0; xVal < 2; ++xVal) {
                            const vec3f p = nucRayIdxToDoseIdx.transformPoint(
                                make_vec3f(nucXVals[xVal], nucYVals[yVal], zVals[zVal])
                            );
                            nucMaxPoint.x = fmaxf(nucMaxPoint.x, p.x);
                            nucMaxPoint.y = fmaxf(nucMaxPoint.y, p.y);
                            nucMaxPoint.z = fmaxf(nucMaxPoint.z, p.z);
                            nucMinPoint.x = fminf(nucMinPoint.x, p.x);
                            nucMinPoint.y = fminf(nucMinPoint.y, p.y);
                            nucMinPoint.z = fminf(nucMinPoint.z, p.z);
                        }
                    }
                }

                const vec3i nucStartIdx = make_vec3i(
                    std::max((static_cast<int>(floorf(nucMinPoint.x)) / 32) * 32, 0),
                    std::max(static_cast<int>(floorf(nucMinPoint.y)), 0),
                    std::max(static_cast<int>(floorf(nucMinPoint.z)), 0)
                );
                const vec3i nucMaxIdx = make_vec3i(
                    std::min(static_cast<int>(ceilf(nucMaxPoint.x)), doseDims.x - 1),
                    std::min(static_cast<int>(ceilf(nucMaxPoint.y)), doseDims.y - 1),
                    std::min(static_cast<int>(ceilf(nucMaxPoint.z)), doseDims.z - 1)
                );
                const Float3ToFanTransform doseIdxToNucRayIdx = nucRayIdxToDoseIdx.invertAndShift(
                    make_vec3f(float(maxSuperpR), float(maxSuperpR), -float(beamFirstInside))
                );
                const TransferParamStructDiv3 nucTransferParams(doseIdxToNucRayIdx);
                dim3 nucTransfGridDim(
                    roundUpTo(std::max(nucMaxIdx.x - nucStartIdx.x + 1, 0), transfBlockDim.x) / transfBlockDim.x,
                    roundUpTo(std::max(nucMaxIdx.y - nucStartIdx.y + 1, 0), transfBlockDim.y) / transfBlockDim.y
                );
                const bool nuclearTransferAuditLayer =
                    (transferAudit || haloAudit) && isRepresentativeLayer(static_cast<int>(layerIdx), numLayers);
                std::vector<float> hDoseBeforeNuc;
                if (nuclearTransferAuditLayer) {
                    hDoseBeforeNuc.resize(doseSize);
                    copyToHost(hDoseBeforeNuc.data(), devDoseVol, doseSize * sizeof(float));
                    std::cout << "[TRANSFER_AUDIT][NUC] layer=" << layerIdx
                              << " energy=" << energy
                              << " startIdx=(" << nucStartIdx.x << "," << nucStartIdx.y << "," << nucStartIdx.z << ")"
                              << " maxIdx=(" << nucMaxIdx.x << "," << nucMaxIdx.y << "," << nucMaxIdx.z << ")"
                              << " grid=(" << nucTransfGridDim.x << "," << nucTransfGridDim.y << ")"
                              << " nucMinPoint=(" << nucMinPoint.x << "," << nucMinPoint.y << "," << nucMinPoint.z << ")"
                              << " nucMaxPoint=(" << nucMaxPoint.x << "," << nucMaxPoint.y << "," << nucMaxPoint.z << ")"
                              << " nucBevDims=(" << bevNucDoseX << "," << bevNucDoseY << "," << bevNucDoseZ << ")"
                              << " haloGridDelta=(" << layerHaloPlan->spotDelta.x << "," << layerHaloPlan->spotDelta.y << ")"
                              << " haloOffset=(" << layerHaloPlan->spotOffset.x << "," << layerHaloPlan->spotOffset.y << ")"
                              << " physicalPBDelta=(" << physicalSpotDeltaForLayer.x << "," << physicalSpotDeltaForLayer.y << ")"
                              << " beamFirstInside=" << beamFirstInside
                              << " transferFirstStep=" << transferFirstStep
                              << " beamFirstCalculatedPassive=" << beamFirstCalculatedPassive
                              << " willLaunch="
                              << ((nucStartIdx.x <= nucMaxIdx.x && nucStartIdx.y <= nucMaxIdx.y &&
                                   nucStartIdx.z <= nucMaxIdx.z && nucTransfGridDim.x > 0 &&
                                   nucTransfGridDim.y > 0) ? 1 : 0)
                              << std::endl;

                    std::cout << "  [TRANSFER_AUDIT][NUC] normDist=("
                              << nucTransferParams.normDist.x << "," << nucTransferParams.normDist.y << ")"
                              << " globalOffset=(" << nucTransferParams.globalOffset.x << ","
                              << nucTransferParams.globalOffset.y << "," << nucTransferParams.globalOffset.z << ")"
                              << std::endl;
                    const vec3f nucBasisDoseOrigin = make_vec3f(float((nucStartIdx.x + nucMaxIdx.x) / 2),
                                                                float((nucStartIdx.y + nucMaxIdx.y) / 2),
                                                                float((nucStartIdx.z + nucMaxIdx.z) / 2));
                    const vec3f nucFanAtOrigin = doseIdxToNucRayIdx.transformPoint(nucBasisDoseOrigin);
                    const vec3f nucFanDx = doseIdxToNucRayIdx.transformPoint(nucBasisDoseOrigin + make_vec3f(1.0f, 0.0f, 0.0f)) - nucFanAtOrigin;
                    const vec3f nucFanDy = doseIdxToNucRayIdx.transformPoint(nucBasisDoseOrigin + make_vec3f(0.0f, 1.0f, 0.0f)) - nucFanAtOrigin;
                    const vec3f nucFanDz = doseIdxToNucRayIdx.transformPoint(nucBasisDoseOrigin + make_vec3f(0.0f, 0.0f, 1.0f)) - nucFanAtOrigin;
                    std::cout << "  [TRANSFER_AUDIT][NUC] basisProbe doseOrigin=("
                              << nucBasisDoseOrigin.x << "," << nucBasisDoseOrigin.y << "," << nucBasisDoseOrigin.z << ")"
                              << " -> nucBev=(" << nucFanAtOrigin.x << "," << nucFanAtOrigin.y << "," << nucFanAtOrigin.z << ")"
                              << std::endl;
                    std::cout << "  [TRANSFER_AUDIT][NUC] doseAxisResponse dX=("
                              << nucFanDx.x << "," << nucFanDx.y << "," << nucFanDx.z << ")"
                              << " dY=(" << nucFanDy.x << "," << nucFanDy.y << "," << nucFanDy.z << ")"
                              << " dZ=(" << nucFanDz.x << "," << nucFanDz.y << "," << nucFanDz.z << ")"
                              << std::endl;

                    auto nucProbe = [&](int i, int j, int k) {
                        TransferParamStructDiv3 p = nucTransferParams;
                        p.init(i, j);
                        const vec3f fan = p.getFanIdx(k);
                        const vec3f tex = fan + make_vec3f(HALF, HALF, HALF);
                        const bool fanIn = (fan.x >= 0.0f && fan.x < float(bevNucDoseX) &&
                                            fan.y >= 0.0f && fan.y < float(bevNucDoseY) &&
                                            fan.z >= 0.0f && fan.z < float(bevNucDoseZ));
                        const bool texIn = (tex.x >= 0.0f && tex.x < float(bevNucDoseX) &&
                                            tex.y >= 0.0f && tex.y < float(bevNucDoseY) &&
                                            tex.z >= 0.0f && tex.z < float(bevNucDoseZ));
                        std::cout << "    [PROBE][NUC] doseIdx=(" << i << "," << j << "," << k << ")"
                                  << " -> nucBevIdx=(" << fan.x << "," << fan.y << "," << fan.z << ")"
                                  << " texIdx=(" << tex.x << "," << tex.y << "," << tex.z << ") "
                                  << (texIn ? "IN" : "OUT")
                                  << " rawFanIn=" << (fanIn ? 1 : 0)
                                  << std::endl;
                    };
                    const int ni0 = nucStartIdx.x;
                    const int nj0 = nucStartIdx.y;
                    const int nic = (nucStartIdx.x + nucMaxIdx.x) / 2;
                    const int njc = (nucStartIdx.y + nucMaxIdx.y) / 2;
                    const int ni1 = nucMaxIdx.x;
                    const int nj1 = nucMaxIdx.y;
                    const int nkA = std::max(0, nucStartIdx.z);
                    const int nkB = std::max(0, nucMaxIdx.z);
                    const int nkMid = std::max(0, (nucStartIdx.z + nucMaxIdx.z) / 2);
                    nucProbe(ni0, nj0, nkA);
                    nucProbe(nic, njc, nkA);
                    nucProbe(ni1, nj1, nkA);
                    if (nkMid != nkA && nkMid != nkB) {
                        nucProbe(ni0, nj0, nkMid);
                        nucProbe(nic, njc, nkMid);
                        nucProbe(ni1, nj1, nkMid);
                    }
                    if (nkB != nkA) {
                        nucProbe(ni0, nj0, nkB);
                        nucProbe(nic, njc, nkB);
                        nucProbe(ni1, nj1, nkB);
                    }
                }

                if (nucStartIdx.x <= nucMaxIdx.x && nucStartIdx.y <= nucMaxIdx.y && nucStartIdx.z <= nucMaxIdx.z &&
                    nucTransfGridDim.x > 0 && nucTransfGridDim.y > 0) {
                    nucTransfDiv<<<nucTransfGridDim, transfBlockDim>>>(
                        devDoseVol,
                        nucTransferParams,
                        nucStartIdx,
                        nucMaxIdx.z,
                        make_vec3i(doseDims.x, doseDims.y, doseDims.z),
                        bevNucDoseTex
                    );
                    checkCudaErrors(cudaDeviceSynchronize());
                } else if (fineTiming) {
                    std::cout << "  [TRANSF] Skipping nucTransfDiv because projected halo dose box is empty" << std::endl;
                }

                if (nuclearTransferAuditLayer) {
                    std::vector<float> hDoseAfterNuc(doseSize);
                    copyToHost(hDoseAfterNuc.data(), devDoseVol, doseSize * sizeof(float));
                    const FloatSummaryStats beforeStats = summarizeFloatVector(hDoseBeforeNuc, 1.0e-12f);
                    const FloatSummaryStats afterStats = summarizeFloatVector(hDoseAfterNuc, 1.0e-12f);
                    double deltaSum = 0.0;
                    float maxDelta = -std::numeric_limits<float>::infinity();
                    float minDelta = std::numeric_limits<float>::infinity();
                    int positiveDeltaCount = 0;
                    int negativeDeltaCount = 0;
                    int finiteDeltaCount = 0;
                    const size_t compareN = std::min(hDoseBeforeNuc.size(), hDoseAfterNuc.size());
                    for (size_t idx = 0; idx < compareN; ++idx) {
                        const float beforeV = hDoseBeforeNuc[idx];
                        const float afterV = hDoseAfterNuc[idx];
                        if (!hostIsFinite(beforeV) || !hostIsFinite(afterV)) continue;
                        const float d = afterV - beforeV;
                        ++finiteDeltaCount;
                        deltaSum += static_cast<double>(d);
                        maxDelta = std::max(maxDelta, d);
                        minDelta = std::min(minDelta, d);
                        if (d > 1.0e-12f) ++positiveDeltaCount;
                        if (d < -1.0e-12f) ++negativeDeltaCount;
                    }
                    if (finiteDeltaCount == 0) {
                        maxDelta = 0.0f;
                        minDelta = 0.0f;
                    }
                    std::cout << "[TRANSFER_AUDIT][NUC_DELTA] layer=" << layerIdx
                              << " beforeSum=" << beforeStats.sumFinite
                              << " afterSum=" << afterStats.sumFinite
                              << " deltaSum=" << deltaSum
                              << " beforeMax=" << beforeStats.maxFinite
                              << " afterMax=" << afterStats.maxFinite
                              << " maxDelta=" << maxDelta
                              << " minDelta=" << minDelta
                              << " positiveDeltaCount=" << positiveDeltaCount
                              << " negativeDeltaCount=" << negativeDeltaCount
                              << " finiteDeltaCount=" << finiteDeltaCount
                              << std::endl;
                }
            }
#endif
            layerPerf.transferMs += perfElapsedMs(transferPerfStart);
            beamTiming.doseTransferMs += layerPerf.transferMs;
            
            // Cleanup texture
            const auto layerCleanupPerfStart = perfNow();
            destroyTextureObjectAndArray(bevPrimDoseTex);
#ifdef NUCLEAR_CORR
            destroyTextureObjectAndArray(bevNucDoseTex);
#endif
            
            if (fineTiming) {
                std::cout << "  Dose transformation completed using primTransfDiv kernel" << std::endl;
                std::cout << "  Dose accumulation completed" << std::endl;
            }
            
            // devRayWeights and primary layer scratch are beam-scoped and cleaned
            // up after all layers.
#ifdef NUCLEAR_CORR
            // Nuclear scratch is also beam-scoped.
#endif
            layerPerf.cleanupMs += perfElapsedMs(layerCleanupPerfStart);
            beamTiming.layerCleanupMs += layerPerf.cleanupMs;
            layerPerf.totalMs = perfElapsedMs(layerPerfStart);
            if (perfProfileLayers) {
                std::cout << "[PERF_LAYER] beam=" << beamIdx
                          << " layer=" << layerPerf.layerIdx << "/" << numLayers
                          << " energy=" << layerPerf.energy
                          << " spots=" << layerPerf.spots
                          << " ms(alloc,iddSigma,superp,tex,transfer,cleanup,total)=("
                          << layerPerf.allocMs << ","
                          << layerPerf.iddSigmaMs << ","
                          << layerPerf.superpositionMs << ","
                          << layerPerf.textureMs << ","
                          << layerPerf.transferMs << ","
                          << layerPerf.cleanupMs << ","
                          << layerPerf.totalMs << ")"
                          << " activeSteps=[" << layerPerf.activeFirst << "," << layerPerf.activeLast << "]"
                          << " activeN=" << layerPerf.activeCount
                          << " rayDims=(" << layerPerf.rayDimsX << "," << layerPerf.rayDimsY << ")"
                          << " superpRayDims=(" << layerPerf.superpRayDimsX << "," << layerPerf.superpRayDimsY << ")"
                          << " transferGrid=(" << layerPerf.transferGridX << "," << layerPerf.transferGridY << ")"
                          << " transferBoxVoxels=" << layerPerf.transferBoxVoxels
                          << std::endl;
            }
            layerPerfRows.push_back(layerPerf);
        }

        freeDeviceMemory(devRayIddScratch);
        freeDeviceMemory(devRayRSigmaEffScratch);
        freeDeviceMemory(devFirstPassiveScratch);
        freeDeviceMemory(devBeamFirstPassiveMaxScratch);
        if (devRayIddPaddedScratch) freeDeviceMemory(devRayIddPaddedScratch);
        if (devRayRSigmaEffPaddedScratch) freeDeviceMemory(devRayRSigmaEffPaddedScratch);
        freeDeviceMemory(devBevPrimDoseScratch);
#ifdef NUCLEAR_CORR
        if (devNucRayWeightsScratch) freeDeviceMemory(devNucRayWeightsScratch);
        if (devNucSpotIdxScratch) freeDeviceMemory(devNucSpotIdxScratch);
        if (devNucIddScratch) freeDeviceMemory(devNucIddScratch);
        if (devNucRSigmaEffScratch) freeDeviceMemory(devNucRSigmaEffScratch);
        if (devBevNucDoseScratch) freeDeviceMemory(devBevNucDoseScratch);
#endif
        freeDeviceMemory(devRayWeightsAllLayers);
        freeDeviceMemory(devBevDensity);
        freeDeviceMemory(devBevCumulSp);
        freeDeviceMemory(devBeamFirstInside);
        freeDeviceMemory(devFirstStepOutside);
        destroyTextureObjectAndArray(profileTex);
        beamTiming.totalMs = perfElapsedMs(beamPerfStart);
        if (perfProfile) {
            double layerTotalMs = 0.0;
            int maxTransferBoxVoxels = 0;
            double maxLayerMs = 0.0;
            size_t maxLayerIdx = 0;
            for (const LayerPerfTiming& row : layerPerfRows) {
                layerTotalMs += row.totalMs;
                if (row.transferBoxVoxels > maxTransferBoxVoxels) {
                    maxTransferBoxVoxels = row.transferBoxVoxels;
                }
                if (row.totalMs > maxLayerMs) {
                    maxLayerMs = row.totalMs;
                    maxLayerIdx = row.layerIdx;
                }
            }
            std::cout << "[PERF_SUMMARY] beam=" << beamIdx
                      << " totalMs=" << beamTiming.totalMs
                      << " numLayers=" << numLayers
                      << " totalSpots=" << totalBeamSpots
                      << " rayDims=(" << rayDims.x << "," << rayDims.y << ")"
                      << " tracerSteps=" << tracerSteps
                      << " maxTransferBoxVoxels=" << maxTransferBoxVoxels
                      << " slowestLayer=" << maxLayerIdx
                      << " slowestLayerMs=" << maxLayerMs
                      << std::endl;
            std::cout << "[PERF_SUMMARY] beam=" << beamIdx
                      << " ms(setup,rayWeight,bevTrace,weplReduce,iddSigma,superp,bevTexture,transfer,layerCleanup,layerTotal)=("
                      << beamTiming.setupMs << ","
                      << beamTiming.rayWeightMs << ","
                      << beamTiming.bevTraceMs << ","
                      << beamTiming.weplReduceMs << ","
                      << beamTiming.iddSigmaMs << ","
                      << beamTiming.superpositionMs << ","
                      << beamTiming.layerTextureMs << ","
                      << beamTiming.doseTransferMs << ","
                      << beamTiming.layerCleanupMs << ","
                      << layerTotalMs << ")"
                      << std::endl;
        }
    }

    if (rtdInputAuditEnabled()) {
        std::vector<float> hDoseVol(doseSize);
        copyToHost(hDoseVol.data(), devDoseVol, doseSize * sizeof(float));
        printStageVolumeSummary("DOSE_PRE_WRITEBACK", hDoseVol, doseDims.x, doseDims.y, doseDims.z, 1.0e-12f, "z");
        printDoseGridSupportSummary("DOSE_PRE_WRITEBACK", hDoseVol, doseDims.x, doseDims.y, doseDims.z, 1.0e-12f);
    }
    
    // Copy accumulated dose from device to host
    copyToHost(doseData, devDoseVol, doseSize * sizeof(float));
    
    // Clean up device memory
    freeDeviceMemory(devDoseVol);
    
    // Clean up textures
    destroyTextureObjectAndArray(imVolTex);
    if (devCtLinear) freeDeviceMemory(devCtLinear);
    destroyTextureObjectAndArray(cumulIddTex);
    destroyTextureObjectAndArray(densityTex);
    destroyTextureObjectAndArray(stoppingPowerTex);
    destroyTextureObjectAndArray(rRadiationLengthTex);
#ifdef NUCLEAR_CORR
    destroyTextureObjectAndArray(nucWeightTex);
    destroyTextureObjectAndArray(nucSqSigmaTex);
#endif
    
    CPU_TIMER_END("RTD Wrapper");
    CPU_TIMER_END_SUMMARY("RTD Wrapper");
}

// Helper functions
extern "C" RTDBeamSettings* createRTDBeamSettings() {
    return new RTDBeamSettings();
}

extern "C" void destroyRTDBeamSettings(RTDBeamSettings* beam) {
    delete beam;
}

extern "C" RTDEnergyStruct* createRTDEnergyStruct() {
    return new RTDEnergyStruct();
}

extern "C" void destroyRTDEnergyStruct(RTDEnergyStruct* energy) {
    delete energy;
}
