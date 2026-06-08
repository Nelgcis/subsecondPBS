/**
 * \file
 * \brief Shared helpers for public/internal nuclear LUT carriage and alignment
 */

#ifndef NUCLEAR_TABLE_UTILS_H
#define NUCLEAR_TABLE_UTILS_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace rtd {
namespace nuclear {

inline bool approxEq(float a, float b, float tol = 1.0e-4f) {
    return std::fabs(a - b) <= tol;
}

inline bool approxVectorEq(const std::vector<float>& a,
                           const std::vector<float>& b,
                           float tol = 1.0e-4f) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!approxEq(a[i], b[i], tol)) return false;
    }
    return true;
}

inline float findFractionalIndexMonotonic(const std::vector<float>& xs, float x) {
    if (xs.empty()) return 0.0f;
    if (xs.size() == 1) return 0.0f;

    const bool ascending = xs.back() >= xs.front();
    if (ascending) {
        if (x <= xs.front()) return 0.0f;
        if (x >= xs.back()) return static_cast<float>(xs.size() - 1);
        for (size_t i = 1; i < xs.size(); ++i) {
            if (x <= xs[i]) {
                const float denom = xs[i] - xs[i - 1];
                const float t = (std::fabs(denom) > 1.0e-12f) ? ((x - xs[i - 1]) / denom) : 0.0f;
                return static_cast<float>(i - 1) + std::min(1.0f, std::max(0.0f, t));
            }
        }
    } else {
        if (x >= xs.front()) return 0.0f;
        if (x <= xs.back()) return static_cast<float>(xs.size() - 1);
        for (size_t i = 1; i < xs.size(); ++i) {
            if (x >= xs[i]) {
                const float denom = xs[i] - xs[i - 1];
                const float t = (std::fabs(denom) > 1.0e-12f) ? ((x - xs[i - 1]) / denom) : 0.0f;
                return static_cast<float>(i - 1) + std::min(1.0f, std::max(0.0f, t));
            }
        }
    }
    return static_cast<float>(xs.size() - 1);
}

template <typename EnergyLike>
inline void clearNuclearTables(EnergyLike& energy) {
    energy.nNucEnergySamples = 0;
    energy.nNucEnergies = 0;
    energy.nucEnergiesPerU.clear();
    energy.nucPeakDepths.clear();
    energy.nucScaleFacts.clear();
    energy.nucWeightMatrix.clear();
    energy.nucSqSigmaMatrix.clear();
    energy.nuclearTablesAlignedToPrimaryAxis = false;
    energy.nuclearTablesResampledFromReference = false;
}

template <typename EnergyLike>
inline bool hasNuclearTables(const EnergyLike& energy) {
    return !energy.nucWeightMatrix.empty() && !energy.nucSqSigmaMatrix.empty();
}

template <typename EnergyLike>
inline bool nuclearPayloadMatchesPrimaryAxis(const EnergyLike& energy, float tol = 1.0e-4f) {
    if (!hasNuclearTables(energy)) return false;
    if (energy.nEnergySamples <= 0 || energy.nEnergies <= 0) return false;

    const size_t expected = static_cast<size_t>(energy.nEnergySamples) * static_cast<size_t>(energy.nEnergies);
    if (energy.nucWeightMatrix.size() != expected || energy.nucSqSigmaMatrix.size() != expected) return false;

    if (energy.nNucEnergySamples != 0 && energy.nNucEnergySamples != energy.nEnergySamples) return false;
    if (energy.nNucEnergies != 0 && energy.nNucEnergies != energy.nEnergies) return false;
    if (!energy.nucEnergiesPerU.empty() && !approxVectorEq(energy.nucEnergiesPerU, energy.energiesPerU, tol)) return false;
    if (!energy.nucPeakDepths.empty() && !approxVectorEq(energy.nucPeakDepths, energy.peakDepths, tol)) return false;
    if (!energy.nucScaleFacts.empty() && !approxVectorEq(energy.nucScaleFacts, energy.scaleFacts, tol)) return false;

    return true;
}

inline float sampleRowLinear(const std::vector<float>& matrix,
                             int row,
                             int cols,
                             float sampleIdx) {
    if (row < 0 || cols <= 0) return 0.0f;
    if (matrix.empty()) return 0.0f;
    const float clamped = std::min(std::max(sampleIdx, 0.0f), static_cast<float>(cols - 1));
    float idxInt = 0.0f;
    const float frac = std::modf(clamped, &idxInt);
    const int i0 = std::max(0, std::min(cols - 1, static_cast<int>(idxInt)));
    const int i1 = std::max(0, std::min(cols - 1, i0 + ((frac > 0.0f) ? 1 : 0)));
    const size_t base = static_cast<size_t>(row) * static_cast<size_t>(cols);
    const float v0 = matrix[base + static_cast<size_t>(i0)];
    const float v1 = matrix[base + static_cast<size_t>(i1)];
    return v0 + (v1 - v0) * frac;
}

template <typename EnergyLike>
inline void setNuclearMetadataToPrimaryAxis(EnergyLike& energy, bool resampled) {
    energy.nNucEnergySamples = energy.nEnergySamples;
    energy.nNucEnergies = energy.nEnergies;
    energy.nucEnergiesPerU = energy.energiesPerU;
    energy.nucPeakDepths = energy.peakDepths;
    energy.nucScaleFacts = energy.scaleFacts;
    energy.nuclearTablesAlignedToPrimaryAxis = true;
    energy.nuclearTablesResampledFromReference = resampled;
}

template <typename RefEnergyLike, typename OutEnergyLike>
inline void alignNuclearTablesToPrimaryAxis(const RefEnergyLike& ref,
                                            OutEnergyLike& out,
                                            const std::string& contextTag) {
    if (!hasNuclearTables(ref)) {
        clearNuclearTables(out);
        return;
    }
    if (out.nEnergies <= 0 || out.nEnergySamples <= 0) {
        throw std::runtime_error(contextTag + ": primary energy axis must be initialized before aligning nuclear tables");
    }
    if (out.energiesPerU.size() != static_cast<size_t>(out.nEnergies) ||
        out.peakDepths.size() != static_cast<size_t>(out.nEnergies) ||
        out.scaleFacts.size() != static_cast<size_t>(out.nEnergies)) {
        throw std::runtime_error(contextTag + ": primary energy axis metadata size mismatch");
    }
    if (ref.nEnergies <= 0 || ref.nEnergySamples <= 0 ||
        ref.energiesPerU.size() != static_cast<size_t>(ref.nEnergies) ||
        ref.scaleFacts.size() != static_cast<size_t>(ref.nEnergies)) {
        throw std::runtime_error(contextTag + ": reference nuclear LUT axis metadata is incomplete");
    }

    const bool identicalAxis =
        ref.nEnergies == out.nEnergies &&
        ref.nEnergySamples == out.nEnergySamples &&
        approxVectorEq(ref.energiesPerU, out.energiesPerU) &&
        approxVectorEq(ref.peakDepths, out.peakDepths) &&
        approxVectorEq(ref.scaleFacts, out.scaleFacts);

    const size_t targetSize = static_cast<size_t>(out.nEnergies) * static_cast<size_t>(out.nEnergySamples);
    out.nucWeightMatrix.assign(targetSize, 0.0f);
    out.nucSqSigmaMatrix.assign(targetSize, 0.0f);

    if (identicalAxis) {
        out.nucWeightMatrix = ref.nucWeightMatrix;
        out.nucSqSigmaMatrix = ref.nucSqSigmaMatrix;
        setNuclearMetadataToPrimaryAxis(out, false);
        return;
    }

    for (int e = 0; e < out.nEnergies; ++e) {
        const float energyPerU = out.energiesPerU[static_cast<size_t>(e)];
        const float energyIdx = findFractionalIndexMonotonic(ref.energiesPerU, energyPerU);
        float energyIdxInt = 0.0f;
        const float energyFrac = std::modf(energyIdx, &energyIdxInt);
        const int e0 = std::max(0, std::min(ref.nEnergies - 1, static_cast<int>(energyIdxInt)));
        const int e1 = std::max(0, std::min(ref.nEnergies - 1, e0 + ((energyFrac > 0.0f) ? 1 : 0)));
        const float targetScale = out.scaleFacts[static_cast<size_t>(e)];

        for (int j = 0; j < out.nEnergySamples; ++j) {
            const float depthMm = (targetScale > 0.0f) ? (static_cast<float>(j) / targetScale) : 0.0f;
            const float srcIdx0 = depthMm * ref.scaleFacts[static_cast<size_t>(e0)];
            const float srcIdx1 = depthMm * ref.scaleFacts[static_cast<size_t>(e1)];
            const float w0 = sampleRowLinear(ref.nucWeightMatrix, e0, ref.nEnergySamples, srcIdx0);
            const float w1 = sampleRowLinear(ref.nucWeightMatrix, e1, ref.nEnergySamples, srcIdx1);
            const float s0 = sampleRowLinear(ref.nucSqSigmaMatrix, e0, ref.nEnergySamples, srcIdx0);
            const float s1 = sampleRowLinear(ref.nucSqSigmaMatrix, e1, ref.nEnergySamples, srcIdx1);
            const size_t outIdx = static_cast<size_t>(e) * static_cast<size_t>(out.nEnergySamples) + static_cast<size_t>(j);
            out.nucWeightMatrix[outIdx] = w0 + (w1 - w0) * energyFrac;
            out.nucSqSigmaMatrix[outIdx] = s0 + (s1 - s0) * energyFrac;
        }
    }

    setNuclearMetadataToPrimaryAxis(out, true);
}

} // namespace nuclear
} // namespace rtd

#endif // NUCLEAR_TABLE_UTILS_H
