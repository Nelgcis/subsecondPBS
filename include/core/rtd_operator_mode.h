/**
 * \file
 * \brief Spot-preserving operator-mode planning helpers.
 *
 * This layer does not change the existing RTD final-dose execution path.
 * It only provides a shared way to keep per-spot identity visible when
 * future CSC / row-norm style operator exports are implemented.
 */

#ifndef RTD_OPERATOR_MODE_H
#define RTD_OPERATOR_MODE_H

#include "core/raytracedicom_integration.h"

#include <algorithm>
#include <vector>

struct RTDOperatorLayerSpan {
    int layerIndex = -1;
    int spotBegin = 0;
    int spotCount = 0;
    float energy = 0.0f;
    float longitudinalCutoff = 0.0f;
};

struct RTDOperatorBatch {
    int batchIndex = -1;
    int spotBegin = 0;
    int spotCount = 0;
    int firstLayerIndex = -1;
    int lastLayerIndex = -1;
};

struct RTDOperatorBatchPlan {
    bool valid = false;
    int totalLayers = 0;
    int totalSpots = 0;
    int maxSpotsPerBatch = 0;
    std::vector<RTDOperatorLayerSpan> layerSpans;
    std::vector<RTDOperatorBatch> batches;
};

inline RTDOperatorBatchPlan buildRTDOperatorBatchPlan(const RTDBeamSettings& beam, int maxSpotsPerBatch) {
    RTDOperatorBatchPlan plan;
    plan.totalLayers = static_cast<int>(beam.layerSpotCounts.size());
    if (plan.totalLayers <= 0 || static_cast<int>(beam.energies.size()) != plan.totalLayers) {
        return plan;
    }

    int totalSpots = 0;
    for (int count : beam.layerSpotCounts) {
        if (count < 0) {
            return plan;
        }
        totalSpots += count;
    }
    plan.totalSpots = totalSpots;
    if (totalSpots <= 0) {
        return plan;
    }
    if (!beam.spotWeights.empty() && static_cast<int>(beam.spotWeights.size()) != totalSpots) {
        return plan;
    }
    if (!beam.spotPositions.empty() && static_cast<int>(beam.spotPositions.size() / 2u) != totalSpots) {
        return plan;
    }
    if (!beam.spotBeamDirections.empty() && static_cast<int>(beam.spotBeamDirections.size() / 3u) != totalSpots) {
        return plan;
    }

    int runningSpot = 0;
    plan.layerSpans.reserve(static_cast<size_t>(plan.totalLayers));
    for (int layer = 0; layer < plan.totalLayers; ++layer) {
        RTDOperatorLayerSpan span;
        span.layerIndex = layer;
        span.spotBegin = runningSpot;
        span.spotCount = beam.layerSpotCounts[static_cast<size_t>(layer)];
        span.energy = beam.energies[static_cast<size_t>(layer)];
        if (layer < static_cast<int>(beam.layerLongitudinalCutoffs.size())) {
            span.longitudinalCutoff = beam.layerLongitudinalCutoffs[static_cast<size_t>(layer)];
        }
        plan.layerSpans.push_back(span);
        runningSpot += span.spotCount;
    }

    plan.maxSpotsPerBatch = std::max(1, maxSpotsPerBatch);
    const auto findLayerForSpot = [&](int spotIdx) {
        for (const RTDOperatorLayerSpan& span : plan.layerSpans) {
            if (span.spotCount <= 0) {
                continue;
            }
            const int spotEnd = span.spotBegin + span.spotCount;
            if (spotIdx >= span.spotBegin && spotIdx < spotEnd) {
                return span.layerIndex;
            }
        }
        return -1;
    };

    for (int batchBegin = 0, batchIndex = 0; batchBegin < totalSpots; batchBegin += plan.maxSpotsPerBatch, ++batchIndex) {
        RTDOperatorBatch batch;
        batch.batchIndex = batchIndex;
        batch.spotBegin = batchBegin;
        batch.spotCount = std::min(plan.maxSpotsPerBatch, totalSpots - batchBegin);
        batch.firstLayerIndex = findLayerForSpot(batch.spotBegin);
        batch.lastLayerIndex = findLayerForSpot(batch.spotBegin + batch.spotCount - 1);
        plan.batches.push_back(batch);
    }

    plan.valid = !plan.layerSpans.empty() && !plan.batches.empty();
    return plan;
}

#endif
