#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cuda_runtime.h>

#include "core/raytracedicom_integration.h"
#include "utils/debug_tools.h"
#include "utils/energy_reader.h"
#include "utils/energy_struct.h"
#include "utils/nuclear_table_utils.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

// -----------------------------
// Helpers
// -----------------------------

static inline int3 to_int3(const std::array<int, 3>& v) {
    return make_int3(v[0], v[1], v[2]);
}

static inline float3 to_float3(const std::array<float, 3>& v) {
    return make_float3(v[0], v[1], v[2]);
}

static inline float2 to_float2(const std::array<float, 2>& v) {
    return make_float2(v[0], v[1]);
}

static void maybe_fill_nuclear_tables(RTDEnergyStruct& out,
                                      const py::dict* energy,
                                      const std::string& tables_dir,
                                      const EnergyStruct* referenceTables);

static std::vector<float> to_float_vector(const py::handle& obj) {
    if (py::isinstance<py::array>(obj)) {
        py::array_t<float, py::array::c_style | py::array::forcecast> arr = py::cast<py::array>(obj);
        py::buffer_info info = arr.request();
        if (info.ndim != 1 && !(info.ndim == 2 && (info.shape[0] == 1 || info.shape[1] == 1))) {
            throw std::runtime_error("Expected 1D float array or vector-like 2D float array");
        }
        const auto n = static_cast<size_t>(info.size);
        const float* ptr = static_cast<const float*>(info.ptr);
        return std::vector<float>(ptr, ptr + n);
    }

    return py::cast<std::vector<float>>(obj);
}

static std::vector<int> to_int_vector(const py::handle& obj) {
    if (py::isinstance<py::array>(obj)) {
        py::array_t<int, py::array::c_style | py::array::forcecast> arr = py::cast<py::array>(obj);
        py::buffer_info info = arr.request();
        if (info.ndim != 1 && !(info.ndim == 2 && (info.shape[0] == 1 || info.shape[1] == 1))) {
            throw std::runtime_error("Expected 1D int array or vector-like 2D int array");
        }
        const auto n = static_cast<size_t>(info.size);
        const int* ptr = static_cast<const int*>(info.ptr);
        return std::vector<int>(ptr, ptr + n);
    }

    return py::cast<std::vector<int>>(obj);
}

static std::vector<float2> to_float2_vector_from_nx2(const py::handle& obj) {
    // Accept: list[tuple(x,y)], list[list], or numpy array shape (N,2)
    if (py::isinstance<py::array>(obj)) {
        py::array_t<float, py::array::c_style | py::array::forcecast> arr = py::cast<py::array>(obj);
        py::buffer_info info = arr.request();
        if (info.ndim != 2 || info.shape[1] != 2) {
            throw std::runtime_error("Expected Nx2 float array");
        }
        const auto n = static_cast<size_t>(info.shape[0]);
        const float* ptr = static_cast<const float*>(info.ptr);
        std::vector<float2> out;
        out.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            float2 v;
            v.x = ptr[i * 2 + 0];
            v.y = ptr[i * 2 + 1];
            out.push_back(v);
        }
        return out;
    }

    std::vector<std::array<float, 2>> tmp = py::cast<std::vector<std::array<float, 2>>>(obj);
    std::vector<float2> out;
    out.reserve(tmp.size());
    for (const auto& t : tmp) {
        float2 v;
        v.x = t[0];
        v.y = t[1];
        out.push_back(v);
    }
    return out;
}

static float3 to_float3_from_seq(const py::handle& obj) {
    if (py::isinstance<py::array>(obj)) {
        py::array_t<float, py::array::forcecast> arr = py::cast<py::array>(obj);
        py::buffer_info info = arr.request();
        const float* ptr = static_cast<const float*>(info.ptr);
        if (info.ndim == 1 && info.size >= 3) {
            return make_float3(ptr[0], ptr[1], ptr[2]);
        }
        if (info.ndim == 2 && info.shape[0] == 3 && info.shape[1] >= 1) {
            return make_float3(ptr[0 * info.shape[1]], ptr[1 * info.shape[1]], ptr[2 * info.shape[1]]);
        }
        if (info.ndim == 2 && info.shape[1] == 3 && info.shape[0] >= 1) {
            return make_float3(ptr[0], ptr[1], ptr[2]);
        }
        throw std::runtime_error("Expected 3-vector input");
    }
    std::array<float, 3> v = py::cast<std::array<float, 3>>(obj);
    return to_float3(v);
}

static float2 to_float2_from_seq(const py::handle& obj) {
    std::array<float, 2> v = py::cast<std::array<float, 2>>(obj);
    return to_float2(v);
}

static std::vector<float> flatten_float_array(const py::handle& obj) {
    py::array_t<float, py::array::c_style | py::array::forcecast> arr = py::cast<py::array>(obj);
    py::buffer_info info = arr.request();
    const float* ptr = static_cast<const float*>(info.ptr);
    return std::vector<float>(ptr, ptr + static_cast<size_t>(info.size));
}

static std::vector<float> to_float3_columns(const py::handle& obj, size_t expected_count = 0) {
    py::array_t<float, py::array::forcecast> arr = py::cast<py::array>(obj);
    py::buffer_info info = arr.request();

    std::vector<float> out;
    if (info.ndim == 1) {
        if ((info.size % 3) != 0) {
            throw std::runtime_error("Expected flattened 3-vector array with length multiple of 3");
        }
        const size_t n = static_cast<size_t>(info.size);
        const float* ptr = static_cast<const float*>(info.ptr);
        out.assign(ptr, ptr + n);
    } else if (info.ndim == 2) {
        const float* ptr = static_cast<const float*>(info.ptr);
        if (info.shape[1] == 3) {
            const size_t n = static_cast<size_t>(info.shape[0]);
            out.resize(n * 3);
            for (size_t i = 0; i < n; ++i) {
                out[i * 3 + 0] = ptr[i * 3 + 0];
                out[i * 3 + 1] = ptr[i * 3 + 1];
                out[i * 3 + 2] = ptr[i * 3 + 2];
            }
        } else if (info.shape[0] == 3) {
            const size_t n = static_cast<size_t>(info.shape[1]);
            out.resize(n * 3);
            for (size_t i = 0; i < n; ++i) {
                out[i * 3 + 0] = ptr[0 * n + i];
                out[i * 3 + 1] = ptr[1 * n + i];
                out[i * 3 + 2] = ptr[2 * n + i];
            }
        } else {
            throw std::runtime_error("Expected shape (N,3) or (3,N) for 3-vector array");
        }
    } else {
        throw std::runtime_error("Expected 1D or 2D float array");
    }

    if (expected_count > 0 && out.size() != expected_count * 3) {
        throw std::runtime_error("3-vector array length does not match expected count");
    }
    return out;
}

static std::vector<float> to_float2_columns(const py::handle& obj, size_t expected_count = 0) {
    py::array_t<float, py::array::forcecast> arr = py::cast<py::array>(obj);
    py::buffer_info info = arr.request();

    std::vector<float> out;
    if (info.ndim == 1) {
        if ((info.size % 2) != 0) {
            throw std::runtime_error("Expected flattened 2-vector array with length multiple of 2");
        }
        const size_t n = static_cast<size_t>(info.size);
        const float* ptr = static_cast<const float*>(info.ptr);
        out.assign(ptr, ptr + n);
    } else if (info.ndim == 2) {
        const float* ptr = static_cast<const float*>(info.ptr);
        if (info.shape[1] == 2) {
            const size_t n = static_cast<size_t>(info.shape[0]);
            out.resize(n * 2);
            for (size_t i = 0; i < n; ++i) {
                out[i * 2 + 0] = ptr[i * 2 + 0];
                out[i * 2 + 1] = ptr[i * 2 + 1];
            }
        } else if (info.shape[0] == 2) {
            const size_t n = static_cast<size_t>(info.shape[1]);
            out.resize(n * 2);
            for (size_t i = 0; i < n; ++i) {
                out[i * 2 + 0] = ptr[0 * n + i];
                out[i * 2 + 1] = ptr[1 * n + i];
            }
        } else {
            throw std::runtime_error("Expected shape (N,2) or (2,N) for 2-vector array");
        }
    } else {
        throw std::runtime_error("Expected 1D or 2D float array");
    }

    if (expected_count > 0 && out.size() != expected_count * 2) {
        throw std::runtime_error("2-vector array length does not match expected count");
    }
    return out;
}

static float3 first_float3_from_columns(const py::handle& obj) {
    const std::vector<float> flat = to_float3_columns(obj);
    if (flat.size() < 3) {
        throw std::runtime_error("Expected at least one 3-vector");
    }
    return make_float3(flat[0], flat[1], flat[2]);
}

static float3 normalized_mean_direction(const std::vector<float>& dirs) {
    if (dirs.empty() || (dirs.size() % 3) != 0) {
        throw std::runtime_error("Direction array must contain one or more 3-vectors");
    }

    double sx = 0.0;
    double sy = 0.0;
    double sz = 0.0;
    const size_t n = dirs.size() / 3;
    for (size_t i = 0; i < n; ++i) {
        sx += dirs[i * 3 + 0];
        sy += dirs[i * 3 + 1];
        sz += dirs[i * 3 + 2];
    }

    float3 d = make_float3(
        static_cast<float>(sx / static_cast<double>(n)),
        static_cast<float>(sy / static_cast<double>(n)),
        static_cast<float>(sz / static_cast<double>(n))
    );
    const float len = std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
    if (len > 0.0f) {
        d.x /= len;
        d.y /= len;
        d.z /= len;
    }
    return d;
}

static bool rtd_input_audit_enabled() {
    const char* v = std::getenv("RTD_INPUT_AUDIT");
    if (!v) return false;
    const std::string s(v);
    return !(s == "0" || s == "false" || s == "FALSE" || s == "off" || s == "OFF");
}

static bool rtd_pybind_audit_enabled() {
    if (rtd_input_audit_enabled()) return true;
    const char* v = std::getenv("RTD_PYBIND_AUDIT");
    if (!v) return false;
    const std::string s(v);
    return !(s == "0" || s == "false" || s == "FALSE" || s == "off" || s == "OFF");
}

static std::string shape_to_string(const py::buffer_info& info) {
    std::ostringstream os;
    os << "(";
    for (ssize_t i = 0; i < info.ndim; ++i) {
        if (i) os << ",";
        os << info.shape[i];
    }
    os << ")";
    return os.str();
}

static std::string strides_to_string(const py::buffer_info& info) {
    std::ostringstream os;
    os << "(";
    for (ssize_t i = 0; i < info.ndim; ++i) {
        if (i) os << ",";
        os << info.strides[i];
    }
    os << ")";
    return os.str();
}

static void print_float_sample_values(const float* ptr, size_t n, size_t max_values = 6) {
    std::cout << "[";
    const size_t show = std::min(n, max_values);
    for (size_t i = 0; i < show; ++i) {
        if (i) std::cout << ",";
        std::cout << ptr[i];
    }
    if (n > show) {
        std::cout << " ... ";
        const size_t tail = std::min<size_t>(3, n - show);
        for (size_t i = n - tail; i < n; ++i) {
            if (i != n - tail) std::cout << ",";
            std::cout << ptr[i];
        }
    }
    std::cout << "]";
}

static void print_int_sample_values(const int* ptr, size_t n, size_t max_values = 6) {
    std::cout << "[";
    const size_t show = std::min(n, max_values);
    for (size_t i = 0; i < show; ++i) {
        if (i) std::cout << ",";
        std::cout << ptr[i];
    }
    if (n > show) {
        std::cout << " ... ";
        const size_t tail = std::min<size_t>(3, n - show);
        for (size_t i = n - tail; i < n; ++i) {
            if (i != n - tail) std::cout << ",";
            std::cout << ptr[i];
        }
    }
    std::cout << "]";
}

static void print_float_vector_audit(const std::string& name, const std::vector<float>& v) {
    if (!rtd_pybind_audit_enabled()) return;
    int finite = 0;
    int positive = 0;
    int zeros = 0;
    int nan = 0;
    int inf = 0;
    double sum = 0.0;
    float min_v = std::numeric_limits<float>::infinity();
    float max_v = -std::numeric_limits<float>::infinity();
    for (float x : v) {
        if (std::isnan(x)) {
            ++nan;
            continue;
        }
        if (!std::isfinite(x)) {
            ++inf;
            continue;
        }
        ++finite;
        if (x > 0.0f) ++positive;
        if (x == 0.0f) ++zeros;
        sum += static_cast<double>(x);
        min_v = std::min(min_v, x);
        max_v = std::max(max_v, x);
    }
    std::cout << "[PYBIND_AUDIT] " << name
              << " len=" << v.size()
              << " finite=" << finite
              << " nan=" << nan
              << " inf=" << inf
              << " positive=" << positive
              << " zero=" << zeros;
    if (finite > 0) {
        std::cout << " min=" << min_v << " max=" << max_v << " sum=" << sum;
    }
    if (!v.empty()) {
        std::cout << " sample=";
        print_float_sample_values(v.data(), v.size());
    }
    std::cout << std::endl;
}

static void print_int_vector_audit(const std::string& name, const std::vector<int>& v) {
    if (!rtd_pybind_audit_enabled()) return;
    long long sum = 0;
    int positive = 0;
    int zeros = 0;
    int min_v = std::numeric_limits<int>::max();
    int max_v = std::numeric_limits<int>::min();
    for (int x : v) {
        sum += x;
        if (x > 0) ++positive;
        if (x == 0) ++zeros;
        min_v = std::min(min_v, x);
        max_v = std::max(max_v, x);
    }
    std::cout << "[PYBIND_AUDIT] " << name
              << " len=" << v.size()
              << " positive=" << positive
              << " zero=" << zeros;
    if (!v.empty()) {
        std::cout << " min=" << min_v << " max=" << max_v << " sum=" << sum << " sample=";
        print_int_sample_values(v.data(), v.size());
    }
    std::cout << std::endl;
}

static void print_py_array_audit(const std::string& name, const py::array& arr) {
    if (!rtd_pybind_audit_enabled()) return;
    py::buffer_info info = arr.request();
    std::cout << "[PYBIND_AUDIT] " << name
              << " raw dtype=" << py::str(arr.dtype())
              << " ndim=" << info.ndim
              << " shape=" << shape_to_string(info)
              << " strides=" << strides_to_string(info)
              << " itemsize=" << info.itemsize
              << " size=" << info.size
              << " ptr=" << info.ptr
              << std::endl;

    try {
        py::array_t<float, py::array::c_style | py::array::forcecast> cast_arr = py::cast<py::array>(arr);
        py::buffer_info cast_info = cast_arr.request();
        const float* ptr = static_cast<const float*>(cast_info.ptr);
        std::vector<float> v(ptr, ptr + static_cast<size_t>(cast_info.size));
        print_float_vector_audit(name + " cast<float32>", v);
    } catch (const std::exception& e) {
        std::cout << "[PYBIND_AUDIT] " << name << " cast<float32> failed: " << e.what() << std::endl;
    }
}

static void print_weq_contract_audit(const std::string& name, const std::vector<float>& weq) {
    if (!rtd_pybind_audit_enabled()) return;
    std::cout << "[PYBIND_AUDIT] " << name
              << " totalLen=" << weq.size();
    if (weq.size() < 9u) {
        std::cout << " missingHeader=1" << std::endl;
        return;
    }
    const int depth_n = static_cast<int>(std::lround(weq[2]));
    const int y_n = static_cast<int>(std::lround(weq[5]));
    const int x_n = static_cast<int>(std::lround(weq[8]));
    const size_t expected_body =
        (depth_n > 0 && y_n > 0 && x_n > 0)
            ? static_cast<size_t>(depth_n) * static_cast<size_t>(y_n) * static_cast<size_t>(x_n)
            : 0u;
    const size_t actual_body = weq.size() - 9u;
    std::cout << " header=(depth0=" << weq[0]
              << ",depthStep=" << weq[1]
              << ",depthN=" << depth_n
              << ",y0=" << weq[3]
              << ",yStep=" << weq[4]
              << ",yN=" << y_n
              << ",x0=" << weq[6]
              << ",xStep=" << weq[7]
              << ",xN=" << x_n
              << ") expectedBody=" << expected_body
              << " actualBody=" << actual_body
              << " bodyMatches=" << (expected_body == actual_body ? 1 : 0)
              << std::endl;
}

static void print_carbonpbs_input_crosscheck(const RTDBeamSettings& beam,
                                             const std::vector<float>& all_spot_energies,
                                             int lut_layers,
                                             int max_subspots,
                                             int profile_rows,
                                             int profile_depth_n,
                                             int profile_channels) {
    if (!rtd_pybind_audit_enabled()) return;
    const int total_spots = std::accumulate(beam.layerSpotCounts.begin(), beam.layerSpotCounts.end(), 0);
    const int spot_pos_rows = static_cast<int>(beam.spotPositions.size() / 2u);
    const int spot_dir_rows = static_cast<int>(beam.spotBeamDirections.size() / 3u);
    const size_t expected_subspot =
        beam.energies.size() * static_cast<size_t>(std::max(max_subspots, 0)) * 5ull;

    std::cout << "[PYBIND_AUDIT] CarbonPBS crosscheck"
              << " layers=" << beam.energies.size()
              << " layerSpotCounts=" << beam.layerSpotCounts.size()
              << " totalSpots=" << total_spots
              << " allEnergies=" << all_spot_energies.size()
              << " nPar=" << beam.spotWeights.size()
              << " idbeamxyRows=" << spot_pos_rows
              << " beamDirRows=" << spot_dir_rows
              << " lutLayers=" << lut_layers
              << " maxSubspots=" << max_subspots
              << " subspotExpectedAfterMap=" << expected_subspot
              << " subspotActual=" << beam.subspotData.size()
              << " profileRawShape=(" << profile_rows << "," << profile_depth_n << "," << profile_channels << ")"
              << " profileData=" << beam.profileData.size()
              << " profileSetting=" << beam.profileSetting.size()
              << " beamParaRows=" << (beam.beamParaData.size() / 3ull)
              << " cutoffs=" << beam.layerLongitudinalCutoffs.size()
              << " roiLinear=" << beam.roiLinearIndices.size()
              << std::endl;

    if (total_spots != static_cast<int>(beam.spotWeights.size())) {
        std::cout << "[PYBIND_AUDIT] WARNING sum(layerInfo)=" << total_spots
                  << " but nPar.size=" << beam.spotWeights.size() << std::endl;
    }
    if (!all_spot_energies.empty() && all_spot_energies.size() != beam.spotWeights.size()) {
        std::cout << "[PYBIND_AUDIT] WARNING all_energies.size=" << all_spot_energies.size()
                  << " but nPar.size=" << beam.spotWeights.size() << std::endl;
    }
    if (spot_pos_rows != static_cast<int>(beam.spotWeights.size())) {
        std::cout << "[PYBIND_AUDIT] WARNING idbeamxy rows=" << spot_pos_rows
                  << " but nPar.size=" << beam.spotWeights.size() << std::endl;
    }
    if (spot_dir_rows != static_cast<int>(beam.spotWeights.size())) {
        std::cout << "[PYBIND_AUDIT] WARNING tmpBeamDir rows=" << spot_dir_rows
                  << " but nPar.size=" << beam.spotWeights.size() << std::endl;
    }
    if (expected_subspot != beam.subspotData.size()) {
        std::cout << "[PYBIND_AUDIT] WARNING subspot data size expected after mapping="
                  << expected_subspot << " actual=" << beam.subspotData.size() << std::endl;
    }
}

static std::vector<float> spacing_values_or_throw(const py::object& obj,
                                                  const std::string& name,
                                                  size_t total_spots) {
    if (obj.is_none()) {
        throw std::runtime_error(name + " is required when parsing explicit spot spacing");
    }
    std::vector<float> values = to_float_vector(py::cast<py::array>(obj));
    print_float_vector_audit(name + " converted", values);
    if (values.size() != 1u && values.size() != total_spots) {
        throw std::runtime_error(
            name + " length must be 1 or total spot count; got length " +
            std::to_string(values.size()) + " totalSpots=" + std::to_string(total_spots)
        );
    }
    for (float v : values) {
        if (!std::isfinite(v)) {
            throw std::runtime_error(name + " contains NaN/Inf");
        }
        if (!(v > 0.0f)) {
            throw std::runtime_error(name + " must contain only positive physical PB spacing values");
        }
    }
    if (values.size() == 1u && total_spots > 1u) {
        values.assign(total_spots, values.front());
    }
    return values;
}

static float require_uniform_layer_spacing(const std::vector<float>& values,
                                           int offset,
                                           int count,
                                           int layer,
                                           const std::string& name) {
    if (count <= 0) {
        throw std::runtime_error(name + " cannot derive layer spacing from an empty layer " + std::to_string(layer));
    }
    const float ref = values[static_cast<size_t>(offset)];
    float min_v = ref;
    float max_v = ref;
    const float abs_tol = 1.0e-3f;
    const float rel_tol = 1.0e-3f;
    for (int i = 0; i < count; ++i) {
        const float v = values[static_cast<size_t>(offset + i)];
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
    }
    const float tol = std::max(abs_tol, std::fabs(ref) * rel_tol);
    if ((max_v - min_v) > tol) {
        std::ostringstream oss;
        oss << name << " is not uniform within layer=" << layer
            << " count=" << count
            << " min=" << min_v
            << " max=" << max_v
            << " tol=" << tol
            << " sample=[";
        const int sample_n = std::min(count, 6);
        for (int i = 0; i < sample_n; ++i) {
            if (i) oss << ",";
            oss << values[static_cast<size_t>(offset + i)];
        }
        oss << "]";
        throw std::runtime_error(oss.str());
    }
    return ref;
}

static void apply_explicit_spot_spacing_if_present(RTDBeamSettings& beam,
                                                   const py::object& spotSpacingX_obj,
                                                   const py::object& spotSpacingZ_obj,
                                                   const char* tag) {
    const bool has_x = !spotSpacingX_obj.is_none();
    const bool has_z = !spotSpacingZ_obj.is_none();
    if (!has_x && !has_z) return;
    if (has_x != has_z) {
        throw std::runtime_error("spotSpacingX and spotSpacingZ must be provided together");
    }

    const int total_spots = std::accumulate(beam.layerSpotCounts.begin(), beam.layerSpotCounts.end(), 0);
    if (total_spots <= 0) {
        throw std::runtime_error("spot spacing cannot be applied before layerSpotCounts define a positive total spot count");
    }

    if (beam.layerSpotCounts.empty()) {
        throw std::runtime_error("spot spacing cannot be applied before layerSpotCounts is populated");
    }
    const std::vector<float> x_values =
        spacing_values_or_throw(spotSpacingX_obj, "spotSpacingX", static_cast<size_t>(total_spots));
    const std::vector<float> z_values =
        spacing_values_or_throw(spotSpacingZ_obj, "spotSpacingZ", static_cast<size_t>(total_spots));

    beam.layerSpotDeltas.clear();
    beam.layerSpotDeltas.reserve(beam.layerSpotCounts.size());
    int offset = 0;
    for (size_t layer = 0; layer < beam.layerSpotCounts.size(); ++layer) {
        const int count = beam.layerSpotCounts[layer];
        const float dx = require_uniform_layer_spacing(x_values, offset, count, static_cast<int>(layer), "spotSpacingX");
        const float dz = require_uniform_layer_spacing(z_values, offset, count, static_cast<int>(layer), "spotSpacingZ");
        if (beam.raySpacing.x > 0.0f && dx < beam.raySpacing.x * 1.5f) {
            throw std::runtime_error(
                "spotSpacingX is too small for nuclear halo spacing at layer=" +
                std::to_string(layer) + ": value=" + std::to_string(dx) +
                " raySpacing.x=" + std::to_string(beam.raySpacing.x) +
                " required>=1.5*raySpacing"
            );
        }
        if (beam.raySpacing.y > 0.0f && dz < beam.raySpacing.y * 1.5f) {
            throw std::runtime_error(
                "spotSpacingZ is too small for nuclear halo spacing at layer=" +
                std::to_string(layer) + ": value=" + std::to_string(dz) +
                " raySpacing.y=" + std::to_string(beam.raySpacing.y) +
                " required>=1.5*raySpacing"
            );
        }
        beam.layerSpotDeltas.push_back(make_float2(dx, dz));
        offset += count;
    }

    if (!beam.layerSpotDeltas.empty()) {
        beam.spotDelta = make_float3(beam.layerSpotDeltas.front().x, beam.layerSpotDeltas.front().y, 0.0f);
    }
    if (rtd_input_audit_enabled() || rtd_pybind_audit_enabled()) {
        std::cout << "[INPUT_AUDIT][" << tag << "] explicit physical PB spacing"
                  << " spotDelta=(" << beam.spotDelta.x << "," << beam.spotDelta.y << "," << beam.spotDelta.z << ")"
                  << " layerDeltas=" << beam.layerSpotDeltas.size();
        if (!beam.layerSpotDeltas.empty()) {
            const float2 first = beam.layerSpotDeltas.front();
            const float2 last = beam.layerSpotDeltas.back();
            std::cout << " first=(" << first.x << "," << first.y << ")"
                      << " last=(" << last.x << "," << last.y << ")";
        }
        std::cout
                  << " source=(spotSpacingX,spotSpacingZ)"
                  << " totalSpots=" << total_spots
                  << " raySpacing=(" << beam.raySpacing.x << "," << beam.raySpacing.y << ")"
                  << std::endl;
    }
}

enum class SliceMode {
    Head,
    Tail
};

static SliceMode parse_slice_mode_env(const char* env_name, SliceMode fallback) {
    const char* v = std::getenv(env_name);
    if (!v || !*v) return fallback;
    std::string s(v);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    if (s == "head" || s == "prefix") return SliceMode::Head;
    if (s == "tail" || s == "suffix") return SliceMode::Tail;
    return fallback;
}

static const char* slice_mode_name(SliceMode mode) {
    return mode == SliceMode::Head ? "head" : "tail";
}

static std::vector<float> head_slice_energies(const std::vector<float>& full_energies, int exported_rows) {
    if (exported_rows <= 0 || full_energies.empty()) return {};
    const size_t keep = std::min(full_energies.size(), static_cast<size_t>(exported_rows));
    return std::vector<float>(full_energies.begin(), full_energies.begin() + static_cast<std::ptrdiff_t>(keep));
}

static std::vector<float> tail_slice_energies(const std::vector<float>& full_energies, int exported_rows) {
    if (exported_rows <= 0 || full_energies.empty()) return {};
    const size_t keep = std::min(full_energies.size(), static_cast<size_t>(exported_rows));
    return std::vector<float>(full_energies.end() - static_cast<std::ptrdiff_t>(keep), full_energies.end());
}

static std::vector<float> select_slice_energies(const std::vector<float>& full_energies, int exported_rows, SliceMode mode) {
    return mode == SliceMode::Head ? head_slice_energies(full_energies, exported_rows)
                                   : tail_slice_energies(full_energies, exported_rows);
}

static float find_fractional_index_monotonic_host(const std::vector<float>& vec, float value) {
    const int n = static_cast<int>(vec.size());
    if (n <= 1) return 0.0f;
    const bool increasing = vec.back() >= vec.front();
    if (increasing) {
        if (value <= vec.front()) return 0.0f;
        if (value >= vec.back()) return static_cast<float>(n - 1);
        for (int i = 1; i < n; ++i) {
            if (value <= vec[i]) {
                const float denom = vec[i] - vec[i - 1];
                const float t = (std::fabs(denom) > 1.0e-12f) ? ((value - vec[i - 1]) / denom) : 0.0f;
                return static_cast<float>(i - 1) + t;
            }
        }
    } else {
        if (value >= vec.front()) return 0.0f;
        if (value <= vec.back()) return static_cast<float>(n - 1);
        for (int i = 1; i < n; ++i) {
            if (value >= vec[i]) {
                const float denom = vec[i] - vec[i - 1];
                const float t = (std::fabs(denom) > 1.0e-12f) ? ((value - vec[i - 1]) / denom) : 0.0f;
                return static_cast<float>(i - 1) + t;
            }
        }
    }
    return static_cast<float>(n - 1);
}

static std::vector<int> representative_indices(int n) {
    if (n <= 0) return {};
    std::vector<int> idx = {0, n / 2, n - 1};
    std::sort(idx.begin(), idx.end());
    idx.erase(std::unique(idx.begin(), idx.end()), idx.end());
    return idx;
}

static void print_sample_window(const std::vector<float>& v, int base, int count) {
    std::cout << "[";
    for (int i = 0; i < count; ++i) {
        const int idx = base + i;
        if (idx < 0 || idx >= static_cast<int>(v.size())) {
            std::cout << "nan";
        } else {
            std::cout << v[static_cast<size_t>(idx)];
        }
        if (i + 1 != count) std::cout << ", ";
    }
    std::cout << "]";
}

static void print_energy_audit(const char* tag, const RTDBeamSettings& beam, const RTDEnergyStruct& energy) {
    if (!rtd_input_audit_enabled()) return;
    std::cout << "[INPUT_AUDIT][" << tag << "] Energy Axes" << std::endl;
    const std::vector<int> layer_sel = representative_indices(static_cast<int>(beam.energies.size()));
    for (int idx : layer_sel) {
        const float layer_energy = beam.energies[static_cast<size_t>(idx)];
        const float energy_idx = energy.energiesPerU.empty() ? 0.0f : find_fractional_index_monotonic_host(energy.energiesPerU, layer_energy);
        const float profile_row_idx = beam.profileEnergies.empty() ? 0.0f : find_fractional_index_monotonic_host(beam.profileEnergies, layer_energy);
        std::cout << "  layer=" << idx
                  << " energy=" << layer_energy
                  << " energyIdx=" << energy_idx
                  << " profileRowIdx=" << profile_row_idx
                  << " longitudalCutoff="
                  << ((idx < static_cast<int>(beam.layerLongitudinalCutoffs.size())) ? beam.layerLongitudinalCutoffs[static_cast<size_t>(idx)] : 0.0f)
                  << std::endl;
    }
    const std::vector<int> energy_sel = representative_indices(static_cast<int>(energy.energiesPerU.size()));
    for (int idx : energy_sel) {
        std::cout << "  energiesPerU[" << idx << "]=" << energy.energiesPerU[static_cast<size_t>(idx)]
                  << " peakDepth=" << (idx < static_cast<int>(energy.peakDepths.size()) ? energy.peakDepths[static_cast<size_t>(idx)] : 0.0f)
                  << " scaleFact=" << (idx < static_cast<int>(energy.scaleFacts.size()) ? energy.scaleFacts[static_cast<size_t>(idx)] : 0.0f)
                  << std::endl;
    }
}

static void print_cidd_audit(const char* tag, const RTDEnergyStruct& energy) {
    if (!rtd_input_audit_enabled()) return;
    if (energy.nEnergies <= 0 || energy.nEnergySamples <= 0 || energy.ciddMatrix.empty()) return;
    std::cout << "[INPUT_AUDIT][" << tag << "] CIDD Rows" << std::endl;
    const std::vector<int> rows = representative_indices(energy.nEnergies);
    for (int row : rows) {
        const size_t base = static_cast<size_t>(row) * static_cast<size_t>(energy.nEnergySamples);
        std::vector<float> row_data(energy.ciddMatrix.begin() + static_cast<std::ptrdiff_t>(base),
                                    energy.ciddMatrix.begin() + static_cast<std::ptrdiff_t>(base + energy.nEnergySamples));
        int peak_idx = 0;
        float max_v = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < energy.nEnergySamples; ++i) {
            if (row_data[static_cast<size_t>(i)] > max_v) {
                max_v = row_data[static_cast<size_t>(i)];
                peak_idx = i;
            }
        }
        std::cout << "  row=" << row
                  << " energy=" << (row < static_cast<int>(energy.energiesPerU.size()) ? energy.energiesPerU[static_cast<size_t>(row)] : 0.0f)
                  << " peakDepth=" << (row < static_cast<int>(energy.peakDepths.size()) ? energy.peakDepths[static_cast<size_t>(row)] : 0.0f)
                  << " cidd_head=";
        print_sample_window(row_data, 0, std::min(8, energy.nEnergySamples));
        std::cout << " cidd_peak=";
        print_sample_window(row_data, std::max(0, peak_idx - 4), std::min(8, energy.nEnergySamples));
        std::cout << " cidd_tail=";
        print_sample_window(row_data, std::max(0, energy.nEnergySamples - 8), std::min(8, energy.nEnergySamples));
        std::cout << std::endl;
    }
}

static void print_profile_audit(const char* tag, const RTDBeamSettings& beam) {
    if (!rtd_input_audit_enabled()) return;
    if (beam.profileSetting.size() < 3 || beam.profileData.empty()) return;
    const int depth_n = std::max(1, static_cast<int>(std::lround(beam.profileSetting[2])));
    int rows = 0;
    if (!beam.profileEnergies.empty()) {
        rows = static_cast<int>(beam.profileEnergies.size());
    } else if (!beam.beamParaData.empty() && (beam.beamParaData.size() % 3) == 0) {
        rows = static_cast<int>(beam.beamParaData.size() / 3);
    } else if (depth_n > 0) {
        rows = static_cast<int>(beam.profileData.size() / static_cast<size_t>(depth_n));
    }
    const int channels = (rows > 0 && depth_n > 0)
        ? static_cast<int>(beam.profileData.size() / (static_cast<size_t>(rows) * static_cast<size_t>(depth_n)))
        : 0;
    if (rows <= 0 || channels <= 0) return;
    const int n_gauss = std::max(0, (channels - 1) / 2);
    std::cout << "[INPUT_AUDIT][" << tag << "] Profile/BeamPara"
              << " rows=" << rows
              << " depthN=" << depth_n
              << " channels=" << channels
              << " profileDepth0=" << beam.profileSetting[0]
              << " profileDepthStep=" << beam.profileSetting[1]
              << std::endl;
    const std::vector<int> layer_sel = representative_indices(static_cast<int>(beam.energies.size()));
    for (int layer : layer_sel) {
        const float layer_energy = beam.energies[static_cast<size_t>(layer)];
        const float profile_row_idx = beam.profileEnergies.empty() ? 0.0f : find_fractional_index_monotonic_host(beam.profileEnergies, layer_energy);
        const int row = std::max(0, std::min(rows - 1, static_cast<int>(std::lround(profile_row_idx))));
        const std::vector<int> depth_sel = representative_indices(depth_n);
        std::cout << "  layer=" << layer
                  << " energy=" << layer_energy
                  << " profileRowIdx=" << profile_row_idx
                  << " beamPara=("
                  << (row * 3 + 0 < static_cast<int>(beam.beamParaData.size()) ? beam.beamParaData[static_cast<size_t>(row * 3 + 0)] : 0.0f) << ","
                  << (row * 3 + 1 < static_cast<int>(beam.beamParaData.size()) ? beam.beamParaData[static_cast<size_t>(row * 3 + 1)] : 0.0f) << ","
                  << (row * 3 + 2 < static_cast<int>(beam.beamParaData.size()) ? beam.beamParaData[static_cast<size_t>(row * 3 + 2)] : 0.0f) << ")"
                  << std::endl;
        for (int depth : depth_sel) {
            const size_t base = (static_cast<size_t>(row) * static_cast<size_t>(depth_n) + static_cast<size_t>(depth)) * static_cast<size_t>(channels);
            std::cout << "    depth=" << depth << " weights=";
            print_sample_window(beam.profileData, static_cast<int>(base), std::min(5, n_gauss));
            std::cout << " sigmas=";
            print_sample_window(beam.profileData, static_cast<int>(base) + n_gauss, std::min(5, n_gauss));
            std::cout << std::endl;
        }
    }
}

static float interpolate_peak_depth_from_reference(const EnergyStruct& ref, float energy) {
    if (ref.energiesPerU.empty() || ref.peakDepths.empty()) return 0.0f;
    const int n = std::min(static_cast<int>(ref.energiesPerU.size()), static_cast<int>(ref.peakDepths.size()));
    if (n <= 0) return 0.0f;
    if (n == 1) return ref.peakDepths[0];

    const bool increasing = ref.energiesPerU[n - 1] >= ref.energiesPerU[0];
    if (increasing) {
        if (energy <= ref.energiesPerU[0]) return ref.peakDepths[0];
        if (energy >= ref.energiesPerU[n - 1]) return ref.peakDepths[n - 1];
        for (int i = 1; i < n; ++i) {
            if (energy <= ref.energiesPerU[i]) {
                const float e0 = ref.energiesPerU[i - 1];
                const float e1 = ref.energiesPerU[i];
                const float t = (std::fabs(e1 - e0) > 1.0e-12f) ? ((energy - e0) / (e1 - e0)) : 0.0f;
                return ref.peakDepths[i - 1] + (ref.peakDepths[i] - ref.peakDepths[i - 1]) * t;
            }
        }
    } else {
        if (energy >= ref.energiesPerU[0]) return ref.peakDepths[0];
        if (energy <= ref.energiesPerU[n - 1]) return ref.peakDepths[n - 1];
        for (int i = 1; i < n; ++i) {
            if (energy >= ref.energiesPerU[i]) {
                const float e0 = ref.energiesPerU[i - 1];
                const float e1 = ref.energiesPerU[i];
                const float t = (std::fabs(e1 - e0) > 1.0e-12f) ? ((energy - e0) / (e1 - e0)) : 0.0f;
                return ref.peakDepths[i - 1] + (ref.peakDepths[i] - ref.peakDepths[i - 1]) * t;
            }
        }
    }
    return ref.peakDepths[n - 1];
}

static std::vector<float> derive_peak_depths_from_idd(
    const std::vector<float>& idd,
    int n_energies,
    int n_energy_samples,
    float start,
    float step
) {
    std::vector<float> peak_depths(static_cast<size_t>(std::max(0, n_energies)), 0.0f);
    if (n_energies <= 0 || n_energy_samples <= 0 || !(step > 0.0f)) {
        return peak_depths;
    }

    for (int e = 0; e < n_energies; ++e) {
        const size_t row_base = static_cast<size_t>(e) * static_cast<size_t>(n_energy_samples);
        int best_idx = 0;
        float best_val = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < n_energy_samples; ++j) {
            const float v = idd[row_base + static_cast<size_t>(j)];
            if (std::isfinite(v) && v > best_val) {
                best_val = v;
                best_idx = j;
            }
        }
        peak_depths[static_cast<size_t>(e)] = start + step * static_cast<float>(best_idx);
    }

    return peak_depths;
}

static RTDEnergyStruct build_energy_from_carbonpbs(
    const py::handle& enelist_obj,
    const py::handle& idddata_obj,
    const py::handle& iddsetting_obj,
    const std::string& tables_dir
) {
    std::cerr << "[DEBUG][cuFinalDose] build_energy_from_carbonpbs entry" << std::endl;
    RTDEnergyStruct out;
    const SliceMode idd_slice_mode = parse_slice_mode_env("RTD_IDD_SLICE_MODE", SliceMode::Tail);
    out.energiesPerU = to_float_vector(enelist_obj);
    std::cerr << "[DEBUG][cuFinalDose] build_energy_from_carbonpbs enelist.size=" << out.energiesPerU.size() << std::endl;
    if (out.energiesPerU.empty()) {
        throw std::runtime_error("enelist must not be empty");
    }

    const std::vector<float> iddSetting = to_float_vector(iddsetting_obj);
    std::cerr << "[DEBUG][cuFinalDose] build_energy_from_carbonpbs iddsetting.size=" << iddSetting.size() << std::endl;
    if (iddSetting.size() < 3) {
        throw std::runtime_error("iddsetting must provide [start, step, nSamples]");
    }

    py::array_t<float, py::array::forcecast> idd_arr = py::cast<py::array>(idddata_obj);
    py::buffer_info idd_info = idd_arr.request();
    std::cerr << "[DEBUG][cuFinalDose] build_energy_from_carbonpbs idddata ndim=" << idd_info.ndim
              << " shape=(" << (idd_info.ndim > 0 ? idd_info.shape[0] : -1)
              << "," << (idd_info.ndim > 1 ? idd_info.shape[1] : -1) << ")"
              << " size=" << idd_info.size << std::endl;
    if (idd_info.ndim != 2) {
        throw std::runtime_error("idddata must be a 2D array with shape (n_energies, n_samples)");
    }

    const int full_rows = static_cast<int>(out.energiesPerU.size());
    const int rows = static_cast<int>(idd_info.shape[0]);
    const int cols = static_cast<int>(idd_info.shape[1]);
    std::cerr << "[DEBUG][cuFinalDose] build_energy_from_carbonpbs full_rows=" << full_rows
              << " rows=" << rows << " cols=" << cols << std::endl;
    if (rows <= 0 || rows > full_rows) {
        throw std::runtime_error("idddata rows must be in the range [1, enelist length]");
    }

    out.energiesPerU = (rows == full_rows)
        ? out.energiesPerU
        : select_slice_energies(out.energiesPerU, rows, idd_slice_mode);
    out.nEnergies = rows;
    out.nEnergySamples = cols;
    std::vector<float> idd(static_cast<size_t>(idd_info.size));
    {
        const float* ptr = static_cast<const float*>(idd_info.ptr);
        std::copy(ptr, ptr + idd.size(), idd.begin());
    }

    const float startRaw = iddSetting[0];
    const float stepRaw = iddSetting[1];
    const int logicalCols = static_cast<int>(std::lround(iddSetting[2]));
    std::cerr << "[DEBUG][cuFinalDose] build_energy_from_carbonpbs startRaw=" << startRaw
              << " stepRaw=" << stepRaw << " logicalCols=" << logicalCols << std::endl;
    const bool compactedDepthAxis = cols > 0 && logicalCols > cols;
    float sliceDepthScale = 1.0f;
    if (compactedDepthAxis) {
        sliceDepthScale = static_cast<float>(logicalCols) / static_cast<float>(cols);
    }
    float iddDepthUnitToMm = 1.0f;
    const float observedDepthSpanRaw =
        (cols > 1) ? ((cols - 1) * stepRaw * sliceDepthScale) : (stepRaw * sliceDepthScale);
    const float maxEnergy = *std::max_element(out.energiesPerU.begin(), out.energiesPerU.end());
    if (!compactedDepthAxis && stepRaw > 0.0f && stepRaw < 1.0f && observedDepthSpanRaw < 20.0f) {
        iddDepthUnitToMm = 10.0f;
    } else if (!compactedDepthAxis && maxEnergy >= 250.0f && observedDepthSpanRaw > 20.0f && observedDepthSpanRaw <= 100.0f) {
        iddDepthUnitToMm = 10.0f;
    }
    const float start = startRaw * iddDepthUnitToMm;
    const float step = stepRaw * sliceDepthScale * iddDepthUnitToMm;

    out.scaleFacts.assign(static_cast<size_t>(out.nEnergies), (step > 0.0f) ? (1.0f / step) : 1.0f);
    out.ciddMatrix.assign(static_cast<size_t>(out.nEnergies) * out.nEnergySamples, 0.0f);
    for (int e = 0; e < out.nEnergies; ++e) {
        float accum = 0.0f;
        for (int j = 0; j < out.nEnergySamples; ++j) {
            accum += idd[static_cast<size_t>(e) * out.nEnergySamples + j] * step;
            out.ciddMatrix[static_cast<size_t>(e) * out.nEnergySamples + j] = accum;
        }
    }

    std::cerr << "[DEBUG][cuFinalDose] build_energy_from_carbonpbs calling energyReader(" << tables_dir << ")" << std::endl;
    EnergyStruct ref = energyReader(tables_dir);
    std::cerr << "[DEBUG][cuFinalDose] build_energy_from_carbonpbs energyReader returned nEnergySamples=" << ref.nEnergySamples
              << " nEnergies=" << ref.nEnergies << std::endl;
    out.nDensitySamples = ref.nDensitySamples;
    out.densityScaleFact = ref.densityScaleFact;
    out.densityVector = ref.densityVector;
    out.nSpSamples = ref.nSpSamples;
    out.spScaleFact = ref.spScaleFact;
    out.spVector = ref.spVector;
    out.nRRlSamples = ref.nRRlSamples;
    out.rRlScaleFact = ref.rRlScaleFact;
    out.rRlVector = ref.rRlVector;
    out.peakDepths = derive_peak_depths_from_idd(idd, out.nEnergies, out.nEnergySamples, start, step);
    std::cerr << "[DEBUG][cuFinalDose] build_energy_from_carbonpbs derived peakDepths size=" << out.peakDepths.size() << std::endl;
    if (std::none_of(out.peakDepths.begin(), out.peakDepths.end(), [](float v) { return v > 0.0f; })) {
        out.peakDepths.assign(static_cast<size_t>(out.nEnergies), 0.0f);
        for (int e = 0; e < out.nEnergies; ++e) {
            const float energy_per_u = out.energiesPerU[static_cast<size_t>(e)];
            out.peakDepths[static_cast<size_t>(e)] = interpolate_peak_depth_from_reference(ref, energy_per_u);
        }
    }
    maybe_fill_nuclear_tables(out, nullptr, tables_dir, &ref);
    if (rtd_input_audit_enabled()) {
        std::cout << "[INPUT_AUDIT][PYBIND] iddSliceMode=" << slice_mode_name(idd_slice_mode)
                  << " rows=" << rows
                  << " cols=" << cols
                  << " rawStep=" << stepRaw
                  << " logicalCols=" << logicalCols
                  << " effectiveStep=" << step
                  << std::endl;
        print_cidd_audit("PYBIND", out);
    }
    return out;
}

static std::vector<float2> derive_layer_sigmas_from_subspots(
    const std::vector<float>& subspot_flat,
    int n_layers,
    int max_subspots
) {
    std::vector<float2> out(static_cast<size_t>(n_layers), make_float2(0.0f, 0.0f));
    for (int layer = 0; layer < n_layers; ++layer) {
        const size_t base = (static_cast<size_t>(layer) * max_subspots) * 5;
        if (base + 4 >= subspot_flat.size()) {
            break;
        }
        out[static_cast<size_t>(layer)] = make_float2(
            subspot_flat[base + 3],
            subspot_flat[base + 4]
        );
    }
    return out;
}

static std::vector<float> derive_layer_longitudinal_cutoffs(
    const std::vector<float>& per_spot_cutoffs,
    const std::vector<int>& layer_spot_counts
) {
    std::vector<float> out;
    out.reserve(layer_spot_counts.size());
    size_t offset = 0;
    for (size_t layer = 0; layer < layer_spot_counts.size(); ++layer) {
        const int count = std::max(0, layer_spot_counts[layer]);
        float layer_cutoff = 0.0f;
        if (count > 0 && offset < per_spot_cutoffs.size()) {
            layer_cutoff = per_spot_cutoffs[offset];
        }
        for (int i = 0; i < count; ++i) {
            const size_t idx = offset + static_cast<size_t>(i);
            if (idx >= per_spot_cutoffs.size()) break;
            if (i == 0) {
                layer_cutoff = per_spot_cutoffs[idx];
            } else if (std::fabs(per_spot_cutoffs[idx] - layer_cutoff) > 1.0e-3f) {
                throw std::runtime_error(
                    "longitudal_cutoff varies within layer " + std::to_string(layer) +
                    ": first=" + std::to_string(layer_cutoff) +
                    ", current=" + std::to_string(per_spot_cutoffs[idx]) +
                    ", spot_offset=" + std::to_string(i) +
                    ". Current RTD compatibility path only supports one longitudinal cutoff per layer."
                );
            }
        }
        out.push_back(layer_cutoff);
        offset += static_cast<size_t>(count);
    }
    return out;
}

static float find_ordered_float_index(const std::vector<float>& xs, float x) {
    if (xs.empty()) return 0.0f;
    if (xs.size() == 1) return 0.0f;

    const bool ascending = xs.front() <= xs.back();
    const float xmin = ascending ? xs.front() : xs.back();
    const float xmax = ascending ? xs.back() : xs.front();
    if (x <= xmin) return ascending ? 0.0f : static_cast<float>(xs.size() - 1);
    if (x >= xmax) return ascending ? static_cast<float>(xs.size() - 1) : 0.0f;

    for (size_t i = 0; i + 1 < xs.size(); ++i) {
        const float a = xs[i];
        const float b = xs[i + 1];
        const float lo = std::min(a, b);
        const float hi = std::max(a, b);
        if (x < lo || x > hi) continue;
        const float denom = b - a;
        const float t = (std::fabs(denom) > 1e-6f) ? ((x - a) / denom) : 0.0f;
        return static_cast<float>(i) + std::min(1.0f, std::max(0.0f, t));
    }

    return ascending ? static_cast<float>(xs.size() - 1) : 0.0f;
}

static void remap_subspot_lut_to_actual_layers(
    const std::vector<float>& lut_energies,
    const std::vector<float>& lut_subspot_data,
    const std::vector<float2>& lut_spot_sigmas,
    int lut_layer_count,
    int max_subspots_per_layer,
    const std::vector<float>& actual_layer_energies,
    std::vector<float>& out_subspot_data,
    std::vector<float2>& out_spot_sigmas
) {
    out_subspot_data.assign(static_cast<size_t>(actual_layer_energies.size()) * max_subspots_per_layer * 5ull, 0.0f);
    out_spot_sigmas.assign(actual_layer_energies.size(), make_float2(0.0f, 0.0f));
    if (lut_layer_count <= 0 || max_subspots_per_layer <= 0 || lut_energies.empty()) {
        return;
    }

    for (size_t layer = 0; layer < actual_layer_energies.size(); ++layer) {
        const float lut_idx = find_ordered_float_index(lut_energies, actual_layer_energies[layer]);
        float idx_int = 0.0f;
        const float frac = std::modf(lut_idx, &idx_int);
        const int i0 = std::max(0, std::min(lut_layer_count - 1, static_cast<int>(idx_int)));
        const int i1 = std::max(0, std::min(lut_layer_count - 1, i0 + ((frac > 0.0f) ? 1 : 0)));

        for (int s = 0; s < max_subspots_per_layer; ++s) {
            for (int c = 0; c < 5; ++c) {
                const size_t base0 = (static_cast<size_t>(i0) * max_subspots_per_layer + static_cast<size_t>(s)) * 5ull + static_cast<size_t>(c);
                const size_t base1 = (static_cast<size_t>(i1) * max_subspots_per_layer + static_cast<size_t>(s)) * 5ull + static_cast<size_t>(c);
                out_subspot_data[(layer * max_subspots_per_layer + static_cast<size_t>(s)) * 5ull + static_cast<size_t>(c)] =
                    lut_subspot_data[base0] + (lut_subspot_data[base1] - lut_subspot_data[base0]) * frac;
            }
        }

        const float2 s0 = lut_spot_sigmas[std::min(i0, static_cast<int>(lut_spot_sigmas.size()) - 1)];
        const float2 s1 = lut_spot_sigmas[std::min(i1, static_cast<int>(lut_spot_sigmas.size()) - 1)];
        out_spot_sigmas[layer] = make_float2(
            s0.x + (s1.x - s0.x) * frac,
            s0.y + (s1.y - s0.y) * frac
        );
    }
}

static void maybe_fill_density_sp_rrl(RTDEnergyStruct& out, const py::dict& energy, const std::string& tables_dir) {
    const bool has_density = energy.contains("density_vector") && energy.contains("density_scale_fact") && energy.contains("n_density_samples");
    const bool has_sp = energy.contains("sp_vector") && energy.contains("sp_scale_fact") && energy.contains("n_sp_samples");
    const bool has_rrl = energy.contains("rrl_vector") && energy.contains("rrl_scale_fact") && energy.contains("n_rrl_samples");

    if (has_density && has_sp && has_rrl) {
        out.nDensitySamples = energy["n_density_samples"].cast<int>();
        out.densityScaleFact = energy["density_scale_fact"].cast<float>();
        out.densityVector = to_float_vector(energy["density_vector"]);

        out.nSpSamples = energy["n_sp_samples"].cast<int>();
        out.spScaleFact = energy["sp_scale_fact"].cast<float>();
        out.spVector = to_float_vector(energy["sp_vector"]);

        out.nRRlSamples = energy["n_rrl_samples"].cast<int>();
        out.rRlScaleFact = energy["rrl_scale_fact"].cast<float>();
        out.rRlVector = to_float_vector(energy["rrl_vector"]);

        return;
    }

    // Fall back to reference tables.
    // NOTE: This only fills density/SP/RRL; energies/IDD must still be provided by caller.
    EnergyStruct ref = energyReader(tables_dir);

    out.nDensitySamples = ref.nDensitySamples;
    out.densityScaleFact = ref.densityScaleFact;
    out.densityVector = ref.densityVector;

    out.nSpSamples = ref.nSpSamples;
    out.spScaleFact = ref.spScaleFact;
    out.spVector = ref.spVector;

    out.nRRlSamples = ref.nRRlSamples;
    out.rRlScaleFact = ref.rRlScaleFact;
    out.rRlVector = ref.rRlVector;
}

static void maybe_fill_nuclear_tables(RTDEnergyStruct& out,
                                      const py::dict* energy,
                                      const std::string& tables_dir,
                                      const EnergyStruct* referenceTables = nullptr) {
    if (energy != nullptr) {
        const bool has_weight = energy->contains("nuclear_weight_matrix");
        const bool has_sigma = energy->contains("nuclear_sq_sigma_matrix");
        if (has_weight || has_sigma) {
            if (!(has_weight && has_sigma)) {
                throw std::runtime_error("nuclear_weight_matrix and nuclear_sq_sigma_matrix must be provided together");
            }

            out.nucWeightMatrix = to_float_vector((*energy)["nuclear_weight_matrix"]);
            out.nucSqSigmaMatrix = to_float_vector((*energy)["nuclear_sq_sigma_matrix"]);
            out.nNucEnergySamples = energy->contains("n_nuclear_energy_samples")
                ? (*energy)["n_nuclear_energy_samples"].cast<int>()
                : out.nEnergySamples;
            out.nNucEnergies = energy->contains("n_nuclear_energies")
                ? (*energy)["n_nuclear_energies"].cast<int>()
                : out.nEnergies;
            out.nucEnergiesPerU = energy->contains("nuclear_energies_per_u")
                ? to_float_vector((*energy)["nuclear_energies_per_u"])
                : out.energiesPerU;
            out.nucPeakDepths = energy->contains("nuclear_peak_depths")
                ? to_float_vector((*energy)["nuclear_peak_depths"])
                : out.peakDepths;
            out.nucScaleFacts = energy->contains("nuclear_scale_facts")
                ? to_float_vector((*energy)["nuclear_scale_facts"])
                : out.scaleFacts;
            out.nuclearTablesResampledFromReference = energy->contains("nuclear_tables_resampled_from_reference")
                ? (*energy)["nuclear_tables_resampled_from_reference"].cast<bool>()
                : false;
            out.nuclearTablesAlignedToPrimaryAxis = energy->contains("nuclear_tables_aligned_to_primary_axis")
                ? (*energy)["nuclear_tables_aligned_to_primary_axis"].cast<bool>()
                : rtd::nuclear::nuclearPayloadMatchesPrimaryAxis(out);
            return;
        }
    }

    const EnergyStruct ref = (referenceTables != nullptr) ? (*referenceTables) : energyReader(tables_dir);
    rtd::nuclear::alignNuclearTablesToPrimaryAxis(ref, out, "pybind nuclear LUT alignment");
}

struct CarbonPBSDoseContext {
    std::array<int, 3> dims = {0, 0, 0};
    std::vector<float> corner;
    std::vector<float> resolution;
    RTDBeamSettings beamSettings;
    RTDEnergyStruct energyData;
};

struct DoseOutputBinding {
    py::array outArray;
    py::array_t<float> nativeDose;
    float* wrapperPtr = nullptr;
    float* outPtr = nullptr;
    std::array<int, 3> dims = {0, 0, 0};
    bool copyToXyz = false;
};

static std::vector<int> to_roi_linear_indices(const py::handle& obj, const std::array<int, 3>& dims) {
    py::array arr = py::cast<py::array>(obj);
    py::buffer_info info = arr.request();
    const int nx = dims[0];
    const int ny = dims[1];
    const int nz = dims[2];
    const int total = nx * ny * nz;

    std::vector<int> out;
    auto xyz_triplets_to_linear = [&](const int* ptr, size_t n_triplets) {
        std::vector<int> linear;
        linear.reserve(n_triplets);
        for (size_t i = 0; i < n_triplets; ++i) {
            const int x = ptr[i * 3 + 0];
            const int y = ptr[i * 3 + 1];
            const int z = ptr[i * 3 + 2];
            linear.push_back(x * ny * nz + y * nz + z);
        }
        return linear;
    };

    if (info.ndim == 1) {
        py::array_t<int, py::array::forcecast> flat_arr = py::cast<py::array>(obj);
        py::buffer_info flat_info = flat_arr.request();
        const int* ptr = static_cast<const int*>(flat_info.ptr);
        if ((flat_info.size % 3) == 0) {
            bool looks_like_xyz_triplets = flat_info.size > 0;
            for (ssize_t i = 0; i < flat_info.size; i += 3) {
                const int x = ptr[i + 0];
                const int y = ptr[i + 1];
                const int z = ptr[i + 2];
                if (x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz) {
                    looks_like_xyz_triplets = false;
                    break;
                }
            }
            if (looks_like_xyz_triplets) {
                return xyz_triplets_to_linear(ptr, static_cast<size_t>(flat_info.size / 3));
            }
        }

        out.reserve(static_cast<size_t>(flat_info.size));
        for (ssize_t i = 0; i < flat_info.size; ++i) {
            const int linear = ptr[i];
            if ((linear < 0 || linear >= total) && rtd_input_audit_enabled()) {
                std::cout << "[INPUT_AUDIT][PYBIND] Warning: roi linear index out of bounds: "
                          << linear << " not in [0," << total << ")" << std::endl;
            }
            out.push_back(linear);
        }
        return out;
    }

    py::array_t<int, py::array::forcecast> roi_arr = py::cast<py::array>(obj);
    py::buffer_info roi_info = roi_arr.request();
    const int* ptr = static_cast<const int*>(roi_info.ptr);
    if (roi_info.ndim != 2) {
        throw std::runtime_error("roi_index must be 1D xyz triplets or 2D (N,3)/(3,N)");
    }

    if (roi_info.shape[1] == 3) {
        return xyz_triplets_to_linear(ptr, static_cast<size_t>(roi_info.shape[0]));
    }

    if (roi_info.shape[0] == 3) {
        const size_t n = static_cast<size_t>(roi_info.shape[1]);
        out.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            const int x = ptr[0 * roi_info.shape[1] + i];
            const int y = ptr[1 * roi_info.shape[1] + i];
            const int z = ptr[2 * roi_info.shape[1] + i];
            out.push_back(x * ny * nz + y * nz + z);
        }
        return out;
    }

    throw std::runtime_error("roi_index must have shape (N,3) or (3,N)");
}

static void zero_numeric_array(py::array& arr) {
    py::buffer_info info = arr.request();
    if (info.ptr == nullptr || info.size == 0) {
        return;
    }

    py::dtype dt = arr.dtype();
    if (dt.is(py::dtype::of<float>())) {
        std::fill_n(static_cast<float*>(info.ptr), static_cast<size_t>(info.size), 0.0f);
        return;
    }
    if (dt.is(py::dtype::of<int>())) {
        std::fill_n(static_cast<int*>(info.ptr), static_cast<size_t>(info.size), 0);
        return;
    }
    if (dt.is(py::dtype::of<uint64_t>())) {
        std::fill_n(static_cast<uint64_t*>(info.ptr), static_cast<size_t>(info.size), static_cast<uint64_t>(0));
        return;
    }

    throw std::runtime_error("Unsupported dtype for zero fill");
}

static bool is_c_contiguous(const py::buffer_info& info) {
    if (info.ndim <= 0) return true;
    ssize_t expected = static_cast<ssize_t>(info.itemsize);
    for (ssize_t axis = info.ndim - 1; axis >= 0; --axis) {
        if (info.shape[axis] > 1 && info.strides[axis] != expected) {
            return false;
        }
        expected *= info.shape[axis];
    }
    return true;
}

static DoseOutputBinding bind_output_dose_array(py::array out_final_dose, const std::array<int, 3>& dims) {
    DoseOutputBinding binding;
    binding.outArray = py::cast<py::array>(out_final_dose);
    binding.dims = dims;

    if (!binding.outArray.dtype().is(py::dtype::of<float>())) {
        throw std::runtime_error("out_final_dose must be a float32 numpy array");
    }

    py::buffer_info info = binding.outArray.request();
    const size_t expected_size = static_cast<size_t>(dims[0]) * dims[1] * dims[2];
    if (static_cast<size_t>(info.size) != expected_size) {
        throw std::runtime_error("out_final_dose size does not match dims");
    }

    binding.outPtr = static_cast<float*>(info.ptr);
    if (info.ndim == 1) {
        binding.wrapperPtr = binding.outPtr;
        return binding;
    }

    if (info.ndim == 3 && is_c_contiguous(info)) {
        if (info.shape[0] == dims[0] && info.shape[1] == dims[1] && info.shape[2] == dims[2]) {
            binding.nativeDose = py::array_t<float>({dims[2], dims[1], dims[0]});
            zero_numeric_array(binding.nativeDose);
            binding.wrapperPtr = static_cast<float*>(binding.nativeDose.request().ptr);
            binding.copyToXyz = true;
            return binding;
        }
        if (info.shape[0] == dims[2] && info.shape[1] == dims[1] && info.shape[2] == dims[0]) {
            binding.wrapperPtr = binding.outPtr;
            return binding;
        }
    }

    throw std::runtime_error(
        "out_final_dose must be either 1D flat, C-contiguous (Nz,Ny,Nx), or C-contiguous (Nx,Ny,Nz)"
    );
}

static void commit_output_dose(DoseOutputBinding& binding) {
    if (!binding.copyToXyz) return;

    py::buffer_info native_info = binding.nativeDose.request();
    const float* src = static_cast<const float*>(native_info.ptr);
    float* dst = binding.outPtr;
    const int nx = binding.dims[0];
    const int ny = binding.dims[1];
    const int nz = binding.dims[2];

    for (int x = 0; x < nx; ++x) {
        for (int y = 0; y < ny; ++y) {
            for (int z = 0; z < nz; ++z) {
                const size_t dst_idx = (static_cast<size_t>(x) * ny + static_cast<size_t>(y)) * nz + static_cast<size_t>(z);
                const size_t src_idx = (static_cast<size_t>(z) * ny + static_cast<size_t>(y)) * nx + static_cast<size_t>(x);
                dst[dst_idx] = src[src_idx];
            }
        }
    }
}

struct FloatBufferSummary {
    double sum = 0.0;
    float max = 0.0f;
    size_t nonZero = 0;
    size_t nanCount = 0;
    size_t infCount = 0;
};

static FloatBufferSummary summarize_float_buffer(const float* data, size_t n) {
    FloatBufferSummary s;
    if (data == nullptr) {
        return s;
    }
    for (size_t i = 0; i < n; ++i) {
        const float v = data[i];
        if (std::isnan(v)) {
            ++s.nanCount;
            continue;
        }
        if (!std::isfinite(v)) {
            ++s.infCount;
            continue;
        }
        s.sum += static_cast<double>(v);
        s.max = std::max(s.max, v);
        if (std::fabs(v) > 1.0e-20f) {
            ++s.nonZero;
        }
    }
    return s;
}

static bool has_positive_spot_weights(const RTDBeamSettings& beam) {
    for (float w : beam.spotWeights) {
        if (w > 0.0f) return true;
    }
    return false;
}

static void require_nonempty_final_dose(const FloatBufferSummary& summary,
                                        const RTDBeamSettings& beam,
                                        const char* entry_name) {
    if (summary.nanCount > 0 || summary.infCount > 0) {
        throw std::runtime_error(std::string(entry_name) + " produced NaN/Inf dose output");
    }
    if (has_positive_spot_weights(beam) && summary.nonZero == 0) {
        throw std::runtime_error(
            std::string(entry_name) +
            " produced an empty dose grid under positive spot weights; wrapper execution likely failed or returned placeholder zeros"
        );
    }
}

static void debug_print_array_info(const std::string& name, const py::array& arr) {
    py::buffer_info info = arr.request();
    std::cout << "[DEBUG][PYBIND] " << name
              << " dtype=" << py::str(arr.dtype())
              << " ndim=" << info.ndim
              << " shape=(";
    for (ssize_t i = 0; i < info.ndim; ++i) {
        if (i > 0) std::cout << ",";
        std::cout << info.shape[i];
    }
    std::cout << ") size=" << info.size
              << " ptr=" << info.ptr;
    if (info.ptr != nullptr && info.size > 0 && info.size <= 16) {
        std::cout << " values=[";
        if (arr.dtype().is(py::dtype::of<float>())) {
            const float* ptr = static_cast<const float*>(info.ptr);
            for (ssize_t i = 0; i < info.size; ++i) {
                if (i) std::cout << ", ";
                std::cout << ptr[i];
            }
        } else if (arr.dtype().is(py::dtype::of<double>())) {
            const double* ptr = static_cast<const double*>(info.ptr);
            for (ssize_t i = 0; i < info.size; ++i) {
                if (i) std::cout << ", ";
                std::cout << ptr[i];
            }
        } else if (arr.dtype().is(py::dtype::of<int>())) {
            const int* ptr = static_cast<const int*>(info.ptr);
            for (ssize_t i = 0; i < info.size; ++i) {
                if (i) std::cout << ", ";
                std::cout << ptr[i];
            }
        }
        std::cout << "]";
    }
    std::cout << std::endl;
}

static CarbonPBSDoseContext build_carbonpbs_context(
    const py::array& rayweq,
    const py::array& roiIdx,
    const py::array& all_energies,
    const py::array& sourcePos,
    const py::array& tmpBeamDir,
    const py::array& bmxdir,
    const py::array& bmydir,
    const py::array& corner,
    const py::array& resolution,
    const py::array& dims,
    py::object longitudalCutoff_obj,
    const py::array& enelist,
    const py::array& idddata,
    const py::array& iddsetting,
    py::object profiledata_obj,
    py::object profilesetting_obj,
    py::object beamparadata_obj,
    const py::array& subspotdata,
    const py::array& layerInfo,
    const py::array& layerEnergy,
    const py::array& idbeamxy,
    const py::array& nPar,
    py::object sadObj,
    py::object beamParaPos_obj,
    py::object spotSpacingX_obj,
    py::object spotSpacingZ_obj,
    std::string tables_dir,
    bool debug = false
) {
    if (debug || rtd_pybind_audit_enabled()) {
        std::cerr << "[DEBUG][cuFinalDose] build_carbonpbs_context entry debug=" << (debug ? 1 : 0) << std::endl;
    }
    if (debug) {
        std::cerr << "[DEBUG][cuFinalDose] build_carbonpbs_context start" << std::endl;
    }
    CarbonPBSDoseContext ctx;
    const SliceMode profile_slice_mode = parse_slice_mode_env("RTD_PROFILE_SLICE_MODE", SliceMode::Tail);

    if (debug) {
        std::cout << "[DEBUG][cuFinalDose] build_carbonpbs_context parse dims/corner/resolution" << std::endl;
    }
    const std::vector<int> dims_vec = to_int_vector(dims);
    print_py_array_audit("dims", dims);
    print_int_vector_audit("dims converted", dims_vec);
    if (dims_vec.size() != 3) {
        throw std::runtime_error("dims must contain exactly 3 integers; got length " + std::to_string(dims_vec.size()));
    }
    ctx.dims = {dims_vec[0], dims_vec[1], dims_vec[2]};
    if (debug) {
        std::cout << "[DEBUG][cuFinalDose] dims=(" << ctx.dims[0] << "," << ctx.dims[1] << "," << ctx.dims[2] << ")" << std::endl;
    }

    ctx.corner = to_float_vector(corner);
    ctx.resolution = to_float_vector(resolution);
    print_py_array_audit("corner", corner);
    print_float_vector_audit("corner converted", ctx.corner);
    print_py_array_audit("resolution", resolution);
    print_float_vector_audit("resolution converted", ctx.resolution);
    if (ctx.corner.size() != 3 || ctx.resolution.size() != 3) {
        throw std::runtime_error(
            "corner and resolution must contain exactly 3 floats; got corner length " +
            std::to_string(ctx.corner.size()) + " resolution length " + std::to_string(ctx.resolution.size())
        );
    }
    if (debug) {
        std::cout << "[DEBUG][cuFinalDose] corner=(" << ctx.corner[0] << "," << ctx.corner[1] << "," << ctx.corner[2] << ")" << std::endl;
        std::cout << "[DEBUG][cuFinalDose] resolution=(" << ctx.resolution[0] << "," << ctx.resolution[1] << "," << ctx.resolution[2] << ")" << std::endl;
    }

    const std::vector<float> all_spot_energies = to_float_vector(all_energies);
    const std::vector<float> lut_energies = to_float_vector(enelist);
    print_float_vector_audit("all_energies converted", all_spot_energies);
    print_float_vector_audit("enelist converted", lut_energies);
    if (debug) {
        std::cout << "[DEBUG][cuFinalDose] build_carbonpbs_context parse rayweq" << std::endl;
    }
    ctx.beamSettings.waterEquivalence = flatten_float_array(rayweq);
    print_py_array_audit("rayweq", rayweq);
    print_weq_contract_audit("rayweq converted", ctx.beamSettings.waterEquivalence);
    if (ctx.beamSettings.waterEquivalence.size() < 9) {
        throw std::runtime_error(
            "rayweq must include the 9-value CarbonPBS header; got flattened length " +
            std::to_string(ctx.beamSettings.waterEquivalence.size())
        );
    }
    ctx.beamSettings.rayWeqHeader.assign(
        ctx.beamSettings.waterEquivalence.begin(),
        ctx.beamSettings.waterEquivalence.begin() + 9
    );
    if (debug) {
        std::cout << "[DEBUG][cuFinalDose] rayWeqHeader=(";
        for (size_t i = 0; i < 9; ++i) {
            if (i) std::cout << ", ";
            std::cout << ctx.beamSettings.rayWeqHeader[i];
        }
        std::cout << ")" << std::endl;
    }
    ctx.beamSettings.raySpacing = make_float2(ctx.beamSettings.rayWeqHeader[7], ctx.beamSettings.rayWeqHeader[4]);
    // spotDelta is the PHYSICAL PB spacing, not the CPB/WEQ spacing. Do not default to
    // raySpacing here — that would produce spotDist=1 and disable nuclear σ compression.
    // Leave as (0,0,0); the wrapper infers spotDelta via buildPhysicalPBLatticeView().
    // Callers with known PB spacing should set beamSettings.spotDelta directly after this call.
    ctx.beamSettings.spotDelta = make_float3(0.0f, 0.0f, 0.0f);
    ctx.beamSettings.steps = std::max(1, static_cast<int>(std::lround(ctx.beamSettings.rayWeqHeader[2])));

    if (debug) {
        std::cout << "[DEBUG][cuFinalDose] build_carbonpbs_context parse ROI/spot/energy data" << std::endl;
    }
    ctx.beamSettings.roiLinearIndices = to_roi_linear_indices(roiIdx, ctx.dims);
    print_py_array_audit("roiIdx", roiIdx);
    print_int_vector_audit("roiIdx linear converted", ctx.beamSettings.roiLinearIndices);
    if (debug) {
        std::cout << "[DEBUG][cuFinalDose] roiLinearIndices.size=" << ctx.beamSettings.roiLinearIndices.size() << std::endl;
        if (!ctx.beamSettings.roiLinearIndices.empty()) {
            std::cout << "[DEBUG][cuFinalDose] roiLinearIndices first=" << ctx.beamSettings.roiLinearIndices.front()
                      << " last=" << ctx.beamSettings.roiLinearIndices.back() << std::endl;
        }
    }
    ctx.beamSettings.energies = to_float_vector(layerEnergy);
    ctx.beamSettings.layerSpotCounts = to_int_vector(layerInfo);
    ctx.beamSettings.spotWeights = to_float_vector(nPar);
    print_float_vector_audit("layerEnergy converted", ctx.beamSettings.energies);
    print_int_vector_audit("layerInfo converted", ctx.beamSettings.layerSpotCounts);
    print_float_vector_audit("nPar converted", ctx.beamSettings.spotWeights);
    if (!all_spot_energies.empty() && all_spot_energies.size() != ctx.beamSettings.spotWeights.size()) {
        throw std::runtime_error(
            "all_energies length must match nPar length; got all_energies=" +
            std::to_string(all_spot_energies.size()) + " nPar=" + std::to_string(ctx.beamSettings.spotWeights.size())
        );
    }
    ctx.beamSettings.spotPositions = to_float2_columns(idbeamxy, ctx.beamSettings.spotWeights.size());
    ctx.beamSettings.spotPositionsAreIndices = true;
    apply_explicit_spot_spacing_if_present(ctx.beamSettings, spotSpacingX_obj, spotSpacingZ_obj, "PYBIND");
    print_py_array_audit("idbeamxy", idbeamxy);
    print_float_vector_audit("idbeamxy converted", ctx.beamSettings.spotPositions);
    if (debug) {
        std::cout << "[DEBUG][cuFinalDose] spotPositions.count=" << (ctx.beamSettings.spotPositions.size() / 2u)
                  << " expected=" << ctx.beamSettings.spotWeights.size() << std::endl;
    }
    ctx.beamSettings.spotBeamDirections = to_float3_columns(tmpBeamDir, ctx.beamSettings.spotWeights.size());
    ctx.beamSettings.beamDirection = normalized_mean_direction(ctx.beamSettings.spotBeamDirections);
    print_py_array_audit("tmpBeamDir", tmpBeamDir);
    print_float_vector_audit("tmpBeamDir converted", ctx.beamSettings.spotBeamDirections);
    if (debug) {
        std::cout << "[DEBUG][cuFinalDose] spotBeamDirections.count=" << (ctx.beamSettings.spotBeamDirections.size() / 3u)
                  << " beamDirection=(" << ctx.beamSettings.beamDirection.x << ","
                  << ctx.beamSettings.beamDirection.y << "," << ctx.beamSettings.beamDirection.z << ")"
                  << std::endl;
    }
    ctx.beamSettings.bmXDirection = to_float3_from_seq(bmxdir);
    ctx.beamSettings.bmYDirection = to_float3_from_seq(bmydir);
    ctx.beamSettings.sourcePosition = first_float3_from_columns(sourcePos);
    print_py_array_audit("bmxdir", bmxdir);
    print_py_array_audit("bmydir", bmydir);
    print_py_array_audit("sourcePos", sourcePos);
    ctx.beamSettings.refPlaneZ = 0.0f;

    if (py::isinstance<py::array>(sadObj)) {
        const std::vector<float> sad_values = to_float_vector(sadObj);
        print_float_vector_audit("sad converted", sad_values);
        if (sad_values.empty()) {
            throw std::runtime_error("sad array must not be empty");
        }
        ctx.beamSettings.sad = sad_values[0];
    } else {
        ctx.beamSettings.sad = py::cast<float>(sadObj);
    }
    ctx.beamSettings.sourceDist = make_float2(0.0f, 0.0f);
    ctx.beamSettings.spotOffset = make_float3(0.0f, 0.0f, 0.0f);
    if (!longitudalCutoff_obj.is_none()) {
        const std::vector<float> per_spot_cutoffs = to_float_vector(py::cast<py::array>(longitudalCutoff_obj));
        print_float_vector_audit("longitudalCutoff converted", per_spot_cutoffs);
        const int total_spots = std::accumulate(
            ctx.beamSettings.layerSpotCounts.begin(),
            ctx.beamSettings.layerSpotCounts.end(),
            0
        );
        if (!per_spot_cutoffs.empty()) {
            if (static_cast<int>(per_spot_cutoffs.size()) != total_spots) {
                throw std::runtime_error(
                    "longitudal_cutoff length must match total spot count; got length " +
                    std::to_string(per_spot_cutoffs.size()) + " total_spots=" + std::to_string(total_spots)
                );
            }
            ctx.beamSettings.layerLongitudinalCutoffs =
                derive_layer_longitudinal_cutoffs(per_spot_cutoffs, ctx.beamSettings.layerSpotCounts);
        }
    }
    int profile_rows = 0;
    int profile_depth_n = 0;
    int profile_channels = 0;
    if (!profiledata_obj.is_none()) {
        py::array_t<float, py::array::forcecast> profile_arr = py::cast<py::array>(profiledata_obj);
        py::buffer_info profile_info = profile_arr.request();
        ctx.beamSettings.profileData = flatten_float_array(py::cast<py::array>(profiledata_obj));
        print_py_array_audit("profiledata", profile_arr);
        print_float_vector_audit("profiledata converted", ctx.beamSettings.profileData);
        if (profile_info.ndim >= 1) profile_rows = static_cast<int>(profile_info.shape[0]);
        if (profile_info.ndim >= 2) profile_depth_n = static_cast<int>(profile_info.shape[1]);
        if (profile_info.ndim >= 3) profile_channels = static_cast<int>(profile_info.shape[2]);
    }
    if (!profilesetting_obj.is_none()) {
        ctx.beamSettings.profileSetting = to_float_vector(py::cast<py::array>(profilesetting_obj));
        print_float_vector_audit("profilesetting converted", ctx.beamSettings.profileSetting);
        if (ctx.beamSettings.profileSetting.size() >= 3u && profile_depth_n > 0) {
            const int logical_depth_n = std::max(0, static_cast<int>(std::lround(ctx.beamSettings.profileSetting[2])));
            if (logical_depth_n > profile_depth_n) {
                ctx.beamSettings.profileSetting[1] *= static_cast<float>(logical_depth_n) /
                                                      static_cast<float>(profile_depth_n);
            }
            ctx.beamSettings.profileSetting[2] = static_cast<float>(profile_depth_n);
        }
    }
    ctx.beamSettings.profileEnergies = lut_energies;
    if (profile_rows > 0 && !lut_energies.empty() && profile_rows != static_cast<int>(lut_energies.size())) {
        ctx.beamSettings.profileEnergies = select_slice_energies(lut_energies, profile_rows, profile_slice_mode);
    }
    if (!beamparadata_obj.is_none()) {
        ctx.beamSettings.beamParaData = flatten_float_array(py::cast<py::array>(beamparadata_obj));
        print_float_vector_audit("beamparadata converted", ctx.beamSettings.beamParaData);
        if (debug) {
            std::cout << "[DEBUG][cuFinalDose] beamParaData.size=" << ctx.beamSettings.beamParaData.size() << std::endl;
        }
    }
    if (!beamParaPos_obj.is_none()) {
        ctx.beamSettings.beamParaPos = py::cast<float>(beamParaPos_obj);
    }

    if (debug) {
        std::cout << "[DEBUG][cuFinalDose] build_carbonpbs_context parse subspotdata" << std::endl;
    }
    py::array_t<float, py::array::c_style | py::array::forcecast> subspot_arr = py::cast<py::array>(subspotdata);
    py::buffer_info subspot_info = subspot_arr.request();
    print_py_array_audit("subspotdata", subspot_arr);
    if (subspot_info.ndim != 3 || subspot_info.shape[2] != 5) {
        throw std::runtime_error(
            "subspotdata must have shape (num_layers, max_subspots_per_layer, 5); got shape " +
            shape_to_string(subspot_info)
        );
    }
    const int lut_layers = static_cast<int>(subspot_info.shape[0]);
    const int max_subspots = static_cast<int>(subspot_info.shape[1]);
    if (static_cast<int>(ctx.beamSettings.layerSpotCounts.size()) != static_cast<int>(ctx.beamSettings.energies.size())) {
        throw std::runtime_error(
            "layerEnergy/layerInfo size mismatch; layerEnergy=" +
            std::to_string(ctx.beamSettings.energies.size()) + " layerInfo=" +
            std::to_string(ctx.beamSettings.layerSpotCounts.size())
        );
    }
    if (debug) {
        std::cout << "[DEBUG][cuFinalDose] lut_layers=" << lut_layers
                  << " max_subspots=" << max_subspots << std::endl;
    }

    const float* subspot_ptr = static_cast<const float*>(subspot_info.ptr);
    ctx.beamSettings.maxSubspotsPerLayer = max_subspots;
    ctx.energyData = build_energy_from_carbonpbs(enelist, idddata, iddsetting, tables_dir);
    if (debug) {
        std::cout << "[DEBUG][cuFinalDose] energyData.nEnergies=" << ctx.energyData.nEnergies
                  << " nEnergySamples=" << ctx.energyData.nEnergySamples << std::endl;
    }
    const std::vector<float> lut_subspot_data(subspot_ptr, subspot_ptr + static_cast<size_t>(subspot_info.size));
    print_float_vector_audit("subspotdata LUT converted", lut_subspot_data);
    const std::vector<float2> lut_spot_sigmas = derive_layer_sigmas_from_subspots(lut_subspot_data, lut_layers, max_subspots);

    if (debug) {
        std::cout << "[DEBUG][cuFinalDose] building subspotData: lut_layers=" << lut_layers
                  << " actual_layers=" << ctx.beamSettings.energies.size()
                  << " max_subspots=" << max_subspots << std::endl;
    }
    if (lut_layers == static_cast<int>(ctx.beamSettings.energies.size())) {
        ctx.beamSettings.subspotData = lut_subspot_data;
        ctx.beamSettings.spotSigmas = lut_spot_sigmas;
    } else {
        remap_subspot_lut_to_actual_layers(
            lut_energies,
            lut_subspot_data,
            lut_spot_sigmas,
            lut_layers,
            max_subspots,
            ctx.beamSettings.energies,
            ctx.beamSettings.subspotData,
            ctx.beamSettings.spotSigmas
        );
        if (debug) {
            std::cout << "[DEBUG][cuFinalDose] remapped subspotData.size=" << ctx.beamSettings.subspotData.size()
                      << " spotSigmas.size=" << ctx.beamSettings.spotSigmas.size() << std::endl;
        }
    }

    if (rtd_input_audit_enabled() || debug) {
        std::cout << "[INPUT_AUDIT][PYBIND] profileSliceMode=" << slice_mode_name(profile_slice_mode)
                  << " profileRowsRaw=" << profile_rows
                  << " profileDepthRaw=" << profile_depth_n
                  << " profileChannelsRaw=" << profile_channels
                  << " profileDataSize=" << ctx.beamSettings.profileData.size()
                  << " beamParaRows=" << (ctx.beamSettings.beamParaData.size() / 3ull)
                  << std::endl;
        print_energy_audit("PYBIND", ctx.beamSettings, ctx.energyData);
        print_profile_audit("PYBIND", ctx.beamSettings);
    }
    print_carbonpbs_input_crosscheck(
        ctx.beamSettings,
        all_spot_energies,
        lut_layers,
        max_subspots,
        profile_rows,
        profile_depth_n,
        profile_channels
    );
    if (debug) {
        std::cout << "[DEBUG][cuFinalDose] build_carbonpbs_context complete" << std::endl;
    }
    return ctx;
}

static py::dict debug_build_carbonpbs_context(
    const py::array& rayweq,
    const py::array& roiIdx,
    const py::array& all_energies,
    const py::array& sourcePos,
    const py::array& tmpBeamDir,
    const py::array& bmxdir,
    const py::array& bmydir,
    const py::array& corner,
    const py::array& resolution,
    const py::array& dims,
    py::object longitudalCutoff_obj,
    const py::array& enelist,
    const py::array& idddata,
    const py::array& iddsetting,
    py::object profiledata_obj,
    py::object profilesetting_obj,
    py::object beamparadata_obj,
    const py::array& subspotdata,
    const py::array& layerInfo,
    const py::array& layerEnergy,
    const py::array& idbeamxy,
    const py::array& nPar,
    py::object sadObj,
    py::object beamParaPos_obj,
    py::object spotSpacingX_obj = py::none(),
    py::object spotSpacingZ_obj = py::none(),
    std::string tables_dir = std::string("tables/")
) {
    CarbonPBSDoseContext ctx = build_carbonpbs_context(
        rayweq,
        roiIdx,
        all_energies,
        sourcePos,
        tmpBeamDir,
        bmxdir,
        bmydir,
        corner,
        resolution,
        dims,
        longitudalCutoff_obj,
        enelist,
        idddata,
        iddsetting,
        profiledata_obj,
        profilesetting_obj,
        beamparadata_obj,
        subspotdata,
        layerInfo,
        layerEnergy,
        idbeamxy,
        nPar,
        sadObj,
        beamParaPos_obj,
        spotSpacingX_obj,
        spotSpacingZ_obj,
        tables_dir,
        true
    );

    const int total_spots = std::accumulate(ctx.beamSettings.layerSpotCounts.begin(), ctx.beamSettings.layerSpotCounts.end(), 0);
    const int pos_count = static_cast<int>(ctx.beamSettings.spotPositions.size() / 2u);
    const int dir_count = static_cast<int>(ctx.beamSettings.spotBeamDirections.size() / 3u);

    py::dict result;
    result["dims"] = py::cast(ctx.dims);
    result["rayWeqHeader"] = py::cast(ctx.beamSettings.rayWeqHeader);
    result["weq_len"] = static_cast<int>(ctx.beamSettings.waterEquivalence.size());
    result["layer_energies"] = py::cast(ctx.beamSettings.energies);
    result["layerSpotCounts"] = py::cast(ctx.beamSettings.layerSpotCounts);
    result["totalSpots"] = total_spots;
    result["spotPositions"] = pos_count;
    result["spotWeights"] = static_cast<int>(ctx.beamSettings.spotWeights.size());
    result["spotDirections"] = dir_count;
    result["spotPositionsAreIndices"] = ctx.beamSettings.spotPositionsAreIndices;
    result["subspotDataSize"] = static_cast<int>(ctx.beamSettings.subspotData.size());
    result["maxSubspotsPerLayer"] = ctx.beamSettings.maxSubspotsPerLayer;
    result["roiLinearIndices"] = static_cast<int>(ctx.beamSettings.roiLinearIndices.size());
    if (!ctx.beamSettings.roiLinearIndices.empty()) {
        int minIdx = std::numeric_limits<int>::max();
        int maxIdx = std::numeric_limits<int>::min();
        for (int idx : ctx.beamSettings.roiLinearIndices) {
            minIdx = std::min(minIdx, idx);
            maxIdx = std::max(maxIdx, idx);
        }
        result["roiLinearMin"] = minIdx;
        result["roiLinearMax"] = maxIdx;
    }
    result["sourcePosition"] = py::cast(std::vector<float>{ctx.beamSettings.sourcePosition.x, ctx.beamSettings.sourcePosition.y, ctx.beamSettings.sourcePosition.z});
    result["sad"] = ctx.beamSettings.sad;
    result["spotDelta"] = py::cast(std::vector<float>{ctx.beamSettings.spotDelta.x, ctx.beamSettings.spotDelta.y, ctx.beamSettings.spotDelta.z});
    result["profileSettingSize"] = static_cast<int>(ctx.beamSettings.profileSetting.size());
    result["beamParaDataSize"] = static_cast<int>(ctx.beamSettings.beamParaData.size());
    result["energyData_nEnergies"] = ctx.energyData.nEnergies;
    result["energyData_nEnergySamples"] = ctx.energyData.nEnergySamples;
    result["energyData_ciddMatrixSize"] = static_cast<int>(ctx.energyData.ciddMatrix.size());
    return result;
}

static py::object run_carbonpbs_final_dose(
    py::array finalDose,
    py::array ct_data,
    py::array rayweq,
    py::array roiIdx,
    py::array all_energies,
    py::array sourcePos,
    py::array tmpBeamDir,
    py::array bmxdir,
    py::array bmydir,
    py::array corner,
    py::array resolution,
    py::array dims,
    py::object longitudalCutoff_obj,
    py::array enelist,
    py::array idddata,
    py::array iddsetting,
    py::object profiledata_obj,
    py::object profilesetting_obj,
    py::object beamparadata_obj,
    py::array subspotdata,
    py::array layerInfo,
    py::array layerEnergy,
    py::array idbeamxy,
    py::array nPar,
    py::object sadObj,
    py::object beamParaPos_obj,
    int gpuId,
    bool nuclear_correction,
    int verbose,
    bool debug,
    py::object spotSpacingX_obj,
    py::object spotSpacingZ_obj,
    std::string tables_dir
) {
    if (debug) {
        std::cout << "[DEBUG][cuFinalDose] entered run_carbonpbs_final_dose" << std::endl;
        debug_print_array_info("finalDose", finalDose);
        debug_print_array_info("ct_data", ct_data);
        debug_print_array_info("rayweq", rayweq);
        debug_print_array_info("roiIdx", roiIdx);
        debug_print_array_info("all_energies", all_energies);
        debug_print_array_info("sourcePos", sourcePos);
        debug_print_array_info("tmpBeamDir", tmpBeamDir);
        debug_print_array_info("bmxdir", bmxdir);
        debug_print_array_info("bmydir", bmydir);
        debug_print_array_info("corner", corner);
        debug_print_array_info("resolution", resolution);
        debug_print_array_info("dims", dims);
        debug_print_array_info("enelist", enelist);
        debug_print_array_info("idddata", idddata);
        debug_print_array_info("iddsetting", iddsetting);
        debug_print_array_info("subspotdata", subspotdata);
        debug_print_array_info("layerInfo", layerInfo);
        debug_print_array_info("layerEnergy", layerEnergy);
        debug_print_array_info("idbeamxy", idbeamxy);
        debug_print_array_info("nPar", nPar);
        std::cout << "[DEBUG][cuFinalDose] gpuId=" << gpuId
                  << " nuclear_correction=" << nuclear_correction
                  << " verbose=" << verbose
                  << " tables_dir=" << tables_dir << std::endl;
    }
    py::array_t<float, py::array::c_style | py::array::forcecast> ct_arr = py::cast<py::array>(ct_data);
    py::buffer_info ct_info = ct_arr.request();

    CarbonPBSDoseContext ctx;
    if (debug) {
        std::cout << "[DEBUG][cuFinalDose] building CarbonPBSDoseContext" << std::endl;
    }
    ctx = build_carbonpbs_context(
        rayweq,
        roiIdx,
        all_energies,
        sourcePos,
        tmpBeamDir,
        bmxdir,
        bmydir,
        corner,
        resolution,
        dims,
        longitudalCutoff_obj,
        enelist,
        idddata,
        iddsetting,
        profiledata_obj,
        profilesetting_obj,
        beamparadata_obj,
        subspotdata,
        layerInfo,
        layerEnergy,
        idbeamxy,
        nPar,
        sadObj,
        beamParaPos_obj,
        spotSpacingX_obj,
        spotSpacingZ_obj,
        tables_dir,
        debug
    );
    if (debug) {
        std::cout << "[DEBUG][cuFinalDose] build_carbonpbs_context succeeded" << std::endl;
        std::cout << "[DEBUG][cuFinalDose] beamSettings summary:" << std::endl;
        std::cout << "  energies=" << ctx.beamSettings.energies.size()
                  << " layerSpotCounts=" << ctx.beamSettings.layerSpotCounts.size()
                  << " totalSpots=" << ctx.beamSettings.spotWeights.size()
                  << " spotPositions=" << (ctx.beamSettings.spotPositions.size() / 2u)
                  << " spotDirections=" << (ctx.beamSettings.spotBeamDirections.size() / 3u)
                  << " maxSubspotsPerLayer=" << ctx.beamSettings.maxSubspotsPerLayer
                  << " roiLinear=" << ctx.beamSettings.roiLinearIndices.size()
                  << " spotPositionsAreIndices=" << (ctx.beamSettings.spotPositionsAreIndices ? 1 : 0)
                  << std::endl;
        std::cout << "  rayWeqHeader=(";
        for (size_t i = 0; i < ctx.beamSettings.rayWeqHeader.size() && i < 9; ++i) {
            if (i) std::cout << ", ";
            std::cout << ctx.beamSettings.rayWeqHeader[i];
        }
        std::cout << ")" << std::endl;
        if (!ctx.beamSettings.roiLinearIndices.empty()) {
            int minIdx = std::numeric_limits<int>::max();
            int maxIdx = std::numeric_limits<int>::min();
            for (int idx : ctx.beamSettings.roiLinearIndices) {
                minIdx = std::min(minIdx, idx);
                maxIdx = std::max(maxIdx, idx);
            }
            std::cout << "  roiLinearIndices range=[" << minIdx << ", " << maxIdx << "]" << std::endl;
        }
        std::cout << "[DEBUG][cuFinalDose] about to bind output dose array" << std::endl;
    }
    DoseOutputBinding output = bind_output_dose_array(finalDose, ctx.dims);
    if (debug) {
        std::cout << "[DEBUG][cuFinalDose] output dose array bound, wrapperPtr=" << output.wrapperPtr
                  << " outPtr=" << output.outPtr << " expected_size="
                  << static_cast<size_t>(ctx.dims[0]) * ctx.dims[1] * ctx.dims[2] << std::endl;
    }

    const size_t expected_size = static_cast<size_t>(ctx.dims[0]) * ctx.dims[1] * ctx.dims[2];
    if (static_cast<size_t>(ct_info.size) != expected_size) {
        throw std::runtime_error("ct_data size does not match dims");
    }
    if (debug) {
        std::cout << "[DEBUG][cuFinalDose] calling subsecondWrapper" << std::endl;
        std::cout << "[DEBUG][cuFinalDose] ct size=" << ct_info.size
                  << " expected_size=" << expected_size
                  << " dims=(" << ctx.dims[0] << "," << ctx.dims[1] << "," << ctx.dims[2] << ")"
                  << std::endl;
    }

    subsecondWrapper(
        static_cast<float*>(ct_info.ptr),
        to_int3(ctx.dims),
        make_float3(ctx.resolution[0], ctx.resolution[1], ctx.resolution[2]),
        make_float3(ctx.corner[0], ctx.corner[1], ctx.corner[2]),
        output.wrapperPtr,
        to_int3(ctx.dims),
        make_float3(ctx.resolution[0], ctx.resolution[1], ctx.resolution[2]),
        make_float3(ctx.corner[0], ctx.corner[1], ctx.corner[2]),
        &ctx.beamSettings,
        1,
        &ctx.energyData,
        gpuId,
        nuclear_correction,
        verbose
    );

    commit_output_dose(output);
    const FloatBufferSummary summary = summarize_float_buffer(output.outPtr, expected_size);
    require_nonempty_final_dose(summary, ctx.beamSettings, "cuFinalDose/calcDose/cuCalDose");

    return finalDose;
}

static py::array_t<float> make_dummy_ct_from_dims(const std::array<int, 3>& dims) {
    const size_t total = static_cast<size_t>(dims[0]) * dims[1] * dims[2];
    py::array_t<float> ct_arr({static_cast<ssize_t>(total)});
    py::buffer_info info = ct_arr.request();
    // Compatibility-only placeholder. The wrapper consumes WEQ for transport when available.
    std::fill_n(static_cast<float*>(info.ptr), static_cast<size_t>(info.size), 1000.0f);
    return ct_arr;
}

static py::array_t<float> make_ct_from_roi_mask(const std::array<int, 3>& dims,
                                                const std::vector<int>& roi_linear_indices) {
    const size_t total = static_cast<size_t>(dims[0]) * dims[1] * dims[2];
    py::array_t<float> ct_arr({static_cast<ssize_t>(total)});
    py::buffer_info info = ct_arr.request();
    float* ptr = static_cast<float*>(info.ptr);
    std::fill_n(ptr, total, 0.0f);
    for (int idx : roi_linear_indices) {
        if (idx >= 0 && static_cast<size_t>(idx) < total) {
            ptr[static_cast<size_t>(idx)] = 1000.0f;
        }
    }
    return ct_arr;
}

// -----------------------------
// Python API
// -----------------------------

/**
 * Python binding for subsecondWrapper (RayTraceDicom GPU dose calculation).
 *
 * Inputs:
 *  - ct: float32/float64 numpy array (flattened or 3D). The wrapper treats it as a flat buffer.
 *  - ct_dims/res/corner, dose_dims/res/corner: geometry.
 *  - beam: dict with keys (minimum recommended):
 *      energies, spot_sigmas, ray_spacing, steps,
 *      max_subspots_per_layer, subspot_data,
 *      beam_direction, beam_xdir, beam_ydir,
 *      sad, source_position, ref_plane_z (optional).
 *  - energy: dict with keys:
 *      energies_per_u, peak_depths, scale_facts, cidd_matrix
 *      plus optionally density/sp/rrl LUTs. If missing, they are loaded from tables_dir.
 */
py::array_t<float> raytracedicom_wrapper_py(
    py::array ct,
    std::array<int, 3> ct_dims,
    std::array<float, 3> ct_resolution,
    std::array<float, 3> ct_corner,
    std::array<int, 3> dose_dims,
    std::array<float, 3> dose_resolution,
    std::array<float, 3> dose_corner,
    py::dict beam,
    py::dict energy,
    int gpu_id,
    bool nuclear_correction,
    int verbose,
    std::string tables_dir
) {
    // Force float32 contiguous.
    if (rtd_pybind_audit_enabled()) {
        std::cout << "[PYBIND_AUDIT] raytracedicom_wrapper entry"
                  << " gpu_id=" << gpu_id
                  << " nuclear_correction=" << (nuclear_correction ? 1 : 0)
                  << " verbose=" << verbose
                  << " tables_dir=" << tables_dir
                  << std::endl;
        print_py_array_audit("ct", py::cast<py::array>(ct));
    }
    py::array_t<float, py::array::c_style | py::array::forcecast> ct_arr = py::cast<py::array>(ct);
    py::buffer_info ct_info = ct_arr.request();

    const size_t ct_size_expected = static_cast<size_t>(ct_dims[0]) * ct_dims[1] * ct_dims[2];
    if (static_cast<size_t>(ct_info.size) != ct_size_expected) {
        throw std::runtime_error(
            "ct buffer size does not match ct_dims; got size " + std::to_string(ct_info.size) +
            " expected " + std::to_string(ct_size_expected)
        );
    }

    // Allocate output dose array with shape (Z,Y,X) for natural linear indexing.
    const size_t dose_size = static_cast<size_t>(dose_dims[0]) * dose_dims[1] * dose_dims[2];
    py::array_t<float> dose_arr({dose_dims[2], dose_dims[1], dose_dims[0]});
    py::buffer_info dose_info = dose_arr.request();
    if (static_cast<size_t>(dose_info.size) != dose_size) {
        throw std::runtime_error("internal error: output size mismatch");
    }

    // Build beam settings
    RTDBeamSettings beamSettings;
    beamSettings.energies = to_float_vector(beam["energies"]);
    beamSettings.spotSigmas = to_float2_vector_from_nx2(beam["spot_sigmas"]);
    print_float_vector_audit("beam.energies converted", beamSettings.energies);
    if (rtd_pybind_audit_enabled()) {
        std::vector<float> sigmas_flat;
        sigmas_flat.reserve(beamSettings.spotSigmas.size() * 2u);
        for (const float2& s : beamSettings.spotSigmas) {
            sigmas_flat.push_back(s.x);
            sigmas_flat.push_back(s.y);
        }
        print_float_vector_audit("beam.spot_sigmas converted", sigmas_flat);
    }

    if (beam.contains("ray_spacing")) {
        beamSettings.raySpacing = to_float2_from_seq(beam["ray_spacing"]);
    } else {
        beamSettings.raySpacing = make_float2(0.0f, 0.0f); // will fall back inside wrapper
    }

    beamSettings.steps = beam["steps"].cast<int>();

    if (beam.contains("source_dist")) {
        beamSettings.sourceDist = to_float2_from_seq(beam["source_dist"]);
    } else {
        beamSettings.sourceDist = make_float2(0.0f, 0.0f); // wrapper fallback: SAD
    }

    if (beam.contains("spot_offset")) {
        beamSettings.spotOffset = to_float3_from_seq(beam["spot_offset"]);
    } else {
        beamSettings.spotOffset = make_float3(0.0f, 0.0f, 0.0f);
    }

    if (beam.contains("spot_delta")) {
        beamSettings.spotDelta = to_float3_from_seq(beam["spot_delta"]);
    } else {
        beamSettings.spotDelta = make_float3(0.0f, 0.0f, 0.0f);
    }

    // CarbonPBS geometry
    beamSettings.beamDirection = to_float3_from_seq(beam["beam_direction"]);
    beamSettings.bmXDirection = to_float3_from_seq(beam["beam_xdir"]);
    beamSettings.bmYDirection = to_float3_from_seq(beam["beam_ydir"]);
    beamSettings.sad = beam["sad"].cast<float>();
    beamSettings.sourcePosition = to_float3_from_seq(beam["source_position"]);
    beamSettings.refPlaneZ = beam.contains("ref_plane_z") ? beam["ref_plane_z"].cast<float>() : 0.0f;
    if (beam.contains("roi_linear_indices")) {
        beamSettings.roiLinearIndices = py::cast<std::vector<int>>(beam["roi_linear_indices"]);
    }
    if (beam.contains("layer_spot_counts")) {
        beamSettings.layerSpotCounts = py::cast<std::vector<int>>(beam["layer_spot_counts"]);
    }
    if (beam.contains("spot_positions")) {
        beamSettings.spotPositions = py::cast<std::vector<float>>(beam["spot_positions"]);
    }
    if (beam.contains("spot_weights")) {
        beamSettings.spotWeights = py::cast<std::vector<float>>(beam["spot_weights"]);
    }
    if (beam.contains("spot_beam_directions")) {
        beamSettings.spotBeamDirections = py::cast<std::vector<float>>(beam["spot_beam_directions"]);
    }
    if (beam.contains("layer_longitudinal_cutoffs")) {
        beamSettings.layerLongitudinalCutoffs = py::cast<std::vector<float>>(beam["layer_longitudinal_cutoffs"]);
    }
    if (beam.contains("water_equivalence")) {
        beamSettings.waterEquivalence = to_float_vector(beam["water_equivalence"]);
        print_float_vector_audit("beam.water_equivalence converted", beamSettings.waterEquivalence);
        print_weq_contract_audit("beam.water_equivalence", beamSettings.waterEquivalence);
        if (beamSettings.waterEquivalence.size() >= 9) {
            beamSettings.rayWeqHeader.assign(
                beamSettings.waterEquivalence.begin(),
                beamSettings.waterEquivalence.begin() + 9
            );
        }
    }
    if (beam.contains("ray_weq_header")) {
        beamSettings.rayWeqHeader = py::cast<std::vector<float>>(beam["ray_weq_header"]);
        print_float_vector_audit("beam.ray_weq_header converted", beamSettings.rayWeqHeader);
        if (beamSettings.waterEquivalence.size() < 9 && beamSettings.rayWeqHeader.size() >= 9) {
            beamSettings.waterEquivalence = beamSettings.rayWeqHeader;
        }
    }
    if (beam.contains("profile_data")) {
        beamSettings.profileData = flatten_float_array(py::cast<py::array>(beam["profile_data"]));
        print_float_vector_audit("beam.profile_data converted", beamSettings.profileData);
    }
    if (beam.contains("profile_energies")) {
        beamSettings.profileEnergies = to_float_vector(py::cast<py::array>(beam["profile_energies"]));
        print_float_vector_audit("beam.profile_energies converted", beamSettings.profileEnergies);
    }
    if (beam.contains("profile_setting")) {
        beamSettings.profileSetting = to_float_vector(py::cast<py::array>(beam["profile_setting"]));
        print_float_vector_audit("beam.profile_setting converted", beamSettings.profileSetting);
    }
    if (beam.contains("beam_para_data")) {
        beamSettings.beamParaData = flatten_float_array(py::cast<py::array>(beam["beam_para_data"]));
        print_float_vector_audit("beam.beam_para_data converted", beamSettings.beamParaData);
    }
    if (beam.contains("beam_para_pos")) {
        beamSettings.beamParaPos = beam["beam_para_pos"].cast<float>();
    }

    // Subspot data
    py::array_t<float, py::array::c_style | py::array::forcecast> subspot = py::cast<py::array>(beam["subspot_data"]);
    py::buffer_info ss_info = subspot.request();
    print_py_array_audit("beam.subspot_data", subspot);
    if (ss_info.ndim != 3 || ss_info.shape[2] != 5) {
        throw std::runtime_error(
            "subspot_data must have shape (num_layers, max_subspots_per_layer, 5); got shape " +
            shape_to_string(ss_info)
        );
    }
    const int num_layers = static_cast<int>(ss_info.shape[0]);
    const int max_subspots = static_cast<int>(ss_info.shape[1]);

    beamSettings.maxSubspotsPerLayer = max_subspots;
    const float* ss_ptr = static_cast<const float*>(ss_info.ptr);
    beamSettings.subspotData.assign(ss_ptr, ss_ptr + static_cast<size_t>(ss_info.size));
    print_float_vector_audit("beam.subspot_data converted", beamSettings.subspotData);

    // Optional: validate energy layer size
    if (static_cast<int>(beamSettings.energies.size()) != num_layers) {
        throw std::runtime_error(
            "beam.energies length must equal subspot_data.shape[0] (num_layers); got energies=" +
            std::to_string(beamSettings.energies.size()) + " num_layers=" + std::to_string(num_layers)
        );
    }
    if (static_cast<int>(beamSettings.spotSigmas.size()) != num_layers) {
        throw std::runtime_error(
            "beam.spot_sigmas length must equal num_layers; got spot_sigmas=" +
            std::to_string(beamSettings.spotSigmas.size()) + " num_layers=" + std::to_string(num_layers)
        );
    }
    if (!beamSettings.layerLongitudinalCutoffs.empty() &&
        static_cast<int>(beamSettings.layerLongitudinalCutoffs.size()) != num_layers) {
        throw std::runtime_error(
            "beam.layer_longitudinal_cutoffs length must equal num_layers; got cutoffs=" +
            std::to_string(beamSettings.layerLongitudinalCutoffs.size()) +
            " num_layers=" + std::to_string(num_layers)
        );
    }

    // Build energy struct
    RTDEnergyStruct energyData;
    energyData.nEnergies = energy["n_energies"].cast<int>();
    energyData.nEnergySamples = energy["n_energy_samples"].cast<int>();

    energyData.energiesPerU = to_float_vector(energy["energies_per_u"]);
    energyData.peakDepths = to_float_vector(energy["peak_depths"]);
    energyData.scaleFacts = to_float_vector(energy["scale_facts"]);
    print_float_vector_audit("energy.energies_per_u converted", energyData.energiesPerU);
    print_float_vector_audit("energy.peak_depths converted", energyData.peakDepths);
    print_float_vector_audit("energy.scale_facts converted", energyData.scaleFacts);

    // cidd_matrix
    py::array_t<float, py::array::c_style | py::array::forcecast> cidd = py::cast<py::array>(energy["cidd_matrix"]);
    py::buffer_info cidd_info = cidd.request();
    print_py_array_audit("energy.cidd_matrix", cidd);
    if (cidd_info.ndim != 2) {
        throw std::runtime_error(
            "cidd_matrix must be 2D (n_energies, n_energy_samples); got shape " +
            shape_to_string(cidd_info)
        );
    }
    if (static_cast<int>(cidd_info.shape[0]) != energyData.nEnergies ||
        static_cast<int>(cidd_info.shape[1]) != energyData.nEnergySamples) {
        throw std::runtime_error(
            "cidd_matrix shape mismatch with n_energies/n_energy_samples; got shape " +
            shape_to_string(cidd_info) + " expected=(" + std::to_string(energyData.nEnergies) +
            "," + std::to_string(energyData.nEnergySamples) + ")"
        );
    }
    const float* cidd_ptr = static_cast<const float*>(cidd_info.ptr);
    energyData.ciddMatrix.assign(cidd_ptr, cidd_ptr + static_cast<size_t>(cidd_info.size));
    print_float_vector_audit("energy.cidd_matrix converted", energyData.ciddMatrix);

    // Density/SP/RRL LUTs
    maybe_fill_density_sp_rrl(energyData, energy, tables_dir);
    maybe_fill_nuclear_tables(energyData, &energy, tables_dir);

    // Call wrapper
    subsecondWrapper(
        static_cast<float*>(ct_info.ptr),
        to_int3(ct_dims),
        to_float3(ct_resolution),
        to_float3(ct_corner),
        static_cast<float*>(dose_info.ptr),
        to_int3(dose_dims),
        to_float3(dose_resolution),
        to_float3(dose_corner),
        &beamSettings,
        1,
        &energyData,
        gpu_id,
        nuclear_correction,
        verbose
    );

    const FloatBufferSummary summary = summarize_float_buffer(static_cast<float*>(dose_info.ptr), dose_size);
    require_nonempty_final_dose(summary, beamSettings, "raytracedicom_wrapper");

    return dose_arr;
}

py::object cu_final_dose_py(
    py::array finalDose,
    py::array rayweq,
    py::array roiIdx,
    py::array all_energies,
    py::array sourcePos,
    py::array tmpBeamDir,
    py::array bmxdir,
    py::array bmydir,
    py::array corner,
    py::array resolution,
    py::array dims,
    py::object longitudalCutoff_obj,
    py::array enelist,
    py::array idddata,
    py::array iddsetting,
    py::object profiledata_obj,
    py::object profilesetting_obj,
    py::object beamparadata_obj,
    py::array subspotdata,
    py::array layerInfo,
    py::array layerEnergy,
    py::array idbeamxy,
    py::array nPar,
    py::object sadObj,
    py::object cutoff_obj,
    py::object beamParaPos_obj,
    int gpuId,
    bool nuclear_correction,
    int verbose,
    bool debug,
    py::object spotSpacingX_obj,
    py::object spotSpacingZ_obj,
    std::string tables_dir = std::string("tables/")
) {
    CPU_TIMER_START_SUMMARY();
    (void)cutoff_obj;  // Kept for CarbonPBS cuFinalDose contract parity; current RTD final-dose path does not consume it.
    const std::vector<int> dims_vec = to_int_vector(dims);
    if (dims_vec.size() != 3) {
        throw std::runtime_error("dims must contain exactly 3 integers");
    }
    const std::array<int, 3> dims_arr = {dims_vec[0], dims_vec[1], dims_vec[2]};
    const std::vector<int> roi_linear = to_roi_linear_indices(roiIdx, dims_arr);
    py::array_t<float> ct_data = roi_linear.empty()
        ? make_dummy_ct_from_dims(dims_arr)
        : make_ct_from_roi_mask(dims_arr, roi_linear);

    py::object result = run_carbonpbs_final_dose(
        finalDose,
        ct_data,
        rayweq,
        roiIdx,
        all_energies,
        sourcePos,
        tmpBeamDir,
        bmxdir,
        bmydir,
        corner,
        resolution,
        dims,
        longitudalCutoff_obj,
        enelist,
        idddata,
        iddsetting,
        profiledata_obj,
        profilesetting_obj,
        beamparadata_obj,
        subspotdata,
        layerInfo,
        layerEnergy,
        idbeamxy,
        nPar,
        sadObj,
        beamParaPos_obj,
        gpuId,
        nuclear_correction,
        verbose,
        debug,
        spotSpacingX_obj,
        spotSpacingZ_obj,
        tables_dir
    );
    CPU_TIMER_END_SUMMARY("cuFinalDose");
    return result;
}

py::object cu_cal_dose3_py(
    py::array out_cscdata,
    py::array out_cscptr,
    py::array out_cscrowind,
    py::array weq_data,
    py::array roi_index,
    py::array source_energies,
    py::array source,
    py::array beam_dir,
    py::array beam_xdir,
    py::array beam_ydir,
    py::array corner,
    py::array resolution,
    py::array dims,
    py::object longitudal_cutoff_obj,
    py::array enelist,
    py::array idddata,
    py::array iddsetting,
    py::object profiledata_obj,
    py::object profilesetting_obj,
    py::object beamparadata_obj,
    py::array subspot_data,
    py::array layer_info,
    py::array layer_energy,
    py::array out_nnz,
    py::array idbeamxy,
    py::array num_particles_per_beam,
    py::object /*cutoff*/,
    py::object sad_obj,
    py::object beam_para_pos_obj,
    py::object /*python_nnz_size*/,
    int gpu_id,
    bool nuclear_correction,
    int verbose,
    std::string tables_dir
) {
    py::array out_cscdata_arr = py::cast<py::array>(out_cscdata);
    py::array out_cscptr_arr = py::cast<py::array>(out_cscptr);
    py::array out_cscrowind_arr = py::cast<py::array>(out_cscrowind);
    py::array out_nnz_arr = py::cast<py::array>(out_nnz);
    zero_numeric_array(out_cscdata_arr);
    zero_numeric_array(out_cscptr_arr);
    zero_numeric_array(out_cscrowind_arr);
    zero_numeric_array(out_nnz_arr);

    const std::vector<int> dims_vec = to_int_vector(dims);
    if (dims_vec.size() != 3) {
        throw std::runtime_error("dims must contain exactly 3 integers");
    }
    const std::array<int, 3> dims_arr = {dims_vec[0], dims_vec[1], dims_vec[2]};
    py::array_t<float> dose_arr({dims_vec[0], dims_vec[1], dims_vec[2]});
    const std::vector<int> roi_linear = to_roi_linear_indices(roi_index, dims_arr);
    py::array_t<float> ct_data = roi_linear.empty()
        ? make_dummy_ct_from_dims(dims_arr)
        : make_ct_from_roi_mask(dims_arr, roi_linear);

    run_carbonpbs_final_dose(
        dose_arr,
        ct_data,
        weq_data,
        roi_index,
        source_energies,
        source,
        beam_dir,
        beam_xdir,
        beam_ydir,
        corner,
        resolution,
        dims,
        longitudal_cutoff_obj,
        enelist,
        idddata,
        iddsetting,
        profiledata_obj,
        profilesetting_obj,
        beamparadata_obj,
        subspot_data,
        layer_info,
        layer_energy,
        idbeamxy,
        num_particles_per_beam,
        sad_obj,
        beam_para_pos_obj,
        gpu_id,
        nuclear_correction,
        verbose,
        false,
        py::none(),
        py::none(),
        tables_dir
    );

    py::dict result;
    result["dose_grid"] = dose_arr;
    result["out_cscdata"] = out_cscdata;
    result["out_cscptr"] = out_cscptr;
    result["out_cscrowind"] = out_cscrowind;
    result["outNNZ"] = out_nnz;
    return result;
}

static const char* compiled_nuclear_mode_name() {
#ifdef NUCLEAR_CORR
#if NUCLEAR_CORR == SOUKUP
    return "SOUKUP";
#elif NUCLEAR_CORR == FLUKA
    return "FLUKA";
#elif NUCLEAR_CORR == GAUSS_FIT
    return "GAUSS_FIT";
#else
    return "UNKNOWN";
#endif
#else
    return "OFF";
#endif
}

static py::dict rtd_support_matrix_py() {
    py::dict result;
    const std::string nuclear_mode = compiled_nuclear_mode_name();
    result["nuclear_corr_compiled"] = nuclear_mode != "OFF";
    result["nuclear_corr_mode"] = nuclear_mode;
    result["runtime_nuclear_correction_arg"] = "nuclear_correction";
    result["pybind_audit_env"] = "RTD_PYBIND_AUDIT";
    result["input_audit_env"] = "RTD_INPUT_AUDIT";
    result["superposition_overflow_debug_env"] = "RTD_SUPERP_OVERFLOW_DEBUG";
    result["perf_profile_env"] = "RTD_PERF_PROFILE";
    result["note"] =
        "NUCLEAR_CORR is a compile-time CMake option. Python can only enable an already-compiled halo path "
        "with nuclear_correction=True; rebuild with -DNUCLEAR_CORR=GAUSS_FIT/SOUKUP/FLUKA to change the macro.";
    return result;
}

PYBIND11_MODULE(cudaCalDoseRTD, m) {
    m.doc() = "RayTraceDicom (CUDA) dose calculation wrapper (pybind11)";

    m.def(
        "rtdSupportMatrix",
        &rtd_support_matrix_py,
        R"pbdoc(
Return compile-time and runtime feature switches for the RTD pybind module.

Use this from Python to confirm whether NUCLEAR_CORR was compiled in and which
environment variables control runtime diagnostics.
)pbdoc"
    );

    m.def(
        "raytracedicom_wrapper",
        &raytracedicom_wrapper_py,
        py::arg("ct"),
        py::arg("ct_dims"),
        py::arg("ct_resolution"),
        py::arg("ct_corner"),
        py::arg("dose_dims"),
        py::arg("dose_resolution"),
        py::arg("dose_corner"),
        py::arg("beam"),
        py::arg("energy"),
        py::arg("gpu_id") = 0,
        py::arg("nuclear_correction") = true,
        py::arg("verbose") = 0,
        py::arg("tables_dir") = std::string("tables/"),
        R"pbdoc(
Wrap subsecondWrapper (RayTraceDicom GPU implementation).

Parameters
----------
verbose : int, optional
    0 = no output, 1 = fine timing (all details), 2 = summary (total time only). Default is 0.

Returns
-------
np.ndarray
    Dose volume as float32 array with shape (Z, Y, X).
)pbdoc"
    );

    m.def(
        "calcDose",
        &cu_final_dose_py,
        py::arg("finalDose"),
        py::arg("rayweq"),
        py::arg("roiIdx"),
        py::arg("all_energies"),
        py::arg("sourcePos"),
        py::arg("tmpBeamDir"),
        py::arg("bmxdir"),
        py::arg("bmydir"),
        py::arg("corner"),
        py::arg("resolution"),
        py::arg("dims"),
        py::arg("longitudalCutoff"),
        py::arg("enelist"),
        py::arg("idddata"),
        py::arg("iddsetting"),
        py::arg("profiledata"),
        py::arg("profilesetting"),
        py::arg("beamparadata"),
        py::arg("subspotdata"),
        py::arg("layerInfo"),
        py::arg("layerEnergy"),
        py::arg("idbeamxy"),
        py::arg("nPar"),
        py::arg("sad"),
        py::arg("cutoff") = py::none(),
        py::arg("beamParaPos") = py::none(),
        py::arg("gpuId") = 0,
        py::arg("nuclear_correction") = true,
        py::arg("verbose") = 0,
        py::arg("debug") = false,
        py::arg("spotSpacingX") = py::none(),
        py::arg("spotSpacingZ") = py::none(),
        py::arg("tables_dir") = std::string("tables/"),
        R"pbdoc(
Compute the full dose distribution in-place into `dose_grid`.

This is the single high-level entry point intended for Python-side integration.
The input arrays carry the CT/plan/beam-model/runtime-derived information.
Returns the same `dose_grid` object after writing the dose values into it.

Parameters
----------
verbose : int, optional
    0 = no output, 1 = fine timing (all details), 2 = summary (total time only). Default is 0.
)pbdoc"
    );

    m.def(
        "cuCalDose3",
        &cu_cal_dose3_py,
        py::arg("out_cscdata"),
        py::arg("out_cscptr"),
        py::arg("out_cscrowind"),
        py::arg("weq_data"),
        py::arg("roi_index"),
        py::arg("source_energies"),
        py::arg("source"),
        py::arg("beam_dir"),
        py::arg("beam_xdir"),
        py::arg("beam_ydir"),
        py::arg("corner"),
        py::arg("resolution"),
        py::arg("dims"),
        py::arg("longitudal_cutoff"),
        py::arg("enelist"),
        py::arg("idddata"),
        py::arg("iddsetting"),
        py::arg("profiledata"),
        py::arg("profilesetting"),
        py::arg("beamparadata"),
        py::arg("subspot_data"),
        py::arg("layer_info"),
        py::arg("layer_energy"),
        py::arg("outNNZ"),
        py::arg("idbeamxy"),
        py::arg("num_particles_per_beam"),
        py::arg("cutoff"),
        py::arg("sad"),
        py::arg("beam_para_pos"),
        py::arg("python_nnz_size"),
        py::arg("gpu_id") = 0,
        py::arg("nuclear_correction") = true,
        py::arg("verbose") = 0,
        py::arg("tables_dir") = std::string("tables/"),
        R"pbdoc(
CarbonPBS-style compatibility wrapper for `cuCalDose3`.

RTD in `patch10_mod` computes final dose directly rather than emitting the
original CarbonPBS CSC spot matrix. The CSC outputs are accepted and zeroed for
interface compatibility, while the returned dict includes `dose_grid` with the
RTD-computed final dose.

Parameters
----------
verbose : int, optional
    0 = no output, 1 = fine timing (all details), 2 = summary (total time only). Default is 0.
)pbdoc"
    );

    m.def(
        "cuCalDose",
        &cu_final_dose_py,
        py::arg("finalDose"),
        py::arg("rayweq"),
        py::arg("roiIdx"),
        py::arg("all_energies"),
        py::arg("sourcePos"),
        py::arg("tmpBeamDir"),
        py::arg("bmxdir"),
        py::arg("bmydir"),
        py::arg("corner"),
        py::arg("resolution"),
        py::arg("dims"),
        py::arg("longitudalCutoff"),
        py::arg("enelist"),
        py::arg("idddata"),
        py::arg("iddsetting"),
        py::arg("profiledata"),
        py::arg("profilesetting"),
        py::arg("beamparadata"),
        py::arg("subspotdata"),
        py::arg("layerInfo"),
        py::arg("layerEnergy"),
        py::arg("idbeamxy"),
        py::arg("nPar"),
        py::arg("sad"),
        py::arg("cutoff") = py::none(),
        py::arg("beamParaPos") = py::none(),
        py::arg("gpuId") = 0,
        py::arg("nuclear_correction") = true,
        py::arg("verbose") = 0,
        py::arg("debug") = false,
        py::arg("spotSpacingX") = py::none(),
        py::arg("spotSpacingZ") = py::none(),
        py::arg("tables_dir") = std::string("tables/"),
        R"pbdoc(
CarbonPBS-style compatibility alias for the wrapper-backed RTD final-dose path.

This entry shares the same implementation as `cuFinalDose` and `calcDose`, so the
public CarbonPBS-style final-dose path stays on one RTD wrapper-backed chain.

Parameters
----------
verbose : int, optional
    0 = no output, 1 = fine timing (all details), 2 = summary (total time only). Default is 0.
)pbdoc"
    );

    m.def(
        "cuFinalDose",
        &cu_final_dose_py,
        py::arg("finalDose"),
        py::arg("rayweq"),
        py::arg("roiIdx"),
        py::arg("all_energies"),
        py::arg("sourcePos"),
        py::arg("tmpBeamDir"),
        py::arg("bmxdir"),
        py::arg("bmydir"),
        py::arg("corner"),
        py::arg("resolution"),
        py::arg("dims"),
        py::arg("longitudalCutoff"),
        py::arg("enelist"),
        py::arg("idddata"),
        py::arg("iddsetting"),
        py::arg("profiledata"),
        py::arg("profilesetting"),
        py::arg("beamparadata"),
        py::arg("subspotdata"),
        py::arg("layerInfo"),
        py::arg("layerEnergy"),
        py::arg("idbeamxy"),
        py::arg("nPar"),
        py::arg("sad"),
        py::arg("cutoff") = py::none(),
        py::arg("beamParaPos") = py::none(),
        py::arg("gpuId") = 0,
        py::arg("nuclear_correction") = true,
        py::arg("verbose") = 0,
        py::arg("debug") = false,
        py::arg("spotSpacingX") = py::none(),
        py::arg("spotSpacingZ") = py::none(),
        py::arg("tables_dir") = std::string("tables/"),
        R"pbdoc(
CarbonPBS-style compatibility wrapper for the RTD final-dose pipeline.

This entry preserves the positional argument contract and keyword names used by
`tps_py/dosecal.py -> cuFinalDose(...)`, while routing the calculation through
the current RTD wrapper-backed final-dose implementation.
)pbdoc"
    );

    m.def(
        "debug_build_carbonpbs_context",
        &debug_build_carbonpbs_context,
        py::arg("rayweq"),
        py::arg("roiIdx"),
        py::arg("all_energies"),
        py::arg("sourcePos"),
        py::arg("tmpBeamDir"),
        py::arg("bmxdir"),
        py::arg("bmydir"),
        py::arg("corner"),
        py::arg("resolution"),
        py::arg("dims"),
        py::arg("longitudalCutoff"),
        py::arg("enelist"),
        py::arg("idddata"),
        py::arg("iddsetting"),
        py::arg("profiledata"),
        py::arg("profilesetting"),
        py::arg("beamparadata"),
        py::arg("subspotdata"),
        py::arg("layerInfo"),
        py::arg("layerEnergy"),
        py::arg("idbeamxy"),
        py::arg("nPar"),
        py::arg("sad"),
        py::arg("beamParaPos") = py::none(),
        py::arg("spotSpacingX") = py::none(),
        py::arg("spotSpacingZ") = py::none(),
        py::arg("tables_dir") = std::string("tables/"),
        R"pbdoc(
Build and inspect the CarbonPBSDoseContext without executing the GPU wrapper.

This helper lets you verify that Python-to-C++ input conversion succeeds and
examine key beam/energy bookkeeping before `subsecondWrapper` is invoked.
)pbdoc"
    );
}
