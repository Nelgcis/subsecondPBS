#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OFF_BUILD_DIR="${REPO_ROOT}/build"
HALO_BUILD_DIR="${REPO_ROOT}/build_halo_audit"
INPUT_DIR=""
REFERENCE_DOSE_BIN=""
OUTPUT_ROOT=""
BEAM_PREFIX=""
REQUIRE_PASS=0
MAX_OFF_BUILD_MAE=""
MAX_OFF_BUILD_RMSE=""
MAX_REF_MAE=""
MAX_REF_RMSE=""
BUILD_JOBS="${RTD_TEST_BUILD_JOBS:-2}"

usage() {
    cat <<'EOF'
Usage:
  tools/run_halo_gate_matrix.sh [options]

Options:
  --input-dir PATH            Fixture case root / dose_inputs_csv / output dir passed to RTD_TEST_INPUT_DIR
  --reference-dose-bin PATH   Explicit RTD_TEST_REFERENCE_DOSE_BIN for gate C
  --output-root PATH          Root directory for OFF / HALO outputs and gate report
  --beam-prefix NAME          Optional RTD_TEST_BEAM_PREFIX override
  --off-build-dir PATH        OFF build directory (default: ./build)
  --halo-build-dir PATH       Halo build directory (default: ./build_halo_audit)
  --require-pass              Export RTD_TEST_HALO_REQUIRE_PASS=1 for the halo run
  --max-off-build-mae VAL     Export RTD_TEST_HALO_MAX_OFF_BUILD_MAE
  --max-off-build-rmse VAL    Export RTD_TEST_HALO_MAX_OFF_BUILD_RMSE
  --max-ref-mae VAL           Export RTD_TEST_HALO_MAX_REF_MAE
  --max-ref-rmse VAL          Export RTD_TEST_HALO_MAX_REF_RMSE
  --build-jobs N              Build parallelism for wrapper_integration_test (default: 2)
  --help                      Show this help

This script runs the halo gate matrix in two steps:
  1. OFF build (`NUCLEAR_CORR=OFF`) -> exports baseline raw dose bin
  2. Halo build (`NUCLEAR_CORR=...`) -> reruns with RTD_TEST_HALO_GATE=1 and
     points RTD_TEST_HALO_OFF_DOSE_BIN at the OFF baseline dose bin

Outputs:
  <output-root>/off/...
  <output-root>/halo/...
  <output-root>/halo_gate_report.txt
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --reference-dose-bin)
            REFERENCE_DOSE_BIN="$2"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --beam-prefix)
            BEAM_PREFIX="$2"
            shift 2
            ;;
        --off-build-dir)
            OFF_BUILD_DIR="$2"
            shift 2
            ;;
        --halo-build-dir)
            HALO_BUILD_DIR="$2"
            shift 2
            ;;
        --require-pass)
            REQUIRE_PASS=1
            shift
            ;;
        --max-off-build-mae)
            MAX_OFF_BUILD_MAE="$2"
            shift 2
            ;;
        --max-off-build-rmse)
            MAX_OFF_BUILD_RMSE="$2"
            shift 2
            ;;
        --max-ref-mae)
            MAX_REF_MAE="$2"
            shift 2
            ;;
        --max-ref-rmse)
            MAX_REF_RMSE="$2"
            shift 2
            ;;
        --build-jobs)
            BUILD_JOBS="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "${OUTPUT_ROOT}" ]]; then
    OUTPUT_ROOT="${REPO_ROOT}/output/halo_gate_matrix_$(date +%Y%m%d_%H%M%S)"
fi

OFF_OUTPUT_DIR="${OUTPUT_ROOT}/off"
HALO_OUTPUT_DIR="${OUTPUT_ROOT}/halo"
REPORT_PATH="${OUTPUT_ROOT}/halo_gate_report.txt"
mkdir -p "${OFF_OUTPUT_DIR}" "${HALO_OUTPUT_DIR}"

ensure_wrapper_test() {
    local build_dir="$1"
    local exe_path="${build_dir}/bin/wrapper_integration_test"
    if [[ ! -x "${exe_path}" ]]; then
        cmake --build "${build_dir}" --target wrapper_integration_test -j "${BUILD_JOBS}"
    fi
    if [[ ! -x "${exe_path}" ]]; then
        echo "Missing wrapper_integration_test under ${build_dir}/bin" >&2
        exit 1
    fi
}

find_single_raw_dose() {
    local dir="$1"
    local pattern="$2"
    mapfile -t matches < <(find "${dir}" -maxdepth 1 -type f -name "${pattern}" | sort)
    if [[ "${#matches[@]}" -eq 0 ]]; then
        echo "No raw dose bin matching ${pattern} under ${dir}" >&2
        exit 1
    fi
    if [[ "${#matches[@]}" -gt 1 ]]; then
        echo "Multiple raw dose bins found under ${dir}:" >&2
        printf '  %s\n' "${matches[@]}" >&2
        echo "Use --beam-prefix to disambiguate." >&2
        exit 1
    fi
    printf '%s\n' "${matches[0]}"
}

run_wrapper_test() {
    local build_dir="$1"
    local output_dir="$2"
    shift 2

    local exe_path="${build_dir}/bin/wrapper_integration_test"
    local -a env_args=()
    env_args+=("RTD_TEST_OUTPUT_DIR=${output_dir}")
    if [[ -n "${INPUT_DIR}" ]]; then
        env_args+=("RTD_TEST_INPUT_DIR=${INPUT_DIR}")
    fi
    if [[ -n "${BEAM_PREFIX}" ]]; then
        env_args+=("RTD_TEST_BEAM_PREFIX=${BEAM_PREFIX}")
    fi
    if [[ -n "${REFERENCE_DOSE_BIN}" ]]; then
        env_args+=("RTD_TEST_REFERENCE_DOSE_BIN=${REFERENCE_DOSE_BIN}")
    fi
    env "${env_args[@]}" "$@" "${exe_path}"
}

echo "[HALO_GATE] output_root=${OUTPUT_ROOT}"
echo "[HALO_GATE] off_build_dir=${OFF_BUILD_DIR}"
echo "[HALO_GATE] halo_build_dir=${HALO_BUILD_DIR}"
if [[ -n "${INPUT_DIR}" ]]; then
    echo "[HALO_GATE] input_dir=${INPUT_DIR}"
fi
if [[ -n "${BEAM_PREFIX}" ]]; then
    echo "[HALO_GATE] beam_prefix=${BEAM_PREFIX}"
fi
if [[ -n "${REFERENCE_DOSE_BIN}" ]]; then
    echo "[HALO_GATE] reference_dose_bin=${REFERENCE_DOSE_BIN}"
fi

ensure_wrapper_test "${OFF_BUILD_DIR}"
ensure_wrapper_test "${HALO_BUILD_DIR}"

echo "[HALO_GATE] step 1/2: running OFF baseline"
run_wrapper_test "${OFF_BUILD_DIR}" "${OFF_OUTPUT_DIR}"

RAW_PATTERN="*_final_dose.bin"
if [[ -n "${BEAM_PREFIX}" ]]; then
    RAW_PATTERN="${BEAM_PREFIX}_final_dose.bin"
fi
OFF_DOSE_BIN="$(find_single_raw_dose "${OFF_OUTPUT_DIR}" "${RAW_PATTERN}")"
echo "[HALO_GATE] off_build_dose_bin=${OFF_DOSE_BIN}"

HALO_ENV=(
    "RTD_TEST_HALO_GATE=1"
    "RTD_TEST_HALO_OFF_DOSE_BIN=${OFF_DOSE_BIN}"
    "RTD_TEST_HALO_REPORT_PATH=${REPORT_PATH}"
)
if [[ "${REQUIRE_PASS}" -eq 1 ]]; then
    HALO_ENV+=("RTD_TEST_HALO_REQUIRE_PASS=1")
fi
if [[ -n "${MAX_OFF_BUILD_MAE}" ]]; then
    HALO_ENV+=("RTD_TEST_HALO_MAX_OFF_BUILD_MAE=${MAX_OFF_BUILD_MAE}")
fi
if [[ -n "${MAX_OFF_BUILD_RMSE}" ]]; then
    HALO_ENV+=("RTD_TEST_HALO_MAX_OFF_BUILD_RMSE=${MAX_OFF_BUILD_RMSE}")
fi
if [[ -n "${MAX_REF_MAE}" ]]; then
    HALO_ENV+=("RTD_TEST_HALO_MAX_REF_MAE=${MAX_REF_MAE}")
fi
if [[ -n "${MAX_REF_RMSE}" ]]; then
    HALO_ENV+=("RTD_TEST_HALO_MAX_REF_RMSE=${MAX_REF_RMSE}")
fi

echo "[HALO_GATE] step 2/2: running halo gate"
run_wrapper_test "${HALO_BUILD_DIR}" "${HALO_OUTPUT_DIR}" "${HALO_ENV[@]}"

echo "[HALO_GATE] halo_output_dir=${HALO_OUTPUT_DIR}"
echo "[HALO_GATE] halo_gate_report=${REPORT_PATH}"
if [[ -f "${REPORT_PATH}" ]]; then
    echo "[HALO_GATE] gate summary:"
    grep -E '^(compiled_halo_mode|gate_ab_status|gate_c_status|tps_halo_enablement|gate_ab_note|gate_c_note)=' "${REPORT_PATH}" || true
fi
