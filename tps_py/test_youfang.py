import time
import os
import csv
import pickle

import numpy as np

from content.dose_operator.multiBeamGroups import MBG
try:
    from content.tools.cudaPKG.cudaCalDoseRTD import rtdSupportMatrix
except Exception:
    rtdSupportMatrix = None

config_dict = dict()
config_dict["gpu_id"] = 0

mbg_obj = MBG(False)
mbg_obj.load_model(config_dict)

import json
with open("config.json") as f:
    content_dict = json.load(f)
content_dict['task_id'] = '123'
start = time.time()
mbg_obj.process(content_dict=content_dict)

MAX_FULL_ELEMENTS = 200000
MAX_ROWS_SUBSPOT = 200000
MAX_NONZERO_ELEMENTS = 200000


def _parse_rayweq_header(water_equivalence):
    arr = np.array(water_equivalence).reshape(-1)
    if arr.size < 9:
        return None
    return {
        "rayweq_n_step": int(arr[2]),
        "rayweq_y0": float(arr[3]),
        "rayweq_dy": float(arr[4]),
        "rayweq_ny": int(arr[5]),
        "rayweq_x0": float(arr[6]),
        "rayweq_dx": float(arr[7]),
        "rayweq_nx": int(arr[8]),
    }


def dump_required_meta_csv(mbg_obj, output_dir="dose_inputs_csv"):
    if not hasattr(mbg_obj, "plan") or not hasattr(mbg_obj.plan, "plan_info"):
        return
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "calc_required_meta.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "beam_group_id",
            "beam_id",
            "num_voxels_x",
            "num_voxels_y",
            "num_voxels_z",
            "resolution_x",
            "resolution_y",
            "resolution_z",
            "corner_x",
            "corner_y",
            "corner_z",
            "nFrac",
            "ext_contour_linear_count",
            "ext_contour_linear_opt_count",
            "source_pos_x",
            "source_pos_y",
            "source_pos_z",
            "sad",
            "rayweq_n_step",
            "rayweq_y0",
            "rayweq_dy",
            "rayweq_ny",
            "rayweq_x0",
            "rayweq_dx",
            "rayweq_nx",
            "idbeamxy_semantics",
        ])
        for bg_id, plan_info in mbg_obj.plan.plan_info.items():
            dose_grid = plan_info.get("dose_grid", {})
            dims = np.array(dose_grid.get("num_voxels", []), dtype=np.int64)
            resolution = np.array(dose_grid.get("resolution", []), dtype=np.float32)
            corner = np.array(dose_grid.get("corner", []), dtype=np.float32)
            nfrac = int(plan_info.get("prescription_config", {}).get("nFrac", 1))
            ext_linear = plan_info.get("ext_contour_linear")
            if ext_linear is None and hasattr(mbg_obj, "ext_contour_linear"):
                ext_linear = mbg_obj.ext_contour_linear
            ext_linear_opt = plan_info.get("ext_contour_linear_opt")
            if ext_linear_opt is None and hasattr(mbg_obj, "ext_contour_linear_opt"):
                ext_linear_opt = mbg_obj.ext_contour_linear_opt
            if dims.size != 3 and hasattr(mbg_obj, "dc") and hasattr(mbg_obj.dc, "doseGrid"):
                dims = np.array(mbg_obj.dc.doseGrid.dims, dtype=np.int64)
            if resolution.size != 3 and hasattr(mbg_obj, "dc") and hasattr(mbg_obj.dc, "doseGrid"):
                resolution = np.array(mbg_obj.dc.doseGrid.resolution).reshape(-1)
            if corner.size != 3 and hasattr(mbg_obj, "dc") and hasattr(mbg_obj.dc, "doseGrid"):
                corner = np.array(mbg_obj.dc.doseGrid.corner).reshape(-1)
            beams = plan_info.get("beams", {})
            for beam_id, beam in beams.items():
                source_pos = np.array(beam.get("source_pos", [np.nan, np.nan, np.nan]), dtype=np.float32).reshape(-1)
                sad = beam.get("sad", "")
                rayweq_meta = _parse_rayweq_header(beam.get("water_equivalence", []))
                writer.writerow([
                    bg_id,
                    beam_id,
                    int(dims[0]),
                    int(dims[1]),
                    int(dims[2]),
                    float(resolution[0]),
                    float(resolution[1]),
                    float(resolution[2]),
                    float(corner[0]),
                    float(corner[1]),
                    float(corner[2]),
                    nfrac,
                    int(np.size(ext_linear)) if ext_linear is not None else 0,
                    int(np.size(ext_linear_opt)) if ext_linear_opt is not None else 0,
                    float(source_pos[0]) if source_pos.size > 0 else np.nan,
                    float(source_pos[1]) if source_pos.size > 1 else np.nan,
                    float(source_pos[2]) if source_pos.size > 2 else np.nan,
                    sad,
                    rayweq_meta["rayweq_n_step"] if rayweq_meta else "",
                    rayweq_meta["rayweq_y0"] if rayweq_meta else "",
                    rayweq_meta["rayweq_dy"] if rayweq_meta else "",
                    rayweq_meta["rayweq_ny"] if rayweq_meta else "",
                    rayweq_meta["rayweq_x0"] if rayweq_meta else "",
                    rayweq_meta["rayweq_dx"] if rayweq_meta else "",
                    rayweq_meta["rayweq_nx"] if rayweq_meta else "",
                    "idbeamxy is rayweq texture index-like coordinate, not direct gantry-plane physical coordinate",
                ])


def dump_variable_name_meta_csv(output_dir="dose_inputs_csv"):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "variable_name_meta.csv")
    rows = [
        ["export_name", "original_variable", "note"],
        ["water_equivalence", "beam['water_equivalence']", "ray lattice header is in first 9 values"],
        ["idbeamxy", "beam['idbeamxy']", "index-like coordinate on rayweq lattice"],
        ["number_particle", "beam['number_particle']", "npermu * weight_vector"],
        ["layer_info", "beam['layer_info']", "spot count per layer"],
        ["beam_dir", "beam['beam_dir']", "direction vector per spot (flattened)"],
        ["profile_setting", "beam['profile_setting']", "profile depth setting, not equivalent to ray spacing"],
        ["subspot_data", "beam['subspot_data']", "subspot parameter table by energy layer"],
        ["spot_energy_sigmaxy", "derived: [beam['all_energies'], interp(beam['subspot_data'][:,0,3/4] by beam['energy_list'])]", "per-spot [energy, sigmaX, sigmaY] for pybind input"],
        ["spot_sigmax", "derived from spot_energy_sigmaxy[:,1]", "per-spot sigmaX aligned to beam['all_energies']"],
        ["spot_sigmay", "derived from spot_energy_sigmaxy[:,2]", "per-spot sigmaY aligned to beam['all_energies']"],
        ["final_dose", "beam['final_dose']", "3D dose grid; numpy shape=(Nx,Ny,Nz); raw .bin uses NumPy C-order flat index ((x*Ny)+y)*Nz+z"],
        ["ext_contour_linear", "plan/body ROI linear indices", "dose calculation ROI linear index"],
        ["ext_contour_linear_opt", "plan/body opt ROI linear indices", "optimization ROI linear index"],
        ["ctgrid_data", "mbg_obj.ct.ctgrid.data", "CT grid data after makeCTGrid (SPR in CT resolution)"],
        ["dosegrid_data", "mbg_obj.dc.doseGrid.data", "Dose grid data used for WEQ tracing (resampled grid)"],
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def dump_subspot_csv(mbg_obj, output_dir="subspot_csv"):
    if not hasattr(mbg_obj, "plan") or not hasattr(mbg_obj.plan, "plan_info"):
        return
    os.makedirs(output_dir, exist_ok=True)
    for bg_id, plan_info in mbg_obj.plan.plan_info.items():
        beams = plan_info.get("beams", {})
        for beam_id, beam in beams.items():
            subspot_data = beam.get("subspot_data")
            if subspot_data is None:
                continue
            enelist = beam.get("energy_list")
            if enelist is None:
                enelist = beam.get("energy")
            if enelist is not None:
                enelist = np.array(enelist)
            csv_path = os.path.join(output_dir, f"subspot_bg{bg_id}_beam{beam_id}.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["#meta", "export_name", "subspot_data"])
                writer.writerow(["#meta", "original_variable", "beam['subspot_data']"])
                writer.writerow(["#meta", "shape", str(tuple(subspot_data.shape))])
                writer.writerow([
                    "beam_group_id",
                    "beam_id",
                    "energy",
                    "energy_idx",
                    "subspot_idx",
                    "deltax",
                    "deltay",
                    "weight",
                    "sigmax",
                    "sigmay",
                ])
                n_ene, n_sub, _ = subspot_data.shape
                rows_written = 0
                need_slice = (n_ene * n_sub) > MAX_ROWS_SUBSPOT
                if need_slice:
                    writer.writerow(["#meta", "export_mode", f"slice_first_{MAX_ROWS_SUBSPOT}_rows"])
                for ene_idx in range(n_ene):
                    energy_val = ""
                    if enelist is not None and enelist.size == n_ene:
                        energy_val = float(enelist[ene_idx])
                    for sub_idx in range(n_sub):
                        if need_slice and rows_written >= MAX_ROWS_SUBSPOT:
                            break
                        deltax, deltay, weight, sigmax, sigmay = subspot_data[ene_idx, sub_idx].tolist()
                        writer.writerow([
                            bg_id,
                            beam_id,
                            energy_val,
                            ene_idx,
                            sub_idx,
                            deltax,
                            deltay,
                            weight,
                            sigmax,
                            sigmay,
                        ])
                        rows_written += 1
                    if need_slice and rows_written >= MAX_ROWS_SUBSPOT:
                        break


dump_subspot_csv(mbg_obj)


def _write_array_csv(array, csv_path, header, nonzero_only=False, export_name="", original_variable=""):
    arr = np.array(array)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        if export_name:
            writer.writerow(["#meta", "export_name", export_name])
        if original_variable:
            writer.writerow(["#meta", "original_variable", original_variable])
        writer.writerow(["#meta", "shape", str(tuple(arr.shape))])
        writer.writerow(["#meta", "ndim", int(arr.ndim)])
        writer.writerow(["#meta", "size", int(arr.size)])
        if nonzero_only:
            writer.writerow(["#meta", "export_mode", "nonzero_only"])
        elif arr.size > MAX_FULL_ELEMENTS:
            writer.writerow(["#meta", "export_mode", "slice"])
        else:
            writer.writerow(["#meta", "export_mode", "full"])
        writer.writerow(header)
        if nonzero_only:
            if arr.ndim <= 3:
                nz = np.argwhere(arr != 0)
                writer.writerow(["#meta", "nonzero_count", int(nz.shape[0])])
                if nz.shape[0] > MAX_NONZERO_ELEMENTS:
                    writer.writerow(["#meta", "nonzero_export_cap", MAX_NONZERO_ELEMENTS])
                    nz = nz[:MAX_NONZERO_ELEMENTS]
                if arr.ndim == 0:
                    if float(arr) != 0:
                        writer.writerow(["", float(arr)])
                    return
                if arr.ndim == 1:
                    for i in nz[:, 0].tolist():
                        writer.writerow([i, arr[i]])
                    return
                if arr.ndim == 2:
                    for i, j in nz.tolist():
                        writer.writerow([i, j, arr[i, j]])
                    return
                if arr.ndim == 3:
                    for i, j, k in nz.tolist():
                        writer.writerow([i, j, k, arr[i, j, k]])
                    return
            flat = arr.reshape(-1)
            nz = np.nonzero(flat)[0]
            writer.writerow(["#meta", "nonzero_count", int(nz.size)])
            if nz.size > MAX_NONZERO_ELEMENTS:
                writer.writerow(["#meta", "nonzero_export_cap", MAX_NONZERO_ELEMENTS])
                nz = nz[:MAX_NONZERO_ELEMENTS]
            for i in nz.tolist():
                writer.writerow([i, flat[i]])
            return
        if arr.ndim == 0:
            writer.writerow(["", float(arr)])
            return
        if arr.ndim == 1:
            max_len = arr.shape[0] if arr.size <= MAX_FULL_ELEMENTS else min(arr.shape[0], 5000)
            for i, v in enumerate(arr[:max_len].tolist()):
                writer.writerow([i, v])
            return
        if arr.ndim == 2:
            if arr.size <= MAX_FULL_ELEMENTS:
                max_i, max_j = arr.shape[0], arr.shape[1]
            else:
                max_i = min(arr.shape[0], 400)
                max_j = min(arr.shape[1], 80)
            for i in range(max_i):
                for j in range(max_j):
                    writer.writerow([i, j, arr[i, j]])
            return
        if arr.ndim == 3:
            if arr.size <= MAX_FULL_ELEMENTS:
                max_i, max_j, max_k = arr.shape[0], arr.shape[1], arr.shape[2]
            else:
                max_i = min(arr.shape[0], 80)
                max_j = min(arr.shape[1], 40)
                max_k = min(arr.shape[2], 20)
            for i in range(max_i):
                for j in range(max_j):
                    for k in range(max_k):
                        writer.writerow([i, j, k, arr[i, j, k]])
            return
        flat = arr.reshape(-1)
        if arr.size > MAX_FULL_ELEMENTS:
            flat = flat[:5000]
        for i, v in enumerate(flat.tolist()):
            writer.writerow([i, v])


def dump_spot_energy_sigma_csv(mbg_obj, output_dir="dose_inputs_csv"):
    if not hasattr(mbg_obj, "plan") or not hasattr(mbg_obj.plan, "plan_info"):
        return
    os.makedirs(output_dir, exist_ok=True)
    for bg_id, plan_info in mbg_obj.plan.plan_info.items():
        beams = plan_info.get("beams", {})
        for beam_id, beam in beams.items():
            all_energies = np.array(beam.get("all_energies", []), dtype=np.float64).reshape(-1)
            if all_energies.size == 0:
                continue

            subspot_data = np.array(beam.get("subspot_data", []), dtype=np.float64)
            energy_list = np.array(beam.get("energy_list", []), dtype=np.float64).reshape(-1)

            if subspot_data.ndim == 3 and subspot_data.shape[2] >= 5 and subspot_data.shape[0] > 0:
                sigma_table_x = subspot_data[:, 0, 3]
                sigma_table_y = subspot_data[:, 0, 4]
            else:
                continue

            if energy_list.size == sigma_table_x.size and energy_list.size > 0:
                order = np.argsort(energy_list)
                sorted_energy = energy_list[order]
                sorted_sigma_x = sigma_table_x[order]
                sorted_sigma_y = sigma_table_y[order]
                spot_sigma_x = np.interp(all_energies, sorted_energy, sorted_sigma_x,
                                         left=sorted_sigma_x[0], right=sorted_sigma_x[-1])
                spot_sigma_y = np.interp(all_energies, sorted_energy, sorted_sigma_y,
                                         left=sorted_sigma_y[0], right=sorted_sigma_y[-1])
            else:
                spot_sigma_x = np.full(all_energies.shape, sigma_table_x[0])
                spot_sigma_y = np.full(all_energies.shape, sigma_table_y[0])

            spot_energy_sigmaxy = np.column_stack((all_energies, spot_sigma_x, spot_sigma_y))

            _write_array_csv(
                spot_energy_sigmaxy,
                os.path.join(output_dir, f"bg{bg_id}_beam{beam_id}_spot_energy_sigmaxy.csv"),
                ["i", "j", "value"],
                export_name="spot_energy_sigmaxy",
                original_variable="derived: [beam['all_energies'], interp(beam['subspot_data'][:,0,3/4] by beam['energy_list'])]",
            )
            _write_array_csv(
                spot_sigma_x,
                os.path.join(output_dir, f"bg{bg_id}_beam{beam_id}_spot_sigmax.csv"),
                ["index", "value"],
                export_name="spot_sigmax",
                original_variable="derived from spot_energy_sigmaxy[:,1]",
            )
            _write_array_csv(
                spot_sigma_y,
                os.path.join(output_dir, f"bg{bg_id}_beam{beam_id}_spot_sigmay.csv"),
                ["index", "value"],
                export_name="spot_sigmay",
                original_variable="derived from spot_energy_sigmaxy[:,2]",
            )


def dump_dose_inputs_csv(mbg_obj, output_dir="dose_inputs_csv"):
    if not hasattr(mbg_obj, "plan") or not hasattr(mbg_obj.plan, "plan_info"):
        return
    os.makedirs(output_dir, exist_ok=True)
    dump_required_meta_csv(mbg_obj, output_dir)
    dump_variable_name_meta_csv(output_dir)
    keys_to_dump = [
        "all_energies",
        "water_equivalence",
        "source_pos",
        "beam_xdir",
        "beam_ydir",
        "beam_dir",
        "longitudal_cutoff",
        "energy_list",
        "idd_data",
        "profile_data",
        "idd_setting",
        "profile_setting",
        "beam_para_data",
        "subspot_data",
        "layer_info",
        "layer_energy",
        "sad",
        "npermu",
        "number_particle",
        "idbeamxy",
    ]
    for bg_id, plan_info in mbg_obj.plan.plan_info.items():
        beams = plan_info.get("beams", {})
        for beam_id, beam in beams.items():
            for key in keys_to_dump:
                if key not in beam:
                    continue
                csv_path = os.path.join(output_dir, f"bg{bg_id}_beam{beam_id}_{key}.csv")
                arr = beam[key]
                if key in ("source_pos", "beam_xdir", "beam_ydir"):
                    _write_array_csv(np.array(arr).reshape(-1), csv_path, ["index", "value"], export_name=key, original_variable=f"beam['{key}']")
                elif key == "beam_dir":
                    _write_array_csv(np.array(arr).reshape(-1), csv_path, ["index", "value"], export_name=key, original_variable=f"beam['{key}']")
                else:
                    ndim = np.array(arr).ndim
                    if ndim == 0:
                        _write_array_csv(arr, csv_path, ["index", "value"], export_name=key, original_variable=f"beam['{key}']")
                    elif ndim == 1:
                        _write_array_csv(arr, csv_path, ["index", "value"], export_name=key, original_variable=f"beam['{key}']")
                    elif ndim == 2:
                        _write_array_csv(arr, csv_path, ["i", "j", "value"], export_name=key, original_variable=f"beam['{key}']")
                    elif ndim == 3:
                        _write_array_csv(arr, csv_path, ["i", "j", "k", "value"], export_name=key, original_variable=f"beam['{key}']")
                    else:
                        _write_array_csv(np.array(arr).reshape(-1), csv_path, ["index", "value"], export_name=key, original_variable=f"beam['{key}']")


dump_dose_inputs_csv(mbg_obj)
dump_spot_energy_sigma_csv(mbg_obj)


def dump_water_equivalence_bin(mbg_obj, output_dir="water_equivalence_bin"):
    if not hasattr(mbg_obj, "plan") or not hasattr(mbg_obj.plan, "plan_info"):
        return
    os.makedirs(output_dir, exist_ok=True)
    meta_path = os.path.join(output_dir, "water_equivalence_meta.csv")
    with open(meta_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "beam_group_id",
            "beam_id",
            "source",
            "dtype",
            "total_len",
            "header_len",
            "data_len_excluding_header",
            "n_step",
            "y0",
            "dy",
            "ny",
            "x0",
            "dx",
            "nx",
            "full_bin_path",
            "data_only_bin_path",
        ])
        for bg_id, plan_info in mbg_obj.plan.plan_info.items():
            beams = plan_info.get("beams", {})
            original_beams = {}
            pickle_path = plan_info.get("beam_group_config", {}).get("picklePath", "")
            if pickle_path and os.path.isfile(pickle_path):
                try:
                    with open(pickle_path, "rb") as pf:
                        loaded = pickle.load(pf)
                    original_beams = loaded.get("beams", {}) if isinstance(loaded, dict) else {}
                except Exception as e:
                    print(f"Failed to load original pickle for beam group {bg_id}: {e}")

            for beam_id, beam in beams.items():
                arr = None
                source = ""

                if "water_equivalence" in beam:
                    candidate = np.asarray(beam["water_equivalence"], dtype=np.float32).reshape(-1)
                    if candidate.size > 0:
                        arr = candidate
                        source = "runtime"

                if arr is None and original_beams:
                    original_beam = original_beams.get(beam_id)
                    if original_beam is None:
                        original_beam = original_beams.get(str(beam_id))
                    if original_beam is None:
                        try:
                            original_beam = original_beams.get(int(beam_id))
                        except Exception:
                            original_beam = None
                    if isinstance(original_beam, dict) and "water_equivalence" in original_beam:
                        candidate = np.asarray(original_beam["water_equivalence"], dtype=np.float32).reshape(-1)
                        if candidate.size > 0:
                            arr = candidate
                            source = "pickle_original"

                if arr is None:
                    print(f"Beam {beam_id} in beam group {bg_id} does not have water equivalence data in runtime or original pickle.")
                    continue

                full_name = f"bg{bg_id}_beam{beam_id}_water_equivalence_full.bin"
                full_path = os.path.join(output_dir, full_name)
                arr.tofile(full_path)

                header_len = 9 if arr.size >= 9 else 0
                data_only = arr[header_len:]
                data_only_name = f"bg{bg_id}_beam{beam_id}_water_equivalence_data_only.bin"
                data_only_path = os.path.join(output_dir, data_only_name)
                data_only.tofile(data_only_path)

                meta = _parse_rayweq_header(arr) if header_len == 9 else None
                writer.writerow([
                    bg_id,
                    beam_id,
                    source,
                    "float32",
                    int(arr.size),
                    header_len,
                    int(data_only.size),
                    meta["rayweq_n_step"] if meta else "",
                    meta["rayweq_y0"] if meta else "",
                    meta["rayweq_dy"] if meta else "",
                    meta["rayweq_ny"] if meta else "",
                    meta["rayweq_x0"] if meta else "",
                    meta["rayweq_dx"] if meta else "",
                    meta["rayweq_nx"] if meta else "",
                    full_name,
                    data_only_name,
                ])


dump_water_equivalence_bin(mbg_obj)


def dump_water_equivalence_original_bin(mbg_obj, output_dir="water_equivalence_original_bin"):
    if not hasattr(mbg_obj, "plan") or not hasattr(mbg_obj.plan, "plan_info"):
        return
    os.makedirs(output_dir, exist_ok=True)
    meta_path = os.path.join(output_dir, "water_equivalence_original_meta.csv")
    with open(meta_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "beam_group_id",
            "beam_id",
            "pickle_path",
            "dtype",
            "total_len",
            "header_len",
            "data_len_excluding_header",
            "n_step",
            "y0",
            "dy",
            "ny",
            "x0",
            "dx",
            "nx",
            "full_bin_path",
            "data_only_bin_path",
        ])

        for bg_id, plan_info in mbg_obj.plan.plan_info.items():
            pickle_path = plan_info.get("beam_group_config", {}).get("picklePath", "")
            if not pickle_path or not os.path.isfile(pickle_path):
                print(f"Beam group {bg_id} missing valid pickle path: {pickle_path}")
                continue

            try:
                with open(pickle_path, "rb") as pf:
                    loaded = pickle.load(pf)
            except Exception as e:
                print(f"Failed to read pickle for beam group {bg_id}: {e}")
                continue

            beams = loaded.get("beams", {}) if isinstance(loaded, dict) else {}
            if not isinstance(beams, dict):
                print(f"Beam group {bg_id} pickle does not contain a valid beams dict.")
                continue

            for beam_id, beam in beams.items():
                if not isinstance(beam, dict) or "water_equivalence" not in beam:
                    continue
                arr = np.asarray(beam["water_equivalence"], dtype=np.float32).reshape(-1)
                if arr.size == 0:
                    continue

                full_name = f"bg{bg_id}_beam{beam_id}_water_equivalence_original_full.bin"
                full_path = os.path.join(output_dir, full_name)
                arr.tofile(full_path)

                header_len = 9 if arr.size >= 9 else 0
                data_only = arr[header_len:]
                data_only_name = f"bg{bg_id}_beam{beam_id}_water_equivalence_original_data_only.bin"
                data_only_path = os.path.join(output_dir, data_only_name)
                data_only.tofile(data_only_path)

                meta = _parse_rayweq_header(arr) if header_len == 9 else None
                writer.writerow([
                    bg_id,
                    beam_id,
                    pickle_path,
                    "float32",
                    int(arr.size),
                    header_len,
                    int(data_only.size),
                    meta["rayweq_n_step"] if meta else "",
                    meta["rayweq_y0"] if meta else "",
                    meta["rayweq_dy"] if meta else "",
                    meta["rayweq_ny"] if meta else "",
                    meta["rayweq_x0"] if meta else "",
                    meta["rayweq_dx"] if meta else "",
                    meta["rayweq_nx"] if meta else "",
                    full_name,
                    data_only_name,
                ])


dump_water_equivalence_original_bin(mbg_obj)


def _write_array_csv_full(array, csv_path, export_name="", original_variable=""):
    arr = np.asarray(array)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        if export_name:
            writer.writerow(["#meta", "export_name", export_name])
        if original_variable:
            writer.writerow(["#meta", "original_variable", original_variable])
        writer.writerow(["#meta", "shape", str(tuple(arr.shape))])
        writer.writerow(["#meta", "ndim", int(arr.ndim)])
        writer.writerow(["#meta", "size", int(arr.size)])
        writer.writerow(["#meta", "export_mode", "full_no_truncation"])

        if arr.ndim == 0:
            writer.writerow(["index", "value"])
            writer.writerow([0, float(arr)])
            return
        if arr.ndim == 1:
            writer.writerow(["index", "value"])
            for i, v in enumerate(arr.tolist()):
                writer.writerow([i, v])
            return
        if arr.ndim == 2:
            writer.writerow(["i", "j", "value"])
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    writer.writerow([i, j, arr[i, j]])
            return
        if arr.ndim == 3:
            writer.writerow(["i", "j", "k", "value"])
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    for k in range(arr.shape[2]):
                        writer.writerow([i, j, k, arr[i, j, k]])
            return

        flat = arr.reshape(-1)
        writer.writerow(["index", "value"])
        for i, v in enumerate(flat.tolist()):
            writer.writerow([i, v])


def dump_dosecal_wrapper_ready_inputs(mbg_obj, output_dir="dosecal_wrapper_inputs"):
    if not hasattr(mbg_obj, "plan") or not hasattr(mbg_obj.plan, "plan_info"):
        return
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.csv")
    support_matrix = {}
    if rtdSupportMatrix is not None:
        try:
            raw_support = dict(rtdSupportMatrix())
            support_matrix = {str(k): dict(v) for k, v in raw_support.items()}
        except Exception as e:
            print(f"Failed to query RTD support matrix: {e}")
            support_matrix = {}
    if support_matrix:
        with open(os.path.join(output_dir, "rtd_support_matrix.json"), "w") as sf:
            json.dump(support_matrix, sf, indent=2, ensure_ascii=False)

    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "beam_group_id",
            "beam_id",
            "dosecal_field",
            "wrapper_field",
            "source",
            "dtype",
            "shape",
            "file_path",
            "note",
        ])

        for bg_id, plan_info in mbg_obj.plan.plan_info.items():
            beams = plan_info.get("beams", {})
            pickle_path = plan_info.get("beam_group_config", {}).get("picklePath", "")
            pickle_beams = {}
            if pickle_path and os.path.isfile(pickle_path):
                try:
                    with open(pickle_path, "rb") as pf:
                        loaded = pickle.load(pf)
                    pickle_beams = loaded.get("beams", {}) if isinstance(loaded, dict) else {}
                except Exception as e:
                    print(f"Failed to load pickle for bg {bg_id}: {e}")

            for beam_id, beam in beams.items():
                base = f"bg{bg_id}_beam{beam_id}"

                original_beam = pickle_beams.get(beam_id)
                if original_beam is None:
                    original_beam = pickle_beams.get(str(beam_id))
                if original_beam is None:
                    try:
                        original_beam = pickle_beams.get(int(beam_id))
                    except Exception:
                        original_beam = None
                if not isinstance(original_beam, dict):
                    original_beam = {}

                requested_nuclear = beam.get(
                    "nuclear_correction",
                    plan_info.get("dose_cal_config", {}).get("nuclear_correction", False),
                )
                requested_nuclear = bool(requested_nuclear)
                nuclear_flag_file = f"{base}_nuclear_correction.txt"
                with open(os.path.join(output_dir, nuclear_flag_file), "w") as nf:
                    nf.write("1\n" if requested_nuclear else "0\n")
                writer.writerow([
                    bg_id,
                    beam_id,
                    "nuclear_correction",
                    "nuclearCorrection",
                    "runtime_or_plan_config",
                    "bool",
                    "(1,)",
                    nuclear_flag_file,
                    "requested RTD halo runtime flag for final-dose path",
                ])

                final_support = dict(support_matrix.get("cuFinalDose", {})) if support_matrix else {}
                if final_support:
                    for key in (
                        "nuclear_correction_build_mode",
                        "nuclear_correction_runtime",
                        "nuclear_correction_detail",
                    ):
                        value = str(final_support.get(key, ""))
                        out_name = f"{base}_{key}.txt"
                        with open(os.path.join(output_dir, out_name), "w") as sf:
                            sf.write(value + "\n")
                        writer.writerow([
                            bg_id,
                            beam_id,
                            f"rtd_support.{key}",
                            key,
                            "rtdSupportMatrix",
                            "str",
                            "(1,)",
                            out_name,
                            "RTD final-dose support metadata captured at export time",
                        ])

                def pick_field(field_name):
                    if field_name in beam and np.size(beam[field_name]) > 0:
                        return beam[field_name], "runtime"
                    if field_name in original_beam and np.size(original_beam[field_name]) > 0:
                        return original_beam[field_name], "pickle_original"
                    return None, "missing"

                # 1) water_equivalence: keep full 1D layout, including 9-header prefix.
                water_equivalence, src = pick_field("water_equivalence")
                if water_equivalence is not None:
                    arr = np.asarray(water_equivalence, dtype=np.float32).reshape(-1)
                    full_name = f"{base}_water_equivalence_full.bin"
                    data_only_name = f"{base}_water_equivalence_data_only.bin"
                    arr.tofile(os.path.join(output_dir, full_name))
                    header_len = 9 if arr.size >= 9 else 0
                    arr[header_len:].tofile(os.path.join(output_dir, data_only_name))
                    writer.writerow([
                        bg_id, beam_id, "water_equivalence", "waterEquivalence", src,
                        "float32", str(tuple(arr.shape)), full_name,
                        "full 1D rayweq with header[0:9]; do not truncate",
                    ])
                    writer.writerow([
                        bg_id, beam_id, "water_equivalence_data_only", "waterEquivalence(body)", src,
                        "float32", str(tuple(arr[header_len:].shape)), data_only_name,
                        "header removed for tools that consume body only",
                    ])

                # 2) subspot_data: canonical source for wrapper subspotData.
                subspot_data, src = pick_field("subspot_data")
                if subspot_data is not None:
                    arr = np.asarray(subspot_data, dtype=np.float32)
                    subspot_csv = f"{base}_subspot_data.csv"
                    subspot_bin = f"{base}_subspot_data.bin"
                    _write_array_csv_full(arr, os.path.join(output_dir, subspot_csv),
                                          export_name="subspot_data",
                                          original_variable="beam['subspot_data']")
                    arr.tofile(os.path.join(output_dir, subspot_bin))
                    writer.writerow([
                        bg_id, beam_id, "subspot_data", "subspotData", src,
                        "float32", str(tuple(arr.shape)), subspot_csv,
                        "primary source; full CSV no truncation",
                    ])
                    writer.writerow([
                        bg_id, beam_id, "subspot_data", "subspotData(flat)", src,
                        "float32", str((arr.size,)), subspot_bin,
                        "flatten in C-order: layer-major, subspot-major, channel-major",
                    ])

                    max_subspots = int(arr.shape[1]) if arr.ndim >= 2 else 0
                    max_subspots_file = f"{base}_maxSubspotsPerLayer.txt"
                    with open(os.path.join(output_dir, max_subspots_file), "w") as sf:
                        sf.write(str(max_subspots) + "\n")
                    writer.writerow([
                        bg_id, beam_id, "subspot_data.shape[1]", "maxSubspotsPerLayer", src,
                        "int32", "(1,)", max_subspots_file,
                        "derived from subspot_data second dimension",
                    ])

                field_map = [
                    ("source_pos", "sourcePosition", np.float32),
                    ("beam_xdir", "bmXDirection", np.float32),
                    ("beam_ydir", "bmYDirection", np.float32),
                    ("beam_dir", "spotBeamDirections", np.float32),
                    ("layer_info", "layerSpotCounts", np.int32),
                    ("layer_energy", "energies", np.float32),
                    ("number_particle", "spotWeights", np.float32),
                    ("idbeamxy", "spotPositions", np.float32),
                    ("sad", "sad", np.float32),
                    ("energy_list", "energyList", np.float32),
                ]

                for field_name, wrapper_name, dtype in field_map:
                    value, src = pick_field(field_name)
                    if value is None:
                        continue
                    arr = np.asarray(value, dtype=dtype)
                    out_name = f"{base}_{field_name}.bin"
                    arr.tofile(os.path.join(output_dir, out_name))
                    writer.writerow([
                        bg_id,
                        beam_id,
                        field_name,
                        wrapper_name,
                        src,
                        str(np.dtype(dtype)),
                        str(tuple(arr.shape)),
                        out_name,
                        "",
                    ])

                # number_particle fallback: npermu * weight_vector
                if "number_particle" not in beam and "number_particle" not in original_beam:
                    if "npermu" in beam and "weight_vector" in beam:
                        npart = np.asarray(beam["npermu"], dtype=np.float32) * np.asarray(beam["weight_vector"], dtype=np.float32)
                        out_name = f"{base}_number_particle.bin"
                        npart.tofile(os.path.join(output_dir, out_name))
                        writer.writerow([
                            bg_id,
                            beam_id,
                            "number_particle",
                            "spotWeights",
                            "runtime_derived",
                            "float32",
                            str(tuple(npart.shape)),
                            out_name,
                            "derived as npermu * weight_vector",
                        ])


dump_dosecal_wrapper_ready_inputs(mbg_obj)


def dump_roi_index_csv(mbg_obj, output_dir="roi_index_csv"):
    if not hasattr(mbg_obj, "plan") or not hasattr(mbg_obj.plan, "plan_info"):
        return
    os.makedirs(output_dir, exist_ok=True)
    for bg_id, plan_info in mbg_obj.plan.plan_info.items():
        ext_linear = plan_info.get("ext_contour_linear")
        if ext_linear is None and hasattr(mbg_obj, "ext_contour_linear"):
            ext_linear = mbg_obj.ext_contour_linear
        if ext_linear is not None:
            csv_path = os.path.join(output_dir, f"bg{bg_id}_ext_contour_linear.csv")
            _write_array_csv(np.array(ext_linear), csv_path, ["index", "linear"], export_name="ext_contour_linear", original_variable="ct.tissues[body]['VOI_indices_linear']")

        ext_linear_opt = plan_info.get("ext_contour_linear_opt")
        if ext_linear_opt is None and hasattr(mbg_obj, "ext_contour_linear_opt"):
            ext_linear_opt = mbg_obj.ext_contour_linear_opt
        if ext_linear_opt is not None:
            csv_path = os.path.join(output_dir, f"bg{bg_id}_ext_contour_linear_opt.csv")
            _write_array_csv(np.array(ext_linear_opt), csv_path, ["index", "linear"], export_name="ext_contour_linear_opt", original_variable="ct.tissues[body]['VOI_indices_linear_opt']")


dump_roi_index_csv(mbg_obj)


def dump_final_dose_csv(mbg_obj, output_dir="dose_final_csv"):
    if not hasattr(mbg_obj, "plan") or not hasattr(mbg_obj.plan, "plan_info"):
        return
    os.makedirs(output_dir, exist_ok=True)
    for bg_id, plan_info in mbg_obj.plan.plan_info.items():
        beams = plan_info.get("beams", {})
        for beam_id, beam in beams.items():
            if "final_dose" not in beam:
                continue
            csv_path = os.path.join(output_dir, f"bg{bg_id}_beam{beam_id}_final_dose.csv")
            _write_array_csv(beam["final_dose"], csv_path, ["i", "j", "k", "value"], nonzero_only=True, export_name="final_dose", original_variable="beam['final_dose']")
            raw_bin_path = os.path.join(output_dir, f"bg{bg_id}_beam{beam_id}_final_dose.bin")
            np.asarray(beam["final_dose"], dtype=np.float32).tofile(raw_bin_path)
            layout_path = os.path.join(output_dir, f"bg{bg_id}_beam{beam_id}_final_dose.layout.txt")
            arr = np.asarray(beam["final_dose"], dtype=np.float32)
            with open(layout_path, "w", newline="") as f:
                f.write("array_name=final_dose\n")
                f.write("shape=(Nx,Ny,Nz)\n")
                f.write(f"dims={tuple(int(v) for v in arr.shape)}\n")
                f.write("dtype=float32\n")
                f.write("flat_order=NumPy C-order\n")
                f.write("flat_index=((x*Ny)+y)*Nz+z\n")
                f.write("note=This raw export is x-major Python-facing layout, not wrapper-native (z,y,x) payload order.\n")


dump_final_dose_csv(mbg_obj)


def dump_ct_grid_bin(mbg_obj, output_dir="ct_grid_bin"):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "ct_grid_meta.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "name",
            "shape",
            "dims_x",
            "dims_y",
            "dims_z",
            "resolution_x",
            "resolution_y",
            "resolution_z",
            "corner_x",
            "corner_y",
            "corner_z",
            "patient_position",
            "bin_path",
            "note",
        ])

        if hasattr(mbg_obj, "ct") and hasattr(mbg_obj.ct, "ctgrid") and hasattr(mbg_obj.ct.ctgrid, "data"):
            arr = np.asarray(mbg_obj.ct.ctgrid.data, dtype=np.float32)
            bin_path = os.path.join(output_dir, "ctgrid_data.bin")
            arr.tofile(bin_path)
            dims = np.array(mbg_obj.ct.ctgrid.dims).reshape(-1)
            res = np.array(mbg_obj.ct.ctgrid.resolution).reshape(-1)
            corner = np.array(mbg_obj.ct.ctgrid.corner).reshape(-1)
            writer.writerow([
                "ctgrid_data",
                str(tuple(arr.shape)),
                int(dims[0]), int(dims[1]), int(dims[2]),
                float(res[0]), float(res[1]), float(res[2]),
                float(corner[0]), float(corner[1]), float(corner[2]),
                getattr(mbg_obj.ct, "patientPosition", ""),
                "ctgrid_data.bin",
                "CT grid after makeCTGrid (usually SPR in CT resolution)",
            ])

        if hasattr(mbg_obj, "dc") and hasattr(mbg_obj.dc, "doseGrid") and hasattr(mbg_obj.dc.doseGrid, "data"):
            arr = np.asarray(mbg_obj.dc.doseGrid.data, dtype=np.float32)
            bin_path = os.path.join(output_dir, "dosegrid_data.bin")
            arr.tofile(bin_path)
            dims = np.array(mbg_obj.dc.doseGrid.dims).reshape(-1)
            res = np.array(mbg_obj.dc.doseGrid.resolution).reshape(-1)
            corner = np.array(mbg_obj.dc.doseGrid.corner).reshape(-1)
            writer.writerow([
                "dosegrid_data",
                str(tuple(arr.shape)),
                int(dims[0]), int(dims[1]), int(dims[2]),
                float(res[0]), float(res[1]), float(res[2]),
                float(corner[0]), float(corner[1]), float(corner[2]),
                getattr(mbg_obj.ct, "patientPosition", ""),
                "dosegrid_data.bin",
                "Dose grid used for WEQ and dose kernels",
            ])


dump_ct_grid_bin(mbg_obj)
end = time.time()
print(end - start)
# import argparse
# parser = argparse.ArgumentParser(prog='TPS')
# parser.add_argument('filename')
# parser.add_argument('-e', '--exceute_type', default="dose", choices=["dose", "commission"])
# args = parser.parse_args()

# if(args.exceute_type=="dose"):
#     from content.dose_operator.multiBeamGroups import MBG

#     config_dict = dict()
#     config_dict["gpu_id"] = 1

#     mbg_obj = MBG(False)
#     mbg_obj.load_model(config_dict)

#     import json
#     # with open("tps_ffsbz.json") as f:
#     with open(args.filename) as f:
#     # with open("initConfig.json") as f:
#         content_dict = json.load(f)
#     content_dict['task_id'] = '123'

#     mbg_obj.process(content_dict=content_dict)
# if(args.exceute_type=="commission"):
#     from content.bm_operator.commisionBeamModel import COMMISION_BEAM_MODEL

#     cbm = COMMISION_BEAM_MODEL()
#     import json
#     with open(args.filename) as f:
#         content_dict = json.load(f)
#     content_dict['task_id'] = '123'

#     cbm.process(content_dict=content_dict)
