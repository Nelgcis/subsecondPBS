"""
Test script demonstrating how to call cuFinalDose with CarbonPBS-style variables.

This shows the equivalent of the CarbonPBS data structures as numpy arrays,
matching the variable names used in CarbonPBS code.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/gadolinite/CASHIM_HL/subsecond/raytracedicom_pybind_stage/patch10_mod/build')

try:
    import raytracedicom_pybind as rtd
except ImportError as e:
    print(f"Failed to import raytracedicom_pybind: {e}")
    print("Make sure the module is compiled and in the build directory")
    sys.exit(1)


def create_test_data():
    """
    Create test data matching CarbonPBS variable names and structure.
    
    These correspond to the variables used in carbonPBS/cudaCalDose.cpp
    """
    
    # =====================================================================
    # CT Data (equivalent to ct_data in CarbonPBS)
    # =====================================================================
    # Example: 10x10x10 CT volume with water-equivalent density (1.0 g/cm³)
    ct_dims = np.array([10, 10, 10], dtype=np.int32)  # [Z, Y, X]
    ct_resolution = np.array([0.2, 0.2, 0.2], dtype=np.float32)  # cm per voxel
    ct_corner = np.array([-1.0, -1.0, -1.0], dtype=np.float32)  # origin in cm
    
    # Water-equivalent CT data (all voxels = 1.0 g/cm³)
    ct_data = np.ones(ct_dims, dtype=np.float32, order='C')
    
    # =====================================================================
    # Dose Grid (same as CT grid for simplicity)
    # =====================================================================
    dose_dims = ct_dims.copy()
    dose_resolution = ct_resolution.copy()
    dose_corner = ct_corner.copy()
    
    # Output dose array (will be filled by calculation)
    out_final_dose = np.zeros(dose_dims, dtype=np.float32, order='C')
    
    # =====================================================================
    # WEQ Data (water equivalence header from CarbonPBS)
    # =====================================================================
    # 9-value header as used in CarbonPBS:
    # [0]: ??? (often 0 or 1)
    # [1]: ??? (often 0)
    # [2]: steps (number of integration steps)
    # [3]: ??? (often 0)
    # [4]: ray_spacing_y
    # [5]: ??? (often 0)
    # [6]: ??? (often 0)
    # [7]: ray_spacing_x
    # [8]: ??? (often 0)
    weq_header = np.array([1.0, 0.0, 100.0, 0.0, 0.2, 0.0, 0.0, 0.2, 0.0], dtype=np.float32)
    
    # Full weq_data: header + flattened CT (or separate density map)
    # For now, just use the header + placeholder
    weq_data = np.concatenate([
        weq_header,
        np.ones(100, dtype=np.float32)  # Placeholder LUT data
    ])
    
    # =====================================================================
    # ROI Index (linear indices of voxels to calculate dose)
    # =====================================================================
    # Calculate dose for all voxels in the grid
    total_voxels = int(np.prod(ct_dims))
    roi_index = np.arange(total_voxels, dtype=np.int32)
    
    # Alternatively, select a subset (e.g., central region)
    # z_center, y_center, x_center = ct_dims[0]//2, ct_dims[1]//2, ct_dims[2]//2
    # roi_index = np.array([z_center * ct_dims[1] * ct_dims[2] + 
    #                       y_center * ct_dims[2] + x_center], dtype=np.int32)
    
    # =====================================================================
    # Beam Geometry (CarbonPBS style)
    # =====================================================================
    source_position = np.array([[0.0, 0.0, -50.0]], dtype=np.float32)  # Source at -50cm Z
    beam_direction = np.array([[0.0, 0.0, 1.0]] * 2601, dtype=np.float32)  # All beams along +Z
    beam_xdir = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # X direction
    beam_ydir = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # Y direction
    sad = np.float64(50.0)  # Source-to-axis distance in cm
    
    # =====================================================================
    # Energy Layers (one energy layer per depth)
    # =====================================================================
    num_layers = 10
    layer_energy = np.linspace(200, 400, num_layers, dtype=np.float32)  # MeV/u
    
    # Layer info: number of spots per layer
    spots_per_layer = 260  # Total 2601 spots across 10 layers
    layer_info = np.full(num_layers, spots_per_layer, dtype=np.int32)
    layer_info[-1] = 2601 - (num_layers - 1) * spots_per_layer  # Adjust last layer
    
    # =====================================================================
    # Spot Positions (idbeamxy in CarbonPBS)
    # =====================================================================
    num_spots = 2601
    # Create a 51x51 grid of spots (-5cm to +5cm in X and Y)
    xx, yy = np.meshgrid(np.linspace(-5, 5, 51), np.linspace(-5, 5, 51))
    idbeamxy = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)
    
    assert len(idbeamxy) == num_spots, f"Expected {num_spots} spots, got {len(idbeamxy)}"
    
    # =====================================================================
    # Number of Particles per Beam (spot weights)
    # =====================================================================
    num_particles_per_beam = np.ones(num_spots, dtype=np.float32) * 1e6  # 1 million particles per spot
    
    # =====================================================================
    # Subspot Data (for Gaussian beam sampling)
    # =====================================================================
    # Shape: (num_layers, max_subspots_per_layer, 5)
    # Columns: [x_offset, y_offset, weight, sigma_x, sigma_y]
    max_subspots_per_layer = 10
    subspot_data = np.zeros((num_layers, max_subspots_per_layer, 5), dtype=np.float32)
    
    for i in range(num_layers):
        # Simple single-subspot approximation (all weight on first subspot)
        subspot_data[i, 0, :] = [0.0, 0.0, 1.0, 0.5, 0.5]  # centered, sigma=0.5cm
    
    # =====================================================================
    # IDD Data (Integrated Depth Dose)
    # =====================================================================
    # enelist: energy levels for IDD table
    enelist = np.linspace(100, 500, 50, dtype=np.float32)  # 50 energy points
    
    # idddata: IDD curves for each energy (shape: n_energies × n_samples)
    n_depth_samples = 100
    idddata = np.zeros((len(enelist), n_depth_samples), dtype=np.float32)
    
    # Simulate Bragg peaks (simplified)
    for i, energy in enumerate(enelist):
        # Approximate range in water: R ≈ 0.0022 * E^1.77 (E in MeV/u, R in cm)
        range_cm = 0.0022 * (energy ** 1.77)
        peak_depth_idx = int(range_cm / 0.1)  # Assuming 0.1cm depth bins
        
        if 0 <= peak_depth_idx < n_depth_samples:
            # Gaussian-like Bragg peak
            depth_axis = np.arange(n_depth_samples) * 0.1
            bragg_peak = np.exp(-((depth_axis - range_cm) ** 2) / (2 * 0.3**2))
            idddata[i, :] = bragg_peak
    
    # iddsetting: metadata for IDD table
    # [depth_start, depth_step, n_samples, ...]
    iddsetting = np.array([0.0, 0.1, n_depth_samples, 0, 0, 0, 0, 0], dtype=np.float32)
    
    # =====================================================================
    # Source Energies (optional, matches num_particles_per_beam)
    # =====================================================================
    # Assign energy to each spot based on layer
    source_energies = np.zeros(num_spots, dtype=np.float32)
    spot_count = 0
    for layer_idx in range(num_layers):
        n_spots_in_layer = layer_info[layer_idx]
        source_energies[spot_count:spot_count+n_spots_in_layer] = layer_energy[layer_idx]
        spot_count += n_spots_in_layer
    
    return {
        'ct_data': ct_data,
        'ct_dims': ct_dims,
        'ct_resolution': ct_resolution,
        'ct_corner': ct_corner,
        'out_final_dose': out_final_dose,
        'weq_data': weq_data,
        'roi_index': roi_index,
        'source_energies': source_energies,
        'source': source_position,
        'beam_dir': beam_direction,
        'beam_xdir': beam_xdir,
        'beam_ydir': beam_ydir,
        'sad': sad,
        'enelist': enelist,
        'idddata': idddata,
        'iddsetting': iddsetting,
        'subspot_data': subspot_data,
        'layer_info': layer_info,
        'layer_energy': layer_energy,
        'idbeamxy': idbeamxy,
        'num_particles_per_beam': num_particles_per_beam,
    }


def test_cu_final_dose():
    """
    Test the cuFinalDose function with CarbonPBS-style data.
    
    This mimics the call signature from carbonPBS/cudaCalDose.cpp
    """
    print("=" * 80)
    print("Testing cuFinalDose with CarbonPBS-style variables")
    print("=" * 80)
    
    # Create test data
    data = create_test_data()
    
    print("\nInput shapes:")
    for key, value in data.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: {value.shape} ({value.dtype})")
        else:
            print(f"  {key}: {value} ({type(value).__name__})")
    
    print("\nCalling cu_final_dose...")
    
    try:
        # Call the binding function
        result = rtd.cu_final_dose(
            out_final_dose=data['out_final_dose'],
            ct_data=data['ct_data'],
            weq_data=data['weq_data'],
            roi_index=data['roi_index'],
            source_energies=data['source_energies'],
            source=data['source'],
            beam_dir=data['beam_dir'],
            beam_xdir=data['beam_xdir'],
            beam_ydir=data['beam_ydir'],
            corner=data['ct_corner'],
            resolution=data['ct_resolution'],
            dims=data['ct_dims'],
            longitudinal_cutoff=None,  # Not used yet
            enelist=data['enelist'],
            idddata=data['idddata'],
            iddsetting=data['iddsetting'],
            profiledata=None,  # Not used yet
            subspot_data=data['subspot_data'],
            layer_info=data['layer_info'],
            layer_energy=data['layer_energy'],
            idbeamxy=data['idbeamxy'],
            num_particles_per_beam=data['num_particles_per_beam'],
            sad=data['sad'],
            cutoff=None,  # Not used yet
            beam_para_pos=None,  # Not used yet
            gpu_id=0,
            nuclear_correction=False,
            fine_timing=False,
            tables_dir="/home/gadolinite/CASHIM_HL/subsecond/raytracedicom_pybind_stage/patch10_mod/tables"
        )
        
        print("\n✓ Calculation completed successfully!")
        
        # Analyze results
        dose_result = np.asarray(result)
        print(f"\nOutput dose shape: {dose_result.shape}")
        print(f"Dose statistics:")
        print(f"  Min: {dose_result.min():.6f} Gy")
        print(f"  Max: {dose_result.max():.6f} Gy")
        print(f"  Mean: {dose_result.mean():.6f} Gy")
        print(f"  Sum: {dose_result.sum():.6f} Gy·voxel")
        
        # Check for non-zero dose
        nonzero_voxels = np.count_nonzero(dose_result)
        print(f"  Non-zero voxels: {nonzero_voxels} / {dose_result.size}")
        
        return dose_result
        
    except Exception as e:
        print(f"\n✗ Error during calculation: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    dose = test_cu_final_dose()
    
    # Optional: Save results
    np.save('test_dose_output.npy', dose)
    print("\nDose distribution saved to 'test_dose_output.npy'")
