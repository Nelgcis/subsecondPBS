#!/usr/bin/env python3
"""
Calculate Z coordinates for dose distribution analysis
"""

import numpy as np

def calculate_z_coordinates():
    """Calculate the actual Z coordinates used in the analysis"""
    
    print("=== Z Coordinate Analysis ===")
    
    # Parameters from wrapper_integration_test
    dose_vol_dims = (400, 400, 32)
    dose_vol_spacing = (0.1, 0.1, 0.1)  # cm
    dose_vol_origin = (-20.0, -20.0, 0.0)  # cm
    
    print(f"Dose Volume Parameters:")
    print(f"  Dimensions: {dose_vol_dims}")
    print(f"  Spacing: {dose_vol_spacing} cm")
    print(f"  Origin: {dose_vol_origin} cm")
    
    # Calculate Z coordinates
    z_coords = np.linspace(dose_vol_origin[2], 
                          dose_vol_origin[2] + dose_vol_dims[2] * dose_vol_spacing[2], 
                          dose_vol_dims[2])
    
    print(f"\nZ Coordinate Range:")
    print(f"  Z min: {z_coords[0]:.3f} cm")
    print(f"  Z max: {z_coords[-1]:.3f} cm")
    print(f"  Z step: {dose_vol_spacing[2]:.3f} cm")
    print(f"  Number of Z slices: {len(z_coords)}")
    
    print(f"\nAll Z Coordinates (cm):")
    for i, z in enumerate(z_coords):
        print(f"  Slice {i:2d}: {z:6.3f} cm")
    
    # Find the slice with maximum dose (slice 0 in our analysis)
    max_slice = 0
    z_at_max_slice = z_coords[max_slice]
    
    print(f"\nAnalysis Results:")
    print(f"  Maximum dose slice: {max_slice}")
    print(f"  Z coordinate at max dose slice: {z_at_max_slice:.3f} cm")
    
    # Calculate depth from surface (assuming surface is at z=0)
    depth_from_surface = z_at_max_slice - dose_vol_origin[2]
    print(f"  Depth from surface: {depth_from_surface:.3f} cm")
    
    # Calculate depth in water equivalent (assuming water density)
    # For protons, 1 cm water ≈ 1 g/cm²
    depth_water_equivalent = depth_from_surface  # Simplified
    print(f"  Water equivalent depth: {depth_water_equivalent:.3f} cm")
    
    return z_coords, z_at_max_slice

if __name__ == "__main__":
    calculate_z_coordinates()
