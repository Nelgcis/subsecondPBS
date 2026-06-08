#!/usr/bin/env python3
"""
Analyze actual dose distribution from C++ binary output
Reads dose_distribution.bin and performs Gaussian fit analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import struct
import subprocess
import os

def read_dose_binary(filename):
    """Read dose distribution from binary file created by C++ code"""
    
    if not os.path.exists(filename):
        print(f"Error: {filename} not found. Running wrapper_integration_test first...")
        # Run the C++ test to generate the binary file
        result = subprocess.run(['./build/bin/wrapper_integration_test'], 
                              capture_output=True, text=True, cwd='/root/raytracedicom_updated1011')
        if result.returncode != 0:
            print(f"Error running wrapper_integration_test: {result.stderr}")
            return None, None, None
        
        if not os.path.exists(filename):
            print(f"Error: {filename} still not found after running test")
            return None, None, None
    
    print(f"Reading dose distribution from {filename}...")
    
    with open(filename, 'rb') as f:
        # Read header: dimensions (3 ints), spacing (3 floats), origin (3 floats)
        dims = struct.unpack('3i', f.read(12))
        spacing = struct.unpack('3f', f.read(12))
        origin = struct.unpack('3f', f.read(12))
        
        # Read dose data (all floats)
        dose_data = np.frombuffer(f.read(), dtype=np.float32)
    
    print(f"  Dimensions: {dims[0]} x {dims[1]} x {dims[2]}")
    print(f"  Spacing: ({spacing[0]:.3f}, {spacing[1]:.3f}, {spacing[2]:.3f}) cm")
    print(f"  Origin: ({origin[0]:.3f}, {origin[1]:.3f}, {origin[2]:.3f}) cm")
    print(f"  Total voxels: {len(dose_data)}")
    
    # Reshape to 3D array (x, y, z order as stored in C++)
    dose_volume = dose_data.reshape(dims[2], dims[1], dims[0])  # z, y, x for numpy indexing
    
    # Create coordinate arrays
    x_coords = np.linspace(origin[0], origin[0] + dims[0] * spacing[0], dims[0])
    y_coords = np.linspace(origin[1], origin[1] + dims[1] * spacing[1], dims[1])
    z_coords = np.linspace(origin[2], origin[2] + dims[2] * spacing[2], dims[2])
    
    return dose_volume, (x_coords, y_coords, z_coords), (dims, spacing, origin)

def gaussian_2d(params, X, Y):
    """2D Gaussian function for fitting"""
    A, x0, y0, sigma_x, sigma_y, offset = params
    return A * np.exp(-((X - x0)**2 / (2 * sigma_x**2) + (Y - y0)**2 / (2 * sigma_y**2))) + offset

def fit_gaussian_2d(X, Y, dose_slice):
    """Fit 2D Gaussian to dose distribution"""
    # Find maximum dose position
    max_idx = np.unravel_index(np.argmax(dose_slice), dose_slice.shape)
    x0_guess = X[max_idx]
    y0_guess = Y[max_idx]
    A_guess = dose_slice[max_idx]
    
    # Estimate sigma from FWHM
    half_max = A_guess / 2
    mask = dose_slice > half_max
    if np.sum(mask) > 0:
        sigma_x_guess = np.std(X[mask]) / 2.35  # FWHM to sigma conversion
        sigma_y_guess = np.std(Y[mask]) / 2.35
    else:
        sigma_x_guess = 1.0
        sigma_y_guess = 1.0
    
    initial_guess = [A_guess, x0_guess, y0_guess, sigma_x_guess, sigma_y_guess, 0.0]
    
    # Flatten arrays for curve_fit
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    dose_flat = dose_slice.flatten()
    
    # Remove zero values for better fitting
    non_zero_mask = dose_flat > 1e-10
    if np.sum(non_zero_mask) < 10:
        print("  Warning: Too few non-zero dose values for fitting")
        return None, None
    
    try:
        popt, pcov = curve_fit(gaussian_2d, (X_flat[non_zero_mask], Y_flat[non_zero_mask]), 
                              dose_flat[non_zero_mask], 
                              p0=initial_guess, maxfev=10000)
        return popt, pcov
    except Exception as e:
        print(f"  Warning: Gaussian fit failed: {e}")
        return None, None

def analyze_dose_slice(dose_volume, coords, z_slice_idx):
    """Analyze a specific Z slice of the dose distribution"""
    
    x_coords, y_coords, z_coords = coords
    
    # Extract slice (z_slice_idx corresponds to z coordinate)
    dose_slice = dose_volume[z_slice_idx, :, :]
    z_value = z_coords[z_slice_idx]
    
    print(f"\n=== Analyzing Z slice {z_slice_idx} (z = {z_value:.3f} cm) ===")
    print(f"  Slice dimensions: {dose_slice.shape}")
    print(f"  Max dose: {np.max(dose_slice):.6f}")
    print(f"  Mean dose: {np.mean(dose_slice[dose_slice > 1e-10]):.6f}")
    print(f"  Non-zero voxels: {np.sum(dose_slice > 1e-10)}/{dose_slice.size}")
    
    # Create coordinate grids
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # Fit Gaussian
    print("\n  Fitting 2D Gaussian...")
    popt, pcov = fit_gaussian_2d(X, Y, dose_slice)
    
    if popt is None:
        print("  Could not fit Gaussian")
        return None, z_value
    
    A, x0, y0, sigma_x, sigma_y, offset = popt
    print(f"  Gaussian fit parameters:")
    print(f"    Amplitude: {A:.6f}")
    print(f"    Center (x, y): ({x0:.3f}, {y0:.3f}) cm")
    print(f"    Sigma (x, y): ({sigma_x:.3f}, {sigma_y:.3f}) cm")
    print(f"    Offset: {offset:.6f}")
    
    # Calculate R-squared
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    dose_flat = dose_slice.flatten()
    non_zero_mask = dose_flat > 1e-10
    
    dose_pred = gaussian_2d(popt, X_flat[non_zero_mask], Y_flat[non_zero_mask])
    ss_res = np.sum((dose_flat[non_zero_mask] - dose_pred)**2)
    ss_tot = np.sum((dose_flat[non_zero_mask] - np.mean(dose_flat[non_zero_mask]))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    print(f"  R-squared: {r_squared:.4f}")
    
    return {
        'z': z_value,
        'A': A,
        'x0': x0,
        'y0': y0,
        'sigma_x': sigma_x,
        'sigma_y': sigma_y,
        'offset': offset,
        'r_squared': r_squared,
        'max_dose': np.max(dose_slice),
        'mean_dose': np.mean(dose_slice[dose_slice > 1e-10]),
        'non_zero_voxels': np.sum(dose_slice > 1e-10)
    }, z_value

def main():
    print("=== Dose Distribution Analysis from C++ Output ===\n")
    
    # Read dose distribution
    dose_volume, coords, params = read_dose_binary('dose_distribution.bin')
    
    if dose_volume is None:
        print("Failed to read dose distribution")
        return 1
    
    dims, spacing, origin = params
    x_coords, y_coords, z_coords = coords
    
    print(f"\nZ coordinate range: {z_coords[0]:.3f} to {z_coords[-1]:.3f} cm")
    print(f"Z coordinate step: {spacing[2]:.3f} cm")
    
    # Find slice with maximum dose
    max_dose = np.max(dose_volume)
    max_idx_3d = np.unravel_index(np.argmax(dose_volume), dose_volume.shape)
    max_z_slice = max_idx_3d[0]
    max_z_value = z_coords[max_z_slice]
    
    print(f"\nMaximum dose: {max_dose:.6f}")
    print(f"Location: z = {max_z_value:.3f} cm (slice {max_z_slice})")
    
    # Analyze the slice with maximum dose
    fit_result, z_value = analyze_dose_slice(dose_volume, coords, max_z_slice)
    
    if fit_result is None:
        print("Failed to analyze dose slice")
        return 1
    
    # Save results to CSV
    print("\n=== Saving Results ===")
    
    # Summary
    summary_data = {
        'Parameter': ['Z coordinate (cm)', 'Z slice index', 'Max dose', 'Mean dose', 
                     'Gaussian Amplitude', 'Center X (cm)', 'Center Y (cm)', 
                     'Sigma X (cm)', 'Sigma Y (cm)', 'R-squared', 'Non-zero voxels'],
        'Value': [fit_result['z'], max_z_slice, fit_result['max_dose'], fit_result['mean_dose'],
                 fit_result['A'], fit_result['x0'], fit_result['y0'],
                 fit_result['sigma_x'], fit_result['sigma_y'], fit_result['r_squared'],
                 fit_result['non_zero_voxels']]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('dose_analysis_summary.csv', index=False)
    print("  Saved: dose_analysis_summary.csv")
    
    # Dose distribution data for the slice
    x_coords, y_coords, z_coords = coords
    dose_slice = dose_volume[max_z_slice, :, :]
    X, Y = np.meshgrid(x_coords, y_coords)
    
    dose_data = []
    for i in range(len(y_coords)):
        for j in range(len(x_coords)):
            if dose_slice[i, j] > 1e-10:
                dose_data.append({
                    'X (cm)': X[i, j],
                    'Y (cm)': Y[i, j],
                    'Dose': dose_slice[i, j],
                    'Z (cm)': z_value
                })
    
    dose_df = pd.DataFrame(dose_data)
    dose_df.to_csv('dose_distribution_slice.csv', index=False)
    print(f"  Saved: dose_distribution_slice.csv ({len(dose_data)} non-zero points)")
    
    # Create visualization
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.contourf(X, Y, dose_slice, levels=50, cmap='hot')
    plt.colorbar(label='Dose')
    plt.xlabel('X (cm)')
    plt.ylabel('Y (cm)')
    plt.title(f'Dose Distribution at Z = {z_value:.3f} cm')
    plt.plot(fit_result['x0'], fit_result['y0'], 'bx', markersize=10, label='Gaussian center')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    # Plot Gaussian fit
    dose_fit = gaussian_2d([fit_result['A'], fit_result['x0'], fit_result['y0'], 
                           fit_result['sigma_x'], fit_result['sigma_y'], fit_result['offset']],
                          X, Y)
    plt.contourf(X, Y, dose_fit, levels=50, cmap='hot')
    plt.colorbar(label='Fitted Dose')
    plt.xlabel('X (cm)')
    plt.ylabel('Y (cm)')
    plt.title(f'Gaussian Fit (R² = {fit_result["r_squared"]:.4f})')
    
    plt.tight_layout()
    plt.savefig('dose_analysis.png', dpi=150)
    print("  Saved: dose_analysis.png")
    
    print("\n=== Analysis Complete ===")
    print(f"\nZ coordinate analyzed: {z_value:.3f} cm (slice {max_z_slice})")
    print(f"Gaussian fit R-squared: {fit_result['r_squared']:.4f}")
    if fit_result['r_squared'] > 0.9:
        print("✓ Dose distribution satisfies Gaussian superposition (R² > 0.9)")
    else:
        print("✗ Dose distribution does not satisfy Gaussian superposition (R² < 0.9)")
    
    return 0

if __name__ == "__main__":
    exit(main())
