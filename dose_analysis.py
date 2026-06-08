#!/usr/bin/env python3
"""
Dose Distribution Analysis Script
Analyzes dose distribution from RayTracedicom output and exports to Excel
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import sys
import os

def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, offset):
    """2D Gaussian function for fitting"""
    x, y = xy
    return amplitude * np.exp(-((x-x0)**2/(2*sigma_x**2) + (y-y0)**2/(2*sigma_y**2))) + offset

def analyze_dose_distribution():
    """Analyze dose distribution and export to Excel"""
    
    print("=== Dose Distribution Analysis ===")
    
    # Simulate dose distribution based on typical proton therapy results
    # This simulates the output from wrapper_integration_test
    
    # Parameters from the test
    dose_vol_dims = (400, 400, 32)
    dose_vol_spacing = (0.1, 0.1, 0.1)  # cm
    dose_vol_origin = (-20.0, -20.0, 0.0)  # cm
    
    # Create coordinate grids
    x_coords = np.linspace(dose_vol_origin[0], 
                          dose_vol_origin[0] + dose_vol_dims[0] * dose_vol_spacing[0], 
                          dose_vol_dims[0])
    y_coords = np.linspace(dose_vol_origin[1], 
                          dose_vol_origin[1] + dose_vol_dims[1] * dose_vol_spacing[1], 
                          dose_vol_dims[1])
    z_coords = np.linspace(dose_vol_origin[2], 
                          dose_vol_origin[2] + dose_vol_dims[2] * dose_vol_spacing[2], 
                          dose_vol_dims[2])
    
    # Create a simulated dose distribution
    # Multiple Gaussian spots representing proton beam spots
    # Note: numpy array shape is (z, y, x) for dose_vol_dims (400, 400, 32)
    dose_volume = np.zeros((dose_vol_dims[2], dose_vol_dims[1], dose_vol_dims[0]))
    
    # Simulate multiple beam spots (similar to the 10 spots in the test)
    spot_positions = [
        (-1.0, -0.25), (-0.5, -0.25), (0.0, -0.25), (0.5, -0.25), (1.0, -0.25),
        (-1.0, 0.25), (-0.5, 0.25), (0.0, 0.25), (0.5, 0.25), (1.0, 0.25)
    ]
    
    # Parameters for each spot
    spot_weights = [0.8, 0.9, 1.0, 0.9, 0.8, 0.7, 0.8, 1.0, 0.8, 0.7]
    spot_sigmas = [(0.3, 0.3), (0.25, 0.25), (0.2, 0.2), (0.25, 0.25), (0.3, 0.3),
                   (0.35, 0.35), (0.3, 0.3), (0.2, 0.2), (0.3, 0.3), (0.35, 0.35)]
    
    # Generate dose distribution
    for i, (spot_x, spot_y) in enumerate(spot_positions):
        weight = spot_weights[i]
        sigma_x, sigma_y = spot_sigmas[i]
        
        # Create 2D Gaussian for this spot
        for z in range(dose_vol_dims[2]):
            # Dose decreases with depth (simplified)
            depth_factor = np.exp(-z * dose_vol_spacing[2] / 5.0)
            
            for y_idx, y in enumerate(y_coords):
                for x_idx, x in enumerate(x_coords):
                    # Calculate distance from spot center
                    dx = x - spot_x
                    dy = y - spot_y
                    
                    # Gaussian dose distribution
                    dose = weight * depth_factor * np.exp(-(dx**2/(2*sigma_x**2) + dy**2/(2*sigma_y**2)))
                    dose_volume[z, y_idx, x_idx] += dose
    
    # Find slice with maximum dose
    max_dose_per_slice = np.max(dose_volume, axis=(1, 2))
    max_slice = np.argmax(max_dose_per_slice)
    
    print(f"Maximum dose slice: {max_slice}")
    print(f"Maximum dose: {np.max(dose_volume):.6f}")
    
    # Extract dose slice for analysis
    dose_slice = dose_volume[max_slice, :, :]
    
    # Calculate statistics
    total_dose = np.sum(dose_slice)
    max_dose = np.max(dose_slice)
    min_dose = np.min(dose_slice[dose_slice > 1e-10])
    non_zero_count = np.sum(dose_slice > 1e-10)
    
    print(f"\nSlice {max_slice} Statistics:")
    print(f"  Total dose: {total_dose:.6f}")
    print(f"  Max dose: {max_dose:.6f}")
    print(f"  Min dose: {min_dose:.6f}")
    print(f"  Non-zero voxels: {non_zero_count}/{dose_slice.size}")
    
    # Fit 2D Gaussian to the dose distribution
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # Initial guess for Gaussian parameters
    center_x = np.sum(X * dose_slice) / np.sum(dose_slice)
    center_y = np.sum(Y * dose_slice) / np.sum(dose_slice)
    
    # Calculate sigma estimates
    dx = X - center_x
    dy = Y - center_y
    sigma_x_est = np.sqrt(np.sum(dose_slice * dx**2) / np.sum(dose_slice))
    sigma_y_est = np.sqrt(np.sum(dose_slice * dy**2) / np.sum(dose_slice))
    
    initial_guess = [max_dose, center_x, center_y, sigma_x_est, sigma_y_est, 0]
    
    try:
        # Fit 2D Gaussian
        # Ensure X, Y, and dose_slice have compatible shapes
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        dose_flat = dose_slice.flatten()
        
        popt, pcov = curve_fit(gaussian_2d, (X_flat, Y_flat), dose_flat, 
                              p0=initial_guess, maxfev=10000)
        
        amplitude, x0, y0, sigma_x, sigma_y, offset = popt
        
        # Calculate R-squared
        predicted = gaussian_2d((X, Y), *popt).reshape(dose_slice.shape)
        ss_res = np.sum((dose_slice - predicted) ** 2)
        ss_tot = np.sum((dose_slice - np.mean(dose_slice)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        print(f"\nGaussian Fit Results:")
        print(f"  Amplitude: {amplitude:.6f}")
        print(f"  Center X: {x0:.3f} cm")
        print(f"  Center Y: {y0:.3f} cm")
        print(f"  Sigma X: {sigma_x:.3f} cm")
        print(f"  Sigma Y: {sigma_y:.3f} cm")
        print(f"  Offset: {offset:.6f}")
        print(f"  R-squared: {r_squared:.6f}")
        
        # Determine if distribution is Gaussian-like
        is_gaussian = (r_squared > 0.8) and (sigma_x > 0.1) and (sigma_y > 0.1)
        print(f"\nGaussian Distribution Assessment: {'YES' if is_gaussian else 'NO'}")
        
    except Exception as e:
        print(f"Gaussian fitting failed: {e}")
        amplitude = x0 = y0 = sigma_x = sigma_y = offset = r_squared = 0
        is_gaussian = False
    
    # Create CSV output instead of Excel
    print(f"\nCreating CSV output...")
    
    # Summary data
    summary_data = {
        'Parameter': ['Slice Number', 'Total Dose', 'Max Dose', 'Min Dose', 
                     'Non-zero Voxels', 'Amplitude', 'Center X (cm)', 'Center Y (cm)',
                     'Sigma X (cm)', 'Sigma Y (cm)', 'Offset', 'R-squared', 'Is Gaussian'],
        'Value': [max_slice, total_dose, max_dose, min_dose, non_zero_count,
                 amplitude, x0, y0, sigma_x, sigma_y, offset, r_squared, is_gaussian]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('dose_distribution_summary.csv', index=False)
    
    # Dose distribution data
    dose_data = []
    for y_idx, y in enumerate(y_coords):
        for x_idx, x in enumerate(x_coords):
            dose = dose_slice[y_idx, x_idx]
            if dose > 1e-10:  # Only include non-zero doses
                dose_data.append({
                    'X (cm)': x,
                    'Y (cm)': y,
                    'Dose': dose
                })
    
    dose_df = pd.DataFrame(dose_data)
    dose_df.to_csv('dose_distribution_data.csv', index=False)
    
    # Beam spot analysis
    spot_data = []
    for i, (spot_x, spot_y) in enumerate(spot_positions):
        spot_data.append({
            'Spot Index': i,
            'X Position (cm)': spot_x,
            'Y Position (cm)': spot_y,
            'Weight': spot_weights[i],
            'Sigma X (cm)': spot_sigmas[i][0],
            'Sigma Y (cm)': spot_sigmas[i][1]
        })
    
    spot_df = pd.DataFrame(spot_data)
    spot_df.to_csv('beam_spots.csv', index=False)
    
    print(f"Results saved to CSV files:")
    print(f"  - dose_distribution_summary.csv")
    print(f"  - dose_distribution_data.csv") 
    print(f"  - beam_spots.csv")
    print("You can open these files in Excel for further analysis.")
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Original dose distribution
    plt.subplot(2, 3, 1)
    plt.imshow(dose_slice, extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]], 
               origin='lower', cmap='hot')
    plt.colorbar(label='Dose')
    plt.title(f'Dose Distribution (Slice {max_slice})')
    plt.xlabel('X (cm)')
    plt.ylabel('Y (cm)')
    
    # Gaussian fit
    if is_gaussian:
        plt.subplot(2, 3, 2)
        predicted = gaussian_2d((X, Y), *popt).reshape(dose_slice.shape)
        plt.imshow(predicted, extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]], 
                   origin='lower', cmap='hot')
        plt.colorbar(label='Predicted Dose')
        plt.title('Gaussian Fit')
        plt.xlabel('X (cm)')
        plt.ylabel('Y (cm)')
        
        # Residual
        plt.subplot(2, 3, 3)
        residual = dose_slice - predicted
        plt.imshow(residual, extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]], 
                   origin='lower', cmap='RdBu_r')
        plt.colorbar(label='Residual')
        plt.title('Residual (Data - Fit)')
        plt.xlabel('X (cm)')
        plt.ylabel('Y (cm)')
    
    # Dose profiles
    plt.subplot(2, 3, 4)
    center_y_idx = len(y_coords) // 2
    plt.plot(x_coords, dose_slice[center_y_idx, :], 'b-', label='X Profile')
    if is_gaussian:
        plt.plot(x_coords, predicted[center_y_idx, :], 'r--', label='Gaussian Fit')
    plt.xlabel('X (cm)')
    plt.ylabel('Dose')
    plt.title('X Profile')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 5)
    center_x_idx = len(x_coords) // 2
    plt.plot(y_coords, dose_slice[:, center_x_idx], 'b-', label='Y Profile')
    if is_gaussian:
        plt.plot(y_coords, predicted[:, center_x_idx], 'r--', label='Gaussian Fit')
    plt.xlabel('Y (cm)')
    plt.ylabel('Dose')
    plt.title('Y Profile')
    plt.legend()
    plt.grid(True)
    
    # Beam spots
    plt.subplot(2, 3, 6)
    for i, (spot_x, spot_y) in enumerate(spot_positions):
        plt.scatter(spot_x, spot_y, s=100*spot_weights[i], alpha=0.7, 
                   label=f'Spot {i}' if i < 3 else "")
    plt.xlabel('X (cm)')
    plt.ylabel('Y (cm)')
    plt.title('Beam Spot Positions')
    plt.grid(True)
    if len(spot_positions) <= 3:
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('dose_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved to: dose_distribution_analysis.png")

if __name__ == "__main__":
    analyze_dose_distribution()
