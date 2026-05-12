#!/usr/bin/env python3
"""
Visualize dose distribution from wrapper_integration_test output — NO interpolation.

This script reads:
1. output/dose_distribution.bin - Final dose output from wrapper_integration_test
2. output/test_output_config.txt - Configuration and subspot positions

Displays:
- Up to 12 dose slices at different z depths
- Subspot initial positions overlaid on each slice

Key change from the original:
- Each pixel is rendered as a single colored square on a regular grid (no interpolation),
  using imshow(..., interpolation='nearest') with correct physical extents so that one
  voxel maps to one cell.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import struct
import re
from pathlib import Path

# Display units for dose values read from .bin
UNITS = 'Gy'


def parse_config_file(config_file):
    """Parse test_output_config.txt and extract all parameters."""
    config = {}
    subspot_positions = []

    with open(config_file, 'r') as f:
        content = f.read()

        # Parse DOSE_VOL_DIMS
        match = re.search(r'DOSE_VOL_DIMS = \(([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\)', content)
        if match:
            config['dose_vol_dims'] = (int(float(match.group(1))), int(float(match.group(2))), int(float(match.group(3))))

        # Parse DOSE_VOL_SPACING
        match = re.search(r'DOSE_VOL_SPACING = \(([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\)', content)
        if match:
            config['dose_vol_spacing'] = (float(match.group(1)), float(match.group(2)), float(match.group(3)))

        # Parse DOSE_VOL_ORIGIN
        match = re.search(r'DOSE_VOL_ORIGIN = \(([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\)', content)
        if match:
            config['dose_vol_origin'] = (float(match.group(1)), float(match.group(2)), float(match.group(3)))

        # Parse NUM_LAYERS
        match = re.search(r'NUM_LAYERS = ([\d]+)', content)
        if match:
            config['num_layers'] = int(match.group(1))

        # Parse MAX_SUBSPOTS_PER_LAYER
        match = re.search(r'MAX_SUBSPOTS_PER_LAYER = ([\d]+)', content)
        if match:
            config['max_subspots_per_layer'] = int(match.group(1))

        # Parse ENERGIES
        match = re.search(r'ENERGIES = \[([\d.,\s]+)\]', content)
        if match:
            energies_str = match.group(1)
            config['energies'] = [float(x.strip()) for x in energies_str.split(',')]

        # Parse SAD
        match = re.search(r'SAD = ([-\d.]+)', content)
        if match:
            config['sad'] = float(match.group(1))

        # Parse SOURCE_POS
        match = re.search(r'SOURCE_POS = \[([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\]', content)
        if match:
            config['source_pos'] = np.array([float(match.group(1)), float(match.group(2)), float(match.group(3))])

        # Parse BEAM_DIR
        match = re.search(r'BEAM_DIR = \[([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\]', content)
        if match:
            config['beam_dir'] = np.array([float(match.group(1)), float(match.group(2)), float(match.group(3))])

        # Parse BM_X_DIR
        match = re.search(r'BM_X_DIR = \[([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\]', content)
        if match:
            config['bm_x_dir'] = np.array([float(match.group(1)), float(match.group(2)), float(match.group(3))])

        # Parse BM_Y_DIR
        match = re.search(r'BM_Y_DIR = \[([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\]', content)
        if match:
            config['bm_y_dir'] = np.array([float(match.group(1)), float(match.group(2)), float(match.group(3))])

        # Parse REF_PLANE_Z
        match = re.search(r'REF_PLANE_Z = ([-\d.]+)', content)
        if match:
            config['ref_plane_z'] = float(match.group(1))

        # Parse SUBSPOT_POSITIONS
        match = re.search(r'SUBSPOT_POSITIONS = \[(.*?)\]', content, re.DOTALL)
        if match:
            positions_str = match.group(1)
            pattern = r'\((\d+),\s*(\d+),\s*([-\d.]+),\s*([-\d.]+)\)'
            for m in re.finditer(pattern, positions_str):
                layer_idx = int(m.group(1))
                subspot_idx = int(m.group(2))
                x = float(m.group(3))
                y = float(m.group(4))
                subspot_positions.append((layer_idx, subspot_idx, x, y))

    return config, subspot_positions


def read_dose_distribution(filename):
    """Read dose distribution from binary file."""
    with open(filename, 'rb') as f:
        # Read header
        dims_x = struct.unpack('i', f.read(4))[0]
        dims_y = struct.unpack('i', f.read(4))[0]
        dims_z = struct.unpack('i', f.read(4))[0]
        spacing_x = struct.unpack('f', f.read(4))[0]
        spacing_y = struct.unpack('f', f.read(4))[0]
        spacing_z = struct.unpack('f', f.read(4))[0]
        origin_x = struct.unpack('f', f.read(4))[0]
        origin_y = struct.unpack('f', f.read(4))[0]
        origin_z = struct.unpack('f', f.read(4))[0]

        print(f"Dose volume dimensions: {dims_x} x {dims_y} x {dims_z}")
        print(f"Dose volume spacing: ({spacing_x}, {spacing_y}, {spacing_z}) cm")
        print(f"Dose volume origin: ({origin_x}, {origin_y}, {origin_z}) cm")

        # Read dose data (stored as x, y, z order in memory)
        dose_data = np.frombuffer(f.read(dims_x * dims_y * dims_z * 4), dtype=np.float32)
        dose_data = dose_data.reshape((dims_z, dims_y, dims_x))  # (z, y, x)

        return dose_data, (dims_x, dims_y, dims_z), (spacing_x, spacing_y, spacing_z), (origin_x, origin_y, origin_z)


def _extent_from_origin_spacing(origin, spacing, dims):
    """Compute imshow extent so each pixel is a proper cell centered on its voxel."""
    x_min = origin[0]
    x_max = origin[0] + (dims[0] - 1) * spacing[0]
    y_min = origin[1]
    y_max = origin[1] + (dims[1] - 1) * spacing[1]
    dx = spacing[0]
    dy = spacing[1]
    # Pad by half a voxel to make pixel edges land exactly on grid lines
    return [x_min - dx/2, x_max + dx/2, y_min - dy/2, y_max + dy/2]


def plot_dose_slice(dose_slice, z_idx, z_coord, subspot_positions, origin, spacing, dims, ax, energies):
    """Plot a single dose slice with subspot positions overlaid (no interpolation)."""
    # Normalize
    vmin = float(np.nanmin(dose_slice)) if np.isfinite(dose_slice).any() else 0.0
    vmax = float(np.nanmax(dose_slice)) if np.isfinite(dose_slice).any() else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Render as grid of square cells — no interpolation
    extent = _extent_from_origin_spacing(origin, spacing, dims)
    im = ax.imshow(
        dose_slice,
        origin='lower',               # y grows upward in physical coords
        interpolation='nearest',      # critical: NO interpolation
        extent=extent,
        cmap='hot',
        norm=norm,
        aspect='equal',
    )

    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_title(f'Dose at Z = {z_coord:.2f} cm (slice {z_idx})')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label=f'Dose ({UNITS})')

    # Overlay subspots that lie near this z
    z_tolerance = spacing[2] * 2  # within 2 voxels of this plane
    ref_plane_z = origin[2]       # subspots defined on reference plane ≈ origin z
    if abs(ref_plane_z - z_coord) <= z_tolerance:
        light_colors = ['lightblue', 'lightgreen', 'yellow', 'lightpink', 'lightcyan']
        seen_layer = set()
        for layer_idx, subspot_idx, x, y in subspot_positions:
            color = light_colors[layer_idx % len(light_colors)]
            label = None
            if layer_idx not in seen_layer and 0 <= layer_idx < len(energies):
                label = f'Layer {layer_idx} ({energies[layer_idx]} MeV)'
                seen_layer.add(layer_idx)
            ax.plot(x, y, marker='+', color=color, markersize=4, markeredgewidth=0.8, alpha=0.7, label=label)
        if seen_layer:
            ax.legend(loc='upper right', fontsize=8)


def main():
    # Find files in output directory
    output_dir = Path('output')
    config_file = output_dir / 'test_output_config.txt'
    dose_file = output_dir / 'dose_distribution.bin'

    if not config_file.exists():
        print(f"Error: Could not find {config_file}")
        print("Please run wrapper_integration_test first to generate the configuration file.")
        return

    if not dose_file.exists():
        print(f"Error: Could not find {dose_file}")
        print("Please run wrapper_integration_test first to generate the dose file.")
        return

    print(f"Reading configuration from: {config_file}")
    print(f"Reading dose distribution from: {dose_file}")

    # Parse configuration file
    config, subspot_positions = parse_config_file(config_file)

    print("\nConfiguration loaded:")
    print(f"  Dose volume dims: {config['dose_vol_dims']}")
    print(f"  Dose volume spacing: {config['dose_vol_spacing']}")
    print(f"  Dose volume origin: {config['dose_vol_origin']}")
    print(f"  Number of layers: {config['num_layers']}")
    print(f"  Max subspots per layer: {config['max_subspots_per_layer']}")
    print(f"  Energies: {config['energies']}")

    # Read dose data
    dose_data, dims, spacing, origin = read_dose_distribution(dose_file)

    print(f"\nDose data shape: {dose_data.shape}")
    print(f"Dose range: [{dose_data.min():.6f}, {dose_data.max():.6f}]")
    print(f"Non-zero voxels: {np.count_nonzero(dose_data)} / {dose_data.size}")

    if dose_data.max() == 0:
        print("\nWARNING: All dose values are zero!")
        print("This might indicate:")
        print("  1. The test hasn't been run yet")
        print("  2. There was an issue with dose calculation")
        print("  3. The dose file is from an old run")
        print("\nThe visualization will still be generated, but dose will appear as zero.")

    print(f"\nLoaded {len(subspot_positions)} subspot positions from config file")

    # Select multiple z-slices to display
    num_slices = min(12, dims[2])
    z_indices = np.linspace(0, dims[2] - 1, num_slices, dtype=int) if dims[2] > 1 else [0]
    z_coords = [origin[2] + z_idx * spacing[2] for z_idx in z_indices]

    print(f"\nDisplaying {len(z_indices)} slices at z indices: {z_indices}")
    print(f"Z coordinates: {[f'{z:.2f}' for z in z_coords]} cm")

    # Create figure layout
    num_cols = 4
    num_rows = (len(z_indices) + num_cols - 1) // num_cols
    if num_rows == 1 and len(z_indices) == 1:
        fig, axes = plt.subplots(1, 1, figsize=(10, 10))
        axes = [axes]
    else:
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5 * num_rows))
        if num_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

    for i, (z_idx, z_coord) in enumerate(zip(z_indices, z_coords)):
        dose_slice = dose_data[z_idx, :, :]  # (y, x)
        plot_dose_slice(dose_slice, z_idx, z_coord, subspot_positions, origin, spacing, dims, axes[i], config['energies'])

    # Hide unused subplots
    for i in range(len(z_indices), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    # Save grid of slices
    output_file = 'dose_distribution_slices.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_file}")

    # Also create a reference-plane visualization (if inside bounds)
    fig2, ax2 = plt.subplots(figsize=(10, 10))
    ref_plane_z = config.get('ref_plane_z', origin[2])
    ref_z_idx = int(round((ref_plane_z - origin[2]) / spacing[2]))
    if 0 <= ref_z_idx < dims[2]:
        ref_slice = dose_data[ref_z_idx, :, :]
        vmin = float(np.nanmin(ref_slice)) if np.isfinite(ref_slice).any() else 0.0
        vmax = float(np.nanmax(ref_slice)) if np.isfinite(ref_slice).any() else 1.0
        if vmax <= vmin:
            vmax = vmin + 1.0
        norm2 = Normalize(vmin=vmin, vmax=vmax)
        extent = _extent_from_origin_spacing(origin, spacing, dims)
        im2 = ax2.imshow(
            ref_slice,
            origin='lower',
            interpolation='nearest',
            extent=extent,
            cmap='hot',
            norm=norm2,
            aspect='equal',
        )
        cbar2 = plt.colorbar(im2, ax=ax2, label=f'Dose ({UNITS})')

        light_colors = ['cyan', 'green', 'pink', 'lightpink', 'lightcyan']
        seen_layer = set()
        for layer_idx, subspot_idx, x, y in subspot_positions:
            color = light_colors[layer_idx % len(light_colors)]
            label = None
            if layer_idx not in seen_layer and 0 <= layer_idx < len(config.get('energies', [])):
                label = f'Energy {config["energies"][layer_idx]} MeV'
                seen_layer.add(layer_idx)
            ax2.plot(x, y, marker='+', color=color, markersize=5, markeredgewidth=1, alpha=0.7, label=label)
        if seen_layer:
            ax2.legend(loc='upper right')

        ax2.set_xlabel('X (cm)')
        ax2.set_ylabel('Y (cm)')
        ax2.set_title(f'Dose(Z = {ref_plane_z:.2f} cm)')

        output_file2 = 'dose_distribution_reference_plane.png'
        fig2.savefig(output_file2, dpi=150, bbox_inches='tight')
        print(f"Saved reference plane visualization to: {output_file2}")

    plt.show()


if __name__ == '__main__':
    main()
