#!/usr/bin/env python
"""
Create a full X-ray map visualization similar to the reference PDF.

Plots the full COSMOS X-ray map with proper WCS coordinates and styling.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import ZScaleInterval, ImageNormalize, LogStretch
from matplotlib import colors
from astropy.coordinates import SkyCoord
from astropy import units as u
from pathlib import Path
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.transforms import Affine2D
from scipy.ndimage import gaussian_filter

matplotlib.use('Agg')  # Non-interactive backend

# Paths
BASE_DIR = Path('/Users/gozalig1/Projects/compact-groups-xray-analysis')
OUTPUT_DIR = BASE_DIR / 'cosmos-web_galaxy-groups-X-ray-properties' / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# X-ray map files
XRAY_MAP = BASE_DIR / 'data' / 'xray-map' / 'cosmos_chaxmm14_520.fits'
XRAY_ERR = BASE_DIR / 'data' / 'xray-map' / 'cosmos_chaxmm14_520_err.fits'
# Contour map file (smoothed/large-scale map for contours)
# Set to None if not available, or specify the path
CONTOUR_MAP = None  # Will be set if provided
# Contour map file (smoothed/large-scale map for contours)
# Set to None if not available, or specify the path
CONTOUR_MAP = None  # Will be set if provided


def plot_full_xray_map():
    """Create full X-ray map visualization with X-ray contours.
    
    Contours are created by smoothing the main map data (same method as showcase plots).
    """
    print("Loading X-ray maps...")
    
    # Load FITS files
    with fits.open(XRAY_MAP) as hdul:
        data = hdul[0].data.astype(np.float64)
        header = hdul[0].header
        wcs = WCS(header)
    
    with fits.open(XRAY_ERR) as hdul:
        error = hdul[0].data.astype(np.float64)
    
    print(f"Map shape: {data.shape}")
    print(f"Data range: [{np.nanmin(data):.3e}, {np.nanmax(data):.3e}]")
    
    # Create figure with WCS projection
    fig = plt.figure(figsize=(12, 10))
    ax = plt.subplot(projection=wcs)
    
    # Zoom in - crop to center region (remove edges)
    crop_factor = 0.85  # Show 85% of the map (zoom in)
    center_x, center_y = data.shape[1] / 2, data.shape[0] / 2
    width = data.shape[1] * crop_factor
    height = data.shape[0] * crop_factor
    ax.set_xlim(center_x - width/2, center_x + width/2)
    ax.set_ylim(center_y - height/2, center_y + height/2)
    
    # Mask invalid values - keep negative values as they're meaningful (background-subtracted map)
    masked_data = np.ma.masked_invalid(data)
    display_data = np.where(masked_data.mask, np.nan, masked_data.data)
    
    # Use symmetric scaling centered at zero to show both positive and negative values
    # This is scientifically accurate for background-subtracted maps
    valid_data = display_data[~np.isnan(display_data)]
    if len(valid_data) > 0:
        # Use symmetric limits based on the absolute maximum
        abs_max = np.max(np.abs(valid_data))
        vmin, vmax = -abs_max, abs_max
        # Use a small percentile-based limit to avoid extreme outliers dominating the scale
        p99 = np.percentile(np.abs(valid_data), 99)
        vmin, vmax = -p99, p99
    else:
        vmin, vmax = -1e-4, 1e-4
    
    # Use linear stretch (log doesn't work with negative values)
    # Use diverging colormap to clearly show positive (red/yellow) vs negative (blue) regions
    norm = ImageNormalize(display_data, vmin=vmin, vmax=vmax)
    
    # Use 'RdBu_r' (Red-Blue reversed) colormap: blue for negative, red for positive
    # Or keep 'jet' if preferred - but RdBu_r is better for showing sign
    im = ax.imshow(display_data, origin='lower', cmap='RdBu_r', norm=norm, aspect='equal')
    
    # Create X-ray contours using the same method as showcase plots
    # Replace NaNs for contour computation (same as showcase plot)
    data_for_contours = np.array(display_data, copy=True)
    finite_mask = np.isfinite(data_for_contours)
    if np.any(finite_mask):
        median_val = np.nanmedian(data_for_contours[finite_mask])
        data_for_contours[~finite_mask] = median_val
        
        # Apply Gaussian smoothing for contours (same parameters as showcase plot)
        contour_smoothing_sigma = 1.2
        contour_smoothing_for_extended = 2.0
        sigma_extended = contour_smoothing_sigma + contour_smoothing_for_extended
        smoothed_for_contours = gaussian_filter(data_for_contours, sigma=sigma_extended)
        
        # Create contour levels - use 4 levels for cleaner appearance
        smin = np.nanmin(smoothed_for_contours)
        smax = np.nanmax(smoothed_for_contours)
        if np.isfinite(smin) and np.isfinite(smax) and smax > smin:
            # Create 4 evenly spaced contour levels from positive values only
            positive_data = smoothed_for_contours[smoothed_for_contours > 0]
            if len(positive_data) > 0:
                # Use percentiles from 60th to 95th for 4 levels
                contour_percentiles = np.linspace(60, 95, 4)
                contour_levels = np.nanpercentile(positive_data, contour_percentiles)
                contour_levels = np.unique(contour_levels[np.isfinite(contour_levels)])
                eps = max(1e-10 * (smax - smin), 1e-15)
                contour_levels = contour_levels[contour_levels > smin + eps]
            else:
                # Fallback: use evenly spaced levels across the range
                eps = max(1e-10 * (smax - smin), 1e-15)
                contour_levels = np.linspace(smin + eps, smax, 4)
            
            if len(contour_levels) > 0:
                # Overlay contours - use cyan color with lower alpha to see underlying pixels
                contours = ax.contour(smoothed_for_contours, levels=contour_levels, colors='cyan',
                                     linewidths=1.5, alpha=0.2, linestyles='solid', origin='lower', zorder=5)
    
    # Set coordinate labels
    ax.set_xlabel('Right Ascension (J2000)', fontsize=12)
    ax.set_ylabel('Declination (J2000)', fontsize=12)
    
    # Format coordinate ticks - use hour:minute format for RA
    ax.coords[0].set_format_unit('deg')
    ax.coords[1].set_format_unit('deg')
    ax.coords[0].set_major_formatter('hh:mm:ss')
    ax.coords[1].set_major_formatter('dd:mm:ss')
    ax.coords[0].set_ticks(spacing=0.1*u.deg, color='white', size=8)
    ax.coords[1].set_ticks(spacing=0.1*u.deg, color='white', size=8)
    ax.coords[0].set_ticklabel(size=10, color='black')
    ax.coords[1].set_ticklabel(size=10, color='black')
    
    # Add grid - no gridlines for A&A style
    # ax.coords.grid(True, color='white', alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add COSMOS-Web footprint overlay (rotated polygon)
    # Define COSMOS-Web survey area properties
    cosmos_center = SkyCoord(ra="10h00m27.92s", dec="+02d12m03.5s", frame='fk5')
    cosmos_width = 0.54**0.5 * u.deg  # Square root of the area
    cosmos_height = 0.54**0.5 * u.deg  # Square root of the area
    angle = 110  # Orientation angle in degrees
    
    # Convert the center position to pixel coordinates using the WCS
    center_pix = wcs.world_to_pixel(cosmos_center)
    
    # Calculate the half-width and half-height in degrees
    half_width_deg = cosmos_width.to(u.deg).value / 2
    half_height_deg = cosmos_height.to(u.deg).value / 2
    
    # Define the corners of the unrotated rectangle in world coordinates (RA, Dec)
    corners_ra = np.array([cosmos_center.ra.deg - half_width_deg, cosmos_center.ra.deg + half_width_deg,
                           cosmos_center.ra.deg + half_width_deg, cosmos_center.ra.deg - half_width_deg])
    corners_dec = np.array([cosmos_center.dec.deg - half_height_deg, cosmos_center.dec.deg - half_height_deg,
                            cosmos_center.dec.deg + half_height_deg, cosmos_center.dec.deg + half_height_deg])
    
    # Convert the corners from world coordinates to pixel coordinates
    corners_pix = wcs.world_to_pixel(SkyCoord(ra=corners_ra*u.deg, dec=corners_dec*u.deg, frame='fk5'))
    
    # Apply rotation to the rectangle using Affine2D
    rotation_transform = Affine2D().rotate_deg_around(center_pix[0], center_pix[1], angle)
    
    # Transform the corners of the rectangle
    rotated_corners = rotation_transform.transform(np.column_stack([corners_pix[0], corners_pix[1]]))
    
    # Create a polygon representing the COSMOS-Web area
    # Bring to front with high z-order
    # Use black color for COSMOS-Web footprint
    footprint_color = 'black'
    polygon = Polygon(rotated_corners, edgecolor=footprint_color, facecolor='none', linestyle='--', lw=2.5, alpha=0.9, zorder=10)
    ax.add_patch(polygon)
    
    # Add text label at center - also bring to front
    ax.text(center_pix[0], center_pix[1], 'COSMOS-Web', color=footprint_color, fontsize=12, 
            ha='center', va='center', weight='bold', bbox=dict(boxstyle='round,pad=0.3', 
            facecolor='white', alpha=0.8, edgecolor=footprint_color, linewidth=1), zorder=11)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Count Rate (counts s$^{-1}$ pixel$^{-1}$)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    # Add legend for COSMOS-Web footprint
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=footprint_color, linestyle='--', linewidth=2.5, 
                              label='COSMOS-Web footprint')]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
    
    # No title for A&A style
    # ax.set_title('COSMOS X-ray Map (0.5-2.0 keV)', fontsize=14, fontweight='bold', pad=20)
    
    # Tight layout with minimal padding
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.95)
    
    output_path = OUTPUT_DIR / 'xray_map_full.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"Saved: {output_path}")
    plt.close()
    
    # Also create a version with RA/Dec labels on axes instead of WCS
    fig2, ax2 = plt.subplots(figsize=(12, 10))
    
    # Get pixel coordinates
    ny, nx = data.shape
    x = np.arange(nx)
    y = np.arange(ny)
    
    # Convert to world coordinates for extent - zoom in to center region
    crop_factor = 0.85  # Show 85% of the map (zoom in)
    center_x, center_y = nx / 2, ny / 2
    width = nx * crop_factor
    height = ny * crop_factor
    x_min = int(center_x - width/2)
    x_max = int(center_x + width/2)
    y_min = int(center_y - height/2)
    y_max = int(center_y + height/2)
    
    ra_min, dec_min = wcs.pixel_to_world_values(x_min, y_min)
    ra_max, dec_max = wcs.pixel_to_world_values(x_max, y_max)
    
    # Crop the display data to match
    display_data_cropped = display_data[y_min:y_max, x_min:x_max]
    
    # Use same symmetric normalization for simple plot (on cropped data)
    valid_cropped = display_data_cropped[~np.isnan(display_data_cropped)]
    if len(valid_cropped) > 0:
        # Use symmetric limits based on absolute maximum
        abs_max2 = np.max(np.abs(valid_cropped))
        p99_2 = np.percentile(np.abs(valid_cropped), 99)
        vmin2, vmax2 = -p99_2, p99_2
    else:
        vmin2, vmax2 = vmin, vmax
    
    # Use linear stretch with symmetric scaling
    norm2 = ImageNormalize(display_data_cropped, vmin=vmin2, vmax=vmax2)
    
    # Plot with extent in degrees - use 'RdBu_r' colormap (same as main plot)
    im2 = ax2.imshow(display_data_cropped, origin='lower', cmap='RdBu_r', norm=norm2,
                     extent=[ra_min, ra_max, dec_min, dec_max], aspect='equal')
    
    # Create X-ray contours for simple plot (same method as main plot)
    data_for_contours2 = np.array(display_data_cropped, copy=True)
    finite_mask2 = np.isfinite(data_for_contours2)
    if np.any(finite_mask2):
        median_val2 = np.nanmedian(data_for_contours2[finite_mask2])
        data_for_contours2[~finite_mask2] = median_val2
        
        # Apply Gaussian smoothing
        contour_smoothing_sigma = 1.2
        contour_smoothing_for_extended = 2.0
        sigma_extended = contour_smoothing_sigma + contour_smoothing_for_extended
        smoothed_for_contours2 = gaussian_filter(data_for_contours2, sigma=sigma_extended)
        
        # Create contour levels - use 4 levels for cleaner appearance
        smin2 = np.nanmin(smoothed_for_contours2)
        smax2 = np.nanmax(smoothed_for_contours2)
        if np.isfinite(smin2) and np.isfinite(smax2) and smax2 > smin2:
            # Create 4 evenly spaced contour levels from positive values only
            positive_data2 = smoothed_for_contours2[smoothed_for_contours2 > 0]
            if len(positive_data2) > 0:
                # Use percentiles from 60th to 95th for 4 levels
                contour_percentiles2 = np.linspace(60, 95, 4)
                contour_levels2 = np.nanpercentile(positive_data2, contour_percentiles2)
                contour_levels2 = np.unique(contour_levels2[np.isfinite(contour_levels2)])
                eps2 = max(1e-10 * (smax2 - smin2), 1e-15)
                contour_levels2 = contour_levels2[contour_levels2 > smin2 + eps2]
            else:
                # Fallback: use evenly spaced levels across the range
                eps2 = max(1e-10 * (smax2 - smin2), 1e-15)
                contour_levels2 = np.linspace(smin2 + eps2, smax2, 4)
            
            if len(contour_levels2) > 0:
                # Create meshgrid for contour plotting with world coordinates
                ny_crop, nx_crop = smoothed_for_contours2.shape
                x_contour = np.linspace(ra_min, ra_max, nx_crop)
                y_contour = np.linspace(dec_min, dec_max, ny_crop)
                X_contour, Y_contour = np.meshgrid(x_contour, y_contour)
                
                contours2 = ax2.contour(X_contour, Y_contour, smoothed_for_contours2, 
                                        levels=contour_levels2, colors='cyan', 
                                        linewidths=1.5, alpha=0.2, linestyles='solid', zorder=5)
    
    ax2.set_xlabel('Right Ascension (deg)', fontsize=12)
    ax2.set_ylabel('Declination (deg)', fontsize=12)
    # No title for A&A style
    # ax2.set_title('COSMOS X-ray Map (0.5-2.0 keV)', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Count Rate (counts s$^{-1}$ pixel$^{-1}$)', fontsize=12)
    cbar2.ax.tick_params(labelsize=10)
    
    # Format axes
    ax2.tick_params(axis='both', labelsize=10)
    
    # Set exact extent to remove gaps
    ax2.set_xlim(ra_min, ra_max)
    ax2.set_ylim(dec_min, dec_max)
    
    # Add COSMOS-Web footprint overlay (rotated polygon)
    # Define COSMOS-Web survey area properties
    cosmos_center = SkyCoord(ra="10h00m27.92s", dec="+02d12m03.5s", frame='fk5')
    cosmos_width = 0.54**0.5 * u.deg  # Square root of the area
    cosmos_height = 0.54**0.5 * u.deg  # Square root of the area
    angle = 110  # Orientation angle in degrees
    
    # Convert the center position to pixel coordinates using the WCS
    center_pix = wcs.world_to_pixel(cosmos_center)
    
    # Calculate the half-width and half-height in degrees
    half_width_deg = cosmos_width.to(u.deg).value / 2
    half_height_deg = cosmos_height.to(u.deg).value / 2
    
    # Define the corners of the unrotated rectangle in world coordinates (RA, Dec)
    corners_ra = np.array([cosmos_center.ra.deg - half_width_deg, cosmos_center.ra.deg + half_width_deg,
                           cosmos_center.ra.deg + half_width_deg, cosmos_center.ra.deg - half_width_deg])
    corners_dec = np.array([cosmos_center.dec.deg - half_height_deg, cosmos_center.dec.deg - half_height_deg,
                            cosmos_center.dec.deg + half_height_deg, cosmos_center.dec.deg + half_height_deg])
    
    # Convert the corners from world coordinates to pixel coordinates
    corners_pix = wcs.world_to_pixel(SkyCoord(ra=corners_ra*u.deg, dec=corners_dec*u.deg, frame='fk5'))
    
    # Apply rotation to the rectangle using Affine2D
    rotation_transform = Affine2D().rotate_deg_around(center_pix[0], center_pix[1], angle)
    
    # Transform the corners of the rectangle
    rotated_corners_pix = rotation_transform.transform(np.column_stack([corners_pix[0], corners_pix[1]]))
    
    # Convert rotated corners back to world coordinates for plotting on ax2
    rotated_corners_world = wcs.pixel_to_world_values(rotated_corners_pix[:, 0], rotated_corners_pix[:, 1])
    
    # Create polygon in world coordinates (RA, Dec) - bring to front
    # Use black color for COSMOS-Web footprint
    footprint_color = 'black'
    polygon2 = Polygon(np.column_stack([rotated_corners_world[0], rotated_corners_world[1]]), 
                       edgecolor=footprint_color, facecolor='none', linestyle='--', lw=2.5, alpha=0.9, zorder=10)
    ax2.add_patch(polygon2)
    
    # Add text label at center (convert center pixel to world coordinates) - bring to front
    center_world = wcs.pixel_to_world_values(center_pix[0], center_pix[1])
    ax2.text(center_world[0], center_world[1], 'COSMOS-Web', color=footprint_color, fontsize=12, 
             ha='center', va='center', weight='bold', bbox=dict(boxstyle='round,pad=0.3', 
             facecolor='white', alpha=0.8, edgecolor=footprint_color, linewidth=1), zorder=11)
    
    # Add legend for COSMOS-Web footprint
    from matplotlib.lines import Line2D
    legend_elements2 = [Line2D([0], [0], color=footprint_color, linestyle='--', linewidth=2.5, 
                               label='COSMOS-Web footprint')]
    ax2.legend(handles=legend_elements2, loc='upper right', fontsize=10, framealpha=0.9)
    
    # Tight layout with minimal padding
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.95)
    
    output_path2 = OUTPUT_DIR / 'xray_map_full_simple.png'
    plt.savefig(output_path2, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"Saved: {output_path2}")
    plt.close()


if __name__ == '__main__':
    plot_full_xray_map()
